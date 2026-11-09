//! Owned strict-context state. Every payload allocation is reserved against a
//! shared per-context budget before a fallible Vec reservation. Checkpoints keep
//! their reservations alive; the ledger is not a total-process memory measure.
use super::{SequenceMemory, SequenceState};
use crate::{Error, Model, Result, llama::KvLayer, mamba::RecurrentLayer};
use std::{
    mem::size_of,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidState(message.into())
}
pub(crate) fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| invalid("state allocation byte count overflow"))
}
pub(crate) fn multiply(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("state allocation byte count overflow"))
}

#[derive(Debug)]
pub(crate) struct StateBudget {
    limit: usize,
    used: AtomicUsize,
    peak: AtomicUsize,
}
impl StateBudget {
    pub(crate) fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit,
            used: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        })
    }
    pub(crate) fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }
    pub(crate) fn peak(&self) -> usize {
        self.peak.load(Ordering::Acquire)
    }
    pub(crate) fn limit(&self) -> usize {
        self.limit
    }
    pub(crate) fn reserve(self: &Arc<Self>, bytes: usize) -> Result<Reservation> {
        let mut used = self.used();
        loop {
            let required = add(used, bytes)?;
            if required > self.limit {
                return Err(invalid(format!(
                    "state allocation requires {required} reserved bytes \
                     ({used} live plus {bytes} requested), exceeding state budget {}",
                    self.limit
                )));
            }
            match self.used.compare_exchange_weak(
                used,
                required,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.peak.fetch_max(required, Ordering::Relaxed);
                    return Ok(Reservation {
                        budget: Arc::clone(self),
                        bytes,
                    });
                }
                Err(actual) => used = actual,
            }
        }
    }
}
#[derive(Debug)]
pub(crate) struct Reservation {
    budget: Arc<StateBudget>,
    bytes: usize,
}
impl Reservation {
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
    fn reduce_to(&mut self, bytes: usize) {
        assert!(
            bytes <= self.bytes,
            "state reservation cannot grow unchecked"
        );
        self.budget
            .used
            .fetch_sub(self.bytes - bytes, Ordering::AcqRel);
        self.bytes = bytes;
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

/// Reserve capacity without infallible collection, clone, or resize growth.
pub(crate) fn allocate<T>(count: usize, what: &str) -> Result<Vec<T>> {
    let bytes = multiply(count, size_of::<T>())?;
    if bytes > isize::MAX as usize {
        return Err(invalid(format!(
            "{what} allocation exceeds addressable size"
        )));
    }
    #[cfg(test)]
    if bytes != 0 {
        allocation_fault(what)?;
    }
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| invalid(format!("unable to reserve {bytes} bytes for {what}")))?;
    Ok(result)
}
fn floats(source: Option<&[f32]>, capacity: usize, initialized: usize) -> Result<Vec<f32>> {
    let initial = source.map_or(initialized, <[f32]>::len);
    if initial > capacity {
        return Err(invalid("state copy exceeds its preflight capacity"));
    }
    let mut values = allocate(capacity, "state float vector")?;
    if let Some(source) = source {
        values.extend_from_slice(source);
    } else {
        values.resize(initialized, 0.0);
    }
    Ok(values)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MemoryShape {
    Attention {
        layers: usize,
        width: usize,
    },
    Recurrent {
        layers: usize,
        convolution: usize,
        ssm: usize,
    },
}
impl MemoryShape {
    pub(crate) fn for_model(model: &Model) -> Result<Self> {
        Ok(match model {
            Model::Llama(model) => Self::Attention {
                layers: model.layer_count(),
                width: model.config().kv_width()?,
            },
            Model::Qwen2(model) => Self::Attention {
                layers: model.layer_count(),
                width: model.config().kv_width()?,
            },
            Model::Mamba(model) => {
                let config = model.config();
                Self::Recurrent {
                    layers: model.layer_count(),
                    convolution: multiply(config.inner, config.convolution_width - 1)?,
                    ssm: multiply(config.inner, config.state_size)?,
                }
            }
        })
    }
    pub(crate) fn layers(self) -> usize {
        match self {
            Self::Attention { layers, .. } | Self::Recurrent { layers, .. } => layers,
        }
    }
    pub(crate) fn kind(self) -> u8 {
        match self {
            Self::Attention { .. } => 0,
            Self::Recurrent { .. } => 1,
        }
    }
    pub(crate) fn lengths(self, position: usize) -> Result<(usize, usize)> {
        match self {
            Self::Attention { width, .. } => {
                let length = multiply(width, position)?;
                Ok((length, length))
            }
            Self::Recurrent {
                convolution, ssm, ..
            } => Ok((convolution, ssm)),
        }
    }
    fn bytes(self, position: usize) -> Result<usize> {
        let (first, second) = self.lengths(position)?;
        let floats = multiply(add(first, second)?, size_of::<f32>())?;
        let header = match self {
            Self::Attention { .. } => size_of::<KvLayer>(),
            Self::Recurrent { .. } => size_of::<RecurrentLayer>(),
        };
        multiply(self.layers(), add(header, floats)?)
    }
    fn copy_memory(
        self,
        source: Option<&SequenceState>,
        capacity_position: usize,
    ) -> Result<SequenceMemory> {
        let (first, second) = self.lengths(capacity_position)?;
        match self {
            Self::Attention { layers, .. } => {
                let previous = match source.map(|state| &state.memory) {
                    Some(SequenceMemory::Attention(values)) if values.len() == layers => {
                        Some(values)
                    }
                    None => None,
                    _ => return Err(invalid("attention state layout mismatch")),
                };
                let mut result = allocate(layers, "attention layer records")?;
                for index in 0..layers {
                    let old = previous.map(|values| &values[index]);
                    result.push(KvLayer {
                        keys: floats(old.map(|layer| layer.keys.as_slice()), first, 0)?,
                        values: floats(old.map(|layer| layer.values.as_slice()), second, 0)?,
                    });
                }
                Ok(SequenceMemory::Attention(result))
            }
            Self::Recurrent { layers, .. } => {
                let previous = match source.map(|state| &state.memory) {
                    Some(SequenceMemory::Recurrent(values)) if values.len() == layers => {
                        Some(values)
                    }
                    None => None,
                    _ => return Err(invalid("recurrent state layout mismatch")),
                };
                let mut result = allocate(layers, "recurrent layer records")?;
                for index in 0..layers {
                    let old = previous.map(|values| &values[index]);
                    result.push(RecurrentLayer {
                        convolution: floats(
                            old.map(|layer| layer.convolution.as_slice()),
                            first,
                            first,
                        )?,
                        ssm: floats(old.map(|layer| layer.ssm.as_slice()), second, second)?,
                    });
                }
                Ok(SequenceMemory::Recurrent(result))
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PlanEntry {
    pub(crate) id: i32,
    /// Source sequence for a copy/fork; a missing source creates empty state.
    pub(crate) source: i32,
    pub(crate) position: usize,
}
#[derive(Debug)]
pub(crate) struct StatePlan {
    pub(crate) entries: Vec<PlanEntry>,
    _reservation: Reservation,
}
impl StatePlan {
    pub(crate) fn new(budget: &Arc<StateBudget>, capacity: usize) -> Result<Self> {
        let reservation = budget.reserve(multiply(capacity, size_of::<PlanEntry>())?)?;
        let entries = allocate(capacity, "state allocation plan")?;
        Ok(Self {
            entries,
            _reservation: reservation,
        })
    }
    pub(crate) fn push(&mut self, entry: PlanEntry) -> Result<()> {
        if self.entries.len() == self.entries.capacity() {
            return Err(invalid("state plan exceeds its preflight capacity"));
        }
        self.entries.push(entry);
        Ok(())
    }
    pub(crate) fn sort_and_check(&mut self) -> Result<()> {
        self.entries.sort_unstable_by_key(|entry| entry.id);
        if self.entries.iter().any(|entry| entry.id < 0)
            || self.entries.windows(2).any(|pair| pair[0].id == pair[1].id)
        {
            return Err(invalid("negative or duplicate sequence ID in state plan"));
        }
        Ok(())
    }
}

/// A sorted Vec avoids BTreeMap's infallible per-node allocation. Its capacity,
/// layer records and float capacities are all included in the reservation.
#[derive(Debug)]
pub(crate) struct StateStorage {
    entries: Vec<(i32, SequenceState)>,
    reservation: Reservation,
    pub(crate) shape: MemoryShape,
}
impl StateStorage {
    pub(crate) fn empty(budget: &Arc<StateBudget>, shape: MemoryShape) -> Self {
        Self {
            entries: Vec::new(),
            reservation: Reservation {
                budget: Arc::clone(budget),
                bytes: 0,
            },
            shape,
        }
    }
    pub(crate) fn budget(&self) -> &Arc<StateBudget> {
        &self.reservation.budget
    }
    pub(crate) fn bytes(&self) -> usize {
        self.reservation.bytes()
    }
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&i32, &SequenceState)> {
        self.entries.iter().map(|(id, state)| (id, state))
    }
    pub(crate) fn get(&self, id: &i32) -> Option<&SequenceState> {
        let at = self
            .entries
            .binary_search_by_key(id, |entry| entry.0)
            .ok()?;
        Some(&self.entries[at].1)
    }
    pub(crate) fn get_mut(&mut self, id: &i32) -> Option<&mut SequenceState> {
        let at = self
            .entries
            .binary_search_by_key(id, |entry| entry.0)
            .ok()?;
        Some(&mut self.entries[at].1)
    }
    pub(crate) fn contains_key(&self, id: &i32) -> bool {
        self.get(id).is_some()
    }
    pub(crate) fn plan(&self, capacity: usize) -> Result<StatePlan> {
        let mut plan = StatePlan::new(self.budget(), capacity)?;
        for (&id, state) in self.iter() {
            plan.push(PlanEntry {
                id,
                source: id,
                position: state.position,
            })?;
        }
        Ok(plan)
    }
    pub(crate) fn copied(
        budget: &Arc<StateBudget>,
        shape: MemoryShape,
        plan: &StatePlan,
        source: Option<&Self>,
    ) -> Result<Self> {
        let mut bytes = multiply(plan.entries.len(), size_of::<(i32, SequenceState)>())?;
        for entry in &plan.entries {
            bytes = add(bytes, shape.bytes(entry.position)?)?;
        }
        let reservation = budget.reserve(bytes)?;
        let mut entries = allocate(plan.entries.len(), "sequence records")?;
        for entry in &plan.entries {
            let previous = source.and_then(|source| source.get(&entry.source));
            if previous.is_some_and(|state| state.position > entry.position) {
                return Err(invalid("state plan would truncate an existing source"));
            }
            entries.push((
                entry.id,
                SequenceState {
                    position: previous.map_or(0, |state| state.position),
                    memory: shape.copy_memory(previous, entry.position)?,
                },
            ));
        }
        let result = Self {
            entries,
            reservation,
            shape,
        };
        if result.capacity_bytes()? != result.bytes() {
            return Err(invalid("allocator returned an unexpected state capacity"));
        }
        Ok(result)
    }
    pub(crate) fn try_copy(&self, budget: &Arc<StateBudget>) -> Result<Self> {
        // Charge the plan to the destination budget as part of this copy.
        let mut plan = StatePlan::new(budget, self.len())?;
        for (&id, state) in self.iter() {
            plan.push(PlanEntry {
                id,
                source: id,
                position: state.position,
            })?;
        }
        Self::copied(budget, self.shape, &plan, Some(self))
    }
    pub(crate) fn remove(&mut self, id: i32) -> Result<bool> {
        let Ok(index) = self.entries.binary_search_by_key(&id, |entry| entry.0) else {
            return Ok(false);
        };
        self.entries.remove(index);
        let bytes = self.capacity_bytes()?;
        self.reservation.reduce_to(bytes);
        if self.entries.is_empty() {
            *self = Self::empty(self.budget(), self.shape);
        }
        Ok(true)
    }
    fn capacity_bytes(&self) -> Result<usize> {
        let mut bytes = multiply(self.entries.capacity(), size_of::<(i32, SequenceState)>())?;
        for (_, state) in &self.entries {
            match &state.memory {
                SequenceMemory::Attention(layers) => {
                    bytes = add(bytes, multiply(layers.capacity(), size_of::<KvLayer>())?)?;
                    for layer in layers {
                        bytes = add(
                            bytes,
                            multiply(
                                add(layer.keys.capacity(), layer.values.capacity())?,
                                size_of::<f32>(),
                            )?,
                        )?;
                    }
                }
                SequenceMemory::Recurrent(layers) => {
                    bytes = add(
                        bytes,
                        multiply(layers.capacity(), size_of::<RecurrentLayer>())?,
                    )?;
                    for layer in layers {
                        bytes = add(
                            bytes,
                            multiply(
                                add(layer.convolution.capacity(), layer.ssm.capacity())?,
                                size_of::<f32>(),
                            )?,
                        )?;
                    }
                }
            }
        }
        Ok(bytes)
    }
}

#[cfg(test)]
thread_local! {
    static FAIL_AFTER: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}
#[cfg(test)]
fn allocation_fault(what: &str) -> Result<()> {
    FAIL_AFTER.with(|remaining| match remaining.get() {
        Some(0) => Err(invalid(format!("injected allocation failure for {what}"))),
        Some(count) => {
            remaining.set(Some(count - 1));
            Ok(())
        }
        None => Ok(()),
    })
}
#[cfg(test)]
pub(super) struct AllocationFailure;
#[cfg(test)]
impl AllocationFailure {
    pub(super) fn after(successful_reservations: usize) -> Self {
        FAIL_AFTER.with(|remaining| {
            assert!(remaining.get().is_none());
            remaining.set(Some(successful_reservations));
        });
        Self
    }
}
#[cfg(test)]
impl Drop for AllocationFailure {
    fn drop(&mut self) {
        FAIL_AFTER.with(|remaining| remaining.set(None));
    }
}
