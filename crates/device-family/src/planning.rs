//! Pure scheduling contracts awaiting T02 integration. They make no GPU calls.
use crate::{DeviceError, Family};
use std::ops::Range;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowShard {
    pub device: usize,
    pub rows: Range<u64>,
}

/// Partition rows by integer weights, with interior boundaries on tile edges.
/// Empty shards are retained so device indices never silently change.
pub fn split_rows(
    rows: u64,
    tile_rows: u64,
    weights: &[u32],
) -> Result<Vec<RowShard>, DeviceError> {
    if tile_rows == 0 || weights.is_empty() || weights.len() > 4096 {
        return Err(DeviceError::InvalidArgument(
            "positive tile size and 1..=4096 weights required".into(),
        ));
    }
    let sum: u64 = weights.iter().map(|w| *w as u64).sum();
    if sum == 0 {
        return Err(DeviceError::InvalidArgument(
            "all device split weights are zero".into(),
        ));
    }
    let blocks = rows.div_ceil(tile_rows);
    let mut previous = 0;
    let mut cumulative = 0u64;
    let mut shards = Vec::with_capacity(weights.len());
    for (device, weight) in weights.iter().enumerate() {
        cumulative += *weight as u64;
        // Widen before multiplication: weights are u32 and at most 4096.
        let end_block = (blocks as u128 * cumulative as u128) / sum as u128;
        let end = (end_block * tile_rows as u128).min(rows as u128) as u64;
        shards.push(RowShard {
            device,
            rows: previous..end,
        });
        previous = end;
    }
    debug_assert_eq!(previous, rows);
    Ok(shards)
}

pub fn shard_bytes(
    shard: &RowShard,
    bytes_per_row: u64,
    available: u64,
) -> Result<u64, DeviceError> {
    let rows = shard
        .rows
        .end
        .checked_sub(shard.rows.start)
        .ok_or_else(|| DeviceError::InvalidArgument("reversed shard range".into()))?;
    let bytes = rows
        .checked_mul(bytes_per_row)
        .ok_or_else(|| DeviceError::InvalidArgument("shard size overflow".into()))?;
    if bytes > available {
        return Err(DeviceError::InvalidArgument(format!(
            "device {} needs {bytes} bytes but has {available}",
            shard.device
        )));
    }
    Ok(bytes)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerTopology {
    devices: usize,
    access: Vec<bool>,
}
impl PeerTopology {
    /// `access[src * devices + dst]` is a measured directional peer capability.
    pub fn new(devices: usize, access: Vec<bool>) -> Result<Self, DeviceError> {
        if devices == 0 || devices.checked_mul(devices) != Some(access.len()) {
            return Err(DeviceError::InvalidArgument(
                "peer matrix shape mismatch".into(),
            ));
        }
        Ok(Self { devices, access })
    }
    pub fn route(
        &self,
        src: usize,
        dst: usize,
        allow_host_staging: bool,
    ) -> Result<CopyRoute, DeviceError> {
        if src >= self.devices || dst >= self.devices {
            return Err(DeviceError::InvalidArgument(
                "peer device index out of range".into(),
            ));
        }
        if src == dst {
            return Ok(CopyRoute::SameDevice);
        }
        if self.access[src * self.devices + dst] {
            return Ok(CopyRoute::PeerDirect);
        }
        if allow_host_staging {
            return Ok(CopyRoute::HostStaging);
        }
        Err(DeviceError::Unsupported(format!(
            "peer copy {src}->{dst}; host staging was not permitted"
        )))
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyRoute {
    SameDevice,
    PeerDirect,
    HostStaging,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubgroupContract {
    pub family: Family,
    pub reported_width: u32,
}
impl SubgroupContract {
    pub fn new(family: Family, reported_width: u32) -> Result<Self, DeviceError> {
        if reported_width == 0 || !reported_width.is_power_of_two() || reported_width > 128 {
            return Err(DeviceError::InvalidArgument(
                "subgroup width must be a measured power of two <=128".into(),
            ));
        }
        match family {
            Family::Cuda if reported_width != 32 => {
                return Err(DeviceError::Unsupported(
                    "CUDA lowering currently requires measured warp width 32".into(),
                ));
            }
            Family::Hip if !matches!(reported_width, 32 | 64) => {
                return Err(DeviceError::Unsupported(
                    "HIP lowering requires measured wave32 or wave64".into(),
                ));
            }
            _ => {}
        }
        Ok(Self {
            family,
            reported_width,
        })
    }
    pub fn require_width(&self, required: u32) -> Result<(), DeviceError> {
        if required == self.reported_width {
            Ok(())
        } else {
            Err(DeviceError::Unsupported(format!(
                "kernel requires subgroup {required}; {} reported {}",
                self.family, self.reported_width
            )))
        }
    }
}

/// These fields are compared structurally, not by a potentially colliding
/// short hash. They must be supplied from T02/T05 ownership generations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphFingerprint {
    pub family: Family,
    pub device: u32,
    pub topology: Vec<(u32, Vec<u32>)>,
    pub tensor_shapes: Vec<[u64; 4]>,
    pub buffer_generations: Vec<(u64, u64)>,
    pub adapter_generation: u64,
    pub code_generation: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayDecision {
    Reuse,
    RebuildDevice,
    RebuildTopology,
    RebuildShapes,
    RebindBuffers,
    RebuildAdapters,
    RecompileKernels,
}
impl GraphFingerprint {
    pub fn compare(&self, next: &Self) -> ReplayDecision {
        if self.family != next.family || self.device != next.device {
            ReplayDecision::RebuildDevice
        } else if self.code_generation != next.code_generation {
            ReplayDecision::RecompileKernels
        } else if self.topology != next.topology {
            ReplayDecision::RebuildTopology
        } else if self.tensor_shapes != next.tensor_shapes {
            ReplayDecision::RebuildShapes
        } else if self.adapter_generation != next.adapter_generation {
            ReplayDecision::RebuildAdapters
        } else if self.buffer_generations != next.buffer_generations {
            ReplayDecision::RebindBuffers
        } else {
            ReplayDecision::Reuse
        }
    }
}

/// Persistent device-code caches must include all code-affecting inputs. The
/// emitted source is kept verbatim to avoid treating a hash collision as parity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelCacheKey {
    pub family: Family,
    pub target: String,
    pub driver_version: i32,
    pub compiler_identity: String,
    pub compiler_flags: Vec<String>,
    pub ir_version: u32,
    pub source: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn row_partition_has_no_gaps_and_preserves_weights() {
        let split = split_rows(101, 16, &[1, 0, 3]).unwrap();
        assert_eq!(split[0].rows, 0..16);
        assert_eq!(split[1].rows, 16..16);
        assert_eq!(split[2].rows, 16..101);
        assert_eq!(shard_bytes(&split[2], 8, 680).unwrap(), 680);
        assert!(shard_bytes(&split[2], 8, 679).is_err());
    }
    #[test]
    fn row_partition_handles_boundaries_without_u64_overflow() {
        let split = split_rows(u64::MAX, 32, &[u32::MAX, u32::MAX]).unwrap();
        assert_eq!(split[0].rows.start, 0);
        assert_eq!(split[0].rows.end, split[1].rows.start);
        assert_eq!(split[1].rows.end, u64::MAX);
        assert_eq!(split[0].rows.end % 32, 0);
        assert!(split_rows(10, 0, &[1]).is_err());
        assert!(split_rows(10, 1, &[0, 0]).is_err());
        assert!(split_rows(10, 1, &[]).is_err());
    }
    #[test]
    fn bounded_partition_property_grid() {
        for rows in 0..130 {
            for tile in [1, 8, 32] {
                for weights in [&[1, 1][..], &[0, 3, 2], &[9, 0, 1, 3]] {
                    let split = split_rows(rows, tile, weights).unwrap();
                    let mut previous = 0;
                    for shard in split {
                        assert_eq!(shard.rows.start, previous);
                        assert!(shard.rows.start <= shard.rows.end);
                        previous = shard.rows.end;
                    }
                    assert_eq!(previous, rows);
                }
            }
        }
    }
    #[test]
    fn peer_matrix_is_directional_and_staging_explicit() {
        let matrix = PeerTopology::new(2, vec![true, true, false, true]).unwrap();
        assert_eq!(matrix.route(0, 1, false).unwrap(), CopyRoute::PeerDirect);
        assert!(matrix.route(1, 0, false).is_err());
        assert_eq!(matrix.route(1, 0, true).unwrap(), CopyRoute::HostStaging);
        assert_eq!(matrix.route(1, 1, false).unwrap(), CopyRoute::SameDevice);
        assert!(matrix.route(2, 1, true).is_err());
    }
    #[test]
    fn waves_are_not_interchanged() {
        let hip = SubgroupContract::new(Family::Hip, 64).unwrap();
        assert!(hip.require_width(32).is_err());
        assert!(SubgroupContract::new(Family::Cuda, 64).is_err());
        assert!(SubgroupContract::new(Family::Musa, 0).is_err());
    }
    fn graph() -> GraphFingerprint {
        GraphFingerprint {
            family: Family::Cuda,
            device: 0,
            topology: vec![(1, vec![0])],
            tensor_shapes: vec![[32, 2, 1, 1]],
            buffer_generations: vec![(1, 4)],
            adapter_generation: 0,
            code_generation: 1,
        }
    }
    #[test]
    fn graph_replay_invalidates_each_relevant_change() {
        let original = graph();
        assert_eq!(original.compare(&graph()), ReplayDecision::Reuse);
        let mut next = graph();
        next.buffer_generations[0].1 += 1;
        assert_eq!(original.compare(&next), ReplayDecision::RebindBuffers);
        let mut next = graph();
        next.tensor_shapes[0][0] += 1;
        assert_eq!(original.compare(&next), ReplayDecision::RebuildShapes);
        let mut next = graph();
        next.adapter_generation += 1;
        assert_eq!(original.compare(&next), ReplayDecision::RebuildAdapters);
        let mut next = graph();
        next.code_generation += 1;
        assert_eq!(original.compare(&next), ReplayDecision::RecompileKernels);
        let mut next = graph();
        next.topology.push((2, vec![1]));
        assert_eq!(original.compare(&next), ReplayDecision::RebuildTopology);
        let mut next = graph();
        next.family = Family::Hip;
        assert_eq!(original.compare(&next), ReplayDecision::RebuildDevice);
    }
}
