//! Owned, bounded GGUF shard loading following frozen llama-model-loader.cpp.
//! Secondary metadata never replaces or augments the primary model metadata.

use crate::{
    Endian, Error, Gguf, HeaderUsage, Limits, Result, TensorInfo, Value, invalid, read_tensor_f32,
    read_tensor_payload,
};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs::File,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone)]
pub struct ShardLimits {
    /// Maximum simultaneously owned files. This is an explicit resource limit,
    /// not a restriction of the GGUF UINT16 split-count field.
    pub max_shards: usize,
    /// Header bytes, metadata entries/values and tensor descriptors are charged
    /// cumulatively across the complete set. String, array and payload-read
    /// limits still apply to each individual object/read.
    pub gguf: Limits,
}
impl Default for ShardLimits {
    fn default() -> Self {
        Self {
            max_shards: 1024,
            gguf: Limits::default(),
        }
    }
}

/// File-specific layout. Tensor offsets are always relative to this file's
/// data_offset, and scalar payloads use this file's endian.
#[derive(Debug, Clone)]
pub struct ShardInfo {
    pub path: PathBuf,
    pub version: u32,
    pub endian: Endian,
    pub alignment: u32,
    pub data_offset: u64,
    pub file_len: u64,
    pub tensor_count: usize,
    pub header_usage: HeaderUsage,
}

struct Shard {
    info: ShardInfo,
    file: Gguf<File>,
}

/// A complete validated model-file set, retaining the handles that were parsed.
/// No tensor payload is loaded until requested. Construction is transactional:
/// any malformed/missing shard drops all staged handles and returns an error.
pub struct GgufSet {
    metadata: BTreeMap<String, Value>,
    shards: Vec<Shard>,
    tensors: Vec<TensorInfo>,
    tensor_shards: Vec<usize>,
    by_name: Vec<usize>,
    usage: HeaderUsage,
    file_bytes: u64,
    tensor_bytes: u64,
    tensor_elements: u64,
}

type ExplicitSplits<'path, 'resolver> = (usize, &'resolver dyn Fn(usize) -> Option<&'path Path>);

impl GgufSet {
    /// Open the primary file and discover siblings using the frozen filename
    /// suffix when its UINT16 split.count exceeds one.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_limits(path, ShardLimits::default())
    }

    pub fn open_with_limits(path: impl AsRef<Path>, limits: ShardLimits) -> Result<Self> {
        Self::load(path.as_ref(), None, limits)
    }

    /// Open an explicitly ordered list. As in the frozen C API, an empty list
    /// fails. If the primary count is absent/zero/one, later paths are ignored.
    pub fn open_splits<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        Self::open_splits_with_limits(paths, ShardLimits::default())
    }

    pub fn open_splits_with_limits<P: AsRef<Path>>(
        paths: &[P],
        limits: ShardLimits,
    ) -> Result<Self> {
        let first = paths
            .first()
            .ok_or_else(|| invalid("list of GGUF splits is empty"))?;
        let resolve = |index: usize| paths.get(index).map(AsRef::as_ref);
        Self::load(first.as_ref(), Some((paths.len(), &resolve)), limits)
    }

    fn load<'a>(
        first_path: &Path,
        explicit: Option<ExplicitSplits<'a, '_>>,
        limits: ShardLimits,
    ) -> Result<Self> {
        if limits.max_shards == 0 {
            return Err(Error::Limit("maximum GGUF shard count is zero".into()));
        }
        let mut first = Gguf::open_with_limits(first_path, limits.gguf.clone())?;
        let declared_count = u16_key(&first.metadata, "split.count", Some(0))?;
        let count = usize::from(declared_count).max(1);
        if count > limits.max_shards {
            return Err(Error::Limit(format!(
                "model declares {count} GGUF shards, exceeding {}",
                limits.max_shards
            )));
        }
        let prefix = if declared_count > 1 {
            if u16_key(&first.metadata, "split.no", None)? != 0 {
                return Err(invalid(
                    "model must be loaded with the first split (split.no=0)",
                ));
            }
            if let Some((given, _)) = explicit {
                // Deliberately no UINT16 narrowing: the source's cast can wrap
                // for oversized lists; that is not a safe input convention.
                if given != count {
                    return Err(invalid(format!(
                        "explicit split count {given} differs from declared {count}"
                    )));
                }
                None
            } else {
                Some(split_prefix(first_path, 0, declared_count)?)
            }
        } else {
            None
        };

        let mut set = Self {
            metadata: std::mem::take(&mut first.metadata),
            shards: Vec::new(),
            tensors: Vec::new(),
            tensor_shards: Vec::new(),
            by_name: Vec::new(),
            usage: HeaderUsage::default(),
            file_bytes: 0,
            tensor_bytes: 0,
            tensor_elements: 0,
        };
        set.shards
            .try_reserve_exact(count)
            .map_err(|_| Error::Limit("GGUF shard owners allocation".into()))?;
        set.append(first_path, first)?;
        for index in 1..count {
            let path = match explicit {
                Some((_, resolve)) => copy_path(
                    resolve(index).ok_or_else(|| invalid("explicit split path is missing"))?,
                )?,
                None => split_path(
                    prefix
                        .as_deref()
                        .ok_or_else(|| invalid("missing split filename prefix"))?,
                    index,
                    declared_count,
                )?,
            };
            let remaining = remaining_limits(&limits.gguf, set.usage)?;
            let file = Gguf::open_with_limits(&path, remaining)?;
            // This is the ONLY subsidiary model metadata field consulted by
            // the original loader. In particular split.count, total, arch,
            // tokenizer and other primary metadata must not be merged.
            if usize::from(u16_key(&file.metadata, "split.no", None)?) != index {
                return Err(invalid(format!("split.no differs from expected {index}")));
            }
            set.append(&path, file)?;
        }
        set.by_name
            .try_reserve_exact(set.tensors.len())
            .map_err(|_| Error::Limit("unified tensor name index allocation".into()))?;
        set.by_name.extend(0..set.tensors.len());
        set.by_name
            .sort_unstable_by(|&left, &right| set.tensors[left].name.cmp(&set.tensors[right].name));
        for pair in set.by_name.windows(2) {
            if set.tensors[pair[0]].name == set.tensors[pair[1]].name {
                return Err(invalid(format!(
                    "tensor {} is duplicated across GGUF splits",
                    set.tensors[pair[0]].name
                )));
            }
        }
        if declared_count > 1 {
            let expected = match set.metadata.get("split.tensors.count") {
                Some(Value::I32(value)) if *value >= 0 => *value as usize,
                Some(Value::I32(_)) => return Err(invalid("split.tensors.count is negative")),
                Some(_) => return Err(invalid("split.tensors.count must be INT32")),
                None => return Err(invalid("missing split.tensors.count")),
            };
            if expected != set.tensors.len() {
                return Err(invalid(format!(
                    "split.tensors.count={expected}, but {} tensors were found",
                    set.tensors.len()
                )));
            }
        }
        Ok(set)
    }

    fn append(&mut self, path: &Path, mut file: Gguf<File>) -> Result<()> {
        let shard_index = self.shards.len();
        self.usage = add_usage(self.usage, file.header_usage)?;
        self.file_bytes = self
            .file_bytes
            .checked_add(file.file_len)
            .ok_or_else(|| invalid("aggregate GGUF file size overflow"))?;
        self.tensors
            .try_reserve(file.tensors.len())
            .map_err(|_| Error::Limit("unified tensor descriptors allocation".into()))?;
        self.tensor_shards
            .try_reserve(file.tensors.len())
            .map_err(|_| Error::Limit("unified tensor origins allocation".into()))?;
        for tensor in &file.tensors {
            self.tensor_bytes = self
                .tensor_bytes
                .checked_add(tensor.byte_len)
                .ok_or_else(|| invalid("aggregate tensor byte count overflow"))?;
            self.tensor_elements = self
                .tensor_elements
                .checked_add(tensor.elements()? as u64)
                .ok_or_else(|| invalid("aggregate tensor element count overflow"))?;
        }
        let info = ShardInfo {
            path: copy_path(path)?,
            version: file.version,
            endian: file.endian,
            alignment: file.alignment,
            data_offset: file.data_offset,
            file_len: file.file_len,
            tensor_count: file.tensors.len(),
            header_usage: file.header_usage,
        };
        self.tensor_shards
            .extend(std::iter::repeat_n(shard_index, file.tensors.len()));
        self.tensors.append(&mut file.tensors);
        // Secondary headers have already been validated and charged. Retain
        // only primary metadata and file-specific layout, as the source does.
        file.metadata.clear();
        self.shards.push(Shard { info, file });
        Ok(())
    }

    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.metadata
    }
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }
    pub fn primary_path(&self) -> &Path {
        &self.shards[0].info.path
    }
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }
    pub fn shard_info(&self, index: usize) -> Option<&ShardInfo> {
        self.shards.get(index).map(|shard| &shard.info)
    }
    pub fn header_usage(&self) -> HeaderUsage {
        self.usage
    }
    /// Sum of the actual file lengths, including every header and padding.
    pub fn file_bytes(&self) -> u64 {
        self.file_bytes
    }
    /// Sum of stored tensor payload lengths, excluding headers and padding.
    pub fn tensor_bytes(&self) -> u64 {
        self.tensor_bytes
    }
    pub fn tensor_elements(&self) -> u64 {
        self.tensor_elements
    }
    fn tensor_index(&self, name: &str) -> Option<usize> {
        self.by_name
            .binary_search_by(|&index| self.tensors[index].name.as_str().cmp(name))
            .ok()
            .map(|index| self.by_name[index])
    }
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index(name).map(|index| &self.tensors[index])
    }
    pub fn tensor_shard(&self, name: &str) -> Option<usize> {
        self.tensor_index(name)
            .map(|index| self.tensor_shards[index])
    }
    pub fn tensor_endian(&self, name: &str) -> Option<Endian> {
        self.tensor_shard(name)
            .map(|index| self.shards[index].info.endian)
    }
    pub fn read_tensor(&mut self, name: &str) -> Result<Vec<u8>> {
        let index = self
            .tensor_index(name)
            .ok_or_else(|| invalid(format!("missing tensor {name}")))?;
        let shard = &mut self.shards[self.tensor_shards[index]].file;
        read_tensor_payload(
            &mut shard.reader,
            shard.data_offset,
            shard.file_len,
            &shard.limits,
            &self.tensors[index],
        )
    }
    pub fn read_f32(&mut self, name: &str) -> Result<Vec<f32>> {
        let index = self
            .tensor_index(name)
            .ok_or_else(|| invalid(format!("missing tensor {name}")))?;
        let shard = &mut self.shards[self.tensor_shards[index]].file;
        read_tensor_f32(
            &mut shard.reader,
            shard.data_offset,
            shard.file_len,
            shard.endian,
            &shard.limits,
            &self.tensors[index],
        )
    }
}

fn u16_key(metadata: &BTreeMap<String, Value>, key: &str, default: Option<u16>) -> Result<u16> {
    match metadata.get(key) {
        Some(Value::U16(value)) => Ok(*value),
        Some(_) => Err(invalid(format!("{key} must be UINT16"))),
        None => default.ok_or_else(|| invalid(format!("missing {key}"))),
    }
}

fn add_usage(a: HeaderUsage, b: HeaderUsage) -> Result<HeaderUsage> {
    let overflow = || Error::Limit("cumulative GGUF header accounting overflow".into());
    Ok(HeaderUsage {
        bytes: a.bytes.checked_add(b.bytes).ok_or_else(overflow)?,
        metadata_entries: a
            .metadata_entries
            .checked_add(b.metadata_entries)
            .ok_or_else(overflow)?,
        metadata_values: a
            .metadata_values
            .checked_add(b.metadata_values)
            .ok_or_else(overflow)?,
        tensors: a.tensors.checked_add(b.tensors).ok_or_else(overflow)?,
    })
}
fn remaining_limits(limits: &Limits, used: HeaderUsage) -> Result<Limits> {
    let exhausted = || Error::Limit("cumulative GGUF header budget".into());
    Ok(Limits {
        max_metadata_bytes: limits
            .max_metadata_bytes
            .checked_sub(used.bytes)
            .ok_or_else(exhausted)?,
        max_total_values: limits
            .max_total_values
            .checked_sub(used.metadata_values)
            .ok_or_else(exhausted)?,
        max_tensors: limits
            .max_tensors
            .checked_sub(used.tensors)
            .ok_or_else(exhausted)?,
        max_metadata_entries: limits
            .max_metadata_entries
            .checked_sub(used.metadata_entries)
            .ok_or_else(exhausted)?,
        ..limits.clone()
    })
}

fn copy_path(path: &Path) -> Result<PathBuf> {
    let mut result = OsString::new();
    result
        .try_reserve_exact(path.as_os_str().len())
        .map_err(|_| Error::Limit("GGUF path allocation".into()))?;
    result.push(path.as_os_str());
    Ok(result.into())
}
fn split_path(prefix: &Path, index: usize, count: u16) -> Result<PathBuf> {
    let suffix = format!("-{:05}-of-{count:05}.gguf", index + 1);
    let mut path = copy_path(prefix)?.into_os_string();
    path.try_reserve_exact(suffix.len())
        .map_err(|_| Error::Limit("GGUF split path allocation".into()))?;
    path.push(suffix);
    Ok(path.into())
}
fn split_prefix(path: &Path, index: usize, count: u16) -> Result<PathBuf> {
    let suffix = format!("-{:05}-of-{count:05}.gguf", index + 1);
    strip_suffix(path, &suffix)
        .ok_or_else(|| invalid(format!("invalid split filename: {}", path.display())))?
}

#[cfg(unix)]
fn strip_suffix(path: &Path, suffix: &str) -> Option<Result<PathBuf>> {
    use std::os::unix::ffi::{OsStrExt, OsStringExt};
    let original = path.as_os_str().as_bytes();
    let prefix = original.strip_suffix(suffix.as_bytes())?;
    if prefix.is_empty() {
        return None;
    }
    let mut bytes = Vec::new();
    if bytes.try_reserve_exact(prefix.len()).is_err() {
        return Some(Err(Error::Limit("GGUF split prefix allocation".into())));
    }
    bytes.extend_from_slice(prefix);
    Some(Ok(OsString::from_vec(bytes).into()))
}
#[cfg(windows)]
fn strip_suffix(path: &Path, suffix: &str) -> Option<Result<PathBuf>> {
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    let mut encoded = Vec::new();
    if encoded.try_reserve_exact(path.as_os_str().len()).is_err() {
        return Some(Err(Error::Limit("GGUF split prefix allocation".into())));
    }
    encoded.extend(path.as_os_str().encode_wide());
    let n = encoded.len().checked_sub(suffix.len())?;
    if n == 0
        || !encoded[n..]
            .iter()
            .copied()
            .eq(suffix.bytes().map(u16::from))
    {
        return None;
    }
    Some(Ok(OsString::from_wide(&encoded[..n]).into()))
}
#[cfg(not(any(unix, windows)))]
fn strip_suffix(path: &Path, suffix: &str) -> Option<Result<PathBuf>> {
    let prefix = path.to_str()?.strip_suffix(suffix)?;
    if prefix.is_empty() {
        None
    } else {
        Some(copy_path(Path::new(prefix)))
    }
}
