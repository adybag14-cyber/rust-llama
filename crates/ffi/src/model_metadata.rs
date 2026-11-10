//! Frozen model scalar metadata APIs. Raw primary wire order, scalar-only
//! emplace semantics, compiler-profile ordering and decimal formatting are
//! separate from the typed model getter projection.
use crate::native_metadata_policy::{self as policy, Limits as PolicyLimits, Profile};
use crate::*;
use llama_gguf::{GgufSet, RawValue, Value};
use std::cmp::Ordering;
use std::fmt::{self, Write};
use std::mem::size_of;

const MAX_METADATA_WORK: u64 = 128 * 1024 * 1024;
const FORMAT_SCRATCH_BYTES: usize = 1024;

pub(crate) struct MetadataCache {
    entries: Vec<Entry>,
    by_key: Vec<usize>,
    exact_profile: bool,
    // Conservative simultaneous raw/projection/cache/temporary charge. The
    // underlying GGUF owner is dropped only after model loading has completed.
    _peak_charge: usize,
    _work_units: u64,
}
struct Entry {
    key: Vec<u8>,
    // Always exactly one trailing NUL; interior NULs were removed by the source
    // C-prefix view before this owned cache was constructed.
    value: Vec<u8>,
}
struct Input<'a> {
    key: &'a [u8],
    value: &'a RawValue,
}
fn prefix(value: &[u8]) -> &[u8] {
    &value[..value
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(value.len())]
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| "model metadata byte accounting overflow".into())
}
fn multiply(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| "model metadata byte accounting overflow".into())
}
fn require(bytes: usize, limit: usize) -> Result<()> {
    if bytes > limit {
        Err(format!(
            "aggregate model metadata requires {bytes} bytes, exceeding {limit}"
        ))
    } else {
        Ok(())
    }
}
fn remaining(used: usize, limit: usize) -> Result<usize> {
    limit
        .checked_sub(used)
        .ok_or_else(|| "aggregate model metadata budget exhausted".into())
}
fn reserve<T>(count: usize, what: &str) -> Result<Vec<T>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| format!("{what} allocation"))?;
    Ok(result)
}
fn copy(value: &[u8], nul: bool) -> Result<Vec<u8>> {
    let length = add(value.len(), usize::from(nul))?;
    let mut result = reserve(length, "metadata byte string")?;
    result.extend_from_slice(value);
    if nul {
        result.push(0);
    }
    Ok(result)
}
fn native_profile() -> Option<Profile> {
    if cfg!(all(
        target_arch = "x86_64",
        target_os = "windows",
        target_env = "msvc"
    )) {
        Some(Profile::Msvc)
    } else if cfg!(all(
        target_arch = "x86_64",
        target_os = "linux",
        target_env = "gnu"
    )) {
        Some(Profile::Gnu)
    } else {
        None
    }
}
/// The existing high-level loader copies these display/template bytes while
/// the raw GGUF owner is still live. Their source storage is charged separately.
fn consumer_copies(files: &GgufSet) -> Result<usize> {
    let mut total = FORMAT_SCRATCH_BYTES;
    for key in ["general.name", "tokenizer.chat_template"] {
        if let Some(bytes) = files.metadata().get(key).and_then(Value::as_bytes) {
            total = add(total, bytes.len())?;
        }
    }
    Ok(total)
}
struct Decimal {
    bytes: [u8; 317],
    len: usize,
}
impl Decimal {
    fn new() -> Self {
        Self {
            bytes: [0; 317],
            len: 0,
        }
    }
    fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut result = Self::new();
        let text =
            std::str::from_utf8(bytes).map_err(|_| "scalar formatter returned non-ASCII text")?;
        result
            .write_str(text)
            .map_err(|_| "scalar formatter exceeded317 bytes")?;
        Ok(result)
    }
}
impl fmt::Write for Decimal {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        let end = self.len.checked_add(value.len()).ok_or(fmt::Error)?;
        if end > self.bytes.len() {
            return Err(fmt::Error);
        }
        self.bytes[self.len..end].copy_from_slice(value.as_bytes());
        self.len = end;
        Ok(())
    }
}
// Keep bounded decimal scratch on the stack; boxing adds an unneeded per-scalar allocation.
#[allow(clippy::large_enum_variant)]
enum Scalar<'a> {
    Borrowed(&'a [u8]),
    Decimal(Decimal),
}
impl Scalar<'_> {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Borrowed(bytes) => bytes,
            Self::Decimal(value) => value.as_bytes(),
        }
    }
}
fn scalar(value: &RawValue, profile: Profile) -> Result<Scalar<'_>> {
    let mut decimal = Decimal::new();
    let written = match value {
        RawValue::U8(value) => write!(decimal, "{value}"),
        RawValue::I8(value) => write!(decimal, "{value}"),
        RawValue::U16(value) => write!(decimal, "{value}"),
        RawValue::I16(value) => write!(decimal, "{value}"),
        RawValue::U32(value) => write!(decimal, "{value}"),
        RawValue::I32(value) => write!(decimal, "{value}"),
        RawValue::U64(value) => write!(decimal, "{value}"),
        RawValue::I64(value) => write!(decimal, "{value}"),
        RawValue::F32(value) => {
            let text = policy::format_f32(*value, profile).map_err(|error| error.to_string())?;
            return Ok(Scalar::Decimal(Decimal::from_bytes(text.as_bytes())?));
        }
        RawValue::F64(value) => {
            let text = policy::format_f64(*value, profile).map_err(|error| error.to_string())?;
            return Ok(Scalar::Decimal(Decimal::from_bytes(text.as_bytes())?));
        }
        RawValue::Bool(value) => {
            return Ok(Scalar::Borrowed(if *value { b"true" } else { b"false" }));
        }
        RawValue::String(value) | RawValue::ByteString(value) => {
            return Ok(Scalar::Borrowed(prefix(value)));
        }
        RawValue::Array { .. } => return Err("array reached scalar metadata cache".into()),
    };
    written.map_err(|_| "integer metadata formatter overflow")?;
    Ok(Scalar::Decimal(decimal))
}
fn charge_work(work: &mut u64, amount: u64, limit: u64) -> Result<()> {
    *work = work
        .checked_add(amount)
        .ok_or("metadata work accounting overflow")?;
    if *work > limit {
        return Err("metadata lookup index exceeds work limit".into());
    }
    Ok(())
}
fn compare(left: &[u8], right: &[u8], work: &mut u64, limit: u64) -> Result<Ordering> {
    for (left, right) in left.iter().zip(right) {
        charge_work(work, 1, limit)?;
        let order = left.cmp(right);
        if order != Ordering::Equal {
            return Ok(order);
        }
    }
    charge_work(work, 1, limit)?;
    Ok(left.len().cmp(&right.len()))
}
/// Bounded bottom-up merge sort for the private lookup index. Native iteration
/// order stays in entries; only lookup references are sorted. Both index vectors
/// and every compared byte are charged before a model handle can escape.
fn lookup_index(entries: &[Entry], work: &mut u64, limit: u64) -> Result<Vec<usize>> {
    charge_work(work, entries.len() as u64 * 2, limit)?;
    let mut sorted = reserve(entries.len(), "metadata lookup indices")?;
    sorted.extend(0..entries.len());
    let mut scratch = reserve(entries.len(), "metadata lookup scratch")?;
    scratch.resize(entries.len(), 0);
    let mut width = 1usize;
    while width < entries.len() {
        let step = width
            .checked_mul(2)
            .ok_or("metadata merge width overflow")?;
        let mut start = 0usize;
        while start < entries.len() {
            let middle = start.saturating_add(width).min(entries.len());
            let end = start.saturating_add(step).min(entries.len());
            let (mut left, mut right) = (start, middle);
            for slot in &mut scratch[start..end] {
                charge_work(work, 1, limit)?;
                if right == end
                    || (left < middle
                        && compare(
                            &entries[sorted[left]].key,
                            &entries[sorted[right]].key,
                            work,
                            limit,
                        )? != Ordering::Greater)
                {
                    *slot = sorted[left];
                    left += 1;
                } else {
                    *slot = sorted[right];
                    right += 1;
                }
            }
            start = end;
        }
        std::mem::swap(&mut sorted, &mut scratch);
        width = step;
    }
    Ok(sorted)
}

impl MetadataCache {
    pub(crate) fn build(files: &GgufSet) -> Result<Self> {
        let Some(profile) = native_profile() else {
            return Self::text_only(files);
        };
        Self::build_profile(
            files,
            profile,
            files.metadata_storage_limit(),
            MAX_METADATA_WORK,
        )
    }
    fn build_profile(
        files: &GgufSet,
        profile: Profile,
        byte_limit: usize,
        work_limit: u64,
    ) -> Result<Self> {
        let byte_limit = byte_limit.min(files.metadata_storage_limit());
        let count = files
            .raw_metadata_order()
            .filter(|key| !matches!(files.raw_metadata().get(*key), Some(RawValue::Array { .. })))
            .count();
        let base = add(files.metadata_storage_bytes(), consumer_copies(files)?)?;
        let input_bytes = multiply(count, add(size_of::<Input<'_>>(), size_of::<&[u8]>())?)?;
        let held = add(base, input_bytes)?;
        require(held, byte_limit)?;
        let mut input = reserve(count, "metadata scalar input references")?;
        for key in files.raw_metadata_order() {
            let value = &files.raw_metadata()[key];
            if !matches!(value, RawValue::Array { .. }) {
                input.push(Input {
                    key: prefix(key),
                    value,
                });
            }
        }
        let mut keys = reserve(count, "metadata policy key references")?;
        keys.extend(input.iter().map(|entry| entry.key));
        let order = policy::default_map_order(
            &keys,
            profile,
            PolicyLimits {
                max_bytes: remaining(held, byte_limit)?,
                max_work: work_limit,
            },
        )
        .map_err(|error| error.to_string())?;
        let policy_peak = add(held, order.peak_bytes)?;
        require(policy_peak, byte_limit)?;
        let mut cache_bytes = add(
            size_of::<Self>(),
            multiply(order.indices.len(), size_of::<Entry>())?,
        )?;
        // Two vectors coexist during lookup sorting, after the policy result.
        cache_bytes = add(
            cache_bytes,
            multiply(order.indices.len(), size_of::<usize>() * 2)?,
        )?;
        for &index in &order.indices {
            let row = input
                .get(index)
                .ok_or("metadata policy returned an invalid input index")?;
            cache_bytes = add(cache_bytes, row.key.len())?;
            cache_bytes = add(
                cache_bytes,
                add(scalar(row.value, profile)?.bytes().len(), 1)?,
            )?;
        }
        let build_peak = add(add(held, order.result_bytes)?, cache_bytes)?;
        require(build_peak, byte_limit)?;
        let mut entries = reserve(order.indices.len(), "metadata scalar entries")?;
        for &index in &order.indices {
            let row = &input[index];
            let value = scalar(row.value, profile)?;
            entries.push(Entry {
                key: copy(row.key, false)?,
                value: copy(value.bytes(), true)?,
            });
        }
        let mut work = order.work_units;
        let by_key = lookup_index(&entries, &mut work, work_limit)?;
        Ok(Self {
            entries,
            by_key,
            exact_profile: true,
            _peak_charge: policy_peak.max(build_peak),
            _work_units: work,
        })
    }
    /// Preserve prior C model loading and byte chat-template access on other
    /// Rust targets. The four compiler-dependent metadata queries explicitly
    /// reject there; no unmeasured iteration/number-format policy is invented.
    fn text_only(files: &GgufSet) -> Result<Self> {
        let count = files
            .metadata()
            .values()
            .filter(|value| value.as_bytes().is_some())
            .count();
        let base = add(files.metadata_storage_bytes(), consumer_copies(files)?)?;
        let mut charge = add(
            base,
            add(
                size_of::<Self>(),
                multiply(count, size_of::<Entry>() + 2 * size_of::<usize>())?,
            )?,
        )?;
        for (key, value) in files.metadata() {
            if let Some(bytes) = value.as_bytes() {
                charge = add(charge, add(key.len(), add(bytes.len(), 1)?)?)?;
            }
        }
        require(charge, files.metadata_storage_limit())?;
        let mut entries = reserve(count, "metadata text fallback entries")?;
        for (key, value) in files.metadata() {
            if let Some(bytes) = value.as_bytes() {
                entries.push(Entry {
                    key: copy(key.as_bytes(), false)?,
                    value: copy(bytes, true)?,
                });
            }
        }
        let mut work = 0;
        let by_key = lookup_index(&entries, &mut work, MAX_METADATA_WORK)?;
        Ok(Self {
            entries,
            by_key,
            exact_profile: false,
            _peak_charge: charge,
            _work_units: work,
        })
    }
    fn exact(&self) -> Result<&Self> {
        if self.exact_profile {
            Ok(self)
        } else {
            Err("exact C metadata ordering/number formatting is currently implemented only for x86_64 Windows/MSVC and Linux/GNU".into())
        }
    }
    fn get(&self, key: &[u8]) -> Option<&Entry> {
        self.by_key
            .binary_search_by(|&index| self.entries[index].key.as_slice().cmp(key))
            .ok()
            .map(|position| &self.entries[self.by_key[position]])
    }
    pub(crate) fn chat_template(&self, key: &[u8]) -> Option<*const c_char> {
        self.get(key).map(|entry| entry.value.as_ptr().cast())
    }
}
unsafe fn copy_optional(value: Option<&[u8]>, output: *mut c_char, capacity: usize) -> Result<i32> {
    if capacity > isize::MAX as usize {
        return Err("metadata output capacity exceeds pointer arithmetic range".into());
    }
    if capacity != 0 {
        unsafe {
            reference_mut(output)?;
        }
        let bytes = value.unwrap_or(&[]);
        let count = bytes.len().min(capacity - 1);
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), output.cast(), count);
            *output.add(count) = 0;
        }
    }
    value.map_or(Ok(-1), |bytes| {
        i32::try_from(bytes.len()).map_err(|_| "metadata length exceeds INT32_MAX".into())
    })
}
fn value_bytes(entry: &Entry) -> &[u8] {
    &entry.value[..entry.value.len() - 1]
}
ffi!(
    llama_model_meta_count(model: *const ModelHandle) -> i32,
    -1,
    {
        let cache = unsafe { reference(model)? }.data.metadata.exact()?;
        i32::try_from(cache.entries.len()).map_err(|_| "metadata count exceeds INT32_MAX".into())
    }
);
ffi!(
    llama_model_meta_val_str(
        model: *const ModelHandle,
        key: *const c_char,
        output: *mut c_char,
        capacity: usize,
    ) -> i32,
    -1,
    {
        let cache = unsafe { reference(model)? }.data.metadata.exact()?;
        if key.is_null() {
            return Err("null metadata key".into());
        }
        let key = unsafe { CStr::from_ptr(key) }.to_bytes();
        unsafe { copy_optional(cache.get(key).map(value_bytes), output, capacity) }
    }
);
ffi!(
    llama_model_meta_key_by_index(
        model: *const ModelHandle,
        index: i32,
        output: *mut c_char,
        capacity: usize,
    ) -> i32,
    -1,
    {
        let cache = unsafe { reference(model)? }.data.metadata.exact()?;
        let entry = usize::try_from(index)
            .ok()
            .and_then(|index| cache.entries.get(index));
        unsafe { copy_optional(entry.map(|entry| entry.key.as_slice()), output, capacity) }
    }
);
ffi!(
    llama_model_meta_val_str_by_index(
        model: *const ModelHandle,
        index: i32,
        output: *mut c_char,
        capacity: usize,
    ) -> i32,
    -1,
    {
        let cache = unsafe { reference(model)? }.data.metadata.exact()?;
        let entry = usize::try_from(index)
            .ok()
            .and_then(|index| cache.entries.get(index));
        unsafe { copy_optional(entry.map(value_bytes), output, capacity) }
    }
);
ffi!(
    llama_model_meta_key_str(key: i32) -> *const c_char,
    ptr::null(),
    {
        let value = match key {
            0 => c"general.sampling.sequence",
            1 => c"general.sampling.top_k",
            2 => c"general.sampling.top_p",
            3 => c"general.sampling.min_p",
            4 => c"general.sampling.xtc_probability",
            5 => c"general.sampling.xtc_threshold",
            6 => c"general.sampling.temp",
            7 => c"general.sampling.penalty_last_n",
            8 => c"general.sampling.penalty_repeat",
            9 => c"general.sampling.mirostat",
            10 => c"general.sampling.mirostat_tau",
            11 => c"general.sampling.mirostat_eta",
            _ => return Ok(ptr::null()),
        };
        Ok(value.as_ptr())
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn raw_optional_copy_preserves_guards_and_null_capacity_contract() {
        for bytes in [b"".as_slice(), &[0xff, 0x80], b"longer"] {
            for capacity in 0..=9 {
                let mut output = [0xa5u8; 32];
                let rc = unsafe {
                    copy_optional(Some(bytes), output.as_mut_ptr().add(8).cast(), capacity)
                }
                .unwrap();
                assert_eq!(rc, bytes.len() as i32);
                assert!(output[..8].iter().all(|&byte| byte == 0xa5));
                assert!(output[8 + capacity..].iter().all(|&byte| byte == 0xa5));
                if capacity > 0 {
                    let n = bytes.len().min(capacity - 1);
                    assert_eq!(&output[8..8 + n], &bytes[..n]);
                    assert_eq!(output[8 + n], 0);
                    assert!(
                        output[8 + n + 1..8 + capacity]
                            .iter()
                            .all(|&byte| byte == 0xa5)
                    );
                }
            }
            assert_eq!(
                unsafe { copy_optional(Some(bytes), ptr::null_mut(), 0) }.unwrap(),
                bytes.len() as i32
            );
        }
        let mut output = [0xa5u8; 8];
        assert_eq!(
            unsafe { copy_optional(None, output.as_mut_ptr().cast(), 8) }.unwrap(),
            -1
        );
        assert_eq!(output, [0, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5]);
        assert!(unsafe { copy_optional(None, ptr::null_mut(), 1) }.is_err());
        assert!(unsafe { copy_optional(None, ptr::null_mut(), usize::MAX) }.is_err());
    }
    #[test]
    fn aggregate_metadata_budget_and_index_work_are_enforced() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/ffi/native-context-model.gguf");
        let files = GgufSet::open(path).unwrap();
        for profile in [Profile::Msvc, Profile::Gnu] {
            let cache = MetadataCache::build_profile(
                &files,
                profile,
                files.metadata_storage_limit(),
                MAX_METADATA_WORK,
            )
            .unwrap();
            let exact = cache._peak_charge;
            assert_eq!(
                MetadataCache::build_profile(&files, profile, exact, MAX_METADATA_WORK)
                    .unwrap()
                    .entries
                    .len(),
                17
            );
            assert!(
                MetadataCache::build_profile(&files, profile, exact - 1, MAX_METADATA_WORK)
                    .is_err()
            );
            assert!(MetadataCache::build_profile(&files, profile, usize::MAX, 0).is_err());
        }
        assert!(add(usize::MAX, 1).is_err());
        assert!(multiply(usize::MAX, 2).is_err());
    }
}
