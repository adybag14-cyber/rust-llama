//! Anonymous owned-file offsets, retained identity and resource regressions.
use llama_gguf::{GgufSet, ShardLimits, TensorData, Value};
use llama_quant::DType;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

struct Work(PathBuf);
impl Work {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "rust-llama-owned-gguf-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Work {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn input(split: bool) -> Vec<u8> {
    let mut metadata = BTreeMap::from([("test".into(), Value::String("owned file".into()))]);
    if split {
        metadata.insert("split.count".into(), Value::U16(3));
        metadata.insert("split.no".into(), Value::U16(2));
        metadata.insert("split.tensors.count".into(), Value::I32(100));
    }
    let payload: Vec<u8> = (0..96).collect();
    let tensors = [
        TensorData {
            name: "a",
            dimensions: &[33],
            dtype: DType::I8,
            data: &payload[..33],
        },
        TensorData {
            name: "b",
            dimensions: &[63],
            dtype: DType::I8,
            data: &payload[33..],
        },
    ];
    let mut bytes = Cursor::new(Vec::new());
    llama_gguf::write(&mut bytes, &metadata, &tensors).unwrap();
    bytes.into_inner()
}
fn embedded(path: &Path, bytes: &[u8], prefix: usize, suffix: bool) -> File {
    let mut file = File::create(path).unwrap();
    file.write_all(&vec![0xa7; prefix]).unwrap();
    file.write_all(bytes).unwrap();
    if suffix {
        file.write_all(&[0xcc; 47]).unwrap();
    }
    drop(file);
    File::open(path).unwrap()
}

#[test]
fn owned_file_origin_alignment_and_handle_identity() {
    let work = Work::new();
    let bytes = input(false);
    for prefix in [0, 1, 31, 32, 77] {
        let path = work.0.join(format!("input-{prefix}.gguf"));
        let file = embedded(&path, &bytes, prefix, true);
        let mut set = GgufSet::from_file_at(file, prefix as u64).unwrap();
        assert_eq!(set.primary_path(), Path::new(""));
        assert_eq!(set.shard_count(), 1);
        assert_eq!(set.file_bytes(), (bytes.len() + 47) as u64);
        assert_eq!(set.shard_info(0).unwrap().data_offset % 32, 0);
        let renamed = path.with_extension("retained");
        fs::rename(&path, renamed).unwrap();
        fs::write(&path, b"replacement is not the opened GGUF").unwrap();
        assert_eq!(
            set.read_tensor("a").unwrap(),
            (0_u8..33).collect::<Vec<_>>()
        );
        let mut region = [0; 17];
        set.read_tensor_range("b", 19, &mut region).unwrap();
        assert_eq!(&region, &(52_u8..69).collect::<Vec<_>>()[..]);
        assert!(set.read_tensor_range("b", 62, &mut [0; 2]).is_err());
        assert!(set.read_tensor_range("b", u64::MAX, &mut [0; 1]).is_err());
    }
}

#[test]
fn anonymous_stream_does_not_discover_sibling_paths() {
    let work = Work::new();
    let path = work.0.join("anonymous.gguf");
    let bytes = input(true);
    let file = embedded(&path, &bytes, 0, false);
    assert!(GgufSet::open(&path).is_err());
    let mut set = GgufSet::from_file_at(file, 0).unwrap();
    assert_eq!(set.metadata().get("split.count"), Some(&Value::U16(3)));
    assert_eq!(set.shard_count(), 1);
    assert_eq!(
        set.read_tensor("b").unwrap(),
        (33_u8..96).collect::<Vec<_>>()
    );
}

#[test]
fn owned_file_rejects_out_of_region_offsets_and_retains_parser_limits() {
    let work = Work::new();
    let path = work.0.join("input.gguf");
    let bytes = input(false);
    for offset in [bytes.len() as u64 + 1, i64::MAX as u64, u64::MAX] {
        let file = embedded(&path, &bytes, 0, false);
        assert!(GgufSet::from_file_at(file, offset).is_err());
    }
    let truncated = embedded(&path, &bytes[..bytes.len() - 9], 7, false);
    assert!(GgufSet::from_file_at(truncated, 7).is_err());
    for limits in [
        ShardLimits {
            max_shards: 0,
            ..ShardLimits::default()
        },
        ShardLimits {
            max_metadata_storage_bytes: 1,
            ..ShardLimits::default()
        },
        {
            let mut limits = ShardLimits::default();
            limits.gguf.max_tensors = 1;
            limits
        },
        {
            let mut limits = ShardLimits::default();
            limits.gguf.max_metadata_bytes = 24;
            limits
        },
    ] {
        let file = embedded(&path, &bytes, 7, true);
        assert!(GgufSet::from_file_at_with_limits(file, 7, limits).is_err());
    }
}
