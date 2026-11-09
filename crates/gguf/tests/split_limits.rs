//! Resource and file-ownership regressions for the streaming split/merge API.
//! These are explicit Rust error contracts; native output parity is exercised
//! independently by the CLI's frozen original corpus.
use llama_gguf::split::{self, MergeOptions, SplitLimit, SplitOptions};
use llama_gguf::{Error, Limits, ShardLimits, TensorData, Value};
use llama_quant::DType;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

struct Work(PathBuf);
impl Work {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "rust-llama-gguf-limits-{}-{}",
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
fn input(path: &Path) -> Vec<u8> {
    let payload = (0..96).collect::<Vec<u8>>();
    let metadata = BTreeMap::from([("test".into(), Value::String("streaming".into()))]);
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
    llama_gguf::write(File::create(path).unwrap(), &metadata, &tensors).unwrap();
    fs::read(path).unwrap()
}
fn files(path: &Path) -> BTreeMap<String, Vec<u8>> {
    fs::read_dir(path)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (
                entry.file_name().into_string().unwrap(),
                fs::read(entry.path()).unwrap(),
            )
        })
        .collect()
}
fn two_shards(work: &Work) -> Vec<PathBuf> {
    let source = work.0.join("input.gguf");
    input(&source);
    let prefix = work.0.join("parts");
    let summary = split::split(
        &source,
        &prefix,
        SplitOptions {
            limit: SplitLimit::Tensors(1),
            ..SplitOptions::default()
        },
        &ShardLimits::default(),
        &mut Vec::new(),
        &mut Vec::new(),
    )
    .unwrap();
    assert_eq!((summary.shards, summary.tensors), (2, 2));
    (0..2)
        .map(|i| split::split_path(&prefix, i, 2).unwrap())
        .collect()
}
#[test]
fn split_resource_failures_precede_file_mutation() {
    let work = Work::new();
    let source = work.0.join("input.gguf");
    input(&source);
    let before = files(&work.0);
    let options = SplitOptions {
        limit: SplitLimit::Tensors(1),
        ..SplitOptions::default()
    };
    let limits = ShardLimits {
        max_shards: 1,
        ..ShardLimits::default()
    };
    assert!(matches!(
        split::split(
            &source,
            &work.0.join("parts"),
            options,
            &limits,
            &mut Vec::new(),
            &mut Vec::new()
        ),
        Err(Error::Limit(_))
    ));
    assert_eq!(files(&work.0), before);

    let limits = ShardLimits {
        gguf: Limits {
            max_tensor_read_bytes: 0,
            ..Limits::default()
        },
        ..ShardLimits::default()
    };
    assert!(matches!(
        split::split(
            &source,
            &work.0.join("parts"),
            options,
            &limits,
            &mut Vec::new(),
            &mut Vec::new()
        ),
        Err(Error::Limit(_))
    ));
    assert_eq!(files(&work.0), before);
    let dry = SplitOptions {
        dry_run: true,
        ..options
    };
    assert!(
        split::split(
            &source,
            &work.0.join("parts"),
            dry,
            &limits,
            &mut Vec::new(),
            &mut Vec::new()
        )
        .is_ok()
    );
    assert_eq!(files(&work.0), before);
}
#[test]
fn merge_cumulative_limits_and_zero_buffer_preserve_inputs() {
    let work = Work::new();
    let shards = two_shards(&work);
    let before = files(&work.0);
    let options = MergeOptions {
        delete_splits: true,
        ..MergeOptions::default()
    };
    // The first header has four entries and the second three: each fits six,
    // but the complete owned set exceeds that explicit combined allowance.
    let limits = ShardLimits {
        gguf: Limits {
            max_metadata_entries: 6,
            ..Limits::default()
        },
        ..ShardLimits::default()
    };
    assert!(matches!(
        split::merge(
            &shards[0],
            &work.0.join("merged.gguf"),
            options,
            &limits,
            &mut Vec::new()
        ),
        Err(Error::Limit(_))
    ));
    assert_eq!(files(&work.0), before);
    let limits = ShardLimits {
        gguf: Limits {
            max_tensor_read_bytes: 0,
            ..Limits::default()
        },
        ..ShardLimits::default()
    };
    assert!(matches!(
        split::merge(
            &shards[0],
            &work.0.join("merged.gguf"),
            options,
            &limits,
            &mut Vec::new()
        ),
        Err(Error::Limit(_))
    ));
    assert_eq!(files(&work.0), before);
    // Native merge dry-run still reads payloads, unlike split dry-run.
    assert!(matches!(
        split::merge(
            &shards[0],
            &work.0.join("merged.gguf"),
            MergeOptions {
                dry_run: true,
                ..options
            },
            &limits,
            &mut Vec::new()
        ),
        Err(Error::Limit(_))
    ));
    assert_eq!(files(&work.0), before);
}
#[test]
fn minimal_streaming_buffer_keeps_payload_bytes() {
    let work = Work::new();
    let shards = two_shards(&work);
    let output = work.0.join("merged.gguf");
    let limits = ShardLimits {
        gguf: Limits {
            max_tensor_read_bytes: 1,
            ..Limits::default()
        },
        ..ShardLimits::default()
    };
    split::merge(
        &shards[0],
        &output,
        MergeOptions::default(),
        &limits,
        &mut Vec::new(),
    )
    .unwrap();
    let mut original = llama_gguf::Gguf::open(work.0.join("input.gguf")).unwrap();
    let mut merged = llama_gguf::Gguf::open(&output).unwrap();
    for name in ["a", "b"] {
        assert_eq!(
            original.read_tensor(name).unwrap(),
            merged.read_tensor(name).unwrap()
        );
    }
}
#[cfg(any(unix, windows))]
#[test]
fn split_cannot_truncate_input_through_a_hard_link() {
    let work = Work::new();
    let source = work.0.join("input.gguf");
    let original = input(&source);
    let prefix = work.0.join("alias");
    let alias = split::split_path(&prefix, 0, 1).unwrap();
    fs::hard_link(&source, &alias).unwrap();
    assert!(
        split::split(
            &source,
            &prefix,
            SplitOptions::default(),
            &ShardLimits::default(),
            &mut Vec::new(),
            &mut Vec::new()
        )
        .is_err()
    );
    assert_eq!(fs::read(&source).unwrap(), original);
    assert_eq!(fs::read(&alias).unwrap(), original);
}

#[cfg(unix)]
#[test]
fn split_checks_retained_handles_after_path_preflight() {
    use std::io::{self, Write};
    struct AliasAfterPreflight {
        source: PathBuf,
        destination: PathBuf,
        swapped: bool,
    }
    impl Write for AliasAfterPreflight {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if !self.swapped {
                fs::remove_file(&self.destination)?;
                fs::hard_link(&self.source, &self.destination)?;
                self.swapped = true;
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let work = Work::new();
    let source = work.0.join("input.gguf");
    let original = input(&source);
    let prefix = work.0.join("swapped");
    let destination = split::split_path(&prefix, 0, 1).unwrap();
    fs::write(&destination, b"unrelated output allowed to be overwritten").unwrap();
    let mut stdout = AliasAfterPreflight {
        source: source.clone(),
        destination: destination.clone(),
        swapped: false,
    };
    assert!(
        split::split(
            &source,
            &prefix,
            SplitOptions::default(),
            &ShardLimits::default(),
            &mut stdout,
            &mut Vec::new()
        )
        .is_err()
    );
    assert!(stdout.swapped);
    assert_eq!(fs::read(source).unwrap(), original);
    assert_eq!(fs::read(destination).unwrap(), original);
}
