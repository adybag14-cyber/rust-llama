#![cfg(any(windows, all(target_os = "linux", target_arch = "x86_64")))]
use llama_checkpoint::{
    Checkpoint, CheckpointOptions, Error, Limits, SourceDirectory, SourceLimits, SourceUsage,
};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "llama-source-directory-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        Self(root)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn tensor_file() -> Vec<u8> {
    let header = br#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header);
    out.extend_from_slice(&0x3f80_0000u32.to_le_bytes());
    out
}
#[test]
fn cached_sidecars_charge_once_and_cumulative_limit_precedes_read() {
    let dir = Temp::new();
    fs::write(dir.0.join("a.json"), b"abc").unwrap();
    fs::write(dir.0.join("b.json"), b"def").unwrap();
    let limits = SourceLimits {
        max_file_bytes: 4,
        max_total_bytes: 5,
        max_files: 2,
        ..SourceLimits::default()
    };
    let mut source = SourceDirectory::open(&dir.0, limits).unwrap();
    assert_eq!(source.read_required("a.json").unwrap(), b"abc");
    assert_eq!(source.read_required("a.json").unwrap(), b"abc");
    assert_eq!(source.usage(), SourceUsage { files: 1, bytes: 3 });
    assert!(matches!(
        source.read_optional("b.json"),
        Err(Error::Limit("cumulative sidecar bytes"))
    ));
    assert_eq!(source.usage(), SourceUsage { files: 1, bytes: 3 });
    assert!(source.read_optional("missing.json").unwrap().is_none());
    assert!(matches!(
        source.read_optional("../a.json"),
        Err(Error::Path(_))
    ));
    assert_eq!(source.read_required("a.json").unwrap(), b"abc");
}
#[test]
fn sidecar_single_file_and_owner_limits_are_checked() {
    let dir = Temp::new();
    fs::write(dir.0.join("large"), [0u8; 128]).unwrap();
    fs::write(dir.0.join("empty"), []).unwrap();
    fs::write(dir.0.join("another"), []).unwrap();
    let limits = SourceLimits {
        max_file_bytes: 16,
        max_files: 1,
        ..SourceLimits::default()
    };
    let mut source = SourceDirectory::open(&dir.0, limits).unwrap();
    assert!(matches!(
        source.read_optional("large"),
        Err(Error::Limit("individual sidecar bytes"))
    ));
    assert_eq!(source.usage(), SourceUsage::default());
    assert_eq!(source.read_required("empty").unwrap(), b"");
    assert!(matches!(
        source.read_optional("another"),
        Err(Error::Limit("sidecar owner count"))
    ));
    assert_eq!(source.usage(), SourceUsage { files: 1, bytes: 0 });
}
#[test]
fn same_retained_root_opens_sidecars_and_checkpoint() {
    let dir = Temp::new();
    fs::write(dir.0.join("config.json"), b"{}").unwrap();
    fs::write(dir.0.join("model.safetensors"), tensor_file()).unwrap();
    let mut source = SourceDirectory::open(&dir.0, SourceLimits::default()).unwrap();
    assert_eq!(source.read_required("config.json").unwrap(), b"{}");
    let checkpoint =
        Checkpoint::open_source(&source, CheckpointOptions::default(), Limits::default()).unwrap();
    assert_eq!(checkpoint.file_count(), 1);
    let (_, file) = checkpoint.file(0).unwrap();
    assert_eq!(file.read_tensor(0).unwrap(), 0x3f80_0000u32.to_le_bytes());
}
#[cfg(unix)]
#[test]
fn replacing_root_path_cannot_redirect_sidecars_or_shards() {
    let dir = Temp::new();
    let path = dir.0.join("root");
    let moved = dir.0.join("retained");
    fs::create_dir(&path).unwrap();
    fs::write(path.join("config.json"), b"original").unwrap();
    fs::write(path.join("model.safetensors"), tensor_file()).unwrap();
    let mut source = SourceDirectory::open(&path, SourceLimits::default()).unwrap();
    fs::rename(&path, &moved).unwrap();
    fs::create_dir(&path).unwrap();
    fs::write(path.join("config.json"), b"replacement").unwrap();
    fs::write(path.join("model.safetensors"), b"invalid replacement").unwrap();
    assert_eq!(source.read_required("config.json").unwrap(), b"original");
    assert_eq!(
        Checkpoint::open_source(&source, CheckpointOptions::default(), Limits::default())
            .unwrap()
            .file_count(),
        1
    );
}
#[cfg(windows)]
#[test]
fn cached_sidecar_keeps_read_only_share_owner() {
    let dir = Temp::new();
    let path = dir.0.join("config.json");
    fs::write(&path, b"original").unwrap();
    let mut source = SourceDirectory::open(&dir.0, SourceLimits::default()).unwrap();
    source.read_required("config.json").unwrap();
    assert!(fs::OpenOptions::new().write(true).open(&path).is_err());
    drop(source);
    assert!(fs::OpenOptions::new().write(true).open(&path).is_ok());
}
#[test]
fn parser_usage_includes_discarded_duplicate_values() {
    let (value, usage) =
        llama_checkpoint::json::parse_with_usage(br#"{"a":1,"a":2}"#, Limits::default()).unwrap();
    assert_eq!(value.as_object().unwrap().len(), 1);
    assert_eq!(usage.json_nodes, 5);
    let remaining = Limits {
        max_json_nodes: 6 - usage.json_nodes,
        ..Limits::default()
    };
    assert!(matches!(
        llama_checkpoint::json::parse(br#"{"b":3}"#, remaining),
        Err(Error::Limit("JSON nodes"))
    ));
}

#[test]
fn checked_subdirectory_enumeration_and_followup_sidecars() {
    let dir = Temp::new();
    fs::create_dir_all(dir.0.join("chat_templates/nested")).unwrap();
    fs::write(dir.0.join("chat_templates/a.jinja"), b"A").unwrap();
    fs::write(dir.0.join("chat_templates/b.jinja"), b"B").unwrap();
    fs::write(dir.0.join("regular"), b"not a directory").unwrap();
    let mut source = SourceDirectory::open(&dir.0, SourceLimits::default()).unwrap();
    let mut names = source
        .entries_under("chat_templates")
        .unwrap()
        .unwrap()
        .into_iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(names, ["a.jinja", "b.jinja", "nested"]);
    assert_eq!(
        source.read_required("chat_templates/a.jinja").unwrap(),
        b"A"
    );
    assert_eq!(source.usage(), SourceUsage { files: 1, bytes: 1 });
    assert!(source.entries_under("missing").unwrap().is_none());
    assert!(source.entries_under("regular").unwrap().is_none());
    assert!(matches!(
        source.entries_under("../outside"),
        Err(Error::Path(_))
    ));
    let limits = SourceLimits {
        paths: Limits {
            max_directory_entries: 1,
            ..Limits::default()
        },
        ..SourceLimits::default()
    };
    let mut bounded = SourceDirectory::open(&dir.0, limits).unwrap();
    assert!(matches!(
        bounded.entries_under("chat_templates"),
        Err(Error::Limit("directory entries"))
    ));
    assert_eq!(bounded.usage(), SourceUsage::default());
}
#[test]
fn optional_subdirectory_links_are_explicit_errors() {
    let dir = Temp::new();
    fs::create_dir(dir.0.join("target")).unwrap();
    fs::write(dir.0.join("target/a.jinja"), b"source").unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(dir.0.join("target"), dir.0.join("alias")).unwrap();
    #[cfg(windows)]
    std::os::windows::fs::symlink_dir(dir.0.join("target"), dir.0.join("alias")).unwrap();
    let mut source = SourceDirectory::open(&dir.0, SourceLimits::default()).unwrap();
    assert!(source.entries_under("alias").is_err());
    assert!(source.read_optional("alias/a.jinja").is_err());
    #[cfg(unix)]
    {
        assert!(matches!(
            source.entries_under("alias"),
            Err(Error::Path("relative directory component is a symlink"))
        ));
        assert!(matches!(
            source.read_optional("alias/a.jinja"),
            Err(Error::Path("relative directory component is a symlink"))
        ));
    }
    assert_eq!(source.usage(), SourceUsage::default());
}
