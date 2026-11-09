use super::*;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "llama-checkpoint-controls-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn wire(header: &str, payload: &[u8]) -> Vec<u8> {
    let mut output = (header.len() as u64).to_le_bytes().to_vec();
    output.extend_from_slice(header.as_bytes());
    output.extend_from_slice(payload);
    output
}
fn one(payload: &[u8]) -> Vec<u8> {
    wire(
        &format!(
            r#"{{"x":{{"dtype":"U8","shape":[{}],"data_offsets":[0,{}]}}}}"#,
            payload.len(),
            payload.len()
        ),
        payload,
    )
}
fn put(path: &Path, bytes: &[u8]) {
    fs::write(path, bytes).unwrap();
}
fn limited(error: Error) {
    assert!(matches!(error, Error::Limit(_)), "{error}");
}
fn text(value: &str) -> Text {
    Text::try_from_str(value).unwrap()
}
fn options() -> CheckpointOptions {
    CheckpointOptions::default()
}

#[test]
fn owned_file_survives_path_replacement_and_checked_truncation() {
    let temp = Temp::new();
    let path = temp.0.join("model.safetensors");
    put(&path, &one(b"original"));
    // An already-opened capability permits replacement on both hosts.
    let owner = File::open(&path).unwrap();
    let file = SafetensorsFile::from_file(owner, Limits::default()).unwrap();
    fs::rename(&path, temp.0.join("old.safetensors")).unwrap();
    put(&path, &one(b"replaced"));
    assert_eq!(file.read_tensor(0).unwrap(), b"original");
    drop(file);
    let file = SafetensorsFile::from_file(File::open(&path).unwrap(), Limits::default()).unwrap();
    OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(file.data_offset() + 2)
        .unwrap();
    assert!(
        matches!(file.read_tensor(0), Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof)
    );
}

#[test]
fn streaming_ranges_seek_and_separate_typed_validation() {
    let temp = Temp::new();
    let path = temp.0.join("model.safetensors");
    put(&path, &one(b"abcdefgh"));
    let file = SafetensorsFile::open(&path, Limits::default()).unwrap();
    let mut reader = file.reader(0).unwrap();
    assert_eq!(reader.len(), 8);
    let mut bytes = [0; 3];
    reader.read_exact(&mut bytes).unwrap();
    assert_eq!(&bytes, b"abc");
    assert_eq!(reader.seek(SeekFrom::End(-3)).unwrap(), 5);
    reader.read_exact(&mut bytes).unwrap();
    assert_eq!(&bytes, b"fgh");
    assert!(reader.seek(SeekFrom::Current(1)).is_err());
    assert!(reader.seek(SeekFrom::Start(u64::MAX)).is_err());
    assert_eq!(reader.read(&mut bytes).unwrap(), 0);
    assert!(file.reader(usize::MAX).is_err());
    let layout = file.tensors()[0].typed_layout().unwrap();
    assert_eq!(
        layout,
        TypedLayout {
            dtype: DType::U8,
            shape: vec![8],
            elements: 8,
            bytes: 8
        }
    );
    drop(file);
    put(
        &path,
        &wire(
            r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}}"#,
            b"abcdefgh",
        ),
    );
    let file = SafetensorsFile::open(&path, Limits::default()).unwrap();
    assert_eq!(file.read_tensor(0).unwrap(), b"abcdefgh");
    assert!(file.tensors()[0].typed_layout().is_err());
}
#[test]
fn raw_read_limit_does_not_disable_bounded_streaming() {
    let temp = Temp::new();
    let path = temp.0.join("model.safetensors");
    put(&path, &one(b"abcdef"));
    let file = SafetensorsFile::open(
        &path,
        Limits {
            max_read_bytes: 3,
            ..Limits::default()
        },
    )
    .unwrap();
    limited(file.read_tensor(0).unwrap_err());
    let mut reader = file.reader(0).unwrap();
    let mut output = [0; 6];
    reader.read_exact(&mut output).unwrap();
    assert_eq!(&output, b"abcdef");
}
#[test]
fn cumulative_header_index_and_descriptor_budgets() {
    let temp = Temp::new();
    let raw = one(b"abcd");
    let header = (raw.len() - 12) as u64;
    put(&temp.0.join("model-a.safetensors"), &raw);
    put(&temp.0.join("model-b.safetensors"), &raw);
    let base = Checkpoint::open_directory(&temp.0, options(), Limits::default()).unwrap();
    assert_eq!(base.file_count(), 2);
    assert_eq!(base.len(), 1);
    assert_eq!(base.usage().files, 2);
    assert_eq!(base.usage().tensors, 2);
    assert_eq!(base.usage().header_bytes, 2 * header);
    drop(base);
    limited(
        Checkpoint::open_directory(
            &temp.0,
            options(),
            Limits {
                max_header_bytes: header - 1,
                ..Limits::default()
            },
        )
        .unwrap_err(),
    );
    limited(
        Checkpoint::open_directory(
            &temp.0,
            options(),
            Limits {
                max_total_header_bytes: header * 2 - 1,
                ..Limits::default()
            },
        )
        .unwrap_err(),
    );
    limited(
        Checkpoint::open_directory(
            &temp.0,
            options(),
            Limits {
                max_files: 1,
                ..Limits::default()
            },
        )
        .unwrap_err(),
    );
    limited(
        Checkpoint::open_directory(
            &temp.0,
            options(),
            Limits {
                max_tensors: 1,
                ..Limits::default()
            },
        )
        .unwrap_err(),
    );
    limited(
        Checkpoint::open_directory(
            &temp.0,
            options(),
            Limits {
                max_directory_entries: 1,
                ..Limits::default()
            },
        )
        .unwrap_err(),
    );

    put(
        &temp.0.join("model.safetensors.index.json"),
        br#"{"weight_map":{}}"#,
    );
    let empty = Checkpoint::open_directory(
        &temp.0,
        options(),
        Limits {
            max_files: 0,
            ..Limits::default()
        },
    )
    .unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.file_count(), 0);
    limited(
        Checkpoint::open_directory(
            &temp.0,
            options(),
            Limits {
                max_index_bytes: 1,
                ..Limits::default()
            },
        )
        .unwrap_err(),
    );
}
#[test]
fn decoded_nodes_text_depth_dimensions_and_numeric_limits() {
    limited(
        json::parse(
            b"[]",
            Limits {
                max_json_nodes: 0,
                ..Limits::default()
            },
        )
        .unwrap_err(),
    );
    limited(
        json::parse(
            br#"["a","b"]"#,
            Limits {
                max_text_bytes: 1,
                ..Limits::default()
            },
        )
        .unwrap_err(),
    );
    limited(
        json::parse(
            br#""abc""#,
            Limits {
                max_string_bytes: 2,
                ..Limits::default()
            },
        )
        .unwrap_err(),
    );
    limited(
        json::parse(
            b"1234",
            Limits {
                max_integer_digits: 3,
                ..Limits::default()
            },
        )
        .unwrap_err(),
    );
    limited(
        json::parse(
            b"[]",
            Limits {
                max_json_depth: 513,
                ..Limits::default()
            },
        )
        .unwrap_err(),
    );
    let nested = format!("{}0{}", "[".repeat(129), "]".repeat(129));
    limited(
        json::parse(
            nested.as_bytes(),
            Limits {
                max_json_depth: 128,
                ..Limits::default()
            },
        )
        .unwrap_err(),
    );
    assert!(json::parse(nested.as_bytes(), Limits::default()).is_ok());
    let temp = Temp::new();
    let path = temp.0.join("model.safetensors");
    put(
        &path,
        &wire(
            r#"{"x":{"dtype":"U8","shape":[2,2],"data_offsets":[0,4]}}"#,
            b"abcd",
        ),
    );
    limited(
        SafetensorsFile::open(
            &path,
            Limits {
                max_dimensions: 1,
                ..Limits::default()
            },
        )
        .unwrap_err(),
    );
    put(
        &path,
        &wire(
            r#"{"x":{"dtype":"U8","shape":[18446744073709551615,2],"data_offsets":[0,4]}}"#,
            b"abcd",
        ),
    );
    let file = SafetensorsFile::open(&path, Limits::default()).unwrap();
    assert_eq!(file.read_tensor(0).unwrap(), b"abcd");
    limited(file.tensors()[0].typed_layout().unwrap_err());
}
#[test]
fn index_override_global_set_and_capability_selection_are_distinct() {
    let temp = Temp::new();
    let a = temp.0.join("a.weights");
    let b = temp.0.join("b.weights");
    put(
        &a,
        &wire(
            r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
            b"a",
        ),
    );
    put(
        &b,
        &wire(
            r#"{"b":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
            b"b",
        ),
    );
    let caps = || {
        vec![
            (text("a.weights"), File::open(&a).unwrap()),
            (text("b.weights"), File::open(&b).unwrap()),
        ]
    };
    let index = br#"{"weight_map":{"a":"b.weights","b":"a.weights"}}"#;
    let checkpoint =
        Checkpoint::from_open_files(caps(), Some(index), options(), Limits::default()).unwrap();
    assert_eq!(checkpoint.find_str("a").unwrap().read().unwrap(), b"a");
    assert_eq!(checkpoint.find_str("b").unwrap().read().unwrap(), b"b");
    assert!(
        Checkpoint::from_open_files(
            caps(),
            Some(br#"{"weight_map":{"a":"a.weights","absent":"b.weights"}}"#),
            options(),
            Limits::default()
        )
        .is_err()
    );
    limited(
        Checkpoint::from_open_files(
            caps(),
            Some(index),
            options(),
            Limits {
                max_files: 1,
                ..Limits::default()
            },
        )
        .unwrap_err(),
    );
    let empty = Checkpoint::from_open_files(
        caps(),
        Some(br#"{"weight_map":{}}"#),
        options(),
        Limits::default(),
    )
    .unwrap();
    assert!(empty.is_empty());
}
#[test]
fn directory_path_policies_and_unicode_names() {
    let temp = Temp::new();
    put(&temp.0.join("model-水🙂.safetensors"), &one(b"unicode"));
    let checkpoint = Checkpoint::open_directory(&temp.0, options(), Limits::default()).unwrap();
    assert_eq!(
        checkpoint.find_str("x").unwrap().read().unwrap(),
        b"unicode"
    );
    drop(checkpoint);
    let root = platform::Root::open(&temp.0).unwrap();
    for name in ["", ".", "..", "../outside", "a/../b", "/absolute", "a\0b"] {
        assert!(
            root.open_relative(&text(name), &mut Budget::new(Limits::default()).unwrap())
                .is_err(),
            "{name:?}"
        );
    }
    #[cfg(windows)]
    for name in ["C:\\outside", "C:outside", "\\outside", "a:stream"] {
        assert!(
            root.open_relative(&text(name), &mut Budget::new(Limits::default()).unwrap())
                .is_err(),
            "{name:?}"
        );
    }
    limited(
        root.open_relative(
            &text("a/b"),
            &mut Budget::new(Limits {
                max_path_components: 1,
                ..Limits::default()
            })
            .unwrap(),
        )
        .unwrap_err(),
    );
    limited(
        root.open_relative(
            &text("abcdef"),
            &mut Budget::new(Limits {
                max_path_bytes: 5,
                ..Limits::default()
            })
            .unwrap(),
        )
        .unwrap_err(),
    );
}
fn symlink_directory(target: &Path, alias: &Path) {
    #[cfg(unix)]
    std::os::unix::fs::symlink(target, alias).unwrap();
    #[cfg(windows)]
    std::os::windows::fs::symlink_dir(target, alias).unwrap();
}
fn handle_count() -> u64 {
    platform::handle_count().unwrap()
}
fn direct_raw(owner: File) -> Vec<u8> {
    SafetensorsFile::from_file(owner, Limits::default())
        .unwrap()
        .read_tensor(0)
        .unwrap()
}
fn assert_count(expected: u64) {
    assert_eq!(handle_count(), expected, "file handle/FD leak");
}

/// A child process isolates handle accounting from other parallel tests.
/// This is deliberately ordinary coverage, not a silent environment skip.
#[test]
fn handles_failures_and_swaps() {
    if std::env::var_os("LLAMA_CHECKPOINT_HANDLE_CHILD").is_none() {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "tests::handles_failures_and_swaps",
                "--test-threads=1",
                "--nocapture",
            ])
            .env("LLAMA_CHECKPOINT_HANDLE_CHILD", "1")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "child failed:\n{}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        return;
    }
    let temp = Temp::new();
    let model = temp.0.join("model");
    let outside = temp.0.join("outside");
    fs::create_dir_all(model.join("nested")).unwrap();
    fs::create_dir(&outside).unwrap();
    put(&model.join("nested/a.safetensors"), &one(b"retained"));
    put(&outside.join("a.safetensors"), &one(b"outside!"));
    let baseline = handle_count();
    {
        let root = platform::Root::open(&model).unwrap();
        let retained = handle_count();
        // Error paths after each native-open stage must release every temporary
        // capability, including a successful final handle not yet returned.
        let prior_panic_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        for stage in 0..if cfg!(windows) { 2 } else { 3 } {
            for _ in 0..8 {
                platform::set_open_hook(Some(Box::new(move |current| {
                    assert_ne!(current, stage, "injected open-stage failure");
                })));
                let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    root.open_relative(
                        &text("nested/a.safetensors"),
                        &mut Budget::new(Limits::default()).unwrap(),
                    )
                }));
                platform::set_open_hook(None);
                assert!(failure.is_err());
                assert_count(retained);
            }
        }
        std::panic::set_hook(prior_panic_hook);
        for _ in 0..16 {
            assert!(
                root.entries(
                    &mut Budget::new(Limits {
                        max_directory_entries: 0,
                        ..Limits::default()
                    })
                    .unwrap()
                )
                .is_err()
            );
            assert!(
                root.open_relative(
                    &text("nested/missing"),
                    &mut Budget::new(Limits::default()).unwrap()
                )
                .is_err()
            );
            assert_count(retained);
        }

        let nested = model.join("nested");
        let parked = model.join("parked");
        let destination = outside.clone();
        let from = nested.clone();
        let to = parked.clone();
        platform::set_open_hook(Some(Box::new(move |stage| {
            if stage == 0 {
                fs::rename(&from, &to).unwrap();
                symlink_directory(&destination, &from);
            }
        })));
        let rejected = root.open_relative(
            &text("nested/a.safetensors"),
            &mut Budget::new(Limits::default()).unwrap(),
        );
        platform::set_open_hook(None);
        assert!(
            rejected.is_err(),
            "reparse/symlink substitution must not read outside"
        );
        assert_count(retained);
        fs::remove_dir(&nested)
            .or_else(|_| fs::remove_file(&nested))
            .unwrap();
        fs::rename(&parked, &nested).unwrap();

        #[cfg(target_os = "linux")]
        {
            // Swap only after the child directory FD is retained. Its original
            // file remains selected; a canonicalized-path reopen would escape.
            let from = nested.clone();
            let to = parked.clone();
            let destination = outside.clone();
            platform::set_open_hook(Some(Box::new(move |stage| {
                if stage == 1 {
                    fs::rename(&from, &to).unwrap();
                    symlink_directory(&destination, &from);
                }
            })));
            let owner = root
                .open_relative(
                    &text("nested/a.safetensors"),
                    &mut Budget::new(Limits::default()).unwrap(),
                )
                .unwrap();
            platform::set_open_hook(None);
            assert_eq!(direct_raw(owner), b"retained");
            assert_count(retained);
            fs::remove_file(&nested).unwrap();
            fs::rename(&parked, &nested).unwrap();

            let file_path = nested.join("a.safetensors");
            let old_path = nested.join("old.safetensors");
            let from = file_path.clone();
            let to = old_path.clone();
            platform::set_open_hook(Some(Box::new(move |stage| {
                if stage == 2 {
                    fs::rename(&from, &to).unwrap();
                    put(&from, &one(b"outside!"));
                }
            })));
            let owner = root
                .open_relative(
                    &text("nested/a.safetensors"),
                    &mut Budget::new(Limits::default()).unwrap(),
                )
                .unwrap();
            platform::set_open_hook(None);
            assert_eq!(direct_raw(owner), b"retained");
            assert_count(retained);
            fs::remove_file(&file_path).unwrap();
            fs::rename(&old_path, &file_path).unwrap();
        }
        #[cfg(windows)]
        {
            let owner = root
                .open_relative(
                    &text("nested/a.safetensors"),
                    &mut Budget::new(Limits::default()).unwrap(),
                )
                .unwrap();
            assert!(
                OpenOptions::new()
                    .write(true)
                    .open(nested.join("a.safetensors"))
                    .is_err()
            );
            assert!(
                fs::rename(
                    nested.join("a.safetensors"),
                    nested.join("renamed.safetensors")
                )
                .is_err()
            );
            assert_eq!(direct_raw(owner), b"retained");
            assert_count(retained);
        }
    }
    assert_count(baseline);

    // Moving the trusted root name cannot replace the retained directory.
    {
        let root = platform::Root::open(&model).unwrap();
        let parked = temp.0.join("retained-root");
        let from = model.clone();
        let to = parked.clone();
        let destination = outside.clone();
        platform::set_open_hook(Some(Box::new(move |stage| {
            if stage == 0 {
                fs::rename(&from, &to).unwrap();
                symlink_directory(&destination, &from);
            }
        })));
        let owner = root
            .open_relative(
                &text("nested/a.safetensors"),
                &mut Budget::new(Limits::default()).unwrap(),
            )
            .unwrap();
        platform::set_open_hook(None);
        assert_eq!(direct_raw(owner), b"retained");
    }
    assert_count(baseline);

    // Partially parsed multi-file checkpoints and invalid supplied capabilities
    // drop all owners, even when a later header or global-set check rejects.
    let invalid_root = temp.0.join("invalid");
    fs::create_dir(&invalid_root).unwrap();
    put(&invalid_root.join("model-a.safetensors"), &one(b"good"));
    put(&invalid_root.join("model-b.safetensors"), b"bad");
    for _ in 0..32 {
        assert!(Checkpoint::open_directory(&invalid_root, options(), Limits::default()).is_err());
        let supplied = vec![
            (
                text("model-a.safetensors"),
                File::open(invalid_root.join("model-a.safetensors")).unwrap(),
            ),
            (
                text("model-b.safetensors"),
                File::open(invalid_root.join("model-b.safetensors")).unwrap(),
            ),
        ];
        assert!(Checkpoint::from_open_files(supplied, None, options(), Limits::default()).is_err());
        assert_count(baseline);
    }
    put(&invalid_root.join("model-b.safetensors"), &one(b"also"));
    put(
        &invalid_root.join("model.safetensors.index.json"),
        br#"{"weight_map":{"x":"model-a.safetensors","absent":"model-b.safetensors"}}"#,
    );
    for _ in 0..32 {
        assert!(Checkpoint::open_directory(&invalid_root, options(), Limits::default()).is_err());
        assert_count(baseline);
    }
    eprintln!(
        "retained path swaps, native-stage unwind injection and parse/index failure cleanup passed at {baseline} process handles/FDs"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn non_utf8_filesystem_labels_roundtrip_source_surrogateescape() {
    use std::os::unix::ffi::OsStringExt;
    let temp = Temp::new();
    let mut bytes = b"model-".to_vec();
    bytes.push(0xff);
    bytes.extend_from_slice(b".safetensors");
    let name = std::ffi::OsString::from_vec(bytes);
    put(&temp.0.join(name), &one(b"raw-name"));
    let checkpoint = Checkpoint::open_directory(&temp.0, options(), Limits::default()).unwrap();
    assert_eq!(
        checkpoint.find_str("x").unwrap().read().unwrap(),
        b"raw-name"
    );
    let label = checkpoint.file(0).unwrap().0;
    assert!(label.as_str().is_none());
    assert!(label.codepoints().any(|code| code == 0xdcff));
}
#[cfg(windows)]
#[test]
fn utf16_native_path_ceiling_precedes_native_open() {
    let temp = Temp::new();
    let root = platform::Root::open(&temp.0).unwrap();
    let name = text(&"a".repeat(32768));
    let error = root
        .open_relative(&name, &mut Budget::new(Limits::default()).unwrap())
        .unwrap_err();
    assert!(matches!(error, Error::Limit("native UTF-16 path bytes")));
}
#[cfg(target_os = "linux")]
#[test]
fn fifo_is_rejected_without_opening_data() {
    if std::env::var_os("LLAMA_CHECKPOINT_FIFO_CHILD").is_some() {
        let temp = Temp::new();
        let path = temp.0.join("model.safetensors");
        platform::make_fifo(&path).unwrap();
        let root = platform::Root::open(&temp.0).unwrap();
        let before = handle_count();
        assert!(
            root.open_relative(
                &text("model.safetensors"),
                &mut Budget::new(Limits::default()).unwrap()
            )
            .is_err()
        );
        assert_count(before);
        return;
    }
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "tests::fifo_is_rejected_without_opening_data",
            "--test-threads=1",
        ])
        .env("LLAMA_CHECKPOINT_FIFO_CHILD", "1")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let started = std::time::Instant::now();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "FIFO rejection child failed");
            break;
        }
        if started.elapsed() > std::time::Duration::from_secs(5) {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("opener blocked on FIFO instead of rejecting its O_PATH metadata");
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}
