//! Deterministic ZIP32 stored-file writer, avoiding a foreign packaging runtime.
use super::*;
use std::io::{Seek, Write};

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for byte in bytes {
        crc ^= *byte as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb88320u32 & 0u32.wrapping_sub(crc & 1));
        }
    }
    !crc
}
fn u16le(output: &mut fs::File, value: u16) -> std::io::Result<()> {
    output.write_all(&value.to_le_bytes())
}
fn u32le(output: &mut fs::File, value: u32) -> std::io::Result<()> {
    output.write_all(&value.to_le_bytes())
}
struct Entry {
    name: Vec<u8>,
    crc: u32,
    size: u32,
    offset: u32,
}
struct PartialCleanup(std::path::PathBuf);
impl Drop for PartialCleanup {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
pub fn write_zip(root: &Path, destination: &Path) -> Result<()> {
    inventory::audit_product(root)?;
    if destination.exists() {
        return Err("ZIP destination already exists; choose a new progress filename".into());
    }
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    if parent.canonicalize()?.starts_with(root.canonicalize()?) {
        return Err("ZIP destination must be outside the packaged source tree".into());
    }
    let temporary = destination.with_extension("zip.partial");
    let created = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    // Declared before `out`, so failure closes the file before cleanup (also on Windows).
    let cleanup = PartialCleanup(temporary.clone());
    let mut out = created;
    let mut files = vec![];
    inventory::walk(root, root, &mut files)?;
    files.sort();
    let mut entries = vec![];
    const DATE: u16 = ((2026 - 1980) << 9) | (11 << 5) | 8;
    for relative in files {
        let raw = relative
            .to_str()
            .ok_or("non-UTF8 source path cannot be packaged")?;
        #[cfg(unix)]
        if raw.contains('\\') {
            return Err("literal backslash in ZIP member is ambiguous".into());
        }
        let path = raw.replace('\\', "/");
        if path.split('/').any(|p| p == ".." || p.is_empty()) {
            return Err("unsafe ZIP member path".into());
        }
        let name = format!("llama-rust-t05-t08/{path}").into_bytes();
        let name_len: u16 = name.len().try_into()?;
        let bytes = fs::read(root.join(&relative))?;
        let size: u32 = bytes.len().try_into()?;
        let crc = crc32(&bytes);
        let offset: u32 = out.stream_position()?.try_into()?;
        u32le(&mut out, 0x04034b50)?;
        u16le(&mut out, 20)?;
        u16le(&mut out, 0x800)?;
        u16le(&mut out, 0)?;
        u16le(&mut out, 0)?;
        u16le(&mut out, DATE)?;
        u32le(&mut out, crc)?;
        u32le(&mut out, size)?;
        u32le(&mut out, size)?;
        u16le(&mut out, name_len)?;
        u16le(&mut out, 0)?;
        out.write_all(&name)?;
        out.write_all(&bytes)?;
        entries.push(Entry {
            name,
            crc,
            size,
            offset,
        });
    }
    let central_start: u32 = out.stream_position()?.try_into()?;
    for e in &entries {
        u32le(&mut out, 0x02014b50)?;
        u16le(&mut out, 0x0314)?;
        u16le(&mut out, 20)?;
        u16le(&mut out, 0x800)?;
        u16le(&mut out, 0)?;
        u16le(&mut out, 0)?;
        u16le(&mut out, DATE)?;
        u32le(&mut out, e.crc)?;
        u32le(&mut out, e.size)?;
        u32le(&mut out, e.size)?;
        u16le(&mut out, e.name.len().try_into()?)?;
        u16le(&mut out, 0)?;
        u16le(&mut out, 0)?;
        u16le(&mut out, 0)?;
        u16le(&mut out, 0)?;
        let executable = e.name.ends_with(b"/bin/track-runner-linux-x86_64");
        u32le(
            &mut out,
            if executable {
                0o100755 << 16
            } else {
                0o100644 << 16
            },
        )?;
        u32le(&mut out, e.offset)?;
        out.write_all(&e.name)?;
    }
    let end: u32 = out.stream_position()?.try_into()?;
    let count: u16 = entries.len().try_into()?;
    u32le(&mut out, 0x06054b50)?;
    u16le(&mut out, 0)?;
    u16le(&mut out, 0)?;
    u16le(&mut out, count)?;
    u16le(&mut out, count)?;
    u32le(&mut out, end - central_start)?;
    u32le(&mut out, central_start)?;
    u16le(&mut out, 0)?;
    out.sync_all()?;
    drop(out);
    // Same-directory hard-link publication is atomic and refuses to replace an
    // existing destination, including one created after the initial check.
    fs::hard_link(&temporary, destination)?;
    drop(cleanup);
    let bytes = fs::read(destination)?;
    println!(
        "ZIP saved: {}\nentries={}\nbytes={}\nsha256={}",
        destination.display(),
        count,
        bytes.len(),
        sha256_hex(&bytes)
    );
    Ok(())
}
