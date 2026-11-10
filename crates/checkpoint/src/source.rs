//! Retained directory capability for bounded converter sidecar inputs.
//!
//! The explicitly supplied root is trusted. Relative paths use the same checked
//! component opener as indexed checkpoints. Sidecar bytes and their original
//! file owners remain cached; repeat reads neither reopen names nor charge bytes
//! again. These are input/descriptor limits, not a process-wide memory limit.
use crate::{Budget, Error, Limits, Result, Text, platform, reserve};
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct SourceLimits {
    /// Maximum bytes in any one retained sidecar (config, vocabulary, card, etc.).
    pub max_file_bytes: u64,
    /// Sum of byte lengths of every successfully cached sidecar.
    pub max_total_bytes: u64,
    /// Maximum number of cached sidecar owners, excluding the root directory.
    pub max_files: usize,
    /// Shared work and path limits for sidecar names and directory enumeration.
    /// Safetensors/index parsing has its own separately supplied Limits.
    pub paths: Limits,
}
impl Default for SourceLimits {
    fn default() -> Self {
        Self {
            max_file_bytes: 64 * 1024 * 1024,
            max_total_bytes: 256 * 1024 * 1024,
            max_files: 128,
            paths: Limits::default(),
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SourceUsage {
    pub files: usize,
    pub bytes: u64,
}
struct Sidecar {
    name: Text,
    _owner: File,
    bytes: Vec<u8>,
}
pub struct SourceDirectory {
    root: platform::Root,
    limits: SourceLimits,
    budget: Budget,
    files: Vec<Sidecar>,
    bytes: u64,
}
impl SourceDirectory {
    pub fn open(path: impl AsRef<Path>, limits: SourceLimits) -> Result<Self> {
        let budget = Budget::new(limits.paths.clone())?;
        let root = platform::Root::open(path.as_ref())?;
        Ok(Self {
            root,
            limits,
            budget,
            files: Vec::new(),
            bytes: 0,
        })
    }
    pub fn usage(&self) -> SourceUsage {
        SourceUsage {
            files: self.files.len(),
            bytes: self.bytes,
        }
    }
    /// Enumerate the retained root. Results include names of directories; this
    /// method does not follow them or turn them into file capabilities.
    pub fn entries(&mut self) -> Result<Vec<Text>> {
        self.root.entries(&mut self.budget)
    }
    /// Enumerate a checked relative directory beneath the retained root.
    /// The enumeration owner is retained until the bounded listing completes.
    /// Missing/non-directory paths return None; links/reparse points fail.
    /// As with ordinary directory enumeration, this is not an atomic snapshot
    /// of names and later file contents. Each subsequent read revalidates the
    /// path beneath the same root and retains the selected regular-file owner.
    pub fn entries_under(&mut self, name: &str) -> Result<Option<Vec<Text>>> {
        let name = Text::from_bytes(name.as_bytes(), &mut self.budget)?;
        self.root.entries_under(&name, &mut self.budget)
    }
    /// Open once, boundedly read, and retain a regular file beneath this root.
    /// Missing paths return None. Symlinks, reparse points, traversal and
    /// nonregular files are checked errors. Unix callers must keep contents
    /// stable during this initial read; short reads are rejected.
    pub fn read_optional(&mut self, name: &str) -> Result<Option<&[u8]>> {
        if let Some(index) = self
            .files
            .iter()
            .position(|file| file.name.as_str() == Some(name))
        {
            return Ok(Some(&self.files[index].bytes));
        }
        let text = Text::from_bytes(name.as_bytes(), &mut self.budget)?;
        let Some(owner) = self.root.optional_file(&text, &mut self.budget)? else {
            return Ok(None);
        };
        if self.files.len() >= self.limits.max_files {
            return Err(Error::Limit("sidecar owner count"));
        }
        let metadata = owner.metadata()?;
        if !metadata.is_file() {
            return Err(Error::Invalid("sidecar owner is not a regular file"));
        }
        let length = metadata.len();
        if length > self.limits.max_file_bytes {
            return Err(Error::Limit("individual sidecar bytes"));
        }
        let total = self
            .bytes
            .checked_add(length)
            .filter(|&n| n <= self.limits.max_total_bytes)
            .ok_or(Error::Limit("cumulative sidecar bytes"))?;
        let length = usize::try_from(length)
            .map_err(|_| Error::Limit("sidecar length is not addressable"))?;
        reserve(&mut self.files, 1, "sidecar owner allocation")?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| Error::Limit("sidecar byte allocation"))?;
        bytes.resize(length, 0);
        platform::read_exact_at(&owner, &mut bytes, 0)?;
        self.files.push(Sidecar {
            name: text,
            _owner: owner,
            bytes,
        });
        self.bytes = total;
        Ok(Some(&self.files.last().expect("inserted sidecar").bytes))
    }
    pub fn read_required(&mut self, name: &str) -> Result<&[u8]> {
        self.read_optional(name)?
            .ok_or(Error::Invalid("required sidecar is missing"))
    }
    /// Names of successfully cached sidecars. These identify retained inputs
    /// for callers that publish an output; enumeration does not reopen paths.
    pub fn cached_names(&self) -> impl Iterator<Item = &Text> {
        self.files.iter().map(|file| &file.name)
    }
    pub(crate) fn root(&self) -> &platform::Root {
        &self.root
    }
}
