//! Frozen converter index selection, global-name validation and overwrite order.
use crate::{
    Budget, Error, Limits, Result, SafetensorsFile, Text, Usage, Value, json, platform, reserve,
};
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NamePolicy {
    /// Preserve the frozen ModelBase.filter_tensors substitutions.
    #[default]
    ConverterBase,
    /// Keep original names for a caller that owns architecture-specific filtering.
    Unchanged,
}
#[derive(Debug, Clone)]
pub struct CheckpointOptions {
    pub prefix: String,
    pub mistral: bool,
    pub names: NamePolicy,
}
impl Default for CheckpointOptions {
    fn default() -> Self {
        Self {
            prefix: "model".into(),
            mistral: false,
            names: NamePolicy::ConverterBase,
        }
    }
}
#[derive(Debug)]
struct NamedFile {
    name: Text,
    file: SafetensorsFile,
}
#[derive(Debug)]
struct Selection {
    name: Text,
    file: usize,
    tensor: usize,
    order: usize,
}
#[derive(Debug)]
struct SourceName {
    original: Text,
    renamed: Text,
    order: usize,
}

/// An owned local checkpoint. The index selects a sorted set of filenames and
/// validates the global original-name set. As in the source, it does not enforce
/// that each name resides in the particular file assigned by weight_map.
#[derive(Debug)]
pub struct Checkpoint {
    files: Vec<NamedFile>,
    tensors: Vec<Selection>,
    lookup: Vec<usize>,
    source_names: Vec<SourceName>,
    usage: Usage,
}
impl Checkpoint {
    /// Open beneath a retained root directory. Absolute paths, parent traversal,
    /// symlinks/reparse points and Windows alternate streams are checked errors.
    /// Hard links are permitted. Namespace changes cannot redirect an already
    /// opened component/owner, but callers must keep file contents stable.
    pub fn open_directory(
        path: impl AsRef<Path>,
        options: CheckpointOptions,
        limits: Limits,
    ) -> Result<Self> {
        let mut budget = Budget::new(limits)?;
        validate_prefix(&options.prefix, &budget)?;
        let root = platform::Root::open(path.as_ref())?;
        let entries = root.entries(&mut budget)?;
        let has_pickle = entries.iter().any(|name| {
            name.as_wtf8().starts_with(b"pytorch_model") && name.as_wtf8().ends_with(b".bin")
        });
        let mut discovered = discover(entries, options.prefix.as_bytes())?;
        let index_name = appended_name(
            options.prefix.as_bytes(),
            b".safetensors.index.json",
            &mut budget,
        )?;
        let mut index_owner = if options.mistral {
            None
        } else {
            root.optional_file(&index_name, &mut budget)?
        };
        let is_safetensors = !discovered.is_empty() || index_owner.is_some();
        if !is_safetensors {
            let pickle_index = Text::from_bytes(b"pytorch_model.bin.index.json", &mut budget)?;
            let has_pickle_index =
                !options.mistral && root.optional_file(&pickle_index, &mut budget)?.is_some();
            if has_pickle || has_pickle_index {
                return Err(Error::Unsupported(
                    "PyTorch pickle checkpoint conversion is not implemented",
                ));
            }
        }
        let mut expected = None;
        if let Some(owner) = index_owner.take() {
            let meta = owner.metadata()?;
            if !meta.is_file() {
                return Err(Error::Invalid("index owner is not a regular file"));
            }
            budget.index(meta.len())?;
            let size = usize::try_from(meta.len())
                .map_err(|_| Error::Limit("index is not addressable"))?;
            let mut bytes = Vec::new();
            reserve(&mut bytes, size, "index bytes allocation")?;
            bytes.resize(size, 0);
            platform::read_exact_at(&owner, &mut bytes, 0)?;
            let plan = index_plan(&bytes, &mut budget)?;
            discovered = plan.files;
            expected = Some(plan.names);
        }
        // Reserve only after the declared file count is within the shared bound.
        if discovered.len() > budget.limits.max_files {
            return Err(Error::Limit("file-owner count"));
        }
        let mut files = Vec::new();
        reserve(
            &mut files,
            discovered.len(),
            "checkpoint file owners allocation",
        )?;
        for name in discovered {
            let owner = root.open_relative(&name, &mut budget)?;
            let file = SafetensorsFile::from_owner(owner, &mut budget)?;
            files.push(NamedFile { name, file });
        }
        Self::assemble(files, expected, options.names, budget)
    }

    /// Assemble caller-chosen already-opened files. Labels are identifiers, not
    /// paths resolved here: this permits trusted symlink-backed caches without
    /// weakening the untrusted directory opener. Every selected File goes
    /// through the same bounded parser. Unselected capabilities are dropped.
    /// Without an index, the source prefix/suffix discovery rules apply to labels.
    /// In Mistral mode any supplied index is ignored, as in the frozen converter.
    pub fn from_open_files(
        mut supplied: Vec<(Text, File)>,
        index: Option<&[u8]>,
        options: CheckpointOptions,
        limits: Limits,
    ) -> Result<Self> {
        let mut budget = Budget::new(limits)?;
        validate_prefix(&options.prefix, &budget)?;
        // Bound all supplied owners, including unused labels, before building
        // a separate selection. Ownership was explicitly granted by the caller.
        if supplied.len() > budget.limits.max_files {
            return Err(Error::Limit("supplied file-owner count"));
        }
        for (label, _) in &supplied {
            budget.entry()?;
            if label.as_wtf8().len() > budget.limits.max_path_bytes {
                return Err(Error::Limit("capability label bytes"));
            }
            budget.text(label.as_wtf8().len())?;
        }
        supplied.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        if supplied.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(Error::Invalid("duplicate supplied capability labels"));
        }
        let (names, expected) = if !options.mistral {
            if let Some(bytes) = index {
                budget.index(bytes.len() as u64)?;
                let plan = index_plan(bytes, &mut budget)?;
                (plan.files, Some(plan.names))
            } else {
                (
                    capability_discovery(&supplied, options.prefix.as_bytes(), &mut budget)?,
                    None,
                )
            }
        } else {
            (
                capability_discovery(&supplied, options.prefix.as_bytes(), &mut budget)?,
                None,
            )
        };
        if names.len() > budget.limits.max_files {
            return Err(Error::Limit("file-owner count"));
        }
        let mut files = Vec::new();
        reserve(&mut files, names.len(), "checkpoint file owners allocation")?;
        let mut candidates = supplied.into_iter().peekable();
        for name in names {
            while candidates.peek().is_some_and(|(label, _)| label < &name) {
                candidates.next();
            }
            let (label, owner) = candidates
                .next()
                .ok_or(Error::Invalid("index references an absent capability"))?;
            if label != name {
                return Err(Error::Invalid("index references an absent capability"));
            }
            let file = SafetensorsFile::from_owner(owner, &mut budget)?;
            files.push(NamedFile { name, file });
        }
        Self::assemble(files, expected, options.names, budget)
    }

    fn assemble(
        files: Vec<NamedFile>,
        expected: Option<Vec<Text>>,
        names: NamePolicy,
        mut budget: Budget,
    ) -> Result<Self> {
        let total = files
            .iter()
            .try_fold(0usize, |total, file| {
                total.checked_add(file.file.tensors().len())
            })
            .ok_or(Error::Limit("tensor contribution count"))?;
        // Each parsed header charged these descriptors already.
        if total > budget.limits.max_tensors {
            return Err(Error::Limit("tensor contribution count"));
        }
        let mut tensors = Vec::new();
        let mut source_names = Vec::new();
        reserve(&mut tensors, total, "tensor selection allocation")?;
        reserve(&mut source_names, total, "source-name map allocation")?;
        for (file_index, file) in files.iter().enumerate() {
            for (tensor_index, tensor) in file.file.tensors().iter().enumerate() {
                let original = tensor.name().copy(&mut budget)?;
                let renamed = filter_name(tensor.name(), names, &mut budget)?;
                let order = tensors.len();
                source_names.push(SourceName {
                    original,
                    renamed: renamed.copy(&mut budget)?,
                    order,
                });
                tensors.push(Selection {
                    name: renamed,
                    file: file_index,
                    tensor: tensor_index,
                    order,
                });
            }
        }
        source_names
            .sort_unstable_by(|a, b| a.original.cmp(&b.original).then(a.order.cmp(&b.order)));
        source_names.dedup_by(|later, earlier| later.original == earlier.original);
        if let Some(expected) = expected
            && !expected.is_empty()
            && (expected.len() != source_names.len()
                || expected
                    .iter()
                    .zip(&source_names)
                    .any(|(key, source)| key != &source.original))
        {
            return Err(Error::Invalid(
                "weight map and global original tensor-name sets differ",
            ));
        }
        source_names.sort_unstable_by_key(|source| source.order);
        // Python dict: later shards win the value, but the first insertion's
        // position remains. This also applies to base-filter name collisions.
        tensors.sort_unstable_by(|a, b| a.name.cmp(&b.name).then(a.order.cmp(&b.order)));
        tensors.dedup_by(|later, earlier| {
            if later.name == earlier.name {
                earlier.file = later.file;
                earlier.tensor = later.tensor;
                true
            } else {
                false
            }
        });
        tensors.sort_unstable_by_key(|tensor| tensor.order);
        let mut lookup = Vec::new();
        reserve(&mut lookup, tensors.len(), "tensor lookup allocation")?;
        lookup.extend(0..tensors.len());
        lookup.sort_unstable_by(|&a, &b| tensors[a].name.cmp(&tensors[b].name));
        Ok(Self {
            files,
            tensors,
            lookup,
            source_names,
            usage: budget.used,
        })
    }
    pub fn usage(&self) -> Usage {
        self.usage
    }
    pub fn file_count(&self) -> usize {
        self.files.len()
    }
    pub fn file(&self, index: usize) -> Option<(&Text, &SafetensorsFile)> {
        self.files.get(index).map(|file| (&file.name, &file.file))
    }
    pub fn len(&self) -> usize {
        self.tensors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
    pub fn tensors(&self) -> impl Iterator<Item = TensorRef<'_>> {
        self.tensors.iter().map(|selection| TensorRef {
            checkpoint: self,
            selection,
        })
    }
    pub fn find(&self, name: &Text) -> Option<TensorRef<'_>> {
        let found = self
            .lookup
            .binary_search_by(|&index| self.tensors[index].name.cmp(name))
            .ok()?;
        Some(TensorRef {
            checkpoint: self,
            selection: &self.tensors[self.lookup[found]],
        })
    }
    pub fn find_str(&self, name: &str) -> Option<TensorRef<'_>> {
        let found = self
            .lookup
            .binary_search_by(|&index| self.tensors[index].name.as_wtf8().cmp(name.as_bytes()))
            .ok()?;
        Some(TensorRef {
            checkpoint: self,
            selection: &self.tensors[self.lookup[found]],
        })
    }
    pub fn source_names(&self) -> impl Iterator<Item = (&Text, &Text)> {
        self.source_names
            .iter()
            .map(|entry| (&entry.original, &entry.renamed))
    }
}
#[derive(Debug, Clone, Copy)]
pub struct TensorRef<'a> {
    checkpoint: &'a Checkpoint,
    selection: &'a Selection,
}
impl<'a> TensorRef<'a> {
    pub fn name(self) -> &'a Text {
        &self.selection.name
    }
    pub fn file_name(self) -> &'a Text {
        &self.checkpoint.files[self.selection.file].name
    }
    pub fn info(self) -> &'a crate::RawTensorInfo {
        &self.checkpoint.files[self.selection.file].file.tensors()[self.selection.tensor]
    }
    pub fn reader(self) -> Result<crate::TensorReader<'a>> {
        self.checkpoint.files[self.selection.file]
            .file
            .reader(self.selection.tensor)
    }
    pub fn read(self) -> Result<Vec<u8>> {
        self.checkpoint.files[self.selection.file]
            .file
            .read_tensor(self.selection.tensor)
    }
}
struct Plan {
    files: Vec<Text>,
    names: Vec<Text>,
}
fn index_plan(bytes: &[u8], budget: &mut Budget) -> Result<Plan> {
    let Value::Object(mut root) = json::parse_charged(bytes, budget)? else {
        return Err(Error::Invalid("index root is not an object"));
    };
    let Some(Value::Object(map)) = root.remove("weight_map") else {
        return Err(Error::Invalid("weight_map is missing or not an object"));
    };
    let mut files = Vec::new();
    let mut names = Vec::new();
    if map.len() > budget.limits.max_tensors {
        return Err(Error::Limit("index tensor names"));
    }
    reserve(&mut files, map.len(), "index filenames allocation")?;
    reserve(&mut names, map.len(), "index tensor names allocation")?;
    for member in map.0 {
        let Value::Text(name) = member.value else {
            return Err(Error::Invalid("index filename is not text"));
        };
        files.push(name);
        names.push(member.key);
    }
    files.sort_unstable();
    files.dedup();
    names.sort_unstable();
    Ok(Plan { files, names })
}
fn validate_prefix(prefix: &str, budget: &Budget) -> Result<()> {
    if prefix.len() > budget.limits.max_path_bytes {
        return Err(Error::Limit("prefix bytes"));
    }
    if prefix.is_empty()
        || prefix
            .bytes()
            .any(|b| b == 0 || b == b'/' || (cfg!(windows) && matches!(b, b'\\' | b':')))
    {
        return Err(Error::Path("prefix must be a nonempty filename prefix"));
    }
    Ok(())
}
fn appended_name(prefix: &[u8], suffix: &[u8], budget: &mut Budget) -> Result<Text> {
    let len = prefix
        .len()
        .checked_add(suffix.len())
        .ok_or(Error::Limit("index filename length"))?;
    if len > budget.limits.max_path_bytes || len > budget.limits.max_string_bytes {
        return Err(Error::Limit("index filename bytes"));
    }
    budget.text(len)?;
    let mut output = Vec::new();
    reserve(&mut output, len, "index filename allocation")?;
    output.extend_from_slice(prefix);
    output.extend_from_slice(suffix);
    Ok(Text(output))
}
fn discover(entries: Vec<Text>, prefix: &[u8]) -> Result<Vec<Text>> {
    let mut result = Vec::new();
    for name in entries {
        let bytes = name.as_wtf8();
        if bytes.starts_with(prefix) && bytes.ends_with(b".safetensors") {
            reserve(&mut result, 1, "discovered filenames allocation")?;
            result.push(name);
        }
    }
    result.sort_unstable();
    Ok(result)
}
fn capability_discovery(
    supplied: &[(Text, File)],
    prefix: &[u8],
    budget: &mut Budget,
) -> Result<Vec<Text>> {
    let mut names = Vec::new();
    for (name, _) in supplied {
        if name.as_wtf8().starts_with(prefix) && name.as_wtf8().ends_with(b".safetensors") {
            reserve(&mut names, 1, "capability discovery allocation")?;
            names.push(name.copy(budget)?);
        }
    }
    Ok(names)
}
fn replace_all(
    text: &Text,
    pattern: &[u8],
    replacement: &[u8],
    budget: &mut Budget,
) -> Result<Text> {
    let bytes = text.as_wtf8();
    let mut result = Vec::new();
    let mut start = 0;
    let mut position = 0;
    while position + pattern.len() <= bytes.len() {
        if &bytes[position..position + pattern.len()] == pattern {
            let amount = position - start + replacement.len();
            budget.text(amount)?;
            reserve(&mut result, amount, "filtered name allocation")?;
            result.extend_from_slice(&bytes[start..position]);
            result.extend_from_slice(replacement);
            position += pattern.len();
            start = position;
        } else {
            position += 1;
        }
    }
    budget.text(bytes.len() - start)?;
    reserve(&mut result, bytes.len() - start, "filtered name allocation")?;
    result.extend_from_slice(&bytes[start..]);
    if result.len() > budget.limits.max_string_bytes {
        return Err(Error::Limit("filtered name bytes"));
    }
    Ok(Text(result))
}
fn filter_name(name: &Text, policy: NamePolicy, budget: &mut Budget) -> Result<Text> {
    if policy == NamePolicy::Unchanged {
        return name.copy(budget);
    }
    let renamed = if name.as_wtf8().ends_with(b"e_score_correction_bias") {
        replace_all(
            name,
            b"e_score_correction_bias",
            b"e_score_correction.bias",
            budget,
        )?
    } else {
        name.copy(budget)?
    };
    replace_all(&renamed, b"language_model.", b"", budget)
}
