//! Lossless primary metadata plus the source model's C-string view.
//!
//! The raw parser's wire limits are unchanged. This separate preflight bounds
//! retained raw/projection payloads and conservative collection storage before
//! allocating the projection. Standard BTreeMap allocator failures retain the
//! standard Rust allocator contract; this is not a recoverable-OOM/RSS promise.
use crate::{Error, RawGguf, RawValue, Result, Value, raw::c_string};
use std::collections::{BTreeMap, btree_map::Entry};
use std::io::{Read, Seek};
use std::mem::size_of;

pub(crate) fn add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| Error::Limit("model metadata storage overflow".into()))
}
fn mul(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| Error::Limit("model metadata storage overflow".into()))
}
pub(crate) fn within_budget(bytes: usize, limit: usize) -> Result<()> {
    if bytes > limit {
        Err(Error::Limit(format!(
            "model metadata storage requires {bytes} bytes, exceeding {limit}"
        )))
    } else {
        Ok(())
    }
}

/// One entry is charged for two complete maximum-capacity nodes. The pinned
/// Rust BTree implementation has at most 11 entries and 12 child pointers per
/// node; the additional 16 pointer words cover links, lengths and alignment.
/// This deliberately overcharges partially occupied nodes and internal nodes.
fn map_entry_charge<K, V>() -> Result<usize> {
    mul(
        2,
        add(mul(11, size_of::<(K, V)>())?, mul(16, size_of::<usize>())?)?,
    )
}
fn raw_heap(value: &RawValue, depth: usize) -> Result<usize> {
    if depth > 1 {
        return Err(Error::Limit("nested model metadata array".into()));
    }
    match value {
        RawValue::String(bytes) | RawValue::ByteString(bytes) => Ok(bytes.capacity()),
        RawValue::Array { values, .. } => {
            let mut total = mul(values.capacity(), size_of::<RawValue>())?;
            for value in values {
                total = add(total, raw_heap(value, depth + 1)?)?;
            }
            Ok(total)
        }
        _ => Ok(0),
    }
}
fn projected_heap(value: &RawValue, depth: usize) -> Result<usize> {
    if depth > 1 {
        return Err(Error::Limit("nested model metadata array".into()));
    }
    match value {
        RawValue::String(bytes) | RawValue::ByteString(bytes) => Ok(c_string(bytes).len()),
        RawValue::Array { values, .. } => {
            let mut total = mul(values.len(), size_of::<Value>())?;
            for value in values {
                total = add(total, projected_heap(value, depth + 1)?)?;
            }
            Ok(total)
        }
        _ => Ok(0),
    }
}
pub(crate) fn raw_storage(
    metadata: &BTreeMap<Vec<u8>, RawValue>,
    order: &[Vec<u8>],
    order_capacity: usize,
) -> Result<usize> {
    let mut total = add(
        size_of::<BTreeMap<Vec<u8>, RawValue>>(),
        mul(order_capacity, size_of::<Vec<u8>>())?,
    )?;
    for key in order {
        total = add(total, key.capacity())?;
    }
    for (key, value) in metadata {
        total = add(total, map_entry_charge::<Vec<u8>, RawValue>()?)?;
        total = add(total, key.capacity())?;
        total = add(total, raw_heap(value, 0)?)?;
    }
    Ok(total)
}

pub(crate) fn project<R: Read + Seek>(
    raw: &RawGguf<R>,
    limit: usize,
) -> Result<(BTreeMap<String, Value>, usize)> {
    let raw_bytes = raw.model_metadata_storage()?;
    let mut projected = size_of::<BTreeMap<String, Value>>();
    for key in raw.metadata_order() {
        let prefix = c_string(key);
        if std::str::from_utf8(prefix).is_err() {
            // The key is still retained losslessly by raw_metadata(). It cannot
            // name a required text key or be inserted into a String-keyed map.
            continue;
        }
        projected = add(projected, map_entry_charge::<String, Value>()?)?;
        projected = add(projected, prefix.len())?;
        projected = add(projected, projected_heap(&raw.metadata[key], 0)?)?;
    }
    // Aliases are deliberately charged before deduplication, bounding key
    // temporaries as well as the retained first-prefix values.
    let total = add(raw_bytes, projected)?;
    within_budget(total, limit)?;
    let mut metadata = BTreeMap::new();
    for key in raw.metadata_order() {
        let Ok(prefix) = std::str::from_utf8(c_string(key)) else {
            continue;
        };
        let copied_key = copy_bytes(prefix.as_bytes())?;
        let copied_key = String::from_utf8(copied_key)
            .map_err(|_| Error::Invalid("validated model key is not UTF-8".into()))?;
        if let Entry::Vacant(entry) = metadata.entry(copied_key) {
            entry.insert(project_value(&raw.metadata[key], 0)?);
        }
    }
    Ok((metadata, total))
}
fn copy_bytes(value: &[u8]) -> Result<Vec<u8>> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(value.len())
        .map_err(|_| Error::Limit("model metadata bytes allocation".into()))?;
    copy.extend_from_slice(value);
    Ok(copy)
}
fn project_value(value: &RawValue, depth: usize) -> Result<Value> {
    if depth > 1 {
        return Err(Error::Limit("nested model metadata array".into()));
    }
    Ok(match value {
        RawValue::U8(value) => Value::U8(*value),
        RawValue::I8(value) => Value::I8(*value),
        RawValue::U16(value) => Value::U16(*value),
        RawValue::I16(value) => Value::I16(*value),
        RawValue::U32(value) => Value::U32(*value),
        RawValue::I32(value) => Value::I32(*value),
        RawValue::F32(value) => Value::F32(*value),
        RawValue::Bool(value) => Value::Bool(*value),
        RawValue::String(value) | RawValue::ByteString(value) => {
            match String::from_utf8(copy_bytes(c_string(value))?) {
                Ok(text) => Value::String(text),
                Err(error) => Value::ByteString(error.into_bytes()),
            }
        }
        RawValue::Array {
            element_type,
            values,
        } => {
            let mut copy = Vec::new();
            copy.try_reserve_exact(values.len())
                .map_err(|_| Error::Limit("model metadata array allocation".into()))?;
            for value in values {
                copy.push(project_value(value, depth + 1)?);
            }
            Value::Array {
                element_type: *element_type,
                values: copy,
            }
        }
        RawValue::U64(value) => Value::U64(*value),
        RawValue::I64(value) => Value::I64(*value),
        RawValue::F64(value) => Value::F64(*value),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Gguf, Limits, RawTensorDescriptor, write_raw_header};
    use std::io::Cursor;
    fn raw(values: &[(&[u8], RawValue)]) -> RawGguf<Cursor<Vec<u8>>> {
        let metadata: Vec<_> = values.iter().map(|(key, value)| (*key, value)).collect();
        let mut bytes = Vec::new();
        write_raw_header(
            &mut bytes,
            &metadata,
            &[] as &[RawTensorDescriptor<'_>],
            &Limits::default(),
        )
        .unwrap();
        RawGguf::read(Cursor::new(bytes), Limits::default()).unwrap()
    }
    #[test]
    fn model_prefixes_keep_raw_bytes_first_alias_and_array_elements() {
        let raw = raw(&[
            (
                b"prefix\0first",
                RawValue::String(b"first\0ignored".to_vec()),
            ),
            (b"prefix", RawValue::String(b"second".to_vec())),
            (b"raw", RawValue::String(vec![0xff, 0x80, 0, b'x'])),
            (
                b"array",
                RawValue::Array {
                    element_type: 8,
                    values: vec![
                        RawValue::String(b"text\0suffix".to_vec()),
                        RawValue::String(vec![0xfe, 0, 0xfd]),
                    ],
                },
            ),
            (b"\xff", RawValue::String(b"kept raw".to_vec())),
            (b"\0nonempty-wire-key", RawValue::U32(7)),
        ]);
        let (view, charge) = project(&raw, usize::MAX).unwrap();
        assert_eq!(view["prefix"].as_str(), Some("first"));
        assert_eq!(view["raw"].type_id(), 8);
        assert_eq!(view["raw"].as_str(), None);
        assert_eq!(view["raw"].as_bytes(), Some([0xff, 0x80].as_slice()));
        assert_eq!(view[""].as_u64(), Some(7));
        let array = view["array"].as_array().unwrap();
        assert_eq!(array[0].as_str(), Some("text"));
        assert_eq!(array[1].as_bytes(), Some([0xfe].as_slice()));
        assert_eq!(
            raw.metadata[b"raw".as_slice()].as_bytes(),
            Some([0xff, 0x80, 0, b'x'].as_slice())
        );
        assert!(raw.metadata.contains_key(b"\xff".as_slice()));
        assert_eq!(raw.metadata_order().count(), 6);
        assert_eq!(project(&raw, charge).unwrap().1, charge);
        assert!(matches!(project(&raw, charge - 1), Err(Error::Limit(_))));
        // The existing strict entry point still rejects NUL/non-UTF-8 keys.
        assert!(Gguf::read(raw.reader, Limits::default()).is_err());
    }
    #[test]
    fn storage_overflow_and_alias_array_copies_are_charged() {
        assert!(add(usize::MAX, 1).is_err());
        assert!(mul(usize::MAX, 2).is_err());
        let make = || RawValue::Array {
            element_type: 8,
            values: vec![
                RawValue::String(vec![0xff; 4096]),
                RawValue::String(vec![b'x'; 4096]),
            ],
        };
        let one = raw(&[(b"same", make())]);
        let aliases = raw(&[(b"same\0a", make()), (b"same\0b", make())]);
        let (_, first_charge) = project(&one, usize::MAX).unwrap();
        let (view, alias_charge) = project(&aliases, usize::MAX).unwrap();
        assert_eq!(view.len(), 1);
        assert!(alias_charge > first_charge * 2 - 1024);
        assert!(project(&aliases, first_charge).is_err());
    }
    #[test]
    fn explicit_byte_string_keeps_string_wire_type_and_exact_payload() {
        let raw = raw(&[(b"bytes", RawValue::ByteString(vec![0xff, 0, 0xfe]))]);
        assert_eq!(raw.metadata[b"bytes".as_slice()].type_id(), 8);
        assert_eq!(
            raw.metadata[b"bytes".as_slice()].as_bytes(),
            Some([0xff, 0, 0xfe].as_slice())
        );
        assert!(raw.into_text().is_err());
    }
}
