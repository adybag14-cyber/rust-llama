//! SentencePiece 0.2.1 ModelProto loading needed by the frozen converter.
//! The wire parser borrows its bounded input; it never runs foreign code.
//! Apache-2.0: validation follows model_interface.cc and normalizer.cc.
use crate::{Error, Result, reserve};
#[derive(Clone, Copy, Debug)]
pub(crate) struct Piece<'a> {
    pub text: &'a str,
    pub score: f32,
    pub kind: u32,
}
#[derive(Debug)]
pub(crate) struct Model<'a> {
    pub pieces: Vec<Piece<'a>>,
    pub special_ids: [i32; 4], // unknown, bos, eos, pad
}
#[derive(Clone, Copy)]
enum Wire<'a> {
    Varint(u64),
    Fixed32(u32),
    Fixed64,
    Bytes(&'a [u8]),
    Group,
}
struct Cursor<'a> {
    input: &'a [u8],
    position: usize,
}
impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(n)
            .ok_or(Error::Invalid("protobuf range overflow"))?;
        let out = self
            .input
            .get(self.position..end)
            .ok_or(Error::Invalid("truncated protobuf field"))?;
        self.position = end;
        Ok(out)
    }
    fn varint(&mut self) -> Result<u64> {
        let mut out = 0u64;
        for index in 0..10 {
            let byte = self.take(1)?[0];
            if index == 9 && byte > 1 {
                return Err(Error::Invalid("protobuf varint overflow"));
            }
            out |= u64::from(byte & 127) << (index * 7);
            if byte & 128 == 0 {
                return Ok(out);
            }
        }
        Err(Error::Invalid("protobuf varint is too long"))
    }
    fn next(
        &mut self,
        budget: &mut Budget,
        depth: usize,
        group: Option<u32>,
    ) -> Result<Option<(u32, Wire<'a>)>> {
        if depth > 100 {
            return Err(Error::Limit("protobuf nesting"));
        }
        if self.position == self.input.len() {
            return if group.is_some() {
                Err(Error::Invalid("unterminated protobuf group"))
            } else {
                Ok(None)
            };
        }
        budget.field()?;
        let tag =
            u32::try_from(self.varint()?).map_err(|_| Error::Invalid("protobuf tag overflow"))?;
        let number = tag >> 3;
        if number == 0 {
            return Err(Error::Invalid("zero protobuf field number"));
        }
        let value = match tag & 7 {
            0 => Wire::Varint(self.varint()?),
            1 => {
                self.take(8)?;
                Wire::Fixed64
            }
            2 => {
                let length = usize::try_from(self.varint()?)
                    .map_err(|_| Error::Limit("protobuf field length"))?;
                Wire::Bytes(self.take(length)?)
            }
            3 => {
                while self.next(budget, depth + 1, Some(number))?.is_some() {}
                Wire::Group
            }
            4 if group == Some(number) => return Ok(None),
            4 => return Err(Error::Invalid("unexpected protobuf end group")),
            5 => Wire::Fixed32(u32::from_le_bytes(
                self.take(4)?.try_into().expect("four bytes"),
            )),
            _ => return Err(Error::Invalid("invalid protobuf wire type")),
        };
        Ok(Some((number, value)))
    }
}
struct Budget {
    fields: usize,
    max_fields: usize,
}
impl Budget {
    fn field(&mut self) -> Result<()> {
        if self.fields == self.max_fields {
            return Err(Error::Limit("protobuf field count"));
        }
        self.fields += 1;
        Ok(())
    }
}
fn cursor(bytes: &[u8]) -> Cursor<'_> {
    Cursor {
        input: bytes,
        position: 0,
    }
}
fn parse_piece<'a>(bytes: &'a [u8], budget: &mut Budget) -> Result<Piece<'a>> {
    let mut out = Piece {
        text: "",
        score: 0.0,
        kind: 1,
    };
    let mut input = cursor(bytes);
    while let Some((number, wire)) = input.next(budget, 1, None)? {
        match (number, wire) {
            (1, Wire::Bytes(value)) => {
                out.text = std::str::from_utf8(value)
                    .map_err(|_| Error::Invalid("SentencePiece piece is not UTF-8"))?
            }
            (2, Wire::Fixed32(value)) => out.score = f32::from_bits(value),
            (3, Wire::Varint(value @ 1..=6)) => out.kind = value as u32,
            // Proto2 unknown enums and mismatched wire types are unknown fields.
            _ => {}
        }
    }
    Ok(out)
}
fn parse_trainer<'a>(
    bytes: &'a [u8],
    budget: &mut Budget,
    fallback: &mut bool,
    names: &mut [&'a [u8]; 4],
) -> Result<()> {
    let mut input = cursor(bytes);
    while let Some((number, wire)) = input.next(budget, 1, None)? {
        match (number, wire) {
            (35, Wire::Varint(value)) => *fallback = value != 0,
            (45..=48, Wire::Bytes(value)) => names[number as usize - 45] = value,
            _ => {}
        }
    }
    Ok(())
}
fn parse_normalizer<'a>(
    bytes: &'a [u8],
    budget: &mut Budget,
    charsmap: &mut &'a [u8],
) -> Result<()> {
    let mut input = cursor(bytes);
    while let Some((number, wire)) = input.next(budget, 1, None)? {
        if let (2, Wire::Bytes(value)) = (number, wire) {
            *charsmap = value;
        }
    }
    Ok(())
}
fn parse_self_test(bytes: &[u8], budget: &mut Budget) -> Result<()> {
    let mut input = cursor(bytes);
    while let Some((number, wire)) = input.next(budget, 1, None)? {
        if let (1, Wire::Bytes(_)) = (number, wire) {
            // A model's self-tests execute SentencePiece, not llama.cpp tokenization.
            // Refuse this separate contract until its native algorithm is ported.
            return Err(Error::Unsupported(
                "SentencePiece embedded execution self-tests",
            ));
        }
    }
    Ok(())
}
fn validate_charsmap(bytes: &[u8]) -> Result<()> {
    if bytes.is_empty() {
        return Ok(());
    }
    let prefix: [u8; 4] = bytes
        .get(..4)
        .ok_or(Error::Invalid("normalizer blob header"))?
        .try_into()
        .expect("four bytes");
    let length = u32::from_le_bytes(prefix) as usize;
    if length < 1024 || length & 1023 != 0 {
        return Err(Error::Invalid("normalizer trie size"));
    }
    let tail = bytes
        .get(4..)
        .and_then(|data| data.get(length..))
        .ok_or(Error::Invalid("normalizer trie exceeds blob"))?;
    if tail.last() != Some(&0) {
        return Err(Error::Invalid("normalizer string block termination"));
    }
    Ok(())
}
impl<'a> Model<'a> {
    pub(crate) fn parse(bytes: &'a [u8], max_pieces: usize, max_fields: usize) -> Result<Self> {
        let mut budget = Budget {
            fields: 0,
            max_fields,
        };
        let mut input = cursor(bytes);
        let mut pieces = Vec::new();
        let mut byte_fallback = false;
        let defaults: [&[u8]; 4] = [b"<unk>", b"<s>", b"</s>", b"<pad>"];
        let mut names = defaults;
        let mut charsmap: &[u8] = &[];
        let mut denormalizer: &[u8] = &[];
        while let Some((number, wire)) = input.next(&mut budget, 0, None)? {
            match (number, wire) {
                (1, Wire::Bytes(value)) => {
                    if pieces.len() == max_pieces {
                        return Err(Error::Limit("SentencePiece vocabulary"));
                    }
                    reserve(&mut pieces, 1, "SentencePiece piece descriptors")?;
                    pieces.push(parse_piece(value, &mut budget)?);
                }
                (2, Wire::Bytes(value)) => {
                    parse_trainer(value, &mut budget, &mut byte_fallback, &mut names)?
                }
                (3, Wire::Bytes(value)) => parse_normalizer(value, &mut budget, &mut charsmap)?,
                (4, Wire::Bytes(value)) => parse_self_test(value, &mut budget)?,
                (5, Wire::Bytes(value)) => parse_normalizer(value, &mut budget, &mut denormalizer)?,
                _ => {}
            }
        }
        // SentencePieceProcessor::status intentionally does not include the
        // separately constructed denormalizer's status. Conversion never uses it.
        let _ = denormalizer;
        validate_charsmap(charsmap)?;
        let mut sorted = Vec::new();
        reserve(&mut sorted, pieces.len(), "SentencePiece name descriptors")?;
        let mut unknown = None;
        let mut found_bytes = [false; 256];
        for (id, piece) in pieces.iter().enumerate() {
            if piece.text.is_empty() {
                return Err(Error::Invalid("empty SentencePiece piece"));
            }
            let ordinary = matches!(piece.kind, 1 | 4 | 5);
            sorted.push((ordinary, piece.text, id));
            if piece.kind == 2 && unknown.replace(id).is_some() {
                return Err(Error::Invalid("multiple SentencePiece unknown pieces"));
            }
            if piece.kind == 6 {
                if !byte_fallback {
                    return Err(Error::Invalid("byte pieces without byte fallback"));
                }
                let value = piece.text.as_bytes();
                if value.len() != 6 || &value[..3] != b"<0x" || value[5] != b'>' {
                    return Err(Error::Invalid("SentencePiece byte spelling"));
                }
                let digit = |x: u8| match x {
                    b'0'..=b'9' => Some(x - b'0'),
                    b'A'..=b'F' => Some(x - b'A' + 10),
                    _ => None,
                };
                let hi = digit(value[3]).ok_or(Error::Invalid("SentencePiece byte spelling"))?;
                let lo = digit(value[4]).ok_or(Error::Invalid("SentencePiece byte spelling"))?;
                found_bytes[usize::from(hi) * 16 + usize::from(lo)] = true;
            }
        }
        sorted.sort_unstable();
        if sorted
            .windows(2)
            .any(|pair| pair[0].0 == pair[1].0 && pair[0].1 == pair[1].1)
        {
            return Err(Error::Invalid(
                "duplicate SentencePiece name in the same map",
            ));
        }
        let unknown = unknown.ok_or(Error::Invalid("missing SentencePiece unknown piece"))?;
        if byte_fallback && found_bytes.contains(&false) {
            return Err(Error::Invalid("incomplete SentencePiece byte fallback"));
        }
        let mut special_ids = [-1; 4];
        for index in 0..4 {
            let name = if names[index].is_empty() {
                defaults[index]
            } else {
                names[index]
            };
            let name = name.split(|byte| *byte == 0).next().unwrap_or_default();
            // Native special getters pass c_str() to string_view, then search
            // reserved entries before ordinary entries and fall back to unk.
            let id = pieces
                .iter()
                .position(|p| !matches!(p.kind, 1 | 4 | 5) && p.text.as_bytes() == name)
                .or_else(|| {
                    pieces
                        .iter()
                        .position(|p| matches!(p.kind, 1 | 4 | 5) && p.text.as_bytes() == name)
                })
                .unwrap_or(unknown);
            if pieces[id].kind == if index == 0 { 2 } else { 3 } {
                special_ids[index] =
                    i32::try_from(id).map_err(|_| Error::Limit("SentencePiece ID range"))?;
            }
        }
        Ok(Self {
            pieces,
            special_ids,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llama_checkpoint::{Limits, Value};
    fn get<'a>(value: &'a Value, key: &str) -> &'a Value {
        value.as_object().unwrap().get(key).unwrap()
    }
    fn str_value(value: &Value) -> &str {
        value.as_text().unwrap().as_str().unwrap()
    }
    fn number(value: &Value) -> i64 {
        match value {
            Value::Number(llama_checkpoint::Number::Integer(n)) => n.as_i64().unwrap(),
            _ => panic!("number"),
        }
    }
    fn bytes(hex: &str) -> Vec<u8> {
        hex.as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
            .collect()
    }
    #[test]
    fn original_sentencepiece_model_acceptance_and_pieces() {
        let fixture = llama_checkpoint::json::parse(
            include_bytes!("../tests/fixtures/sentencepiece/reference.json"),
            Limits::default(),
        )
        .unwrap();
        let mut accepted = 0;
        for record in get(&fixture, "cases").as_array().unwrap() {
            let name = str_value(get(record, "name"));
            let raw = bytes(str_value(get(record, "input_hex")));
            let expected = get(record, "result");
            let parsed = Model::parse(&raw, 1_000_000, 4_000_000);
            if get(expected, "ok") == &Value::Bool(false) {
                assert!(parsed.is_err(), "{name}");
                continue;
            }
            let parsed = parsed.unwrap_or_else(|e| panic!("{name}: {e}"));
            accepted += 1;
            let rows = get(expected, "rows").as_array().unwrap();
            assert_eq!(parsed.pieces.len(), rows.len(), "{name}");
            for (id, (piece, row)) in parsed.pieces.iter().zip(rows).enumerate() {
                assert_eq!(piece.text, str_value(get(row, "piece")), "{name}/{id}");
                assert_eq!(
                    piece.score.to_le_bytes().as_slice(),
                    bytes(str_value(get(row, "score_bits"))),
                    "{name}/{id}"
                );
                for (flag, kind) in [("unknown", 2), ("control", 3), ("unused", 5), ("byte", 6)] {
                    assert_eq!(
                        piece.kind == kind,
                        get(row, flag) == &Value::Bool(true),
                        "{name}/{id}/{flag}"
                    );
                }
            }
            for (slot, key) in ["unk", "bos", "eos", "pad"].iter().enumerate() {
                assert_eq!(
                    i64::from(parsed.special_ids[slot]),
                    number(get(expected, key)),
                    "{name}/{key}"
                );
            }
        }
        assert!(accepted >= 27);
    }
    #[test]
    fn wire_resource_limits_reject_before_growth() {
        let raw = b"\x0a\x09\x0a\x05<unk>\x18\x02";
        assert!(Model::parse(raw, 0, 100).is_err());
        assert!(Model::parse(raw, 10, 0).is_err());
        let mut groups = vec![0x0b; 102];
        groups.extend(vec![0x0c; 102]);
        assert!(Model::parse(&groups, 10, 1000).is_err());
        assert!(Model::parse(b"\x0a\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01", 10, 100).is_err());
    }
}
