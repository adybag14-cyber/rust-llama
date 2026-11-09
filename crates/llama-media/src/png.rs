//! PNG scanlines and zlib/DEFLATE, authored in safe Rust.
//!
//! Format semantics were inspected in the frozen `vendor/stb/stb_image.h`.
//! This decoder supports noninterlaced standard PNG at all legal color depths.
//! Adam7, Apple CgBI and animation playback remain explicit open capabilities.
use crate::{
    DecodeLimits, MediaError, Result, bytes, check_limit,
    image::{ImageSize, RgbImage},
    product, zeros,
};

const SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
fn be32(input: &[u8], at: usize) -> Result<u32> {
    let b = bytes(input, at, 4)?;
    Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}
const fn crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 != 0 {
                0xedb88320 ^ (crc >> 1)
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}
const CRC_TABLE: [u32; 256] = crc_table();
fn crc32(input: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for &b in input {
        crc = CRC_TABLE[((crc ^ b as u32) & 255) as usize] ^ (crc >> 8);
    }
    !crc
}
fn adler32(input: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in input.chunks(5552) {
        for &x in chunk {
            a += x as u32;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

#[derive(Debug, Clone, Copy)]
struct Header {
    size: ImageSize,
    depth: u8,
    color: u8,
    channels: usize,
}
pub(crate) fn decode(input: &[u8], limits: DecodeLimits) -> Result<RgbImage> {
    if bytes(input, 0, 8)? != SIGNATURE {
        return Err(MediaError::Invalid("PNG signature"));
    }
    let mut cursor = 8usize;
    let mut header = None;
    let mut palette = Vec::<[u8; 3]>::new();
    let mut compressed = Vec::<u8>::new();
    let mut seen_data = false;
    let mut data_ended = false;
    let mut seen_transparency = false;
    loop {
        let length = be32(input, cursor)? as usize;
        let kind = bytes(input, cursor + 4, 4)?;
        if !kind.iter().all(u8::is_ascii_alphabetic) || !kind[2].is_ascii_uppercase() {
            return Err(MediaError::Invalid("PNG chunk type"));
        }
        let body_at = cursor.checked_add(8).ok_or(MediaError::Overflow)?;
        let chunk = bytes(input, body_at, length)?;
        let crc_at = body_at.checked_add(length).ok_or(MediaError::Overflow)?;
        let expected_crc = be32(input, crc_at)?;
        if crc32(bytes(
            input,
            cursor + 4,
            length.checked_add(4).ok_or(MediaError::Overflow)?,
        )?) != expected_crc
        {
            return Err(MediaError::Invalid("PNG chunk CRC"));
        }
        cursor = crc_at.checked_add(4).ok_or(MediaError::Overflow)?;
        if header.is_none() && kind != b"IHDR" {
            return Err(MediaError::Invalid("PNG must start with IHDR"));
        }
        if seen_data && kind != b"IDAT" {
            data_ended = true;
        }
        match kind {
            b"IHDR" => {
                if header.is_some() || length != 13 {
                    return Err(MediaError::Invalid("PNG IHDR"));
                }
                let size = ImageSize::new(be32(chunk, 0)? as usize, be32(chunk, 4)? as usize)?;
                check_limit("image pixels", size.pixels(), limits.max_image_pixels)?;
                let (depth, color) = (chunk[8], chunk[9]);
                let channels = match color {
                    0 | 3 => 1,
                    2 => 3,
                    4 => 2,
                    6 => 4,
                    _ => return Err(MediaError::Invalid("PNG color type")),
                };
                let legal = match color {
                    0 => [1, 2, 4, 8, 16].contains(&depth),
                    3 => [1, 2, 4, 8].contains(&depth),
                    _ => [8, 16].contains(&depth),
                };
                if !legal {
                    return Err(MediaError::Invalid("PNG depth for color type"));
                }
                if chunk[10] != 0 || chunk[11] != 0 {
                    return Err(MediaError::Unsupported("PNG compression/filter method"));
                }
                if chunk[12] != 0 {
                    return Err(if chunk[12] == 1 {
                        MediaError::Unsupported("PNG Adam7 interlacing")
                    } else {
                        MediaError::Invalid("PNG interlace method")
                    });
                }
                header = Some(Header {
                    size,
                    depth,
                    color,
                    channels,
                });
            }
            b"PLTE" => {
                if seen_data
                    || !palette.is_empty()
                    || length == 0
                    || !length.is_multiple_of(3)
                    || length > 768
                {
                    return Err(MediaError::Invalid("PNG palette"));
                }
                palette
                    .try_reserve(length / 3)
                    .map_err(|_| MediaError::Allocation)?;
                for value in chunk.as_chunks::<3>().0 {
                    palette.push(*value);
                }
            }
            b"tRNS" => {
                if seen_transparency || seen_data {
                    return Err(MediaError::Invalid("PNG transparency chunk order"));
                }
                let h = header.ok_or(MediaError::Invalid("missing PNG header"))?;
                if !match h.color {
                    0 => length == 2,
                    2 => length == 6,
                    3 => !palette.is_empty() && length <= palette.len(),
                    _ => false,
                } {
                    return Err(MediaError::Invalid("PNG transparency size"));
                }
                // Requested output is RGB, exactly as mtmd's stbi_load(...,3).
                seen_transparency = true;
            }
            b"IDAT" => {
                if data_ended {
                    return Err(MediaError::Invalid("nonconsecutive PNG IDAT"));
                }
                let count = compressed
                    .len()
                    .checked_add(length)
                    .ok_or(MediaError::Overflow)?;
                check_limit(
                    "PNG compressed bytes",
                    count,
                    limits.max_input_bytes.min(crate::MAX_ALLOCATION_BYTES),
                )?;
                compressed
                    .try_reserve(length)
                    .map_err(|_| MediaError::Allocation)?;
                compressed.extend_from_slice(chunk);
                seen_data = true;
            }
            b"IEND" => {
                if length != 0 || !seen_data {
                    return Err(MediaError::Invalid("PNG IEND"));
                }
                if cursor != input.len() {
                    return Err(MediaError::Invalid("bytes after PNG IEND"));
                }
                break;
            }
            _ => {
                if kind[0].is_ascii_uppercase() {
                    return Err(MediaError::Unsupported("unknown critical PNG chunk"));
                }
            }
        }
    }
    let header = header.ok_or(MediaError::Invalid("missing PNG header"))?;
    if header.color == 3 && (palette.is_empty() || palette.len() > 1usize << header.depth) {
        return Err(MediaError::Invalid("missing or oversized PNG palette"));
    }
    let row_bits = product(
        product(header.size.width(), header.channels)?,
        header.depth as usize,
    )?;
    let row_bytes = row_bits.div_ceil(8);
    let filtered_size = product(
        row_bytes.checked_add(1).ok_or(MediaError::Overflow)?,
        header.size.height(),
    )?;
    check_limit(
        "PNG inflated bytes",
        filtered_size,
        crate::MAX_ALLOCATION_BYTES,
    )?;
    let mut inflated = inflate_zlib(&compressed, filtered_size)?;
    let bytes_per_pixel = (header.channels * header.depth as usize).div_ceil(8);
    for y in 0..header.size.height() {
        let offset = y * (row_bytes + 1);
        let filter = inflated[offset];
        if filter > 4 {
            return Err(MediaError::Invalid("PNG row filter"));
        }
        for x in 0..row_bytes {
            let position = offset + 1 + x;
            let left = if x >= bytes_per_pixel {
                inflated[position - bytes_per_pixel]
            } else {
                0
            };
            let above = if y > 0 {
                inflated[position - row_bytes - 1]
            } else {
                0
            };
            let upper_left = if y > 0 && x >= bytes_per_pixel {
                inflated[position - row_bytes - 1 - bytes_per_pixel]
            } else {
                0
            };
            let prediction = match filter {
                0 => 0,
                1 => left,
                2 => above,
                3 => ((left as u16 + above as u16) / 2) as u8,
                4 => paeth(left, above, upper_left),
                _ => unreachable!(),
            };
            inflated[position] = inflated[position].wrapping_add(prediction);
        }
    }
    let mut rgb = zeros(product(header.size.pixels(), 3)?, 0u8)?;
    for y in 0..header.size.height() {
        let row = &inflated[y * (row_bytes + 1) + 1..(y + 1) * (row_bytes + 1)];
        for x in 0..header.size.width() {
            let dest = (y * header.size.width() + x) * 3;
            let sample = |channel: usize| -> u8 {
                let index = x * header.channels + channel;
                if header.depth == 16 {
                    row[index * 2]
                } else if header.depth == 8 {
                    row[index]
                } else {
                    let bit = index * header.depth as usize;
                    let mask = (1u8 << header.depth) - 1;
                    (row[bit / 8] >> (8 - header.depth - bit as u8 % 8)) & mask
                }
            };
            match header.color {
                0 | 4 => {
                    let grey = sample(0);
                    let value = if header.depth < 8 {
                        grey * (255 / ((1u8 << header.depth) - 1))
                    } else {
                        grey
                    };
                    rgb[dest..dest + 3].fill(value);
                }
                2 | 6 => {
                    rgb[dest] = sample(0);
                    rgb[dest + 1] = sample(1);
                    rgb[dest + 2] = sample(2);
                }
                3 => {
                    let index = sample(0) as usize;
                    let color = palette
                        .get(index)
                        .ok_or(MediaError::Invalid("PNG palette index"))?;
                    rgb[dest..dest + 3].copy_from_slice(color);
                }
                _ => return Err(MediaError::Invalid("PNG color type")),
            }
        }
    }
    RgbImage::new(header.size.width(), header.size.height(), rgb)
}

fn paeth(left: u8, above: u8, upper_left: u8) -> u8 {
    let p = left as i32 + above as i32 - upper_left as i32;
    let a = (p - left as i32).abs();
    let b = (p - above as i32).abs();
    let c = (p - upper_left as i32).abs();
    if a <= b && a <= c {
        left
    } else if b <= c {
        above
    } else {
        upper_left
    }
}

struct Bits<'a> {
    data: &'a [u8],
    position: usize,
    value: u64,
    available: u8,
}
impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            position: 0,
            value: 0,
            available: 0,
        }
    }
    fn read(&mut self, count: u8) -> Result<u32> {
        while self.available < count {
            let byte = *self.data.get(self.position).ok_or(MediaError::Truncated {
                offset: self.position,
                required: 1,
            })?;
            self.position += 1;
            self.value |= (byte as u64) << self.available;
            self.available += 8;
        }
        let value = (self.value & ((1u64 << count) - 1)) as u32;
        self.value >>= count;
        self.available -= count;
        Ok(value)
    }
    fn align(&mut self) {
        self.value = 0;
        self.available = 0;
    }
}
#[derive(Clone, Default)]
struct Node {
    children: [Option<usize>; 2],
    symbol: Option<u16>,
}
struct Huffman {
    nodes: Vec<Node>,
}
impl Huffman {
    fn new(lengths: &[u8]) -> Result<Self> {
        let mut counts = [0u32; 16];
        for &length in lengths {
            if length > 15 {
                return Err(MediaError::Invalid("DEFLATE code length"));
            }
            if length > 0 {
                counts[length as usize] += 1;
            }
        }
        let mut remaining = 1i32;
        for &count in &counts[1..] {
            remaining = remaining * 2 - count as i32;
            if remaining < 0 {
                return Err(MediaError::Invalid("oversubscribed DEFLATE tree"));
            }
        }
        let mut next = [0u32; 16];
        let mut code = 0u32;
        for bit in 1..16 {
            code = (code + counts[bit - 1]) << 1;
            next[bit] = code;
        }
        let mut nodes = Vec::new();
        // Every canonical tree has at most 1 + 15 * symbols nodes, bounded here.
        nodes
            .try_reserve(lengths.len() * 15 + 1)
            .map_err(|_| MediaError::Allocation)?;
        nodes.push(Node::default());
        for (symbol, &length) in lengths.iter().enumerate() {
            if length == 0 {
                continue;
            }
            let code = next[length as usize];
            next[length as usize] += 1;
            let mut cursor = 0usize;
            for shift in (0..length).rev() {
                if nodes[cursor].symbol.is_some() {
                    return Err(MediaError::Invalid("overlapping DEFLATE code"));
                }
                let bit = ((code >> shift) & 1) as usize;
                let child = if let Some(child) = nodes[cursor].children[bit] {
                    child
                } else {
                    let child = nodes.len();
                    nodes.push(Node::default());
                    nodes[cursor].children[bit] = Some(child);
                    child
                };
                cursor = child;
            }
            if nodes[cursor].symbol.is_some() || nodes[cursor].children.iter().any(Option::is_some)
            {
                return Err(MediaError::Invalid("overlapping DEFLATE leaf"));
            }
            nodes[cursor].symbol = Some(symbol as u16);
        }
        Ok(Self { nodes })
    }
    fn symbol(&self, bits: &mut Bits<'_>) -> Result<u16> {
        let mut cursor = 0usize;
        for _ in 0..15 {
            let bit = bits.read(1)? as usize;
            cursor = self.nodes[cursor].children[bit]
                .ok_or(MediaError::Invalid("invalid DEFLATE code"))?;
            if let Some(symbol) = self.nodes[cursor].symbol {
                return Ok(symbol);
            }
        }
        Err(MediaError::Invalid("unterminated DEFLATE code"))
    }
}
fn dynamic_trees(bits: &mut Bits<'_>) -> Result<(Huffman, Huffman)> {
    let literals = bits.read(5)? as usize + 257;
    let distances = bits.read(5)? as usize + 1;
    let code_count = bits.read(4)? as usize + 4;
    if literals > 286 {
        return Err(MediaError::Invalid("DEFLATE literal count"));
    }
    const ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    // Explicit values preserve the format's permutation.
    let mut code_lengths = [0u8; 19];
    for &index in &ORDER[..code_count] {
        code_lengths[index] = bits.read(3)? as u8;
    }
    let tree = Huffman::new(&code_lengths)?;
    let mut lengths = Vec::new();
    lengths
        .try_reserve_exact(literals + distances)
        .map_err(|_| MediaError::Allocation)?;
    while lengths.len() < literals + distances {
        let symbol = tree.symbol(bits)?;
        let (value, count) = match symbol {
            0..=15 => (symbol as u8, 1usize),
            16 => (
                *lengths
                    .last()
                    .ok_or(MediaError::Invalid("DEFLATE repeat without previous code"))?,
                bits.read(2)? as usize + 3,
            ),
            17 => (0, bits.read(3)? as usize + 3),
            18 => (0, bits.read(7)? as usize + 11),
            _ => return Err(MediaError::Invalid("DEFLATE length alphabet")),
        };
        if lengths.len() + count > literals + distances {
            return Err(MediaError::Invalid("DEFLATE length repeat overflow"));
        }
        lengths.resize(lengths.len() + count, value);
    }
    if lengths[256] == 0 {
        return Err(MediaError::Invalid("DEFLATE tree lacks end-of-block"));
    }
    Ok((
        Huffman::new(&lengths[..literals])?,
        Huffman::new(&lengths[literals..])?,
    ))
}
fn fixed_trees() -> Result<(Huffman, Huffman)> {
    let mut lengths = [8u8; 288];
    lengths[144..256].fill(9);
    lengths[256..280].fill(7);
    lengths[280..].fill(8);
    Ok((Huffman::new(&lengths)?, Huffman::new(&[5u8; 32])?))
}
fn inflate_zlib(input: &[u8], expected: usize) -> Result<Vec<u8>> {
    if input.len() < 6 {
        return Err(MediaError::Invalid("short zlib stream"));
    }
    let cmf = input[0];
    let flags = input[1];
    if cmf & 15 != 8 || cmf >> 4 > 7 || !(cmf as u16 * 256 + flags as u16).is_multiple_of(31) {
        return Err(MediaError::Invalid("zlib header"));
    }
    if flags & 32 != 0 {
        return Err(MediaError::Unsupported("zlib preset dictionary"));
    }
    let window = 1usize << ((cmf >> 4) + 8);
    let mut bits = Bits::new(&input[2..input.len() - 4]);
    let mut output = Vec::<u8>::new();
    check_limit(
        "DEFLATE output bytes",
        expected,
        crate::MAX_ALLOCATION_BYTES,
    )?;
    output
        .try_reserve_exact(expected)
        .map_err(|_| MediaError::Allocation)?;
    loop {
        let final_block = bits.read(1)? != 0;
        let block_type = bits.read(2)?;
        if block_type == 0 {
            bits.align();
            let length = bits.read(16)? as usize;
            let inverse = bits.read(16)? as u16;
            if (length as u16) ^ inverse != u16::MAX {
                return Err(MediaError::Invalid("DEFLATE stored block length"));
            }
            if output
                .len()
                .checked_add(length)
                .ok_or(MediaError::Overflow)?
                > expected
            {
                return Err(MediaError::Invalid("DEFLATE output exceeds PNG shape"));
            }
            for _ in 0..length {
                output.push(bits.read(8)? as u8);
            }
        } else if block_type == 1 || block_type == 2 {
            let (literals, distances) = if block_type == 1 {
                fixed_trees()?
            } else {
                dynamic_trees(&mut bits)?
            };
            loop {
                let symbol = literals.symbol(&mut bits)?;
                if symbol == 256 {
                    break;
                }
                if symbol < 256 {
                    if output.len() >= expected {
                        return Err(MediaError::Invalid("DEFLATE output exceeds PNG shape"));
                    }
                    output.push(symbol as u8);
                    continue;
                }
                if symbol > 285 {
                    return Err(MediaError::Invalid("reserved DEFLATE length"));
                }
                const LENGTH: [usize; 29] = [
                    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67,
                    83, 99, 115, 131, 163, 195, 227, 258,
                ];
                const LENGTH_EXTRA: [u8; 29] = [
                    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5,
                    5, 5, 0,
                ];
                let index = (symbol - 257) as usize;
                let length = LENGTH[index] + bits.read(LENGTH_EXTRA[index])? as usize;
                let distance_symbol = distances.symbol(&mut bits)? as usize;
                if distance_symbol >= 30 {
                    return Err(MediaError::Invalid("reserved DEFLATE distance"));
                }
                const DISTANCE: [usize; 30] = [
                    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769,
                    1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
                ];
                const DISTANCE_EXTRA: [u8; 30] = [
                    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11,
                    11, 12, 12, 13, 13,
                ];
                let distance = DISTANCE[distance_symbol]
                    + bits.read(DISTANCE_EXTRA[distance_symbol])? as usize;
                if distance > output.len() || distance > window {
                    return Err(MediaError::Invalid("DEFLATE distance before history"));
                }
                if output
                    .len()
                    .checked_add(length)
                    .ok_or(MediaError::Overflow)?
                    > expected
                {
                    return Err(MediaError::Invalid("DEFLATE output exceeds PNG shape"));
                }
                for _ in 0..length {
                    output.push(output[output.len() - distance]);
                }
            }
        } else {
            return Err(MediaError::Invalid("reserved DEFLATE block type"));
        }
        if final_block {
            break;
        }
    }
    if output.len() != expected {
        return Err(MediaError::Invalid("DEFLATE output shorter than PNG shape"));
    }
    if bits.position != bits.data.len() {
        return Err(MediaError::Invalid("trailing zlib bytes"));
    }
    if adler32(&output) != be32(input, input.len() - 4)? {
        return Err(MediaError::Invalid("zlib Adler-32"));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stored_blocks_checksum_and_expansion_bound() {
        let mut valid = vec![0x78, 0x01, 0x01, 5, 0, 0xfa, 0xff];
        valid.extend(b"hello");
        valid.extend(adler32(b"hello").to_be_bytes());
        assert_eq!(inflate_zlib(&valid, 5).unwrap(), b"hello");
        assert!(inflate_zlib(&valid, 4).is_err());
        assert!(inflate_zlib(&valid, 6).is_err());
        let mut bad = valid.clone();
        bad[5] ^= 1;
        assert!(inflate_zlib(&bad, 5).is_err());
        let mut bad = valid.clone();
        bad[2] = 7;
        assert!(inflate_zlib(&bad, 5).is_err());
        let mut bad = valid.clone();
        *bad.last_mut().unwrap() ^= 1;
        assert!(inflate_zlib(&bad, 5).is_err());
        for end in 0..valid.len() {
            assert!(inflate_zlib(&valid[..end], 5).is_err());
        }
    }
    #[test]
    fn bounded_malformed_deflate_corpus_has_no_panics() {
        let mut seed = 0x123456789abcdef0u64;
        for len in 0..64 {
            for _ in 0..32 {
                let mut input = vec![0x78, 0x01];
                for _ in 0..len {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    input.push(seed as u8);
                }
                input.extend([0u8; 4]);
                let _ = inflate_zlib(&input, 17);
            }
        }
    }
    #[test]
    fn standard_checksums() {
        assert_eq!(crc32(b"123456789"), 0xcbf43926);
        assert_eq!(adler32(b"Wikipedia"), 0x11e60398);
    }
    #[test]
    fn corrupt_huffman_tables_are_rejected() {
        assert!(Huffman::new(&[1, 1, 1]).is_err());
        assert!(Huffman::new(&[16]).is_err());
        assert!(
            Huffman::new(&[])
                .unwrap()
                .symbol(&mut Bits::new(&[0]))
                .is_err()
        );
    }
}
