//! Rust ports of bounded specialty-backend utilities. These do not constitute
//! running a DSP, ET accelerator or virtual GPU.
use crate::{Error, Result};

pub const ET_FORBIDDEN_INSTRUCTIONS: [&str; 14] = [
    "fdiv.s",
    "fsqrt.s",
    "fcvt.l.s",
    "fcvt.lu.s",
    "fcvt.s.l",
    "fcvt.s.lu",
    "fdiv.pi",
    "fdivu.pi",
    "fremu.pi",
    "frem.pi",
    "fdiv.ps",
    "fsqrt.ps",
    "frsq.ps",
    "fsin.ps",
];
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForbiddenInstruction {
    pub line: usize,
    pub mnemonic: &'static str,
}
/// Implements the frozen ET build guard's mnemonic deny-list in Rust. Input
/// must be disassembly/assembly text; this does not decode binary opcodes.
pub fn et_instruction_audit(text: &str) -> Vec<ForbiddenInstruction> {
    let mut found = Vec::new();
    for (line, source) in text.lines().enumerate() {
        let source = source
            .split('#')
            .next()
            .unwrap_or("")
            .split("//")
            .next()
            .unwrap_or("");
        for token in source.split_whitespace() {
            let token = token.trim_matches(|c: char| c == ',' || c == '{' || c == '}');
            if let Some(&mnemonic) = ET_FORBIDDEN_INSTRUCTIONS.iter().find(|&&p| p == token) {
                found.push(ForbiddenInstruction {
                    line: line + 1,
                    mnemonic,
                });
            }
        }
    }
    found
}

pub const SWIV_SECTION_TYPE: u32 = 0xD3574956;
const MAX_ELF_BYTES: usize = 256 * 1024 * 1024;
const MAX_LOADED_BYTES: u64 = 256 * 1024 * 1024;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElfClass {
    Elf32,
    Elf64,
}
#[derive(Clone, Debug)]
pub struct Segment {
    pub kind: u32,
    pub offset: u64,
    pub virtual_address: u64,
    pub file_size: u64,
    pub memory_size: u64,
    pub flags: u32,
    pub alignment: u64,
}
#[derive(Clone, Debug)]
pub struct Section {
    pub kind: u32,
    pub offset: u64,
    pub size: u64,
}
#[derive(Clone, Debug)]
pub struct ElfImage<'a> {
    pub class: ElfClass,
    pub machine: u16,
    pub segments: Vec<Segment>,
    pub sections: Vec<Section>,
    data: &'a [u8],
    header_size: usize,
    phoff: usize,
    phsize: usize,
    shoff: usize,
    shsize: usize,
}
fn range(data: &[u8], offset: u64, size: u64) -> Result<&[u8]> {
    let end = offset
        .checked_add(size)
        .ok_or(Error::Bounds("ELF range overflow"))?;
    if end > data.len() as u64 {
        return Err(Error::Bounds("ELF range exceeds file"));
    }
    Ok(&data[offset as usize..end as usize])
}
fn u16_at(data: &[u8], o: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        range(data, o as u64, 2)?
            .try_into()
            .expect("checked length"),
    ))
}
fn u32_at(data: &[u8], o: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        range(data, o as u64, 4)?
            .try_into()
            .expect("checked length"),
    ))
}
fn u64_at(data: &[u8], o: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        range(data, o as u64, 8)?
            .try_into()
            .expect("checked length"),
    ))
}
impl<'a> ElfImage<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        if data.len() > MAX_ELF_BYTES {
            return Err(Error::Bounds("ELF file size limit"));
        }
        if range(data, 0, 16)?[..4] != *b"\x7fELF" || data[6] != 1 {
            return Err(Error::Protocol("invalid ELF identity/version".into()));
        }
        if data[5] != 1 {
            return Err(Error::Unsupported("ELF big endian encoding"));
        }
        let class = match data[4] {
            1 => ElfClass::Elf32,
            2 => ElfClass::Elf64,
            _ => return Err(Error::Protocol("invalid ELF class".into())),
        };
        let (header_size, phoff, shoff, ehsize, phsize, phcount, shsize, shcount) = match class {
            ElfClass::Elf32 => (
                52,
                u32_at(data, 28)? as u64,
                u32_at(data, 32)? as u64,
                u16_at(data, 40)?,
                u16_at(data, 42)?,
                u16_at(data, 44)?,
                u16_at(data, 46)?,
                u16_at(data, 48)?,
            ),
            ElfClass::Elf64 => (
                64,
                u64_at(data, 32)?,
                u64_at(data, 40)?,
                u16_at(data, 52)?,
                u16_at(data, 54)?,
                u16_at(data, 56)?,
                u16_at(data, 58)?,
                u16_at(data, 60)?,
            ),
        };
        if u32_at(data, 20)? != 1 || ehsize as usize != header_size {
            return Err(Error::Protocol("invalid ELF header version/size".into()));
        }
        if phcount == 0 || phcount > 100 {
            return Err(Error::Bounds("ELF must have 1..=100 program headers"));
        }
        let (expected_ph, expected_sh) = match class {
            ElfClass::Elf32 => (32, 40),
            ElfClass::Elf64 => (56, 64),
        };
        if phsize != expected_ph
            || shsize != expected_sh
            || shcount == 0
            || shcount > 4096
            || phoff == 0
            || shoff == 0
        {
            return Err(Error::Bounds("invalid ELF table sizes/counts"));
        }
        range(data, phoff, u64::from(phsize) * u64::from(phcount))?;
        range(data, shoff, u64::from(shsize) * u64::from(shcount))?;
        let mut segments = Vec::new();
        let mut loaded = 0u64;
        for i in 0..phcount as usize {
            let p = range(data, phoff + i as u64 * u64::from(phsize), phsize as u64)?;
            let segment = match class {
                ElfClass::Elf32 => Segment {
                    kind: u32_at(p, 0)?,
                    offset: u32_at(p, 4)? as u64,
                    virtual_address: u32_at(p, 8)? as u64,
                    file_size: u32_at(p, 16)? as u64,
                    memory_size: u32_at(p, 20)? as u64,
                    flags: u32_at(p, 24)?,
                    alignment: u32_at(p, 28)? as u64,
                },
                ElfClass::Elf64 => Segment {
                    kind: u32_at(p, 0)?,
                    flags: u32_at(p, 4)?,
                    offset: u64_at(p, 8)?,
                    virtual_address: u64_at(p, 16)?,
                    file_size: u64_at(p, 32)?,
                    memory_size: u64_at(p, 40)?,
                    alignment: u64_at(p, 48)?,
                },
            };
            range(data, segment.offset, segment.file_size)?;
            if segment.kind == 1 {
                if segment.memory_size < segment.file_size {
                    return Err(Error::Bounds("ELF load memory smaller than file data"));
                }
                loaded = loaded
                    .checked_add(segment.memory_size)
                    .ok_or(Error::Bounds("ELF load size overflow"))?;
                if loaded > MAX_LOADED_BYTES {
                    return Err(Error::Bounds("ELF load/BSS CRC limit"));
                }
            }
            if segment.alignment > 1
                && (!segment.alignment.is_power_of_two()
                    || segment.offset % segment.alignment
                        != segment.virtual_address % segment.alignment)
            {
                return Err(Error::Bounds("ELF segment alignment"));
            }
            segments.push(segment);
        }
        let mut sections = Vec::new();
        for i in 0..shcount as usize {
            let p = range(data, shoff + i as u64 * u64::from(shsize), shsize as u64)?;
            let section = match class {
                ElfClass::Elf32 => Section {
                    kind: u32_at(p, 4)?,
                    offset: u32_at(p, 16)? as u64,
                    size: u32_at(p, 20)? as u64,
                },
                ElfClass::Elf64 => Section {
                    kind: u32_at(p, 4)?,
                    offset: u64_at(p, 24)?,
                    size: u64_at(p, 32)?,
                },
            };
            if section.kind != 8 && section.size != 0 {
                range(data, section.offset, section.size)?;
            }
            sections.push(section);
        }
        Ok(Self {
            class,
            machine: u16_at(data, 18)?,
            segments,
            sections,
            data,
            header_size,
            phoff: phoff as usize,
            phsize: phsize as usize,
            shoff: shoff as usize,
            shsize: shsize as usize,
        })
    }
    pub fn loaded_crc32(&self) -> Result<u32> {
        let mut crc = 0u32;
        let zeros = [0u8; 4096];
        for p in self.segments.iter().filter(|p| p.kind == 1) {
            crc = crc32_update(crc, range(self.data, p.offset, p.file_size)?);
            let mut remaining = p.memory_size - p.file_size;
            while remaining > 0 {
                let n = remaining.min(4096) as usize;
                crc = crc32_update(crc, &zeros[..n]);
                remaining -= n as u64;
            }
        }
        Ok(crc)
    }
    pub fn verify_swiv(&self) -> Result<bool> {
        let matching: Vec<_> = self
            .sections
            .iter()
            .filter(|s| s.kind == SWIV_SECTION_TYPE)
            .collect();
        if matching.len() != 1 {
            return Ok(false);
        }
        let s = matching[0];
        if s.size != 16 {
            return Ok(false);
        }
        let bytes = range(self.data, s.offset, 16)?;
        Ok(&bytes[..4] == b"SWIV"
            && u32_at(bytes, 4)? == self.loaded_crc32()?
            && bytes[8..].iter().all(|b| *b == 0))
    }
}

/// Port of htp-swiv.py's section generation for checked ELF32/ELF64 LE images.
/// Copies segments and sections in the same order as the frozen script, hashes
/// loadable data plus BSS zeros, and writes a 16-byte SWIV record.
pub fn append_swiv(data: &[u8]) -> Result<Vec<u8>> {
    let elf = ElfImage::parse(data)?;
    if elf.sections.iter().any(|s| {
        s.kind == SWIV_SECTION_TYPE
            || s.size == 16 && range(data, s.offset, 4).is_ok_and(|b| b == b"SWIV")
    }) || elf
        .segments
        .iter()
        .any(|s| s.memory_size == 16 && range(data, s.offset, 4).is_ok_and(|b| b == b"SWIV"))
    {
        return Err(Error::Protocol("SWIV already present".into()));
    }
    let end_table = elf
        .shoff
        .checked_add((elf.sections.len() + 1) * elf.shsize)
        .ok_or(Error::Bounds("SWIV table overflow"))?;
    let end = end_table
        .checked_add(16)
        .ok_or(Error::Bounds("SWIV output overflow"))?;
    if end > MAX_ELF_BYTES {
        return Err(Error::Bounds("SWIV output size limit"));
    }
    let extension_start = elf.shoff + elf.sections.len() * elf.shsize;
    // The original assumes free space after the section table. Reject an input
    // whose new metadata would overwrite a meaningful segment or section.
    for (off, size) in elf.segments.iter().map(|s| (s.offset, s.file_size)).chain(
        elf.sections
            .iter()
            .filter(|s| s.kind != 8)
            .map(|s| (s.offset, s.size)),
    ) {
        if size > 0 && off < (end as u64) && off + size > extension_start as u64 {
            return Err(Error::Unsupported(
                "SWIV table growth overlaps existing ELF contents",
            ));
        }
    }
    let mut max = end;
    for p in &elf.segments {
        max = max.max((p.offset + p.file_size) as usize);
    }
    for s in elf.sections.iter().filter(|s| s.kind != 8) {
        max = max.max((s.offset + s.size) as usize);
    }
    if max > MAX_ELF_BYTES {
        return Err(Error::Bounds("SWIV output size limit"));
    }
    let mut out = vec![0u8; max];
    for p in &elf.segments {
        let start = p.offset as usize;
        let size = p.file_size as usize;
        out[start..start + size].copy_from_slice(range(data, p.offset, p.file_size)?);
    }
    out[..elf.header_size].copy_from_slice(&data[..elf.header_size]);
    let count = (elf.sections.len() + 1) as u16;
    let count_off = match elf.class {
        ElfClass::Elf32 => 48,
        ElfClass::Elf64 => 60,
    };
    out[count_off..count_off + 2].copy_from_slice(&count.to_le_bytes());
    let phlen = elf.phsize * elf.segments.len();
    out[elf.phoff..elf.phoff + phlen].copy_from_slice(&data[elf.phoff..elf.phoff + phlen]);
    for s in elf.sections.iter().filter(|s| s.kind != 8 && s.size != 0) {
        let start = s.offset as usize;
        let size = s.size as usize;
        out[start..start + size].copy_from_slice(range(data, s.offset, s.size)?);
    }
    let oldlen = elf.shsize * elf.sections.len();
    out[elf.shoff..elf.shoff + oldlen].copy_from_slice(&data[elf.shoff..elf.shoff + oldlen]);
    let s = &mut out[extension_start..end_table];
    s.fill(0);
    s[4..8].copy_from_slice(&SWIV_SECTION_TYPE.to_le_bytes());
    match elf.class {
        ElfClass::Elf32 => {
            s[16..20].copy_from_slice(&(end_table as u32).to_le_bytes());
            s[20..24].copy_from_slice(&16u32.to_le_bytes());
        }
        ElfClass::Elf64 => {
            s[24..32].copy_from_slice(&(end_table as u64).to_le_bytes());
            s[32..40].copy_from_slice(&16u64.to_le_bytes());
        }
    }
    let crc = ElfImage::parse(&out)?.loaded_crc32()?;
    out[end_table..end_table + 4].copy_from_slice(b"SWIV");
    out[end_table + 4..end_table + 8].copy_from_slice(&crc.to_le_bytes());
    Ok(out)
}
const fn crc_table() -> [u32; 256] {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        let mut n = i as u32;
        let mut bit = 0;
        while bit < 8 {
            n = if n & 1 != 0 {
                (n >> 1) ^ 0xEDB88320
            } else {
                n >> 1
            };
            bit += 1;
        }
        table[i] = n;
        i += 1;
    }
    table
}
const CRC_TABLE: [u32; 256] = crc_table();
pub fn crc32_update(previous: u32, data: &[u8]) -> u32 {
    let mut crc = !previous;
    for b in data {
        crc = (crc >> 8) ^ CRC_TABLE[((crc ^ u32::from(*b)) & 255) as usize];
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;
    fn elf64() -> Vec<u8> {
        let mut b = vec![0u8; 320];
        b[..7].copy_from_slice(&[127, 69, 76, 70, 2, 1, 1]);
        b[16..18].copy_from_slice(&3u16.to_le_bytes());
        b[18..20].copy_from_slice(&164u16.to_le_bytes());
        b[20..24].copy_from_slice(&1u32.to_le_bytes());
        b[32..40].copy_from_slice(&64u64.to_le_bytes());
        b[40..48].copy_from_slice(&256u64.to_le_bytes());
        b[52..54].copy_from_slice(&64u16.to_le_bytes());
        b[54..56].copy_from_slice(&56u16.to_le_bytes());
        b[56..58].copy_from_slice(&1u16.to_le_bytes());
        b[58..60].copy_from_slice(&64u16.to_le_bytes());
        b[60..62].copy_from_slice(&1u16.to_le_bytes());
        b[64..68].copy_from_slice(&1u32.to_le_bytes());
        b[72..80].copy_from_slice(&128u64.to_le_bytes());
        b[96..104].copy_from_slice(&4u64.to_le_bytes());
        b[104..112].copy_from_slice(&8u64.to_le_bytes());
        b[128..132].copy_from_slice(b"rust");
        b
    }
    #[test]
    fn et_guard_matches_exact_mnemonics() {
        let result = et_instruction_audit(
            "100: fdiv.s f1,f2,f3\n# fsqrt.ps ignored\nadd t0,t1,t2\n104: 000 fsin.ps f1,f2\nnot_fdiv.s_suffix\n",
        );
        assert_eq!(
            result,
            vec![
                ForbiddenInstruction {
                    line: 1,
                    mnemonic: "fdiv.s"
                },
                ForbiddenInstruction {
                    line: 4,
                    mnemonic: "fsin.ps"
                }
            ]
        );
    }
    #[test]
    fn crc_standard_vector_and_incremental() {
        assert_eq!(crc32_update(0, b"123456789"), 0xcbf43926);
        assert_eq!(crc32_update(crc32_update(0, b"1234"), b"56789"), 0xcbf43926);
    }
    #[test]
    fn elf_bss_and_swiv_roundtrip() {
        let b = elf64();
        let elf = ElfImage::parse(&b).unwrap();
        assert_eq!(elf.machine, 164);
        assert_eq!(
            elf.loaded_crc32().unwrap(),
            crc32_update(0, b"rust\0\0\0\0")
        );
        let mut out = append_swiv(&b).unwrap();
        assert_eq!(out.len(), 400);
        assert!(ElfImage::parse(&out).unwrap().verify_swiv().unwrap());
        assert!(append_swiv(&out).is_err());
        out[128] ^= 1;
        assert!(!ElfImage::parse(&out).unwrap().verify_swiv().unwrap());
    }
    #[test]
    fn malformed_elf_rejected() {
        let b = elf64();
        for n in [0, 4, 16, 63, 100, 200] {
            assert!(ElfImage::parse(&b[..n]).is_err());
        }
        let mut b = elf64();
        b[72..80].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(ElfImage::parse(&b).is_err());
        let mut b = elf64();
        b[104..112].copy_from_slice(&3u64.to_le_bytes());
        assert!(ElfImage::parse(&b).is_err());
        let mut b = elf64();
        b[5] = 2;
        assert!(ElfImage::parse(&b).is_err());
    }
}
