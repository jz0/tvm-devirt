use anyhow::{Context, Result, bail};
use crate::binary::pe_format as fmt;

#[derive(Debug, Clone)]
pub struct Section {
    pub name: String,
    pub virtual_address: u32,
    pub virtual_size: u32,
    pub raw_address: u32,
    pub raw_size: u32,
    pub characteristics: u32,
}

impl Section {
    pub fn contains_rva(&self, rva: u32) -> bool {
        rva >= self.virtual_address
            && rva < self.virtual_address + self.virtual_size.max(self.raw_size)
    }

    pub fn is_executable(&self) -> bool {
        self.characteristics & fmt::IMAGE_SCN_MEM_EXECUTE != 0
    }

    pub fn is_writable(&self) -> bool {
        self.characteristics & fmt::IMAGE_SCN_MEM_WRITE != 0
    }
}

#[derive(Debug)]
pub struct PeFile {
    pub data: Vec<u8>,
    pub image_base: u64,
    pub entry_point_rva: u32,
    pub size_of_image: u32,
    pub sections: Vec<Section>,
    pub opt_header_offset: usize,
    pub section_table_offset: usize,
    pub file_alignment: u32,
    pub section_alignment: u32,
    pub size_of_headers: u32,
    pub loader_bound: Vec<(u64, u64)>,
}

impl PeFile {
    pub fn parse(data: Vec<u8>) -> Result<Self> {
        if data.len() < 0x40 || &data[0..2] != b"MZ" {
            bail!("not an MZ image");
        }
        let pe_off = rd_u32(&data, 0x3c)? as usize;
        if data.get(pe_off..pe_off + 4) != Some(b"PE\0\0") {
            bail!("bad PE signature at {pe_off:#x}");
        }
        let machine = rd_u16(&data, pe_off + 4)?;
        if machine != fmt::IMAGE_FILE_MACHINE_AMD64 {
            bail!("only AMD64 images are supported (machine={machine:#x})");
        }
        let num_sections = rd_u16(&data, pe_off + 6)? as usize;
        let opt_size = rd_u16(&data, pe_off + 20)? as usize;
        let opt = pe_off + 24;
        let magic = rd_u16(&data, opt)?;
        if magic != fmt::PE32_PLUS_MAGIC {
            bail!("only PE32+ is supported (magic={magic:#x})");
        }

        let entry_point_rva = rd_u32(&data, opt + 16)?;
        let section_alignment = rd_u32(&data, opt + 32)?;
        let file_alignment = rd_u32(&data, opt + 36)?;
        let size_of_image = rd_u32(&data, opt + fmt::OPTIONAL_HEADER_SIZE_OF_IMAGE_OFFSET)?;
        let size_of_headers = rd_u32(&data, opt + 60)?;
        let image_base = rd_u64(&data, opt + 24)?;

        let section_table_offset = opt + opt_size;
        let mut sections = Vec::with_capacity(num_sections);
        for i in 0..num_sections {
            let o = section_table_offset + i * fmt::SECTION_HEADER_SIZE;
            let raw_name = data.get(o..o + 8).context("truncated section header")?;
            let name = String::from_utf8_lossy(raw_name)
                .trim_end_matches('\0')
                .to_string();
            sections.push(Section {
                name,
                virtual_size: rd_u32(&data, o + 8)?,
                virtual_address: rd_u32(&data, o + 12)?,
                raw_size: rd_u32(&data, o + 16)?,
                raw_address: rd_u32(&data, o + 20)?,
                characteristics: rd_u32(&data, o + fmt::SECTION_CHARACTERISTICS_OFFSET)?,
            });
        }

        Ok(Self {
            data,
            image_base,
            entry_point_rva,
            size_of_image,
            sections,
            opt_header_offset: opt,
            section_table_offset,
            file_alignment,
            section_alignment,
            size_of_headers,
            loader_bound: Vec::new(),
        })
        .map(|mut pe: Self| {
            pe.loader_bound = pe.compute_loader_bound();
            pe
        })
    }

    /// Ranges the loader binds: the IAT directory, plus every descriptor's address table.
    fn compute_loader_bound(&self) -> Vec<(u64, u64)> {
        let mut out: Vec<(u64, u64)> = Vec::new();

        let o = fmt::data_directory_offset(self.opt_header_offset, fmt::IMAGE_DIRECTORY_ENTRY_IAT);
        if let Some(b) = self.data.get(o..o + fmt::DATA_DIRECTORY_ENTRY_SIZE) {
            let rva = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            let size = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
            if rva != 0 && size != 0 {
                let lo = self.rva_to_va(rva);
                out.push((lo, lo + size as u64));
            }
        }

        // Each import descriptor's address table, walked to its null terminator.
        for (slot, _, _) in self.imports() {
            out.push((slot, slot + 8));
        }

        out.sort_unstable();
        // Coalesce touching or overlapping ranges so the membership test stays short.
        let mut merged: Vec<(u64, u64)> = Vec::new();
        for (lo, hi) in out {
            match merged.last_mut() {
                Some(last) if lo <= last.1 => last.1 = last.1.max(hi),
                _ => merged.push((lo, hi)),
            }
        }
        merged
    }

    /// Whether reading `len` bytes at `va` touches anything the loader rewrites.
    pub fn is_loader_bound(&self, va: u64, len: u64) -> bool {
        let end = va.saturating_add(len.max(1));
        self.loader_bound
            .iter()
            .any(|&(lo, hi)| va < hi && lo < end)
    }

    pub fn load(path: &std::path::Path) -> Result<Self> {
        let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(data)
    }

    pub fn section_by_name(&self, name: &str) -> Option<&Section> {
        self.sections.iter().find(|s| s.name == name)
    }

    pub fn section_for_rva(&self, rva: u32) -> Option<&Section> {
        self.sections.iter().find(|s| s.contains_rva(rva))
    }

    pub fn section_for_va(&self, va: u64) -> Option<&Section> {
        let rva = self.va_to_rva(va)?;
        self.section_for_rva(rva)
    }

    pub fn va_to_rva(&self, va: u64) -> Option<u32> {
        va.checked_sub(self.image_base)?.try_into().ok()
    }

    pub fn rva_to_va(&self, rva: u32) -> u64 {
        self.image_base + rva as u64
    }

    /// Raw file bytes backing a section, clipped to what is actually present.
    pub fn section_bytes(&self, s: &Section) -> &[u8] {
        let start = s.raw_address as usize;
        let end = (start + s.raw_size as usize).min(self.data.len());
        if start >= self.data.len() {
            &[]
        } else {
            &self.data[start..end]
        }
    }

    /// Read `len` bytes at a virtual address, following the section mapping.
    /// Returns a short slice when the request runs off the end of raw data.
    pub fn read_va(&self, va: u64, len: usize) -> Option<&[u8]> {
        let rva = self.va_to_rva(va)?;
        let s = self.section_for_rva(rva)?;
        let delta = (rva - s.virtual_address) as usize;
        if delta >= s.raw_size as usize {
            return None; // lives in the zero-filled tail
        }
        let start = s.raw_address as usize + delta;
        let end = (start + len)
            .min(s.raw_address as usize + s.raw_size as usize)
            .min(self.data.len());
        self.data.get(start..end)
    }

    pub fn read_u8(&self, va: u64) -> Option<u8> {
        self.read_va(va, 1).and_then(|b| b.first().copied())
    }

    /// Read one byte from the loaded image, including zero-filled section tails.
    pub fn image_u8(&self, va: u64) -> Option<u8> {
        let rva = self.va_to_rva(va)?;
        let s = self.section_for_rva(rva)?;
        let delta = (rva - s.virtual_address) as usize;
        if delta >= s.raw_size as usize {
            return Some(0);
        }
        self.data.get(s.raw_address as usize + delta).copied()
    }

    pub fn read_u64(&self, va: u64) -> Option<u64> {
        let b = self.read_va(va, 8)?;
        (b.len() == 8).then(|| u64::from_le_bytes(b.try_into().unwrap()))
    }

    /// Map each import address table slot to `("dll", "symbol")`. Needed because the VM's guest calls to imported functions go through `call qword ptr [slot]`, and the qword in the file is an unbound hint the loader overwrites.
    pub fn imports(&self) -> Vec<(u64, String, String)> {
        let mut out = Vec::new();
        // The PE32+ import directory is data-directory entry 1.
        let opt =
            fmt::data_directory_offset(self.opt_header_offset, fmt::IMAGE_DIRECTORY_ENTRY_IMPORT);
        let Some(dir) = self.data.get(opt..opt + fmt::DATA_DIRECTORY_ENTRY_SIZE) else {
            return out;
        };
        let import_rva = u32::from_le_bytes([dir[0], dir[1], dir[2], dir[3]]);
        if import_rva == 0 {
            return out;
        }

        // Walk the import descriptors: 20 bytes each, terminated by an all-zero one.
        let mut desc = self.rva_to_va(import_rva);
        loop {
            let Some(b) = self.read_va(desc, 20) else {
                break;
            };
            let lookup_rva = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            let name_rva = u32::from_le_bytes([b[12], b[13], b[14], b[15]]);
            let iat_rva = u32::from_le_bytes([b[16], b[17], b[18], b[19]]);
            if lookup_rva == 0 && name_rva == 0 && iat_rva == 0 {
                break;
            }
            let dll = self.cstr_at(self.rva_to_va(name_rva)).unwrap_or_default();
            // The lookup table holds the names; the address table holds the slots
            // the code actually calls through. They run in parallel.
            let names = if lookup_rva != 0 { lookup_rva } else { iat_rva };
            let mut nva = self.rva_to_va(names);
            let mut sva = self.rva_to_va(if iat_rva != 0 { iat_rva } else { names });
            loop {
                let Some(entry) = self.read_u64(nva) else {
                    break;
                };
                if entry == 0 {
                    break;
                }
                let sym = if entry & (1 << 63) != 0 {
                    // Import by ordinal.
                    format!("#{}", entry & 0xffff)
                } else {
                    // Hint/name table entry: 2-byte hint then the name.
                    let hn = self.rva_to_va((entry & 0x7fff_ffff) as u32);
                    self.cstr_at(hn + 2).unwrap_or_default()
                };
                out.push((sva, dll.clone(), sym));
                nva += 8;
                sva += 8;
            }
            desc += 20;
        }
        out
    }

    pub fn function_table(&self) -> Vec<(u64, u64)> {
        let mut out: Vec<(u64, u64)> = Vec::new();

        let o = fmt::data_directory_offset(
            self.opt_header_offset,
            fmt::IMAGE_DIRECTORY_ENTRY_EXCEPTION,
        );
        if let Some(dir) = self.data.get(o..o + fmt::DATA_DIRECTORY_ENTRY_SIZE) {
            let rva = u32::from_le_bytes([dir[0], dir[1], dir[2], dir[3]]);
            let size = u32::from_le_bytes([dir[4], dir[5], dir[6], dir[7]]);
            if rva != 0 && size >= fmt::RUNTIME_FUNCTION_SIZE as u32 {
                for i in 0..size / fmt::RUNTIME_FUNCTION_SIZE as u32 {
                    let va = self.rva_to_va(rva) + i as u64 * fmt::RUNTIME_FUNCTION_SIZE as u64;
                    let Some(b) = self.read_va(va, fmt::RUNTIME_FUNCTION_SIZE) else {
                        break;
                    };
                    if b.len() < fmt::RUNTIME_FUNCTION_SIZE {
                        break;
                    }
                    let begin = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    let end = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                    if begin == 0 {
                        break;
                    }
                    out.push((self.rva_to_va(begin), self.rva_to_va(end)));
                }
            }
        }

        if let Some(sec) = self.section_by_name(".pdata") {
            let bytes = self.section_bytes(sec);
            for chunk in bytes.chunks_exact(fmt::RUNTIME_FUNCTION_SIZE) {
                let begin = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let end = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                if begin == 0 {
                    continue;
                }
                out.push((self.rva_to_va(begin), self.rva_to_va(end)));
            }
        }

        out.sort_unstable();
        out.dedup();
        out
    }

    fn cstr_at(&self, va: u64) -> Option<String> {
        let mut n = 0usize;
        while self.read_u8(va + n as u64).is_some_and(|b| b != 0) && n < 256 {
            n += 1;
        }
        let b = self.read_va(va, n)?;
        Some(String::from_utf8_lossy(b).into_owned())
    }

    pub fn is_executable(&self, va: u64) -> bool {
        self.section_for_va(va).is_some_and(Section::is_executable)
    }

    pub fn is_writable(&self, va: u64) -> bool {
        self.section_for_va(va).is_some_and(Section::is_writable)
    }

    /// Whether `va` lies in the BSS region.
    pub fn is_bss(&self, va: u64) -> bool {
        if let Some(section) = self.section_for_va(va) {
            let rva = match self.va_to_rva(va) {
                Some(r) => r,
                None => return false,
            };
            // BSS is the region between raw_size and virtual_size
            let section_offset = rva.saturating_sub(section.virtual_address);
            section_offset >= section.raw_size && section_offset < section.virtual_size
        } else {
            false
        }
    }
}

fn rd_u16(d: &[u8], o: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        d.get(o..o + 2).context("truncated u16")?.try_into()?,
    ))
}
fn rd_u32(d: &[u8], o: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        d.get(o..o + 4).context("truncated u32")?.try_into()?,
    ))
}
fn rd_u64(d: &[u8], o: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        d.get(o..o + 8).context("truncated u64")?.try_into()?,
    ))
}
