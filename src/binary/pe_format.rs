//! PE/COFF constants shared by the reader, discovery, and writer.

pub const IMAGE_FILE_MACHINE_AMD64: u16 = 0x8664;
pub const PE32_PLUS_MAGIC: u16 = 0x20b;

pub const SECTION_HEADER_SIZE: usize = 40;
pub const SECTION_CHARACTERISTICS_OFFSET: usize = 36;

pub const PE32_PLUS_DATA_DIRECTORY_OFFSET: usize = 0x70;
pub const DATA_DIRECTORY_ENTRY_SIZE: usize = 8;
pub const IMAGE_DIRECTORY_ENTRY_IMPORT: usize = 1;
pub const IMAGE_DIRECTORY_ENTRY_EXCEPTION: usize = 3;
pub const IMAGE_DIRECTORY_ENTRY_SECURITY: usize = 4;
pub const IMAGE_DIRECTORY_ENTRY_IAT: usize = 12;

pub const OPTIONAL_HEADER_CHECKSUM_OFFSET: usize = 64;
pub const OPTIONAL_HEADER_SIZE_OF_IMAGE_OFFSET: usize = 56;

pub const RUNTIME_FUNCTION_SIZE: usize = 12;

pub const IMAGE_SCN_CNT_CODE: u32 = 0x0000_0020;
pub const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
pub const IMAGE_SCN_MEM_READ: u32 = 0x4000_0000;
pub const IMAGE_SCN_MEM_WRITE: u32 = 0x8000_0000;

pub const fn data_directory_offset(optional_header_offset: usize, entry: usize) -> usize {
    optional_header_offset + PE32_PLUS_DATA_DIRECTORY_OFFSET + entry * DATA_DIRECTORY_ENTRY_SIZE
}
