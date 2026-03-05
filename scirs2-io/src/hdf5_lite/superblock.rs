//! HDF5 superblock parsing
//!
//! The superblock is the first structure in an HDF5 file. It contains metadata about
//! the file including version information and the address of the root group.

use crate::error::{IoError, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{Read, Seek, SeekFrom};

/// HDF5 file signature: 8 bytes at the start of every valid HDF5 file
pub const HDF5_SIGNATURE: [u8; 8] = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];

/// Parsed superblock info
#[derive(Debug, Clone)]
pub struct SuperblockInfo {
    /// Superblock version (0, 1, 2, or 3)
    pub version: u8,
    /// Size of offsets in bytes (typically 8)
    pub offset_size: u8,
    /// Size of lengths in bytes (typically 8)
    pub length_size: u8,
    /// Address of the root group symbol table entry (v0/v1) or root group object header (v2/v3)
    pub root_group_address: u64,
    /// Base address of the file (usually 0)
    pub base_address: u64,
    /// End of file address
    pub eof_address: u64,
}

/// Parse the HDF5 superblock from the beginning of the file.
pub fn parse_superblock<R: Read + Seek>(reader: &mut R) -> Result<SuperblockInfo> {
    reader
        .seek(SeekFrom::Start(0))
        .map_err(|e| IoError::FormatError(format!("Failed to seek to start of HDF5 file: {e}")))?;

    // Read and verify signature
    let mut sig = [0u8; 8];
    reader
        .read_exact(&mut sig)
        .map_err(|e| IoError::FormatError(format!("Failed to read HDF5 signature: {e}")))?;
    if sig != HDF5_SIGNATURE {
        return Err(IoError::FormatError(
            "Not a valid HDF5 file: signature mismatch".to_string(),
        ));
    }

    // Read superblock version
    let sb_version = reader
        .read_u8()
        .map_err(|e| IoError::FormatError(format!("Failed to read superblock version: {e}")))?;

    match sb_version {
        0 | 1 => parse_superblock_v0v1(reader, sb_version),
        2 | 3 => parse_superblock_v2v3(reader, sb_version),
        _ => Err(IoError::UnsupportedFormat(format!(
            "HDF5 superblock version {sb_version} is not supported"
        ))),
    }
}

/// Parse version 0 or 1 superblock
fn parse_superblock_v0v1<R: Read + Seek>(reader: &mut R, version: u8) -> Result<SuperblockInfo> {
    // Free-space storage version (skip)
    let _free_space_version = reader
        .read_u8()
        .map_err(|e| IoError::FormatError(format!("Failed to read free-space version: {e}")))?;

    // Root group symbol table entry version (skip)
    let _root_group_version = reader
        .read_u8()
        .map_err(|e| IoError::FormatError(format!("Failed to read root group version: {e}")))?;

    // Reserved byte
    let _reserved = reader
        .read_u8()
        .map_err(|e| IoError::FormatError(format!("Failed to read reserved byte: {e}")))?;

    // Shared header message format version (skip)
    let _shared_header_version = reader
        .read_u8()
        .map_err(|e| IoError::FormatError(format!("Failed to read shared header version: {e}")))?;

    // Size of offsets
    let offset_size = reader
        .read_u8()
        .map_err(|e| IoError::FormatError(format!("Failed to read offset size: {e}")))?;

    // Size of lengths
    let length_size = reader
        .read_u8()
        .map_err(|e| IoError::FormatError(format!("Failed to read length size: {e}")))?;

    // Reserved byte
    let _reserved2 = reader
        .read_u8()
        .map_err(|e| IoError::FormatError(format!("Failed to read reserved byte: {e}")))?;

    // Group leaf node K
    let _group_leaf_k = reader
        .read_u16::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read group leaf K: {e}")))?;

    // Group internal node K
    let _group_internal_k = reader
        .read_u16::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read group internal K: {e}")))?;

    // File consistency flags (4 bytes)
    let _consistency_flags = reader
        .read_u32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read consistency flags: {e}")))?;

    // Version 1 has additional indexed storage internal node K
    if version == 1 {
        let _indexed_k = reader
            .read_u16::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read indexed K: {e}")))?;
        // Reserved (2 bytes)
        let _reserved3 = reader
            .read_u16::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read reserved: {e}")))?;
    }

    // Base address
    let base_address = read_offset(reader, offset_size)?;

    // Address of file free-space info (skip)
    let _free_space_address = read_offset(reader, offset_size)?;

    // End of file address
    let eof_address = read_offset(reader, offset_size)?;

    // Driver information block address (skip)
    let _driver_info_address = read_offset(reader, offset_size)?;

    // Root group symbol table entry
    // The entry contains: link name offset (offset_size), object header address (offset_size),
    // cache type (4 bytes), reserved (4 bytes), scratch-pad (16 bytes)
    let _link_name_offset = read_offset(reader, offset_size)?;
    let root_group_address = read_offset(reader, offset_size)?;

    Ok(SuperblockInfo {
        version,
        offset_size,
        length_size,
        root_group_address,
        base_address,
        eof_address,
    })
}

/// Parse version 2 or 3 superblock
fn parse_superblock_v2v3<R: Read + Seek>(reader: &mut R, version: u8) -> Result<SuperblockInfo> {
    // Size of offsets
    let offset_size = reader
        .read_u8()
        .map_err(|e| IoError::FormatError(format!("Failed to read offset size: {e}")))?;

    // Size of lengths
    let length_size = reader
        .read_u8()
        .map_err(|e| IoError::FormatError(format!("Failed to read length size: {e}")))?;

    // File consistency flags (1 byte for v2/v3)
    let _consistency_flags = reader
        .read_u8()
        .map_err(|e| IoError::FormatError(format!("Failed to read consistency flags: {e}")))?;

    // Base address
    let base_address = read_offset(reader, offset_size)?;

    // Superblock extension address (skip)
    let _sb_ext_address = read_offset(reader, offset_size)?;

    // End of file address
    let eof_address = read_offset(reader, offset_size)?;

    // Root group object header address
    let root_group_address = read_offset(reader, offset_size)?;

    // Superblock checksum (4 bytes, skip)
    let _checksum = reader
        .read_u32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read superblock checksum: {e}")))?;

    Ok(SuperblockInfo {
        version,
        offset_size,
        length_size,
        root_group_address,
        base_address,
        eof_address,
    })
}

/// Read an offset value of the specified size from the reader.
pub fn read_offset<R: Read>(reader: &mut R, size: u8) -> Result<u64> {
    match size {
        2 => reader
            .read_u16::<LittleEndian>()
            .map(u64::from)
            .map_err(|e| IoError::FormatError(format!("Failed to read 2-byte offset: {e}"))),
        4 => reader
            .read_u32::<LittleEndian>()
            .map(u64::from)
            .map_err(|e| IoError::FormatError(format!("Failed to read 4-byte offset: {e}"))),
        8 => reader
            .read_u64::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read 8-byte offset: {e}"))),
        _ => Err(IoError::FormatError(format!(
            "Unsupported offset size: {size}"
        ))),
    }
}

/// Read a length value of the specified size from the reader.
pub fn read_length<R: Read>(reader: &mut R, size: u8) -> Result<u64> {
    read_offset(reader, size)
}

/// Undefined address constant (all bits set)
pub fn undefined_address(offset_size: u8) -> u64 {
    match offset_size {
        2 => u16::MAX as u64,
        4 => u32::MAX as u64,
        8 => u64::MAX,
        _ => u64::MAX,
    }
}
