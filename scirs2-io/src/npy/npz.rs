//! NPZ (NumPy zip archive) format support.
//!
//! NPZ files are ZIP archives containing multiple .npy files.
//! We use a simple pure Rust implementation for reading/writing the
//! ZIP container format (local file headers + central directory).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::error::{IoError, Result};

use super::npy_reader::read_npy_from_reader;
use super::npy_writer::write_npy_to_writer;
use super::types::NpyArray;

/// ZIP local file header signature
const ZIP_LOCAL_HEADER_SIG: u32 = 0x04034b50;
/// ZIP central directory header signature
const ZIP_CENTRAL_DIR_SIG: u32 = 0x02014b50;
/// ZIP end of central directory signature
const ZIP_END_CENTRAL_SIG: u32 = 0x06054b50;

/// A collection of named arrays from an .npz file
#[derive(Debug, Clone)]
pub struct NpzArchive {
    /// Named arrays
    pub arrays: HashMap<String, NpyArray>,
}

impl NpzArchive {
    /// Create a new empty archive
    pub fn new() -> Self {
        NpzArchive {
            arrays: HashMap::new(),
        }
    }

    /// Add an array
    pub fn add(&mut self, name: impl Into<String>, array: NpyArray) {
        self.arrays.insert(name.into(), array);
    }

    /// Get an array by name
    pub fn get(&self, name: &str) -> Result<&NpyArray> {
        self.arrays
            .get(name)
            .ok_or_else(|| IoError::NotFound(format!("Array '{}' not found in npz", name)))
    }

    /// Get array names
    pub fn names(&self) -> Vec<&str> {
        self.arrays.keys().map(|s| s.as_str()).collect()
    }

    /// Number of arrays
    pub fn len(&self) -> usize {
        self.arrays.len()
    }

    /// Is the archive empty
    pub fn is_empty(&self) -> bool {
        self.arrays.is_empty()
    }
}

impl Default for NpzArchive {
    fn default() -> Self {
        Self::new()
    }
}

/// Read an .npz file
pub fn read_npz<P: AsRef<Path>>(path: P) -> Result<NpzArchive> {
    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut reader = BufReader::new(file);

    let mut archive = NpzArchive::new();

    // Read entries by scanning local file headers
    while let Ok(sig) = reader.read_u32::<LittleEndian>() {
        if sig == ZIP_CENTRAL_DIR_SIG || sig == ZIP_END_CENTRAL_SIG {
            break;
        }

        if sig != ZIP_LOCAL_HEADER_SIG {
            return Err(IoError::FormatError(format!(
                "Invalid ZIP local file header signature: 0x{:08x}",
                sig
            )));
        }

        // Read local file header fields
        let _version = reader
            .read_u16::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read version: {}", e)))?;
        let _flags = reader
            .read_u16::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read flags: {}", e)))?;
        let compression = reader
            .read_u16::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read compression: {}", e)))?;
        let _mod_time = reader
            .read_u16::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read mod time: {}", e)))?;
        let _mod_date = reader
            .read_u16::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read mod date: {}", e)))?;
        let _crc32 = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read CRC32: {}", e)))?;
        let compressed_size = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read compressed size: {}", e)))?
            as usize;
        let _uncompressed_size = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read uncompressed size: {}", e)))?
            as usize;
        let filename_len = reader
            .read_u16::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read filename length: {}", e)))?
            as usize;
        let extra_len = reader
            .read_u16::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read extra length: {}", e)))?
            as usize;

        // Read filename
        let mut filename_buf = vec![0u8; filename_len];
        reader
            .read_exact(&mut filename_buf)
            .map_err(|e| IoError::FormatError(format!("Failed to read filename: {}", e)))?;
        let filename = String::from_utf8(filename_buf)
            .map_err(|e| IoError::FormatError(format!("Invalid filename: {}", e)))?;

        // Skip extra field
        if extra_len > 0 {
            let mut extra = vec![0u8; extra_len];
            reader
                .read_exact(&mut extra)
                .map_err(|e| IoError::FormatError(format!("Failed to read extra: {}", e)))?;
        }

        // Only support stored (no compression) for NPZ
        if compression != 0 {
            // Skip this entry
            let mut skip_buf = vec![0u8; compressed_size];
            reader.read_exact(&mut skip_buf).map_err(|e| {
                IoError::FormatError(format!("Failed to skip compressed entry: {}", e))
            })?;
            continue;
        }

        // Read file data
        let mut data_buf = vec![0u8; compressed_size];
        reader
            .read_exact(&mut data_buf)
            .map_err(|e| IoError::FormatError(format!("Failed to read entry data: {}", e)))?;

        // Parse as npy if it has .npy extension
        if filename.ends_with(".npy") {
            let array_name = filename.trim_end_matches(".npy").to_string();
            let mut cursor = Cursor::new(data_buf);
            match read_npy_from_reader(&mut cursor) {
                Ok(array) => {
                    archive.add(array_name, array);
                }
                Err(e) => {
                    return Err(IoError::FormatError(format!(
                        "Failed to parse array '{}': {}",
                        filename, e
                    )));
                }
            }
        }
    }

    Ok(archive)
}

/// Write an .npz file (uncompressed)
pub fn write_npz<P: AsRef<Path>>(path: P, archive: &NpzArchive) -> Result<()> {
    let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut writer = BufWriter::new(file);

    // Collect sorted names for deterministic output
    let mut names: Vec<&String> = archive.arrays.keys().collect();
    names.sort();

    // Track each entry for central directory
    struct EntryInfo {
        filename: String,
        offset: u64,
        crc32: u32,
        size: u32,
    }
    let mut entries: Vec<EntryInfo> = Vec::new();
    let mut offset: u64 = 0;

    for name in &names {
        let array = &archive.arrays[*name];
        let filename = format!("{}.npy", name);

        // Serialize npy to buffer
        let mut npy_buf = Vec::new();
        write_npy_to_writer(&mut npy_buf, array)?;

        let crc = crc32fast::hash(&npy_buf);
        let size = npy_buf.len() as u32;

        // Write local file header
        writer
            .write_u32::<LittleEndian>(ZIP_LOCAL_HEADER_SIG)
            .map_err(|e| IoError::FileError(format!("Failed to write local header sig: {}", e)))?;
        writer
            .write_u16::<LittleEndian>(20)
            .map_err(|e| IoError::FileError(format!("Failed to write version: {}", e)))?; // version needed
        writer
            .write_u16::<LittleEndian>(0)
            .map_err(|e| IoError::FileError(format!("Failed to write flags: {}", e)))?; // flags
        writer
            .write_u16::<LittleEndian>(0)
            .map_err(|e| IoError::FileError(format!("Failed to write compression: {}", e)))?; // stored
        writer
            .write_u16::<LittleEndian>(0)
            .map_err(|e| IoError::FileError(format!("Failed to write mod time: {}", e)))?; // mod time
        writer
            .write_u16::<LittleEndian>(0)
            .map_err(|e| IoError::FileError(format!("Failed to write mod date: {}", e)))?; // mod date
        writer
            .write_u32::<LittleEndian>(crc)
            .map_err(|e| IoError::FileError(format!("Failed to write crc: {}", e)))?;
        writer
            .write_u32::<LittleEndian>(size)
            .map_err(|e| IoError::FileError(format!("Failed to write compressed size: {}", e)))?;
        writer
            .write_u32::<LittleEndian>(size)
            .map_err(|e| IoError::FileError(format!("Failed to write uncompressed size: {}", e)))?;
        writer
            .write_u16::<LittleEndian>(filename.len() as u16)
            .map_err(|e| IoError::FileError(format!("Failed to write filename len: {}", e)))?;
        writer
            .write_u16::<LittleEndian>(0)
            .map_err(|e| IoError::FileError(format!("Failed to write extra len: {}", e)))?; // extra length

        writer
            .write_all(filename.as_bytes())
            .map_err(|e| IoError::FileError(format!("Failed to write filename: {}", e)))?;

        // Write data
        writer
            .write_all(&npy_buf)
            .map_err(|e| IoError::FileError(format!("Failed to write npy data: {}", e)))?;

        entries.push(EntryInfo {
            filename,
            offset,
            crc32: crc,
            size,
        });

        // 30 = fixed local header size
        offset += 30 + entries.last().map(|e| e.filename.len() as u64).unwrap_or(0) + size as u64;
    }

    // Write central directory
    let central_dir_offset = offset;
    let mut central_dir_size: u64 = 0;

    for entry in &entries {
        writer
            .write_u32::<LittleEndian>(ZIP_CENTRAL_DIR_SIG)
            .map_err(|e| IoError::FileError(format!("Failed to write central dir sig: {}", e)))?;
        writer
            .write_u16::<LittleEndian>(20)
            .map_err(|e| IoError::FileError(format!("Failed to write version made by: {}", e)))?;
        writer
            .write_u16::<LittleEndian>(20)
            .map_err(|e| IoError::FileError(format!("Failed to write version needed: {}", e)))?;
        writer
            .write_u16::<LittleEndian>(0)
            .map_err(|e| IoError::FileError(format!("Failed to write flags: {}", e)))?;
        writer
            .write_u16::<LittleEndian>(0)
            .map_err(|e| IoError::FileError(format!("Failed to write compression: {}", e)))?;
        writer
            .write_u16::<LittleEndian>(0)
            .map_err(|e| IoError::FileError(format!("Failed to write time: {}", e)))?;
        writer
            .write_u16::<LittleEndian>(0)
            .map_err(|e| IoError::FileError(format!("Failed to write date: {}", e)))?;
        writer
            .write_u32::<LittleEndian>(entry.crc32)
            .map_err(|e| IoError::FileError(format!("Failed to write crc: {}", e)))?;
        writer
            .write_u32::<LittleEndian>(entry.size)
            .map_err(|e| IoError::FileError(format!("Failed to write size: {}", e)))?;
        writer
            .write_u32::<LittleEndian>(entry.size)
            .map_err(|e| IoError::FileError(format!("Failed to write uncompressed: {}", e)))?;
        writer
            .write_u16::<LittleEndian>(entry.filename.len() as u16)
            .map_err(|e| IoError::FileError(format!("Failed to write filename len: {}", e)))?;
        writer
            .write_u16::<LittleEndian>(0)
            .map_err(|e| IoError::FileError(format!("Failed to write extra len: {}", e)))?;
        writer
            .write_u16::<LittleEndian>(0)
            .map_err(|e| IoError::FileError(format!("Failed to write comment len: {}", e)))?;
        writer
            .write_u16::<LittleEndian>(0)
            .map_err(|e| IoError::FileError(format!("Failed to write disk number: {}", e)))?;
        writer
            .write_u16::<LittleEndian>(0)
            .map_err(|e| IoError::FileError(format!("Failed to write internal attrs: {}", e)))?;
        writer
            .write_u32::<LittleEndian>(0)
            .map_err(|e| IoError::FileError(format!("Failed to write external attrs: {}", e)))?;
        writer
            .write_u32::<LittleEndian>(entry.offset as u32)
            .map_err(|e| IoError::FileError(format!("Failed to write offset: {}", e)))?;
        writer
            .write_all(entry.filename.as_bytes())
            .map_err(|e| IoError::FileError(format!("Failed to write filename: {}", e)))?;

        central_dir_size += 46 + entry.filename.len() as u64;
    }

    // Write end of central directory
    writer
        .write_u32::<LittleEndian>(ZIP_END_CENTRAL_SIG)
        .map_err(|e| IoError::FileError(format!("Failed to write EOCD sig: {}", e)))?;
    writer
        .write_u16::<LittleEndian>(0)
        .map_err(|e| IoError::FileError(format!("Failed to write disk: {}", e)))?; // disk number
    writer
        .write_u16::<LittleEndian>(0)
        .map_err(|e| IoError::FileError(format!("Failed to write start disk: {}", e)))?;
    writer
        .write_u16::<LittleEndian>(entries.len() as u16)
        .map_err(|e| IoError::FileError(format!("Failed to write entries on disk: {}", e)))?;
    writer
        .write_u16::<LittleEndian>(entries.len() as u16)
        .map_err(|e| IoError::FileError(format!("Failed to write total entries: {}", e)))?;
    writer
        .write_u32::<LittleEndian>(central_dir_size as u32)
        .map_err(|e| IoError::FileError(format!("Failed to write central dir size: {}", e)))?;
    writer
        .write_u32::<LittleEndian>(central_dir_offset as u32)
        .map_err(|e| IoError::FileError(format!("Failed to write central dir offset: {}", e)))?;
    writer
        .write_u16::<LittleEndian>(0)
        .map_err(|e| IoError::FileError(format!("Failed to write comment len: {}", e)))?; // comment length

    writer
        .flush()
        .map_err(|e| IoError::FileError(format!("Failed to flush: {}", e)))?;

    Ok(())
}
