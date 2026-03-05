//! Reader for the columnar binary format.

use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::error::{IoError, Result};

use super::encoding::decode_column;
use super::types::{
    Column, ColumnTypeTag, ColumnarTable, EncodingType, COLUMNAR_MAGIC, FORMAT_VERSION,
};

/// Read a columnar table from a file
pub fn read_columnar<P: AsRef<Path>>(path: P) -> Result<ColumnarTable> {
    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut reader = BufReader::new(file);

    // Read and verify magic bytes
    let mut magic = [0u8; 8];
    reader
        .read_exact(&mut magic)
        .map_err(|e| IoError::FormatError(format!("Failed to read magic bytes: {}", e)))?;
    if &magic != COLUMNAR_MAGIC {
        return Err(IoError::FormatError(
            "Invalid columnar file: bad magic bytes".to_string(),
        ));
    }

    // Read version
    let version = reader
        .read_u32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read version: {}", e)))?;
    if version > FORMAT_VERSION {
        return Err(IoError::FormatError(format!(
            "Unsupported columnar format version: {} (max supported: {})",
            version, FORMAT_VERSION
        )));
    }

    // Read column count
    let col_count = reader
        .read_u32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read column count: {}", e)))?
        as usize;

    // Read row count
    let row_count = reader
        .read_u64::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read row count: {}", e)))?
        as usize;

    // Read each column
    let mut columns = Vec::with_capacity(col_count);
    for col_idx in 0..col_count {
        // Read column name
        let name_len = reader.read_u32::<LittleEndian>().map_err(|e| {
            IoError::FormatError(format!(
                "Failed to read column {} name length: {}",
                col_idx, e
            ))
        })? as usize;
        let mut name_buf = vec![0u8; name_len];
        reader.read_exact(&mut name_buf).map_err(|e| {
            IoError::FormatError(format!("Failed to read column {} name: {}", col_idx, e))
        })?;
        let name = String::from_utf8(name_buf).map_err(|e| {
            IoError::FormatError(format!("Invalid column {} name UTF-8: {}", col_idx, e))
        })?;

        // Read type tag
        let type_tag_byte = reader.read_u8().map_err(|e| {
            IoError::FormatError(format!("Failed to read column {} type tag: {}", col_idx, e))
        })?;
        let type_tag = ColumnTypeTag::try_from(type_tag_byte)?;

        // Read encoding type
        let encoding_byte = reader.read_u8().map_err(|e| {
            IoError::FormatError(format!(
                "Failed to read column {} encoding type: {}",
                col_idx, e
            ))
        })?;
        let encoding = EncodingType::try_from(encoding_byte)?;

        // Read data size
        let data_size = reader.read_u64::<LittleEndian>().map_err(|e| {
            IoError::FormatError(format!(
                "Failed to read column {} data size: {}",
                col_idx, e
            ))
        })? as usize;

        // Read data bytes
        let mut data_buf = vec![0u8; data_size];
        reader.read_exact(&mut data_buf).map_err(|e| {
            IoError::FormatError(format!("Failed to read column {} data: {}", col_idx, e))
        })?;

        // Decode data
        let mut cursor = std::io::Cursor::new(data_buf);
        let data = decode_column(&mut cursor, type_tag, encoding, row_count)?;

        columns.push(Column { name, data });
    }

    ColumnarTable::from_columns(columns)
}
