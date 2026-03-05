//! Feather v1 file format — simplified columnar storage.
//!
//! Feather is a fast, lightweight, language-agnostic columnar file format
//! designed for storing data frames.  This module implements a self-contained
//! Feather v1 writer/reader in pure Rust without any Arrow/IPC dependency.
//!
//! ## Wire layout
//!
//! ```text
//! [MAGIC "FEA1"] [columns...] [FOOTER] [footer_len: u32 LE] [MAGIC "FEA1"]
//! ```
//!
//! Each column is laid out as:
//! ```text
//! [type: u8] [n_rows: u64 LE]
//! [data_len: u64 LE] [data bytes]          // raw little-endian values
//! [null_mask_present: u8]
//! (if present) [mask_len: u64 LE] [mask bytes]  // 1 bit per row, LSB first
//! ```
//!
//! The footer is:
//! ```text
//! [n_cols: u32 LE]
//! for each column:
//!   [name_len: u16 LE] [name bytes (UTF-8)]
//!   [col_offset: u64 LE]    // absolute byte offset of the column block
//! ```

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::error::{IoError, Result};

// ─────────────────────────────── constants ───────────────────────────────────

const MAGIC: &[u8; 4] = b"FEA1";

// Column type tags
const TYPE_INT8: u8 = 0;
const TYPE_INT16: u8 = 1;
const TYPE_INT32: u8 = 2;
const TYPE_INT64: u8 = 3;
const TYPE_UINT8: u8 = 4;
const TYPE_UINT16: u8 = 5;
const TYPE_UINT32: u8 = 6;
const TYPE_UINT64: u8 = 7;
const TYPE_FLOAT32: u8 = 8;
const TYPE_FLOAT64: u8 = 9;
const TYPE_UTF8: u8 = 10;
const TYPE_BOOL: u8 = 11;

// ─────────────────────────────── FeatherData ─────────────────────────────────

/// Column data variants.
#[derive(Debug, Clone, PartialEq)]
pub enum FeatherData {
    Int8(Vec<i8>),
    Int16(Vec<i16>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    UInt8(Vec<u8>),
    UInt16(Vec<u16>),
    UInt32(Vec<u32>),
    UInt64(Vec<u64>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
    /// Variable-length UTF-8 strings: stored as 4-byte LE length + bytes.
    Utf8(Vec<String>),
    Bool(Vec<bool>),
}

impl FeatherData {
    fn type_tag(&self) -> u8 {
        match self {
            FeatherData::Int8(_) => TYPE_INT8,
            FeatherData::Int16(_) => TYPE_INT16,
            FeatherData::Int32(_) => TYPE_INT32,
            FeatherData::Int64(_) => TYPE_INT64,
            FeatherData::UInt8(_) => TYPE_UINT8,
            FeatherData::UInt16(_) => TYPE_UINT16,
            FeatherData::UInt32(_) => TYPE_UINT32,
            FeatherData::UInt64(_) => TYPE_UINT64,
            FeatherData::Float32(_) => TYPE_FLOAT32,
            FeatherData::Float64(_) => TYPE_FLOAT64,
            FeatherData::Utf8(_) => TYPE_UTF8,
            FeatherData::Bool(_) => TYPE_BOOL,
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            FeatherData::Int8(v) => v.len(),
            FeatherData::Int16(v) => v.len(),
            FeatherData::Int32(v) => v.len(),
            FeatherData::Int64(v) => v.len(),
            FeatherData::UInt8(v) => v.len(),
            FeatherData::UInt16(v) => v.len(),
            FeatherData::UInt32(v) => v.len(),
            FeatherData::UInt64(v) => v.len(),
            FeatherData::Float32(v) => v.len(),
            FeatherData::Float64(v) => v.len(),
            FeatherData::Utf8(v) => v.len(),
            FeatherData::Bool(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Serialize the data payload to bytes (little-endian for numerics).
    fn to_bytes(&self) -> Vec<u8> {
        match self {
            FeatherData::Int8(v) => v.iter().map(|&x| x as u8).collect(),
            FeatherData::Int16(v) => v.iter().flat_map(|&x| x.to_le_bytes()).collect(),
            FeatherData::Int32(v) => v.iter().flat_map(|&x| x.to_le_bytes()).collect(),
            FeatherData::Int64(v) => v.iter().flat_map(|&x| x.to_le_bytes()).collect(),
            FeatherData::UInt8(v) => v.clone(),
            FeatherData::UInt16(v) => v.iter().flat_map(|&x| x.to_le_bytes()).collect(),
            FeatherData::UInt32(v) => v.iter().flat_map(|&x| x.to_le_bytes()).collect(),
            FeatherData::UInt64(v) => v.iter().flat_map(|&x| x.to_le_bytes()).collect(),
            FeatherData::Float32(v) => v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect(),
            FeatherData::Float64(v) => v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect(),
            FeatherData::Utf8(v) => {
                let mut buf = Vec::new();
                for s in v {
                    let b = s.as_bytes();
                    buf.extend_from_slice(&(b.len() as u32).to_le_bytes());
                    buf.extend_from_slice(b);
                }
                buf
            }
            FeatherData::Bool(v) => {
                let n_bytes = (v.len() + 7) / 8;
                let mut buf = vec![0u8; n_bytes];
                for (i, &bit) in v.iter().enumerate() {
                    if bit {
                        buf[i / 8] |= 1 << (i % 8);
                    }
                }
                buf
            }
        }
    }
}

// ─────────────────────────────── FeatherColumn ───────────────────────────────

/// A single column in a Feather file.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatherColumn {
    /// Column name.
    pub name: String,
    /// Column data.
    pub data: FeatherData,
    /// Optional validity bitmap (bit=1 → valid, bit=0 → null).  One bit per row, LSB-first.
    pub null_mask: Option<Vec<u8>>,
}

impl FeatherColumn {
    /// Create a non-nullable column.
    pub fn new(name: impl Into<String>, data: FeatherData) -> Self {
        Self {
            name: name.into(),
            data,
            null_mask: None,
        }
    }

    /// Create a nullable column with an explicit validity bitmap.
    pub fn with_nulls(name: impl Into<String>, data: FeatherData, mask: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            data,
            null_mask: Some(mask),
        }
    }
}

// ─────────────────────────────── FeatherFile ─────────────────────────────────

/// An in-memory representation of a Feather file.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatherFile {
    /// Columns — all must have `data.len() == n_rows`.
    pub columns: Vec<FeatherColumn>,
    /// Number of rows.
    pub n_rows: usize,
}

impl FeatherFile {
    /// Create an empty file with the given row count.
    pub fn new(n_rows: usize) -> Self {
        Self {
            columns: Vec::new(),
            n_rows,
        }
    }

    /// Add a column.  Returns `Err` if the length does not match `n_rows`.
    pub fn add_column(&mut self, col: FeatherColumn) -> Result<()> {
        if col.data.len() != self.n_rows {
            return Err(IoError::FormatError(format!(
                "Feather: column '{}' has {} rows but file has {}",
                col.name,
                col.data.len(),
                self.n_rows
            )));
        }
        self.columns.push(col);
        Ok(())
    }

    // ──────────────────────────────── Writer ─────────────────────────────────

    /// Write the Feather file to `path`.
    pub fn write(path: &Path, file: &FeatherFile) -> Result<()> {
        let f = File::create(path).map_err(IoError::Io)?;
        let mut w = BufWriter::new(f);

        // Leading magic
        w.write_all(MAGIC).map_err(IoError::Io)?;

        // Track byte offsets for the footer
        let mut offsets: Vec<u64> = Vec::with_capacity(file.columns.len());

        let mut cursor: u64 = 4; // after leading MAGIC

        for col in &file.columns {
            offsets.push(cursor);
            let data_bytes = col.data.to_bytes();
            let n_rows = col.data.len() as u64;

            // type tag
            w.write_all(&[col.data.type_tag()]).map_err(IoError::Io)?;
            cursor += 1;

            // n_rows
            w.write_all(&n_rows.to_le_bytes()).map_err(IoError::Io)?;
            cursor += 8;

            // data
            let data_len = data_bytes.len() as u64;
            w.write_all(&data_len.to_le_bytes()).map_err(IoError::Io)?;
            w.write_all(&data_bytes).map_err(IoError::Io)?;
            cursor += 8 + data_len;

            // null mask
            match &col.null_mask {
                None => {
                    w.write_all(&[0u8]).map_err(IoError::Io)?;
                    cursor += 1;
                }
                Some(mask) => {
                    w.write_all(&[1u8]).map_err(IoError::Io)?;
                    let mask_len = mask.len() as u64;
                    w.write_all(&mask_len.to_le_bytes()).map_err(IoError::Io)?;
                    w.write_all(mask).map_err(IoError::Io)?;
                    cursor += 1 + 8 + mask_len;
                }
            }
        }

        // Footer
        let _footer_start = cursor;
        let n_cols = file.columns.len() as u32;
        w.write_all(&n_cols.to_le_bytes()).map_err(IoError::Io)?;
        let mut footer_len: u64 = 4; // n_cols field

        for (col, &offset) in file.columns.iter().zip(offsets.iter()) {
            let name_bytes = col.name.as_bytes();
            let name_len = name_bytes.len() as u16;
            w.write_all(&name_len.to_le_bytes()).map_err(IoError::Io)?;
            w.write_all(name_bytes).map_err(IoError::Io)?;
            w.write_all(&offset.to_le_bytes()).map_err(IoError::Io)?;
            footer_len += 2 + name_bytes.len() as u64 + 8;
        }

        // footer_len as u32 LE then trailing magic
        w.write_all(&(footer_len as u32).to_le_bytes())
            .map_err(IoError::Io)?;
        w.write_all(MAGIC).map_err(IoError::Io)?;

        w.flush().map_err(IoError::Io)?;
        Ok(())
    }

    // ──────────────────────────────── Reader ─────────────────────────────────

    /// Read a Feather file from `path`.
    pub fn read(path: &Path) -> Result<FeatherFile> {
        let f = File::open(path).map_err(IoError::Io)?;
        let mut r = BufReader::new(f);

        // Verify leading magic
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic).map_err(IoError::Io)?;
        if &magic != MAGIC {
            return Err(IoError::FormatError(
                "Feather: bad magic bytes at start".into(),
            ));
        }

        // Seek to trailing footer_len (last 8 bytes: 4 for len + 4 for MAGIC)
        r.seek(SeekFrom::End(-8)).map_err(IoError::Io)?;
        let mut len_buf = [0u8; 4];
        r.read_exact(&mut len_buf).map_err(IoError::Io)?;
        let footer_len = u32::from_le_bytes(len_buf) as u64;

        let mut trail_magic = [0u8; 4];
        r.read_exact(&mut trail_magic).map_err(IoError::Io)?;
        if &trail_magic != MAGIC {
            return Err(IoError::FormatError(
                "Feather: bad magic bytes at end".into(),
            ));
        }

        // Seek to footer start
        r.seek(SeekFrom::End(-(8 + footer_len as i64)))
            .map_err(IoError::Io)?;

        let mut n_cols_buf = [0u8; 4];
        r.read_exact(&mut n_cols_buf).map_err(IoError::Io)?;
        let n_cols = u32::from_le_bytes(n_cols_buf) as usize;

        let mut col_meta: Vec<(String, u64)> = Vec::with_capacity(n_cols);
        for _ in 0..n_cols {
            let mut nl_buf = [0u8; 2];
            r.read_exact(&mut nl_buf).map_err(IoError::Io)?;
            let name_len = u16::from_le_bytes(nl_buf) as usize;

            let mut name_bytes = vec![0u8; name_len];
            r.read_exact(&mut name_bytes).map_err(IoError::Io)?;
            let name = String::from_utf8(name_bytes).map_err(|e| {
                IoError::FormatError(format!("Feather: invalid column name UTF-8: {e}"))
            })?;

            let mut off_buf = [0u8; 8];
            r.read_exact(&mut off_buf).map_err(IoError::Io)?;
            let offset = u64::from_le_bytes(off_buf);
            col_meta.push((name, offset));
        }

        // Read each column
        let mut columns: Vec<FeatherColumn> = Vec::with_capacity(n_cols);
        let mut n_rows_global: Option<usize> = None;

        for (col_name, offset) in col_meta {
            r.seek(SeekFrom::Start(offset)).map_err(IoError::Io)?;

            let mut type_buf = [0u8; 1];
            r.read_exact(&mut type_buf).map_err(IoError::Io)?;
            let type_tag = type_buf[0];

            let mut nr_buf = [0u8; 8];
            r.read_exact(&mut nr_buf).map_err(IoError::Io)?;
            let n_rows = u64::from_le_bytes(nr_buf) as usize;

            n_rows_global = Some(match n_rows_global {
                None => n_rows,
                Some(prev) if prev == n_rows => n_rows,
                Some(prev) => {
                    return Err(IoError::FormatError(format!(
                        "Feather: column '{col_name}' has {n_rows} rows but previous columns had {prev}"
                    )))
                }
            });

            let mut dl_buf = [0u8; 8];
            r.read_exact(&mut dl_buf).map_err(IoError::Io)?;
            let data_len = u64::from_le_bytes(dl_buf) as usize;

            let mut data_bytes = vec![0u8; data_len];
            r.read_exact(&mut data_bytes).map_err(IoError::Io)?;

            let data = decode_column_data(type_tag, n_rows, &data_bytes)?;

            let mut mask_flag = [0u8; 1];
            r.read_exact(&mut mask_flag).map_err(IoError::Io)?;
            let null_mask = if mask_flag[0] != 0 {
                let mut ml_buf = [0u8; 8];
                r.read_exact(&mut ml_buf).map_err(IoError::Io)?;
                let mask_len = u64::from_le_bytes(ml_buf) as usize;
                let mut mask = vec![0u8; mask_len];
                r.read_exact(&mut mask).map_err(IoError::Io)?;
                Some(mask)
            } else {
                None
            };

            columns.push(FeatherColumn {
                name: col_name,
                data,
                null_mask,
            });
        }

        Ok(FeatherFile {
            n_rows: n_rows_global.unwrap_or(0),
            columns,
        })
    }
}

// ─────────────────────────────── Decode helpers ──────────────────────────────

fn decode_column_data(type_tag: u8, n_rows: usize, bytes: &[u8]) -> Result<FeatherData> {
    match type_tag {
        TYPE_INT8 => {
            check_len(bytes, n_rows, 1, "i8")?;
            Ok(FeatherData::Int8(bytes.iter().map(|&b| b as i8).collect()))
        }
        TYPE_INT16 => {
            check_len(bytes, n_rows, 2, "i16")?;
            Ok(FeatherData::Int16(
                bytes
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]))
                    .collect(),
            ))
        }
        TYPE_INT32 => {
            check_len(bytes, n_rows, 4, "i32")?;
            Ok(FeatherData::Int32(
                bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            ))
        }
        TYPE_INT64 => {
            check_len(bytes, n_rows, 8, "i64")?;
            Ok(FeatherData::Int64(
                bytes
                    .chunks_exact(8)
                    .map(|c| {
                        Ok(i64::from_le_bytes(c.try_into().map_err(|_| {
                            IoError::FormatError("Feather: bad i64 slice".into())
                        })?))
                    })
                    .collect::<Result<Vec<_>>>()?,
            ))
        }
        TYPE_UINT8 => Ok(FeatherData::UInt8(bytes.to_vec())),
        TYPE_UINT16 => {
            check_len(bytes, n_rows, 2, "u16")?;
            Ok(FeatherData::UInt16(
                bytes
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect(),
            ))
        }
        TYPE_UINT32 => {
            check_len(bytes, n_rows, 4, "u32")?;
            Ok(FeatherData::UInt32(
                bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            ))
        }
        TYPE_UINT64 => {
            check_len(bytes, n_rows, 8, "u64")?;
            Ok(FeatherData::UInt64(
                bytes
                    .chunks_exact(8)
                    .map(|c| {
                        Ok(u64::from_le_bytes(c.try_into().map_err(|_| {
                            IoError::FormatError("Feather: bad u64 slice".into())
                        })?))
                    })
                    .collect::<Result<Vec<_>>>()?,
            ))
        }
        TYPE_FLOAT32 => {
            check_len(bytes, n_rows, 4, "f32")?;
            Ok(FeatherData::Float32(
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                    .collect(),
            ))
        }
        TYPE_FLOAT64 => {
            check_len(bytes, n_rows, 8, "f64")?;
            Ok(FeatherData::Float64(
                bytes
                    .chunks_exact(8)
                    .map(|c| {
                        Ok(f64::from_bits(u64::from_le_bytes(c.try_into().map_err(
                            |_| IoError::FormatError("Feather: bad f64 slice".into()),
                        )?)))
                    })
                    .collect::<Result<Vec<_>>>()?,
            ))
        }
        TYPE_UTF8 => {
            let mut strs = Vec::with_capacity(n_rows);
            let mut pos = 0usize;
            for _ in 0..n_rows {
                if pos + 4 > bytes.len() {
                    return Err(IoError::FormatError(
                        "Feather: truncated UTF-8 column".into(),
                    ));
                }
                let slen = u32::from_le_bytes([
                    bytes[pos],
                    bytes[pos + 1],
                    bytes[pos + 2],
                    bytes[pos + 3],
                ]) as usize;
                pos += 4;
                if pos + slen > bytes.len() {
                    return Err(IoError::FormatError(
                        "Feather: truncated string data".into(),
                    ));
                }
                let s = std::str::from_utf8(&bytes[pos..pos + slen])
                    .map_err(|e| IoError::FormatError(format!("Feather: UTF-8 error: {e}")))?
                    .to_owned();
                pos += slen;
                strs.push(s);
            }
            Ok(FeatherData::Utf8(strs))
        }
        TYPE_BOOL => {
            let mut bools = Vec::with_capacity(n_rows);
            for i in 0..n_rows {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                let bit = if byte_idx < bytes.len() {
                    (bytes[byte_idx] >> bit_idx) & 1 == 1
                } else {
                    false
                };
                bools.push(bit);
            }
            Ok(FeatherData::Bool(bools))
        }
        other => Err(IoError::FormatError(format!(
            "Feather: unknown type tag {other}"
        ))),
    }
}

fn check_len(bytes: &[u8], n_rows: usize, elem_size: usize, type_name: &str) -> Result<()> {
    if bytes.len() != n_rows * elem_size {
        return Err(IoError::FormatError(format!(
            "Feather: expected {} bytes for {n_rows} {type_name} values, got {}",
            n_rows * elem_size,
            bytes.len()
        )));
    }
    Ok(())
}

// ─────────────────────────────── Tests ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        temp_dir().join(name)
    }

    #[test]
    fn test_roundtrip_numeric() {
        let path = tmp_path("feather_test_numeric.fea");
        let mut ff = FeatherFile::new(4);
        ff.add_column(FeatherColumn::new(
            "ints",
            FeatherData::Int32(vec![1, 2, -3, 100]),
        ))
        .expect("add column");
        ff.add_column(FeatherColumn::new(
            "floats",
            FeatherData::Float64(vec![1.1, 2.2, 3.3, 4.4]),
        ))
        .expect("add column");

        FeatherFile::write(&path, &ff).expect("write");
        let loaded = FeatherFile::read(&path).expect("read");

        assert_eq!(loaded.n_rows, 4);
        assert_eq!(loaded.columns[0].name, "ints");
        assert_eq!(
            loaded.columns[0].data,
            FeatherData::Int32(vec![1, 2, -3, 100])
        );
        assert_eq!(loaded.columns[1].name, "floats");
        assert_eq!(
            loaded.columns[1].data,
            FeatherData::Float64(vec![1.1, 2.2, 3.3, 4.4])
        );
    }

    #[test]
    fn test_roundtrip_strings() {
        let path = tmp_path("feather_test_strings.fea");
        let mut ff = FeatherFile::new(3);
        ff.add_column(FeatherColumn::new(
            "names",
            FeatherData::Utf8(vec!["Alice".into(), "Bob".into(), "Charlie".into()]),
        ))
        .expect("add column");

        FeatherFile::write(&path, &ff).expect("write");
        let loaded = FeatherFile::read(&path).expect("read");

        match &loaded.columns[0].data {
            FeatherData::Utf8(v) => {
                assert_eq!(v, &["Alice", "Bob", "Charlie"]);
            }
            other => panic!("expected Utf8, got {other:?}"),
        }
    }

    #[test]
    fn test_roundtrip_bool() {
        let path = tmp_path("feather_test_bool.fea");
        let mut ff = FeatherFile::new(5);
        ff.add_column(FeatherColumn::new(
            "flags",
            FeatherData::Bool(vec![true, false, true, true, false]),
        ))
        .expect("add column");

        FeatherFile::write(&path, &ff).expect("write");
        let loaded = FeatherFile::read(&path).expect("read");

        assert_eq!(
            loaded.columns[0].data,
            FeatherData::Bool(vec![true, false, true, true, false])
        );
    }

    #[test]
    fn test_roundtrip_null_mask() {
        let path = tmp_path("feather_test_nullmask.fea");
        // 4 rows, first is null (bit=0)
        let mask = vec![0b00001110u8]; // bits 1,2,3 set → rows 1,2,3 valid; row 0 null
        let mut ff = FeatherFile::new(4);
        ff.add_column(FeatherColumn::with_nulls(
            "nullable_ints",
            FeatherData::Int64(vec![0, 10, 20, 30]),
            mask.clone(),
        ))
        .expect("add column");

        FeatherFile::write(&path, &ff).expect("write");
        let loaded = FeatherFile::read(&path).expect("read");

        assert_eq!(loaded.columns[0].null_mask, Some(mask));
    }

    #[test]
    fn test_empty_file() {
        let path = tmp_path("feather_test_empty.fea");
        let ff = FeatherFile::new(0);
        FeatherFile::write(&path, &ff).expect("write");
        let loaded = FeatherFile::read(&path).expect("read");
        assert_eq!(loaded.n_rows, 0);
        assert!(loaded.columns.is_empty());
    }

    #[test]
    fn test_length_mismatch() {
        let mut ff = FeatherFile::new(3);
        let result = ff.add_column(FeatherColumn::new(
            "bad",
            FeatherData::Int32(vec![1, 2]), // only 2 rows
        ));
        assert!(result.is_err());
    }
}
