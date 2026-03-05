//! Encoding and decoding routines for columnar data.
//!
//! Implements run-length encoding, dictionary encoding, and delta encoding
//! for efficient column-oriented storage.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::collections::HashMap;
use std::io::{Read, Write};

use crate::error::{IoError, Result};

use super::types::{ColumnData, EncodingType};

// =============================================================================
// Plain encoding
// =============================================================================

/// Write f64 values in plain encoding
pub fn write_plain_f64<W: Write>(writer: &mut W, data: &[f64]) -> Result<()> {
    for &val in data {
        writer
            .write_f64::<LittleEndian>(val)
            .map_err(|e| IoError::FileError(format!("Failed to write f64: {}", e)))?;
    }
    Ok(())
}

/// Read f64 values in plain encoding
pub fn read_plain_f64<R: Read>(reader: &mut R, count: usize) -> Result<Vec<f64>> {
    let mut data = Vec::with_capacity(count);
    for _ in 0..count {
        let val = reader
            .read_f64::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read f64: {}", e)))?;
        data.push(val);
    }
    Ok(data)
}

/// Write i64 values in plain encoding
pub fn write_plain_i64<W: Write>(writer: &mut W, data: &[i64]) -> Result<()> {
    for &val in data {
        writer
            .write_i64::<LittleEndian>(val)
            .map_err(|e| IoError::FileError(format!("Failed to write i64: {}", e)))?;
    }
    Ok(())
}

/// Read i64 values in plain encoding
pub fn read_plain_i64<R: Read>(reader: &mut R, count: usize) -> Result<Vec<i64>> {
    let mut data = Vec::with_capacity(count);
    for _ in 0..count {
        let val = reader
            .read_i64::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read i64: {}", e)))?;
        data.push(val);
    }
    Ok(data)
}

/// Write string values in plain encoding (length-prefixed)
pub fn write_plain_str<W: Write>(writer: &mut W, data: &[String]) -> Result<()> {
    for s in data {
        let bytes = s.as_bytes();
        writer
            .write_u32::<LittleEndian>(bytes.len() as u32)
            .map_err(|e| IoError::FileError(format!("Failed to write string length: {}", e)))?;
        writer
            .write_all(bytes)
            .map_err(|e| IoError::FileError(format!("Failed to write string data: {}", e)))?;
    }
    Ok(())
}

/// Read string values in plain encoding
pub fn read_plain_str<R: Read>(reader: &mut R, count: usize) -> Result<Vec<String>> {
    let mut data = Vec::with_capacity(count);
    for _ in 0..count {
        let len = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read string length: {}", e)))?
            as usize;
        let mut buf = vec![0u8; len];
        reader
            .read_exact(&mut buf)
            .map_err(|e| IoError::FormatError(format!("Failed to read string data: {}", e)))?;
        let s = String::from_utf8(buf)
            .map_err(|e| IoError::FormatError(format!("Invalid UTF-8 string: {}", e)))?;
        data.push(s);
    }
    Ok(data)
}

/// Write bool values in plain encoding (packed bits)
pub fn write_plain_bool<W: Write>(writer: &mut W, data: &[bool]) -> Result<()> {
    // Pack booleans into bytes, 8 per byte
    let num_bytes = (data.len() + 7) / 8;
    let mut packed = vec![0u8; num_bytes];
    for (i, &val) in data.iter().enumerate() {
        if val {
            packed[i / 8] |= 1 << (i % 8);
        }
    }
    writer
        .write_all(&packed)
        .map_err(|e| IoError::FileError(format!("Failed to write bool data: {}", e)))?;
    Ok(())
}

/// Read bool values in plain encoding
pub fn read_plain_bool<R: Read>(reader: &mut R, count: usize) -> Result<Vec<bool>> {
    let num_bytes = (count + 7) / 8;
    let mut packed = vec![0u8; num_bytes];
    reader
        .read_exact(&mut packed)
        .map_err(|e| IoError::FormatError(format!("Failed to read bool data: {}", e)))?;
    let mut data = Vec::with_capacity(count);
    for i in 0..count {
        data.push((packed[i / 8] >> (i % 8)) & 1 == 1);
    }
    Ok(data)
}

// =============================================================================
// Run-length encoding
// =============================================================================

/// Write f64 values with RLE
pub fn write_rle_f64<W: Write>(writer: &mut W, data: &[f64]) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    let mut i = 0;
    while i < data.len() {
        let val = data[i];
        let mut run_len: u32 = 1;
        while i + (run_len as usize) < data.len() && data[i + (run_len as usize)] == val {
            run_len += 1;
        }
        writer
            .write_u32::<LittleEndian>(run_len)
            .map_err(|e| IoError::FileError(format!("Failed to write RLE run length: {}", e)))?;
        writer
            .write_f64::<LittleEndian>(val)
            .map_err(|e| IoError::FileError(format!("Failed to write RLE value: {}", e)))?;
        i += run_len as usize;
    }
    Ok(())
}

/// Read f64 values with RLE
pub fn read_rle_f64<R: Read>(reader: &mut R, total_count: usize) -> Result<Vec<f64>> {
    let mut data = Vec::with_capacity(total_count);
    while data.len() < total_count {
        let run_len = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read RLE run length: {}", e)))?
            as usize;
        let val = reader
            .read_f64::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read RLE value: {}", e)))?;
        for _ in 0..run_len {
            data.push(val);
        }
    }
    if data.len() != total_count {
        return Err(IoError::FormatError(format!(
            "RLE decoded {} values, expected {}",
            data.len(),
            total_count
        )));
    }
    Ok(data)
}

/// Write i64 values with RLE
pub fn write_rle_i64<W: Write>(writer: &mut W, data: &[i64]) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    let mut i = 0;
    while i < data.len() {
        let val = data[i];
        let mut run_len: u32 = 1;
        while i + (run_len as usize) < data.len() && data[i + (run_len as usize)] == val {
            run_len += 1;
        }
        writer
            .write_u32::<LittleEndian>(run_len)
            .map_err(|e| IoError::FileError(format!("Failed to write RLE run length: {}", e)))?;
        writer
            .write_i64::<LittleEndian>(val)
            .map_err(|e| IoError::FileError(format!("Failed to write RLE value: {}", e)))?;
        i += run_len as usize;
    }
    Ok(())
}

/// Read i64 values with RLE
pub fn read_rle_i64<R: Read>(reader: &mut R, total_count: usize) -> Result<Vec<i64>> {
    let mut data = Vec::with_capacity(total_count);
    while data.len() < total_count {
        let run_len = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read RLE run length: {}", e)))?
            as usize;
        let val = reader
            .read_i64::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read RLE value: {}", e)))?;
        for _ in 0..run_len {
            data.push(val);
        }
    }
    if data.len() != total_count {
        return Err(IoError::FormatError(format!(
            "RLE decoded {} values, expected {}",
            data.len(),
            total_count
        )));
    }
    Ok(data)
}

/// Write string values with RLE
pub fn write_rle_str<W: Write>(writer: &mut W, data: &[String]) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    let mut i = 0;
    while i < data.len() {
        let val = &data[i];
        let mut run_len: u32 = 1;
        while i + (run_len as usize) < data.len() && &data[i + (run_len as usize)] == val {
            run_len += 1;
        }
        writer
            .write_u32::<LittleEndian>(run_len)
            .map_err(|e| IoError::FileError(format!("Failed to write RLE run length: {}", e)))?;
        let bytes = val.as_bytes();
        writer
            .write_u32::<LittleEndian>(bytes.len() as u32)
            .map_err(|e| IoError::FileError(format!("Failed to write RLE string length: {}", e)))?;
        writer
            .write_all(bytes)
            .map_err(|e| IoError::FileError(format!("Failed to write RLE string data: {}", e)))?;
        i += run_len as usize;
    }
    Ok(())
}

/// Read string values with RLE
pub fn read_rle_str<R: Read>(reader: &mut R, total_count: usize) -> Result<Vec<String>> {
    let mut data = Vec::with_capacity(total_count);
    while data.len() < total_count {
        let run_len = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read RLE run length: {}", e)))?
            as usize;
        let str_len = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read RLE string length: {}", e)))?
            as usize;
        let mut buf = vec![0u8; str_len];
        reader
            .read_exact(&mut buf)
            .map_err(|e| IoError::FormatError(format!("Failed to read RLE string data: {}", e)))?;
        let s = String::from_utf8(buf)
            .map_err(|e| IoError::FormatError(format!("Invalid UTF-8 in RLE string: {}", e)))?;
        for _ in 0..run_len {
            data.push(s.clone());
        }
    }
    if data.len() != total_count {
        return Err(IoError::FormatError(format!(
            "RLE decoded {} values, expected {}",
            data.len(),
            total_count
        )));
    }
    Ok(data)
}

/// Write bool values with RLE
pub fn write_rle_bool<W: Write>(writer: &mut W, data: &[bool]) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    let mut i = 0;
    while i < data.len() {
        let val = data[i];
        let mut run_len: u32 = 1;
        while i + (run_len as usize) < data.len() && data[i + (run_len as usize)] == val {
            run_len += 1;
        }
        writer
            .write_u32::<LittleEndian>(run_len)
            .map_err(|e| IoError::FileError(format!("Failed to write RLE run length: {}", e)))?;
        writer
            .write_u8(if val { 1 } else { 0 })
            .map_err(|e| IoError::FileError(format!("Failed to write RLE bool value: {}", e)))?;
        i += run_len as usize;
    }
    Ok(())
}

/// Read bool values with RLE
pub fn read_rle_bool<R: Read>(reader: &mut R, total_count: usize) -> Result<Vec<bool>> {
    let mut data = Vec::with_capacity(total_count);
    while data.len() < total_count {
        let run_len = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read RLE run length: {}", e)))?
            as usize;
        let val = reader
            .read_u8()
            .map_err(|e| IoError::FormatError(format!("Failed to read RLE bool value: {}", e)))?
            != 0;
        for _ in 0..run_len {
            data.push(val);
        }
    }
    if data.len() != total_count {
        return Err(IoError::FormatError(format!(
            "RLE decoded {} values, expected {}",
            data.len(),
            total_count
        )));
    }
    Ok(data)
}

// =============================================================================
// Dictionary encoding (for strings)
// =============================================================================

/// Write string values with dictionary encoding
pub fn write_dict_str<W: Write>(writer: &mut W, data: &[String]) -> Result<()> {
    // Build dictionary
    let mut dictionary: Vec<String> = Vec::new();
    let mut dict_map: HashMap<String, u32> = HashMap::new();
    let mut indices: Vec<u32> = Vec::with_capacity(data.len());

    for s in data {
        let idx = if let Some(&existing) = dict_map.get(s) {
            existing
        } else {
            let new_idx = dictionary.len() as u32;
            dict_map.insert(s.clone(), new_idx);
            dictionary.push(s.clone());
            new_idx
        };
        indices.push(idx);
    }

    // Write dictionary size
    writer
        .write_u32::<LittleEndian>(dictionary.len() as u32)
        .map_err(|e| IoError::FileError(format!("Failed to write dictionary size: {}", e)))?;

    // Write dictionary entries
    for entry in &dictionary {
        let bytes = entry.as_bytes();
        writer
            .write_u32::<LittleEndian>(bytes.len() as u32)
            .map_err(|e| IoError::FileError(format!("Failed to write dict entry length: {}", e)))?;
        writer
            .write_all(bytes)
            .map_err(|e| IoError::FileError(format!("Failed to write dict entry: {}", e)))?;
    }

    // Write indices
    for &idx in &indices {
        writer
            .write_u32::<LittleEndian>(idx)
            .map_err(|e| IoError::FileError(format!("Failed to write dict index: {}", e)))?;
    }

    Ok(())
}

/// Read string values with dictionary encoding
pub fn read_dict_str<R: Read>(reader: &mut R, count: usize) -> Result<Vec<String>> {
    // Read dictionary
    let dict_size = reader
        .read_u32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read dictionary size: {}", e)))?
        as usize;

    let mut dictionary = Vec::with_capacity(dict_size);
    for _ in 0..dict_size {
        let len = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read dict entry length: {}", e)))?
            as usize;
        let mut buf = vec![0u8; len];
        reader
            .read_exact(&mut buf)
            .map_err(|e| IoError::FormatError(format!("Failed to read dict entry: {}", e)))?;
        let s = String::from_utf8(buf)
            .map_err(|e| IoError::FormatError(format!("Invalid UTF-8 in dict entry: {}", e)))?;
        dictionary.push(s);
    }

    // Read indices
    let mut data = Vec::with_capacity(count);
    for _ in 0..count {
        let idx = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read dict index: {}", e)))?
            as usize;
        if idx >= dictionary.len() {
            return Err(IoError::FormatError(format!(
                "Dictionary index {} out of range (dict size {})",
                idx,
                dictionary.len()
            )));
        }
        data.push(dictionary[idx].clone());
    }

    Ok(data)
}

// =============================================================================
// Delta encoding (for sorted numeric columns)
// =============================================================================

/// Write f64 values with delta encoding
pub fn write_delta_f64<W: Write>(writer: &mut W, data: &[f64]) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    // Write first value
    writer
        .write_f64::<LittleEndian>(data[0])
        .map_err(|e| IoError::FileError(format!("Failed to write delta base: {}", e)))?;
    // Write deltas
    for i in 1..data.len() {
        let delta = data[i] - data[i - 1];
        writer
            .write_f64::<LittleEndian>(delta)
            .map_err(|e| IoError::FileError(format!("Failed to write delta: {}", e)))?;
    }
    Ok(())
}

/// Read f64 values with delta encoding
pub fn read_delta_f64<R: Read>(reader: &mut R, count: usize) -> Result<Vec<f64>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut data = Vec::with_capacity(count);
    let base = reader
        .read_f64::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read delta base: {}", e)))?;
    data.push(base);
    for _ in 1..count {
        let delta = reader
            .read_f64::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read delta: {}", e)))?;
        let prev = data[data.len() - 1];
        data.push(prev + delta);
    }
    Ok(data)
}

/// Write i64 values with delta encoding
pub fn write_delta_i64<W: Write>(writer: &mut W, data: &[i64]) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    writer
        .write_i64::<LittleEndian>(data[0])
        .map_err(|e| IoError::FileError(format!("Failed to write delta base: {}", e)))?;
    for i in 1..data.len() {
        let delta = data[i].wrapping_sub(data[i - 1]);
        writer
            .write_i64::<LittleEndian>(delta)
            .map_err(|e| IoError::FileError(format!("Failed to write delta: {}", e)))?;
    }
    Ok(())
}

/// Read i64 values with delta encoding
pub fn read_delta_i64<R: Read>(reader: &mut R, count: usize) -> Result<Vec<i64>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut data = Vec::with_capacity(count);
    let base = reader
        .read_i64::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read delta base: {}", e)))?;
    data.push(base);
    for _ in 1..count {
        let delta = reader
            .read_i64::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read delta: {}", e)))?;
        let prev = data[data.len() - 1];
        data.push(prev.wrapping_add(delta));
    }
    Ok(data)
}

// =============================================================================
// Unified encode/decode
// =============================================================================

/// Encode column data to a writer using the specified encoding
pub fn encode_column<W: Write>(
    writer: &mut W,
    data: &ColumnData,
    encoding: EncodingType,
) -> Result<()> {
    match (data, encoding) {
        // Plain
        (ColumnData::Float64(v), EncodingType::Plain) => write_plain_f64(writer, v),
        (ColumnData::Int64(v), EncodingType::Plain) => write_plain_i64(writer, v),
        (ColumnData::Str(v), EncodingType::Plain) => write_plain_str(writer, v),
        (ColumnData::Bool(v), EncodingType::Plain) => write_plain_bool(writer, v),
        // RLE
        (ColumnData::Float64(v), EncodingType::Rle) => write_rle_f64(writer, v),
        (ColumnData::Int64(v), EncodingType::Rle) => write_rle_i64(writer, v),
        (ColumnData::Str(v), EncodingType::Rle) => write_rle_str(writer, v),
        (ColumnData::Bool(v), EncodingType::Rle) => write_rle_bool(writer, v),
        // Dictionary (strings only)
        (ColumnData::Str(v), EncodingType::Dictionary) => write_dict_str(writer, v),
        // Delta (numeric only)
        (ColumnData::Float64(v), EncodingType::Delta) => write_delta_f64(writer, v),
        (ColumnData::Int64(v), EncodingType::Delta) => write_delta_i64(writer, v),
        // Unsupported combinations
        (data, enc) => Err(IoError::FormatError(format!(
            "Encoding {:?} not supported for column type {:?}",
            enc,
            data.type_tag()
        ))),
    }
}

/// Decode column data from a reader
pub fn decode_column<R: Read>(
    reader: &mut R,
    type_tag: super::types::ColumnTypeTag,
    encoding: EncodingType,
    count: usize,
) -> Result<ColumnData> {
    use super::types::ColumnTypeTag;

    match (type_tag, encoding) {
        // Plain
        (ColumnTypeTag::Float64, EncodingType::Plain) => {
            Ok(ColumnData::Float64(read_plain_f64(reader, count)?))
        }
        (ColumnTypeTag::Int64, EncodingType::Plain) => {
            Ok(ColumnData::Int64(read_plain_i64(reader, count)?))
        }
        (ColumnTypeTag::Str, EncodingType::Plain) => {
            Ok(ColumnData::Str(read_plain_str(reader, count)?))
        }
        (ColumnTypeTag::Bool, EncodingType::Plain) => {
            Ok(ColumnData::Bool(read_plain_bool(reader, count)?))
        }
        // RLE
        (ColumnTypeTag::Float64, EncodingType::Rle) => {
            Ok(ColumnData::Float64(read_rle_f64(reader, count)?))
        }
        (ColumnTypeTag::Int64, EncodingType::Rle) => {
            Ok(ColumnData::Int64(read_rle_i64(reader, count)?))
        }
        (ColumnTypeTag::Str, EncodingType::Rle) => {
            Ok(ColumnData::Str(read_rle_str(reader, count)?))
        }
        (ColumnTypeTag::Bool, EncodingType::Rle) => {
            Ok(ColumnData::Bool(read_rle_bool(reader, count)?))
        }
        // Dictionary
        (ColumnTypeTag::Str, EncodingType::Dictionary) => {
            Ok(ColumnData::Str(read_dict_str(reader, count)?))
        }
        // Delta
        (ColumnTypeTag::Float64, EncodingType::Delta) => {
            Ok(ColumnData::Float64(read_delta_f64(reader, count)?))
        }
        (ColumnTypeTag::Int64, EncodingType::Delta) => {
            Ok(ColumnData::Int64(read_delta_i64(reader, count)?))
        }
        // Unsupported
        (tt, enc) => Err(IoError::FormatError(format!(
            "Encoding {:?} not supported for type {:?}",
            enc, tt
        ))),
    }
}
