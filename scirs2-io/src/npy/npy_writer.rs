//! Writer for NumPy .npy files.

use byteorder::{LittleEndian, WriteBytesExt};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::{IoError, Result};

use super::types::{
    NpyArray, NpyDtype, NpyHeader, NPY_MAGIC, NPY_MAJOR_VERSION, NPY_MINOR_VERSION,
};

/// Write an NpyArray to a .npy file
pub fn write_npy<P: AsRef<Path>>(path: P, array: &NpyArray) -> Result<()> {
    let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut writer = BufWriter::new(file);
    write_npy_to_writer(&mut writer, array)
}

/// Write an NpyArray to any writer (used by npz writer too)
pub fn write_npy_to_writer<W: Write>(writer: &mut W, array: &NpyArray) -> Result<()> {
    let header = NpyHeader {
        dtype: array.dtype(),
        byte_order: super::types::ByteOrder::LittleEndian,
        fortran_order: false,
        shape: array.shape().to_vec(),
    };

    write_npy_header(writer, &header)?;
    write_npy_data(writer, array)?;

    Ok(())
}

/// Write the .npy header
fn write_npy_header<W: Write>(writer: &mut W, header: &NpyHeader) -> Result<()> {
    // Write magic
    writer
        .write_all(NPY_MAGIC)
        .map_err(|e| IoError::FileError(format!("Failed to write npy magic: {}", e)))?;

    // Write version
    writer
        .write_u8(NPY_MAJOR_VERSION)
        .map_err(|e| IoError::FileError(format!("Failed to write major version: {}", e)))?;
    writer
        .write_u8(NPY_MINOR_VERSION)
        .map_err(|e| IoError::FileError(format!("Failed to write minor version: {}", e)))?;

    // Build header string
    let header_str = header.to_header_string();

    // Pad header to align to 64 bytes
    // Total header size = 6 (magic) + 2 (version) + 2 (header_len) + header_bytes
    // Must be divisible by 64 for v1
    let preamble_len = 10; // 6 + 2 + 2
    let raw_len = header_str.len() + 1; // +1 for trailing newline
    let pad_len = (64 - (preamble_len + raw_len) % 64) % 64;
    let total_header_len = raw_len + pad_len;

    // Write header length (u16 for v1)
    writer
        .write_u16::<LittleEndian>(total_header_len as u16)
        .map_err(|e| IoError::FileError(format!("Failed to write header length: {}", e)))?;

    // Write header string
    writer
        .write_all(header_str.as_bytes())
        .map_err(|e| IoError::FileError(format!("Failed to write header string: {}", e)))?;

    // Write padding spaces
    for _ in 0..pad_len {
        writer
            .write_all(b" ")
            .map_err(|e| IoError::FileError(format!("Failed to write padding: {}", e)))?;
    }

    // Write trailing newline
    writer
        .write_all(b"\n")
        .map_err(|e| IoError::FileError(format!("Failed to write newline: {}", e)))?;

    Ok(())
}

/// Write array data in little-endian format
fn write_npy_data<W: Write>(writer: &mut W, array: &NpyArray) -> Result<()> {
    match array {
        NpyArray::Float32 { data, .. } => {
            for &val in data {
                writer
                    .write_f32::<LittleEndian>(val)
                    .map_err(|e| IoError::FileError(format!("Failed to write f32: {}", e)))?;
            }
        }
        NpyArray::Float64 { data, .. } => {
            for &val in data {
                writer
                    .write_f64::<LittleEndian>(val)
                    .map_err(|e| IoError::FileError(format!("Failed to write f64: {}", e)))?;
            }
        }
        NpyArray::Int32 { data, .. } => {
            for &val in data {
                writer
                    .write_i32::<LittleEndian>(val)
                    .map_err(|e| IoError::FileError(format!("Failed to write i32: {}", e)))?;
            }
        }
        NpyArray::Int64 { data, .. } => {
            for &val in data {
                writer
                    .write_i64::<LittleEndian>(val)
                    .map_err(|e| IoError::FileError(format!("Failed to write i64: {}", e)))?;
            }
        }
    }

    Ok(())
}

/// Helper to write f64 data as a 1D .npy file
pub fn write_npy_f64<P: AsRef<Path>>(path: P, data: &[f64]) -> Result<()> {
    let array = NpyArray::Float64 {
        data: data.to_vec(),
        shape: vec![data.len()],
    };
    write_npy(path, &array)
}

/// Helper to write f32 data as a 1D .npy file
pub fn write_npy_f32<P: AsRef<Path>>(path: P, data: &[f32]) -> Result<()> {
    let array = NpyArray::Float32 {
        data: data.to_vec(),
        shape: vec![data.len()],
    };
    write_npy(path, &array)
}

/// Helper to write i32 data as a 1D .npy file
pub fn write_npy_i32<P: AsRef<Path>>(path: P, data: &[i32]) -> Result<()> {
    let array = NpyArray::Int32 {
        data: data.to_vec(),
        shape: vec![data.len()],
    };
    write_npy(path, &array)
}

/// Helper to write i64 data as a 1D .npy file
pub fn write_npy_i64<P: AsRef<Path>>(path: P, data: &[i64]) -> Result<()> {
    let array = NpyArray::Int64 {
        data: data.to_vec(),
        shape: vec![data.len()],
    };
    write_npy(path, &array)
}

/// Helper to write a 2D f64 array
pub fn write_npy_f64_2d<P: AsRef<Path>>(
    path: P,
    data: &[f64],
    rows: usize,
    cols: usize,
) -> Result<()> {
    if data.len() != rows * cols {
        return Err(IoError::FormatError(format!(
            "Data length {} does not match shape {}x{}",
            data.len(),
            rows,
            cols
        )));
    }
    let array = NpyArray::Float64 {
        data: data.to_vec(),
        shape: vec![rows, cols],
    };
    write_npy(path, &array)
}
