//! Reader for NumPy .npy files.

use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::error::{IoError, Result};

use super::types::{parse_header_dict, ByteOrder, NpyArray, NpyDtype, NpyHeader, NPY_MAGIC};

/// Read a .npy file and return the array data
pub fn read_npy<P: AsRef<Path>>(path: P) -> Result<NpyArray> {
    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut reader = BufReader::new(file);
    read_npy_from_reader(&mut reader)
}

/// Read a .npy array from any reader (used by npz reader too)
pub fn read_npy_from_reader<R: Read>(reader: &mut R) -> Result<NpyArray> {
    let header = read_npy_header(reader)?;
    read_npy_data(reader, &header)
}

/// Read just the header from a .npy file
fn read_npy_header<R: Read>(reader: &mut R) -> Result<NpyHeader> {
    // Read magic
    let mut magic = [0u8; 6];
    reader
        .read_exact(&mut magic)
        .map_err(|e| IoError::FormatError(format!("Failed to read npy magic: {}", e)))?;
    if &magic != NPY_MAGIC {
        return Err(IoError::FormatError(
            "Not a valid .npy file (bad magic)".to_string(),
        ));
    }

    // Read version
    let major = reader
        .read_u8()
        .map_err(|e| IoError::FormatError(format!("Failed to read npy major version: {}", e)))?;
    let _minor = reader
        .read_u8()
        .map_err(|e| IoError::FormatError(format!("Failed to read npy minor version: {}", e)))?;

    // Read header length
    let header_len = if major <= 1 {
        reader
            .read_u16::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read header length: {}", e)))?
            as usize
    } else {
        reader
            .read_u32::<LittleEndian>()
            .map_err(|e| IoError::FormatError(format!("Failed to read header length v2: {}", e)))?
            as usize
    };

    // Read header bytes
    let mut header_bytes = vec![0u8; header_len];
    reader
        .read_exact(&mut header_bytes)
        .map_err(|e| IoError::FormatError(format!("Failed to read header data: {}", e)))?;

    let header_str = String::from_utf8(header_bytes)
        .map_err(|e| IoError::FormatError(format!("Invalid UTF-8 in header: {}", e)))?;

    parse_header_dict(&header_str)
}

/// Read array data based on header info
fn read_npy_data<R: Read>(reader: &mut R, header: &NpyHeader) -> Result<NpyArray> {
    let num_elements = header.num_elements();

    match header.dtype {
        NpyDtype::Float32 => {
            let data = read_f32_data(reader, num_elements, header.byte_order)?;
            Ok(NpyArray::Float32 {
                data,
                shape: header.shape.clone(),
            })
        }
        NpyDtype::Float64 => {
            let data = read_f64_data(reader, num_elements, header.byte_order)?;
            Ok(NpyArray::Float64 {
                data,
                shape: header.shape.clone(),
            })
        }
        NpyDtype::Int32 => {
            let data = read_i32_data(reader, num_elements, header.byte_order)?;
            Ok(NpyArray::Int32 {
                data,
                shape: header.shape.clone(),
            })
        }
        NpyDtype::Int64 => {
            let data = read_i64_data(reader, num_elements, header.byte_order)?;
            Ok(NpyArray::Int64 {
                data,
                shape: header.shape.clone(),
            })
        }
    }
}

fn read_f32_data<R: Read>(reader: &mut R, count: usize, byte_order: ByteOrder) -> Result<Vec<f32>> {
    let mut data = Vec::with_capacity(count);
    for _ in 0..count {
        let val = match byte_order {
            ByteOrder::BigEndian => reader.read_f32::<BigEndian>(),
            _ => reader.read_f32::<LittleEndian>(),
        }
        .map_err(|e| IoError::FormatError(format!("Failed to read f32: {}", e)))?;
        data.push(val);
    }
    Ok(data)
}

fn read_f64_data<R: Read>(reader: &mut R, count: usize, byte_order: ByteOrder) -> Result<Vec<f64>> {
    let mut data = Vec::with_capacity(count);
    for _ in 0..count {
        let val = match byte_order {
            ByteOrder::BigEndian => reader.read_f64::<BigEndian>(),
            _ => reader.read_f64::<LittleEndian>(),
        }
        .map_err(|e| IoError::FormatError(format!("Failed to read f64: {}", e)))?;
        data.push(val);
    }
    Ok(data)
}

fn read_i32_data<R: Read>(reader: &mut R, count: usize, byte_order: ByteOrder) -> Result<Vec<i32>> {
    let mut data = Vec::with_capacity(count);
    for _ in 0..count {
        let val = match byte_order {
            ByteOrder::BigEndian => reader.read_i32::<BigEndian>(),
            _ => reader.read_i32::<LittleEndian>(),
        }
        .map_err(|e| IoError::FormatError(format!("Failed to read i32: {}", e)))?;
        data.push(val);
    }
    Ok(data)
}

fn read_i64_data<R: Read>(reader: &mut R, count: usize, byte_order: ByteOrder) -> Result<Vec<i64>> {
    let mut data = Vec::with_capacity(count);
    for _ in 0..count {
        let val = match byte_order {
            ByteOrder::BigEndian => reader.read_i64::<BigEndian>(),
            _ => reader.read_i64::<LittleEndian>(),
        }
        .map_err(|e| IoError::FormatError(format!("Failed to read i64: {}", e)))?;
        data.push(val);
    }
    Ok(data)
}
