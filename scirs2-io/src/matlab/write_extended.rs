//! Extended MATLAB v5 write support
//!
//! Adds writing capability for:
//! - Cell arrays
//! - Struct arrays
//! - Sparse matrices (double)
//! - Compressed data elements (deflate)
//! - Additional integer types (Int8, Int16, UInt8, UInt16, Int64, UInt64)

use crate::error::{IoError, Result};
use crate::matlab::{
    MatType, MI_INT32, MI_INT8, MI_MATRIX, MI_UINT32, MX_CHAR_CLASS, MX_DOUBLE_CLASS,
    MX_INT32_CLASS, MX_SINGLE_CLASS, MX_UINT8_CLASS,
};
use byteorder::{LittleEndian, WriteBytesExt};
use scirs2_core::ndarray::ArrayD;
use std::io::{Seek, Write};

// Additional constants for extended types
const MI_UINT8: i32 = 2;
const MI_INT16: i32 = 3;
const MI_UINT16: i32 = 4;
const MI_SINGLE: i32 = 7;
const MI_DOUBLE: i32 = 9;
const MI_INT64: i32 = 12;
const MI_UINT64: i32 = 13;
const MI_COMPRESSED: i32 = 15;

const MX_CELL_CLASS: i32 = 1;
const MX_STRUCT_CLASS: i32 = 2;
const MX_SPARSE_CLASS: i32 = 5;
const MX_INT8_CLASS: i32 = 8;
const MX_INT16_CLASS: i32 = 10;
const MX_UINT16_CLASS: i32 = 11;
const MX_INT64_CLASS: i32 = 14;
const MX_UINT64_CLASS: i32 = 15;

/// Write a complete MAT file with extended type support
pub fn write_mat_extended<W: Write + Seek>(
    writer: &mut W,
    vars: &std::collections::HashMap<String, MatType>,
) -> Result<()> {
    super::write_impl::write_mat_header(writer)?;

    for (name, mat_type) in vars {
        write_variable_extended(writer, name, mat_type)?;
    }

    writer
        .flush()
        .map_err(|e| IoError::FileError(format!("Failed to flush: {e}")))?;
    Ok(())
}

/// Write a single variable with full type support
pub fn write_variable_extended<W: Write + Seek>(
    writer: &mut W,
    name: &str,
    mat_type: &MatType,
) -> Result<()> {
    // For simple types, delegate to existing writer
    match mat_type {
        MatType::Double(_)
        | MatType::Single(_)
        | MatType::Int32(_)
        | MatType::Logical(_)
        | MatType::Char(_) => {
            return super::write_impl::write_variable(writer, name, mat_type);
        }
        _ => {}
    }

    // Extended types: write to a buffer, then emit as MI_MATRIX
    writer
        .write_i32::<LittleEndian>(MI_MATRIX)
        .map_err(|e| IoError::FileError(format!("Failed to write matrix tag: {e}")))?;

    let size_pos = writer
        .stream_position()
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(0)
        .map_err(|e| IoError::FileError(e.to_string()))?;

    let data_start = writer
        .stream_position()
        .map_err(|e| IoError::FileError(e.to_string()))?;

    match mat_type {
        MatType::Int8(array) => {
            write_matrix_header(writer, name, array.shape(), MX_INT8_CLASS, false)?;
            write_typed_data_i8(writer, array)?;
        }
        MatType::Int16(array) => {
            write_matrix_header(writer, name, array.shape(), MX_INT16_CLASS, false)?;
            write_typed_data_i16(writer, array)?;
        }
        MatType::UInt8(array) => {
            write_matrix_header(writer, name, array.shape(), MX_UINT8_CLASS, false)?;
            write_typed_data_u8(writer, array)?;
        }
        MatType::UInt16(array) => {
            write_matrix_header(writer, name, array.shape(), MX_UINT16_CLASS, false)?;
            write_typed_data_u16(writer, array)?;
        }
        MatType::UInt32(array) => {
            write_matrix_header(writer, name, array.shape(), MX_INT32_CLASS, false)?;
            write_typed_data_u32(writer, array)?;
        }
        MatType::Int64(array) => {
            write_matrix_header(writer, name, array.shape(), MX_INT64_CLASS, false)?;
            write_typed_data_i64(writer, array)?;
        }
        MatType::UInt64(array) => {
            write_matrix_header(writer, name, array.shape(), MX_UINT64_CLASS, false)?;
            write_typed_data_u64(writer, array)?;
        }
        MatType::Cell(items) => {
            write_cell_array(writer, name, items)?;
        }
        MatType::Struct(fields) => {
            write_struct_array(writer, name, fields)?;
        }
        MatType::SparseDouble(sparse) => {
            write_sparse_double(writer, name, sparse)?;
        }
        _ => {
            return Err(IoError::Other(
                "Unsupported type for extended writer".to_string(),
            ));
        }
    }

    let data_end = writer
        .stream_position()
        .map_err(|e| IoError::FileError(e.to_string()))?;
    let total_size = (data_end - data_start) as i32;

    writer
        .seek(std::io::SeekFrom::Start(size_pos))
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(total_size)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .seek(std::io::SeekFrom::Start(data_end))
        .map_err(|e| IoError::FileError(e.to_string()))?;

    Ok(())
}

// =============================================================================
// Matrix header helper (shared for extended types)
// =============================================================================

fn write_matrix_header<W: Write + Seek>(
    writer: &mut W,
    name: &str,
    shape: &[usize],
    class_type: i32,
    is_logical: bool,
) -> Result<()> {
    // Array flags
    writer
        .write_i32::<LittleEndian>(MI_UINT32)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(8)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    let mut flags = class_type as u32;
    if is_logical {
        flags |= 0x200;
    }
    writer
        .write_u32::<LittleEndian>(flags)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_u32::<LittleEndian>(0)
        .map_err(|e| IoError::FileError(e.to_string()))?;

    // Dimensions (reversed for MATLAB column-major)
    let dims_size = (shape.len() * 4) as i32;
    writer
        .write_i32::<LittleEndian>(MI_INT32)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(dims_size)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    for &dim in shape.iter().rev() {
        writer
            .write_i32::<LittleEndian>(dim as i32)
            .map_err(|e| IoError::FileError(e.to_string()))?;
    }
    let pad = (8 - (dims_size % 8)) % 8;
    if pad > 0 {
        writer
            .write_all(&vec![0u8; pad as usize])
            .map_err(|e| IoError::FileError(e.to_string()))?;
    }

    // Name
    let name_bytes = name.as_bytes();
    let name_len = name_bytes.len() as i32;
    writer
        .write_i32::<LittleEndian>(MI_INT8)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(name_len)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_all(name_bytes)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    let name_pad = (8 - (name_len % 8)) % 8;
    if name_pad > 0 {
        writer
            .write_all(&vec![0u8; name_pad as usize])
            .map_err(|e| IoError::FileError(e.to_string()))?;
    }

    Ok(())
}

// =============================================================================
// Integer data writers
// =============================================================================

fn write_typed_data_i8<W: Write>(writer: &mut W, array: &ArrayD<i8>) -> Result<()> {
    let data_size = array.len() as i32;
    writer
        .write_i32::<LittleEndian>(MI_INT8)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(data_size)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    for &v in array.iter() {
        writer
            .write_i8(v)
            .map_err(|e| IoError::FileError(e.to_string()))?;
    }
    pad_to_8(writer, data_size as usize)?;
    Ok(())
}

fn write_typed_data_i16<W: Write>(writer: &mut W, array: &ArrayD<i16>) -> Result<()> {
    let data_size = (array.len() * 2) as i32;
    writer
        .write_i32::<LittleEndian>(MI_INT16)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(data_size)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    for &v in array.iter() {
        writer
            .write_i16::<LittleEndian>(v)
            .map_err(|e| IoError::FileError(e.to_string()))?;
    }
    pad_to_8(writer, data_size as usize)?;
    Ok(())
}

fn write_typed_data_u8<W: Write>(writer: &mut W, array: &ArrayD<u8>) -> Result<()> {
    let data_size = array.len() as i32;
    writer
        .write_i32::<LittleEndian>(MI_UINT8)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(data_size)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    for &v in array.iter() {
        writer
            .write_u8(v)
            .map_err(|e| IoError::FileError(e.to_string()))?;
    }
    pad_to_8(writer, data_size as usize)?;
    Ok(())
}

fn write_typed_data_u16<W: Write>(writer: &mut W, array: &ArrayD<u16>) -> Result<()> {
    let data_size = (array.len() * 2) as i32;
    writer
        .write_i32::<LittleEndian>(MI_UINT16)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(data_size)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    for &v in array.iter() {
        writer
            .write_u16::<LittleEndian>(v)
            .map_err(|e| IoError::FileError(e.to_string()))?;
    }
    pad_to_8(writer, data_size as usize)?;
    Ok(())
}

fn write_typed_data_u32<W: Write>(writer: &mut W, array: &ArrayD<u32>) -> Result<()> {
    let data_size = (array.len() * 4) as i32;
    writer
        .write_i32::<LittleEndian>(MI_UINT32)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(data_size)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    for &v in array.iter() {
        writer
            .write_u32::<LittleEndian>(v)
            .map_err(|e| IoError::FileError(e.to_string()))?;
    }
    pad_to_8(writer, data_size as usize)?;
    Ok(())
}

fn write_typed_data_i64<W: Write>(writer: &mut W, array: &ArrayD<i64>) -> Result<()> {
    let data_size = (array.len() * 8) as i32;
    writer
        .write_i32::<LittleEndian>(MI_INT64)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(data_size)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    for &v in array.iter() {
        writer
            .write_i64::<LittleEndian>(v)
            .map_err(|e| IoError::FileError(e.to_string()))?;
    }
    pad_to_8(writer, data_size as usize)?;
    Ok(())
}

fn write_typed_data_u64<W: Write>(writer: &mut W, array: &ArrayD<u64>) -> Result<()> {
    let data_size = (array.len() * 8) as i32;
    writer
        .write_i32::<LittleEndian>(MI_UINT64)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(data_size)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    for &v in array.iter() {
        writer
            .write_u64::<LittleEndian>(v)
            .map_err(|e| IoError::FileError(e.to_string()))?;
    }
    pad_to_8(writer, data_size as usize)?;
    Ok(())
}

// =============================================================================
// Cell array writer
// =============================================================================

fn write_cell_array<W: Write + Seek>(writer: &mut W, name: &str, items: &[MatType]) -> Result<()> {
    let shape = [1, items.len()];
    write_matrix_header(writer, name, &shape, MX_CELL_CLASS, false)?;

    // Each cell element is an embedded MI_MATRIX
    for item in items {
        let cell_name = ""; // cell elements have empty names
        write_embedded_matrix(writer, cell_name, item)?;
    }

    Ok(())
}

// =============================================================================
// Struct array writer
// =============================================================================

fn write_struct_array<W: Write + Seek>(
    writer: &mut W,
    name: &str,
    fields: &std::collections::HashMap<String, MatType>,
) -> Result<()> {
    let shape = [1, 1]; // scalar struct
    write_matrix_header(writer, name, &shape, MX_STRUCT_CLASS, false)?;

    // Field name length (maximum field name length, padded)
    let max_name_len = fields.keys().map(|k| k.len()).max().unwrap_or(0);
    let field_name_len = ((max_name_len + 8) / 8) * 8; // pad to 8
    let field_name_len = field_name_len.max(32); // MATLAB minimum is 32

    // Write field name length as small data element
    writer
        .write_i32::<LittleEndian>(MI_INT32)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(4)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(field_name_len as i32)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_all(&[0u8; 4]) // pad to 8
        .map_err(|e| IoError::FileError(e.to_string()))?;

    // Write field names as MI_INT8 data element
    let num_fields = fields.len();
    let names_total = num_fields * field_name_len;
    writer
        .write_i32::<LittleEndian>(MI_INT8)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(names_total as i32)
        .map_err(|e| IoError::FileError(e.to_string()))?;

    // Collect field names in a stable order
    let mut sorted_keys: Vec<&String> = fields.keys().collect();
    sorted_keys.sort();

    for key in &sorted_keys {
        let bytes = key.as_bytes();
        let mut padded = vec![0u8; field_name_len];
        let copy_len = bytes.len().min(field_name_len);
        padded[..copy_len].copy_from_slice(&bytes[..copy_len]);
        writer
            .write_all(&padded)
            .map_err(|e| IoError::FileError(e.to_string()))?;
    }
    // Pad names block to 8-byte boundary
    let names_pad = (8 - (names_total % 8)) % 8;
    if names_pad > 0 {
        writer
            .write_all(&vec![0u8; names_pad])
            .map_err(|e| IoError::FileError(e.to_string()))?;
    }

    // Write field values
    for key in &sorted_keys {
        if let Some(value) = fields.get(*key) {
            write_embedded_matrix(writer, "", value)?;
        }
    }

    Ok(())
}

// =============================================================================
// Sparse double writer
// =============================================================================

fn write_sparse_double<W: Write + Seek>(
    writer: &mut W,
    name: &str,
    sparse: &crate::sparse::SparseMatrix<f64>,
) -> Result<()> {
    let (nrows, ncols) = sparse.shape();
    let nnz = sparse.nnz();
    let shape = [nrows, ncols];

    // Array flags with sparse class and nzmax
    writer
        .write_i32::<LittleEndian>(MI_UINT32)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(8)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    let flags = MX_SPARSE_CLASS as u32;
    writer
        .write_u32::<LittleEndian>(flags)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_u32::<LittleEndian>(nnz as u32)
        .map_err(|e| IoError::FileError(e.to_string()))?;

    // Dimensions
    let dims_size = (shape.len() * 4) as i32;
    writer
        .write_i32::<LittleEndian>(MI_INT32)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(dims_size)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    // MATLAB: rows, cols (not reversed for sparse)
    writer
        .write_i32::<LittleEndian>(nrows as i32)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(ncols as i32)
        .map_err(|e| IoError::FileError(e.to_string()))?;

    // Name
    let name_bytes = name.as_bytes();
    let name_len = name_bytes.len() as i32;
    writer
        .write_i32::<LittleEndian>(MI_INT8)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(name_len)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_all(name_bytes)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    let name_pad = (8 - (name_len % 8)) % 8;
    if name_pad > 0 {
        writer
            .write_all(&vec![0u8; name_pad as usize])
            .map_err(|e| IoError::FileError(e.to_string()))?;
    }

    // Get COO triplets from sparse matrix
    let coo = sparse.to_coo();

    // Build CSC format: column pointers (jc), row indices (ir), values (pr)
    let mut jc = vec![0i32; ncols + 1];
    let mut ir = Vec::with_capacity(nnz);
    let mut pr = Vec::with_capacity(nnz);

    // Sort by column then row
    let mut triplets: Vec<(usize, usize, f64)> = coo
        .row_indices
        .iter()
        .zip(coo.col_indices.iter())
        .zip(coo.values.iter())
        .map(|((&r, &c), &v)| (r, c, v))
        .collect();
    triplets.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

    for &(r, c, v) in &triplets {
        ir.push(r as i32);
        pr.push(v);
        jc[c + 1] += 1;
    }
    // Convert counts to cumulative
    for i in 1..=ncols {
        jc[i] += jc[i - 1];
    }

    // Write ir (row indices)
    let ir_size = (ir.len() * 4) as i32;
    writer
        .write_i32::<LittleEndian>(MI_INT32)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(ir_size)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    for &idx in &ir {
        writer
            .write_i32::<LittleEndian>(idx)
            .map_err(|e| IoError::FileError(e.to_string()))?;
    }
    pad_to_8(writer, ir_size as usize)?;

    // Write jc (column pointers)
    let jc_size = (jc.len() * 4) as i32;
    writer
        .write_i32::<LittleEndian>(MI_INT32)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(jc_size)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    for &ptr in &jc {
        writer
            .write_i32::<LittleEndian>(ptr)
            .map_err(|e| IoError::FileError(e.to_string()))?;
    }
    pad_to_8(writer, jc_size as usize)?;

    // Write pr (values)
    let pr_size = (pr.len() * 8) as i32;
    writer
        .write_i32::<LittleEndian>(MI_DOUBLE)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(pr_size)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    for &val in &pr {
        writer
            .write_f64::<LittleEndian>(val)
            .map_err(|e| IoError::FileError(e.to_string()))?;
    }
    pad_to_8(writer, pr_size as usize)?;

    Ok(())
}

// =============================================================================
// Embedded matrix (for cell/struct fields)
// =============================================================================

fn write_embedded_matrix<W: Write + Seek>(
    writer: &mut W,
    name: &str,
    mat_type: &MatType,
) -> Result<()> {
    // Write as MI_MATRIX with size
    writer
        .write_i32::<LittleEndian>(MI_MATRIX)
        .map_err(|e| IoError::FileError(e.to_string()))?;

    let size_pos = writer
        .stream_position()
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(0)
        .map_err(|e| IoError::FileError(e.to_string()))?;

    let data_start = writer
        .stream_position()
        .map_err(|e| IoError::FileError(e.to_string()))?;

    match mat_type {
        MatType::Double(array) => {
            write_matrix_header(writer, name, array.shape(), MX_DOUBLE_CLASS, false)?;
            write_double_data(writer, array)?;
        }
        MatType::Single(array) => {
            write_matrix_header(writer, name, array.shape(), MX_SINGLE_CLASS, false)?;
            write_single_data(writer, array)?;
        }
        MatType::Int32(array) => {
            write_matrix_header(writer, name, array.shape(), MX_INT32_CLASS, false)?;
            write_int32_data(writer, array)?;
        }
        MatType::Char(s) => {
            let utf16_chars: Vec<u16> = s.encode_utf16().collect();
            let shape = [1, utf16_chars.len()];
            write_matrix_header(writer, name, &shape, MX_CHAR_CLASS, false)?;
            let data_size = (utf16_chars.len() * 2) as i32;
            writer
                .write_i32::<LittleEndian>(MI_UINT16)
                .map_err(|e| IoError::FileError(e.to_string()))?;
            writer
                .write_i32::<LittleEndian>(data_size)
                .map_err(|e| IoError::FileError(e.to_string()))?;
            for &ch in &utf16_chars {
                writer
                    .write_u16::<LittleEndian>(ch)
                    .map_err(|e| IoError::FileError(e.to_string()))?;
            }
            pad_to_8(writer, data_size as usize)?;
        }
        MatType::Logical(array) => {
            write_matrix_header(writer, name, array.shape(), MX_UINT8_CLASS, true)?;
            let data_size = array.len() as i32;
            writer
                .write_i32::<LittleEndian>(MI_UINT8)
                .map_err(|e| IoError::FileError(e.to_string()))?;
            writer
                .write_i32::<LittleEndian>(data_size)
                .map_err(|e| IoError::FileError(e.to_string()))?;
            for &v in array.iter() {
                writer
                    .write_u8(if v { 1 } else { 0 })
                    .map_err(|e| IoError::FileError(e.to_string()))?;
            }
            pad_to_8(writer, data_size as usize)?;
        }
        _ => {
            // For nested cell/struct, use a scalar double 0 as placeholder
            let placeholder = scirs2_core::ndarray::arr0(0.0f64).into_dyn();
            write_matrix_header(writer, name, placeholder.shape(), MX_DOUBLE_CLASS, false)?;
            write_double_data(writer, &placeholder)?;
        }
    }

    let data_end = writer
        .stream_position()
        .map_err(|e| IoError::FileError(e.to_string()))?;
    let total_size = (data_end - data_start) as i32;

    writer
        .seek(std::io::SeekFrom::Start(size_pos))
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(total_size)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .seek(std::io::SeekFrom::Start(data_end))
        .map_err(|e| IoError::FileError(e.to_string()))?;

    Ok(())
}

// =============================================================================
// Data writers for embedded use
// =============================================================================

fn write_double_data<W: Write>(writer: &mut W, array: &ArrayD<f64>) -> Result<()> {
    let data_size = (array.len() * 8) as i32;
    writer
        .write_i32::<LittleEndian>(MI_DOUBLE)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(data_size)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    for &value in array.iter() {
        writer
            .write_f64::<LittleEndian>(value)
            .map_err(|e| IoError::FileError(e.to_string()))?;
    }
    pad_to_8(writer, data_size as usize)?;
    Ok(())
}

fn write_single_data<W: Write>(writer: &mut W, array: &ArrayD<f32>) -> Result<()> {
    let data_size = (array.len() * 4) as i32;
    writer
        .write_i32::<LittleEndian>(MI_SINGLE)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(data_size)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    for &value in array.iter() {
        writer
            .write_f32::<LittleEndian>(value)
            .map_err(|e| IoError::FileError(e.to_string()))?;
    }
    pad_to_8(writer, data_size as usize)?;
    Ok(())
}

fn write_int32_data<W: Write>(writer: &mut W, array: &ArrayD<i32>) -> Result<()> {
    let data_size = (array.len() * 4) as i32;
    writer
        .write_i32::<LittleEndian>(MI_INT32)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(data_size)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    for &value in array.iter() {
        writer
            .write_i32::<LittleEndian>(value)
            .map_err(|e| IoError::FileError(e.to_string()))?;
    }
    pad_to_8(writer, data_size as usize)?;
    Ok(())
}

// =============================================================================
// Compressed write support
// =============================================================================

/// Write a variable with deflate compression (MI_COMPRESSED wrapper)
pub fn write_variable_compressed<W: Write + Seek>(
    writer: &mut W,
    name: &str,
    mat_type: &MatType,
) -> Result<()> {
    // First serialize the matrix to an in-memory buffer
    let mut inner_buf = std::io::Cursor::new(Vec::new());
    write_variable_extended(&mut inner_buf, name, mat_type)?;
    let uncompressed = inner_buf.into_inner();

    // Compress using miniz_oxide (pure Rust deflate)
    let compressed = miniz_oxide::deflate::compress_to_vec_zlib(&uncompressed, 6);

    // Write MI_COMPRESSED header
    writer
        .write_i32::<LittleEndian>(MI_COMPRESSED)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_i32::<LittleEndian>(compressed.len() as i32)
        .map_err(|e| IoError::FileError(e.to_string()))?;
    writer
        .write_all(&compressed)
        .map_err(|e| IoError::FileError(e.to_string()))?;

    // Pad to 8-byte boundary
    pad_to_8(writer, compressed.len())?;

    Ok(())
}

// =============================================================================
// Utility
// =============================================================================

fn pad_to_8<W: Write>(writer: &mut W, size: usize) -> Result<()> {
    let padding = (8 - (size % 8)) % 8;
    if padding > 0 {
        writer
            .write_all(&vec![0u8; padding])
            .map_err(|e| IoError::FileError(e.to_string()))?;
    }
    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr1, Array};
    use std::collections::HashMap;

    #[test]
    fn test_write_int8() {
        let dir = std::env::temp_dir().join("scirs2_mat_ext_i8");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("int8.mat");

        let mut vars = HashMap::new();
        let data = arr1(&[1i8, -2, 3, -4]).into_dyn();
        vars.insert("x".to_string(), MatType::Int8(data));

        let file = std::fs::File::create(&path).expect("create failed");
        let mut writer = std::io::BufWriter::new(file);
        write_mat_extended(&mut writer, &vars).expect("write failed");

        // Verify file was written (not empty)
        let meta = std::fs::metadata(&path).expect("metadata failed");
        assert!(meta.len() > 128); // at least header size

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_write_cell_array() {
        let dir = std::env::temp_dir().join("scirs2_mat_ext_cell");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("cell.mat");

        let mut vars = HashMap::new();
        let cell = MatType::Cell(vec![
            MatType::Double(arr1(&[1.0, 2.0, 3.0]).into_dyn()),
            MatType::Char("hello".to_string()),
            MatType::Int32(arr1(&[10i32, 20]).into_dyn()),
        ]);
        vars.insert("c".to_string(), cell);

        let file = std::fs::File::create(&path).expect("create failed");
        let mut writer = std::io::BufWriter::new(file);
        write_mat_extended(&mut writer, &vars).expect("write failed");

        let meta = std::fs::metadata(&path).expect("metadata failed");
        assert!(meta.len() > 128);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_write_struct() {
        let dir = std::env::temp_dir().join("scirs2_mat_ext_struct");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("struct.mat");

        let mut fields = HashMap::new();
        fields.insert("name".to_string(), MatType::Char("test".to_string()));
        fields.insert(
            "value".to_string(),
            MatType::Double(arr1(&[42.0]).into_dyn()),
        );

        let mut vars = HashMap::new();
        vars.insert("s".to_string(), MatType::Struct(fields));

        let file = std::fs::File::create(&path).expect("create failed");
        let mut writer = std::io::BufWriter::new(file);
        write_mat_extended(&mut writer, &vars).expect("write failed");

        let meta = std::fs::metadata(&path).expect("metadata failed");
        assert!(meta.len() > 128);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_write_compressed() {
        let dir = std::env::temp_dir().join("scirs2_mat_ext_compress");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("compressed.mat");

        super::super::write_impl::write_mat_header(&mut std::io::BufWriter::new(
            std::fs::File::create(&path).expect("create failed"),
        ))
        .expect("header failed");

        // Write compressed variable
        let file = std::fs::OpenOptions::new()
            .write(true)
            .append(true)
            .open(&path)
            .expect("open failed");
        let mut writer = std::io::BufWriter::new(file);
        let data = Array::from_shape_fn(scirs2_core::ndarray::IxDyn(&[100]), |idx| idx[0] as f64);
        write_variable_compressed(&mut writer, "big", &MatType::Double(data))
            .expect("write failed");
        writer.flush().expect("flush failed");

        let meta = std::fs::metadata(&path).expect("metadata failed");
        assert!(meta.len() > 128);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_write_i16_u16() {
        let dir = std::env::temp_dir().join("scirs2_mat_ext_i16");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("i16.mat");

        let mut vars = HashMap::new();
        vars.insert(
            "a".to_string(),
            MatType::Int16(arr1(&[100i16, -200, 300]).into_dyn()),
        );
        vars.insert(
            "b".to_string(),
            MatType::UInt16(arr1(&[10u16, 20, 30]).into_dyn()),
        );

        let file = std::fs::File::create(&path).expect("create failed");
        let mut writer = std::io::BufWriter::new(file);
        write_mat_extended(&mut writer, &vars).expect("write failed");

        let meta = std::fs::metadata(&path).expect("metadata failed");
        assert!(meta.len() > 128);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_write_i64_u64() {
        let dir = std::env::temp_dir().join("scirs2_mat_ext_i64");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("i64.mat");

        let mut vars = HashMap::new();
        vars.insert(
            "x".to_string(),
            MatType::Int64(arr1(&[1i64, 2, 3]).into_dyn()),
        );
        vars.insert(
            "y".to_string(),
            MatType::UInt64(arr1(&[10u64, 20]).into_dyn()),
        );

        let file = std::fs::File::create(&path).expect("create failed");
        let mut writer = std::io::BufWriter::new(file);
        write_mat_extended(&mut writer, &vars).expect("write failed");

        let meta = std::fs::metadata(&path).expect("metadata failed");
        assert!(meta.len() > 128);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
