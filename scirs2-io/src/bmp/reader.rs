//! BMP file reader

use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::{IoError, Result};

use super::{row_stride, BmpFileHeader, BmpImage, BmpInfoHeader};

/// Read a BMP file and return the image data
///
/// Supports 24-bit uncompressed RGB BMP files.
/// Pixel data is returned in RGB order, row-major (top-to-bottom, left-to-right).
pub fn read_bmp<P: AsRef<Path>>(path: P) -> Result<BmpImage> {
    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut reader = BufReader::new(file);

    let file_header = read_file_header(&mut reader)?;
    let info_header = read_info_header(&mut reader)?;

    // Validate
    if info_header.bits_per_pixel != 24 {
        return Err(IoError::UnsupportedFormat(format!(
            "Only 24-bit BMP is supported, got {}-bit",
            info_header.bits_per_pixel
        )));
    }

    if info_header.compression != 0 {
        return Err(IoError::UnsupportedFormat(format!(
            "Only uncompressed BMP is supported, got compression={}",
            info_header.compression
        )));
    }

    let width = info_header.width.unsigned_abs();
    let height = info_header.height.unsigned_abs();
    let bottom_up = info_header.height > 0;

    // Seek to pixel data
    reader
        .seek(SeekFrom::Start(file_header.pixel_offset as u64))
        .map_err(|e| IoError::FileError(format!("Failed to seek to pixel data: {}", e)))?;

    // Read pixel data
    let stride = row_stride(width, 24);
    let mut pixels = vec![0u8; (width * height * 3) as usize];

    for row in 0..height {
        // BMP stores rows bottom-to-top by default
        let target_row = if bottom_up { height - 1 - row } else { row };
        let row_offset = (target_row * width * 3) as usize;

        // Read the row (BGR in BMP format)
        let mut row_buf = vec![0u8; stride];
        reader
            .read_exact(&mut row_buf)
            .map_err(|e| IoError::FormatError(format!("Failed to read row {}: {}", row, e)))?;

        // Convert BGR to RGB
        for x in 0..width as usize {
            let bmp_idx = x * 3;
            let out_idx = row_offset + x * 3;
            pixels[out_idx] = row_buf[bmp_idx + 2]; // R
            pixels[out_idx + 1] = row_buf[bmp_idx + 1]; // G
            pixels[out_idx + 2] = row_buf[bmp_idx]; // B
        }
    }

    Ok(BmpImage {
        pixels,
        width,
        height,
    })
}

/// Read BMP file header (14 bytes)
fn read_file_header<R: Read>(reader: &mut R) -> Result<BmpFileHeader> {
    let mut sig = [0u8; 2];
    reader
        .read_exact(&mut sig)
        .map_err(|e| IoError::FormatError(format!("Failed to read BMP signature: {}", e)))?;

    if sig != *b"BM" {
        return Err(IoError::FormatError(format!(
            "Not a BMP file: invalid signature 0x{:02x}{:02x}",
            sig[0], sig[1]
        )));
    }

    let file_size = reader
        .read_u32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read file size: {}", e)))?;

    // Skip reserved fields (4 bytes)
    let _reserved = reader
        .read_u32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read reserved: {}", e)))?;

    let pixel_offset = reader
        .read_u32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read pixel offset: {}", e)))?;

    Ok(BmpFileHeader {
        file_size,
        pixel_offset,
    })
}

/// Read BMP info header (BITMAPINFOHEADER)
fn read_info_header<R: Read>(reader: &mut R) -> Result<BmpInfoHeader> {
    let header_size = reader
        .read_u32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read header size: {}", e)))?;

    if header_size < 40 {
        return Err(IoError::FormatError(format!(
            "Unsupported BMP header size: {} (expected >= 40)",
            header_size
        )));
    }

    let width = reader
        .read_i32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read width: {}", e)))?;

    let height = reader
        .read_i32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read height: {}", e)))?;

    let _planes = reader
        .read_u16::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read planes: {}", e)))?;

    let bits_per_pixel = reader
        .read_u16::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read bits per pixel: {}", e)))?;

    let compression = reader
        .read_u32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read compression: {}", e)))?;

    let image_size = reader
        .read_u32::<LittleEndian>()
        .map_err(|e| IoError::FormatError(format!("Failed to read image size: {}", e)))?;

    // Skip remaining header fields (h_res, v_res, colors_used, colors_important)
    // = 4 * 4 = 16 bytes
    let mut skip_buf = [0u8; 16];
    reader
        .read_exact(&mut skip_buf)
        .map_err(|e| IoError::FormatError(format!("Failed to read remaining header: {}", e)))?;

    // If header is larger than 40 bytes, skip the extra
    if header_size > 40 {
        let extra = (header_size - 40) as usize;
        let mut extra_buf = vec![0u8; extra];
        reader.read_exact(&mut extra_buf).map_err(|e| {
            IoError::FormatError(format!("Failed to skip extra header bytes: {}", e))
        })?;
    }

    Ok(BmpInfoHeader {
        width,
        height,
        bits_per_pixel,
        compression,
        image_size,
    })
}
