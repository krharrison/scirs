//! BMP file writer

use byteorder::{LittleEndian, WriteBytesExt};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::{IoError, Result};

use super::row_stride;

/// Write pixel data to a 24-bit BMP file
///
/// # Arguments
///
/// * `path` - Output file path
/// * `pixels` - Pixel data in RGB order, row-major. Length must be width * height * 3.
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
///
/// Writes an uncompressed 24-bit RGB BMP file with bottom-up row ordering.
pub fn write_bmp<P: AsRef<Path>>(path: P, pixels: &[u8], width: u32, height: u32) -> Result<()> {
    let expected_len = (width * height * 3) as usize;
    if pixels.len() != expected_len {
        return Err(IoError::FormatError(format!(
            "Pixel data length {} does not match {}x{}x3 = {}",
            pixels.len(),
            width,
            height,
            expected_len
        )));
    }

    let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut writer = BufWriter::new(file);

    let stride = row_stride(width, 24);
    let image_data_size = stride * height as usize;
    let file_size = 14 + 40 + image_data_size;

    // Write BMP file header (14 bytes)
    writer
        .write_all(b"BM")
        .map_err(|e| IoError::FileError(format!("Failed to write signature: {}", e)))?;
    writer
        .write_u32::<LittleEndian>(file_size as u32)
        .map_err(|e| IoError::FileError(format!("Failed to write file size: {}", e)))?;
    writer
        .write_u32::<LittleEndian>(0)
        .map_err(|e| IoError::FileError(format!("Failed to write reserved: {}", e)))?; // reserved
    writer
        .write_u32::<LittleEndian>(54)
        .map_err(|e| IoError::FileError(format!("Failed to write pixel offset: {}", e)))?; // 14 + 40

    // Write BITMAPINFOHEADER (40 bytes)
    writer
        .write_u32::<LittleEndian>(40)
        .map_err(|e| IoError::FileError(format!("Failed to write header size: {}", e)))?;
    writer
        .write_i32::<LittleEndian>(width as i32)
        .map_err(|e| IoError::FileError(format!("Failed to write width: {}", e)))?;
    writer
        .write_i32::<LittleEndian>(height as i32)
        .map_err(|e| IoError::FileError(format!("Failed to write height: {}", e)))?; // positive = bottom-up
    writer
        .write_u16::<LittleEndian>(1)
        .map_err(|e| IoError::FileError(format!("Failed to write planes: {}", e)))?; // planes
    writer
        .write_u16::<LittleEndian>(24)
        .map_err(|e| IoError::FileError(format!("Failed to write bpp: {}", e)))?; // bits per pixel
    writer
        .write_u32::<LittleEndian>(0)
        .map_err(|e| IoError::FileError(format!("Failed to write compression: {}", e)))?; // no compression
    writer
        .write_u32::<LittleEndian>(image_data_size as u32)
        .map_err(|e| IoError::FileError(format!("Failed to write image size: {}", e)))?;
    writer
        .write_i32::<LittleEndian>(2835)
        .map_err(|e| IoError::FileError(format!("Failed to write h_res: {}", e)))?; // 72 DPI
    writer
        .write_i32::<LittleEndian>(2835)
        .map_err(|e| IoError::FileError(format!("Failed to write v_res: {}", e)))?;
    writer
        .write_u32::<LittleEndian>(0)
        .map_err(|e| IoError::FileError(format!("Failed to write colors used: {}", e)))?;
    writer
        .write_u32::<LittleEndian>(0)
        .map_err(|e| IoError::FileError(format!("Failed to write colors important: {}", e)))?;

    // Write pixel data (bottom-up, BGR)
    let padding_bytes = stride - (width as usize * 3);
    let pad = vec![0u8; padding_bytes];

    for row in (0..height).rev() {
        let row_offset = (row * width * 3) as usize;
        for x in 0..width as usize {
            let idx = row_offset + x * 3;
            // Convert RGB to BGR
            writer
                .write_u8(pixels[idx + 2])
                .map_err(|e| IoError::FileError(format!("Failed to write B: {}", e)))?;
            writer
                .write_u8(pixels[idx + 1])
                .map_err(|e| IoError::FileError(format!("Failed to write G: {}", e)))?;
            writer
                .write_u8(pixels[idx])
                .map_err(|e| IoError::FileError(format!("Failed to write R: {}", e)))?;
        }
        // Write padding
        if !pad.is_empty() {
            writer
                .write_all(&pad)
                .map_err(|e| IoError::FileError(format!("Failed to write padding: {}", e)))?;
        }
    }

    writer
        .flush()
        .map_err(|e| IoError::FileError(format!("Failed to flush: {}", e)))?;

    Ok(())
}
