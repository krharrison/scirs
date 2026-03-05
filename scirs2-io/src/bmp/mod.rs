//! Pure Rust BMP (Bitmap) image file format support
//!
//! Provides reading and writing of uncompressed 24-bit RGB BMP files
//! without any external image library dependencies.
//!
//! # Features
//!
//! - Read 24-bit uncompressed BMP files
//! - Write 24-bit uncompressed BMP files
//! - Pure Rust implementation (no C dependencies)
//! - Proper row padding handling (rows padded to 4-byte boundaries)
//! - Bottom-up and top-down image orientation
//!
//! # Examples
//!
//! ```rust,no_run
//! use scirs2_io::bmp::{read_bmp, write_bmp};
//!
//! // Read a BMP file
//! let bmp = read_bmp("image.bmp").expect("Read failed");
//! println!("Image: {}x{}", bmp.width, bmp.height);
//!
//! // Access pixel (row=0, col=0)
//! let r = bmp.pixels[0];
//! let g = bmp.pixels[1];
//! let b = bmp.pixels[2];
//!
//! // Write a BMP file
//! write_bmp("output.bmp", &bmp.pixels, bmp.width, bmp.height).expect("Write failed");
//! ```

mod reader;
mod writer;

pub use reader::read_bmp;
pub use writer::write_bmp;

/// BMP image data
#[derive(Debug, Clone)]
pub struct BmpImage {
    /// Pixel data in RGB order, row-major: [row0_r, row0_g, row0_b, row0_r, ...]
    /// Total length = width * height * 3
    pub pixels: Vec<u8>,
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
}

impl BmpImage {
    /// Create a new BMP image with the given dimensions filled with a color
    pub fn new(width: u32, height: u32, r: u8, g: u8, b: u8) -> Self {
        let pixel_count = (width * height) as usize;
        let mut pixels = Vec::with_capacity(pixel_count * 3);
        for _ in 0..pixel_count {
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
        BmpImage {
            pixels,
            width,
            height,
        }
    }

    /// Get pixel at (x, y) as (r, g, b)
    pub fn get_pixel(&self, x: u32, y: u32) -> Option<(u8, u8, u8)> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = ((y * self.width + x) * 3) as usize;
        if idx + 2 >= self.pixels.len() {
            return None;
        }
        Some((self.pixels[idx], self.pixels[idx + 1], self.pixels[idx + 2]))
    }

    /// Set pixel at (x, y)
    pub fn set_pixel(&mut self, x: u32, y: u32, r: u8, g: u8, b: u8) {
        if x < self.width && y < self.height {
            let idx = ((y * self.width + x) * 3) as usize;
            if idx + 2 < self.pixels.len() {
                self.pixels[idx] = r;
                self.pixels[idx + 1] = g;
                self.pixels[idx + 2] = b;
            }
        }
    }
}

/// BMP file header (14 bytes)
#[derive(Debug, Clone)]
pub(crate) struct BmpFileHeader {
    /// File size
    pub file_size: u32,
    /// Offset to pixel data
    pub pixel_offset: u32,
}

/// BMP info header (BITMAPINFOHEADER, 40 bytes)
#[derive(Debug, Clone)]
pub(crate) struct BmpInfoHeader {
    /// Width in pixels
    pub width: i32,
    /// Height in pixels (negative = top-down)
    pub height: i32,
    /// Bits per pixel
    pub bits_per_pixel: u16,
    /// Compression type (0 = uncompressed)
    pub compression: u32,
    /// Image data size (can be 0 for uncompressed)
    pub image_size: u32,
}

/// Calculate row stride with padding to 4-byte boundary
pub(crate) fn row_stride(width: u32, bits_per_pixel: u16) -> usize {
    let raw_bytes = (width as usize * bits_per_pixel as usize + 7) / 8;
    (raw_bytes + 3) & !3 // round up to 4-byte boundary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bmp_roundtrip_solid_color() {
        let dir = std::env::temp_dir().join("scirs2_bmp_test_solid");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("solid.bmp");

        let img = BmpImage::new(10, 10, 255, 0, 128);
        write_bmp(&path, &img.pixels, img.width, img.height).expect("Write failed");

        let loaded = read_bmp(&path).expect("Read failed");
        assert_eq!(loaded.width, 10);
        assert_eq!(loaded.height, 10);
        assert_eq!(loaded.pixels.len(), 10 * 10 * 3);

        // Check all pixels are the same color
        for y in 0..10 {
            for x in 0..10 {
                let (r, g, b) = loaded.get_pixel(x, y).expect("Pixel out of range");
                assert_eq!((r, g, b), (255, 0, 128), "Pixel mismatch at ({}, {})", x, y);
            }
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_bmp_roundtrip_gradient() {
        let dir = std::env::temp_dir().join("scirs2_bmp_test_gradient");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("gradient.bmp");

        let width = 256u32;
        let height = 1u32;
        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for x in 0..width {
            pixels.push(x as u8); // R
            pixels.push(0); // G
            pixels.push(0); // B
        }

        write_bmp(&path, &pixels, width, height).expect("Write failed");
        let loaded = read_bmp(&path).expect("Read failed");

        assert_eq!(loaded.width, 256);
        assert_eq!(loaded.height, 1);

        for x in 0..256u32 {
            let (r, _g, _b) = loaded.get_pixel(x, 0).expect("Pixel out of range");
            assert_eq!(r, x as u8, "Red mismatch at x={}", x);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_bmp_odd_width_padding() {
        // Width=3 means 9 bytes per row, padded to 12
        let dir = std::env::temp_dir().join("scirs2_bmp_test_padding");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("odd_width.bmp");

        let width = 3u32;
        let height = 2u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        // Set distinct colors
        // Row 0
        pixels[0] = 255;
        pixels[1] = 0;
        pixels[2] = 0; // red
        pixels[3] = 0;
        pixels[4] = 255;
        pixels[5] = 0; // green
        pixels[6] = 0;
        pixels[7] = 0;
        pixels[8] = 255; // blue
                         // Row 1
        pixels[9] = 128;
        pixels[10] = 128;
        pixels[11] = 0;
        pixels[12] = 0;
        pixels[13] = 128;
        pixels[14] = 128;
        pixels[15] = 128;
        pixels[16] = 0;
        pixels[17] = 128;

        write_bmp(&path, &pixels, width, height).expect("Write failed");
        let loaded = read_bmp(&path).expect("Read failed");

        assert_eq!(loaded.get_pixel(0, 0), Some((255, 0, 0)));
        assert_eq!(loaded.get_pixel(1, 0), Some((0, 255, 0)));
        assert_eq!(loaded.get_pixel(2, 0), Some((0, 0, 255)));
        assert_eq!(loaded.get_pixel(0, 1), Some((128, 128, 0)));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_bmp_large_image() {
        let dir = std::env::temp_dir().join("scirs2_bmp_test_large");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("large.bmp");

        let width = 200u32;
        let height = 150u32;
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        // Create a pattern
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                pixels[idx] = (x % 256) as u8;
                pixels[idx + 1] = (y % 256) as u8;
                pixels[idx + 2] = ((x + y) % 256) as u8;
            }
        }

        write_bmp(&path, &pixels, width, height).expect("Write failed");
        let loaded = read_bmp(&path).expect("Read failed");

        assert_eq!(loaded.width, width);
        assert_eq!(loaded.height, height);
        assert_eq!(loaded.pixels, pixels);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_bmp_1x1() {
        let dir = std::env::temp_dir().join("scirs2_bmp_test_1x1");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("tiny.bmp");

        let pixels = vec![42u8, 84, 126];
        write_bmp(&path, &pixels, 1, 1).expect("Write failed");

        let loaded = read_bmp(&path).expect("Read failed");
        assert_eq!(loaded.width, 1);
        assert_eq!(loaded.height, 1);
        assert_eq!(loaded.get_pixel(0, 0), Some((42, 84, 126)));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_bmp_pixel_access() {
        let mut img = BmpImage::new(5, 5, 0, 0, 0);
        img.set_pixel(2, 3, 100, 200, 50);
        assert_eq!(img.get_pixel(2, 3), Some((100, 200, 50)));
        assert_eq!(img.get_pixel(0, 0), Some((0, 0, 0)));
        assert_eq!(img.get_pixel(10, 10), None);
    }

    #[test]
    fn test_bmp_invalid_pixel_count() {
        let dir = std::env::temp_dir().join("scirs2_bmp_test_invalid");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("invalid.bmp");

        // Wrong number of pixels
        let pixels = vec![0u8; 10]; // not width*height*3
        let result = write_bmp(&path, &pixels, 5, 5);
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_bmp_nonexistent_file() {
        let result = read_bmp("/nonexistent/path/image.bmp");
        assert!(result.is_err());
    }

    #[test]
    fn test_row_stride_calculation() {
        // Width=1, 24bpp: 3 bytes -> padded to 4
        assert_eq!(row_stride(1, 24), 4);
        // Width=2, 24bpp: 6 bytes -> padded to 8
        assert_eq!(row_stride(2, 24), 8);
        // Width=4, 24bpp: 12 bytes -> already aligned
        assert_eq!(row_stride(4, 24), 12);
        // Width=5, 24bpp: 15 bytes -> padded to 16
        assert_eq!(row_stride(5, 24), 16);
    }
}
