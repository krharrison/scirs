//! WAV file format handling module
//!
//! This module provides comprehensive functionality for reading and writing WAV audio files
//! supporting PCM (8/16/24/32-bit) and IEEE float (32/64-bit) formats.
//!
//! # Supported Formats
//!
//! | Format | Bits | Read | Write |
//! |--------|------|------|-------|
//! | PCM    | 8    | Yes  | Yes   |
//! | PCM    | 16   | Yes  | Yes   |
//! | PCM    | 24   | Yes  | Yes   |
//! | PCM    | 32   | Yes  | Yes   |
//! | Float  | 32   | Yes  | Yes   |
//! | Float  | 64   | Yes  | Yes   |
//!
//! # Examples
//!
//! ```rust,no_run
//! use scirs2_io::wavfile::{read_wav, write_wav_config, WavWriteConfig, WavOutputFormat};
//! use scirs2_core::ndarray::Array2;
//! use std::path::Path;
//!
//! // Read a WAV file
//! let (header, data) = read_wav(Path::new("input.wav")).expect("Read failed");
//!
//! // Write as 16-bit PCM
//! let config = WavWriteConfig {
//!     format: WavOutputFormat::Pcm16,
//!     ..Default::default()
//! };
//! write_wav_config(Path::new("output.wav"), header.sample_rate, &data.into_dyn(), config)
//!     .expect("Write failed");
//! ```

/// Sample rate conversion utilities for WAV audio data
///
/// Provides high-quality resampling using linear interpolation,
/// windowed sinc interpolation, and configurable quality presets.
pub mod resample;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use scirs2_core::ndarray::ArrayD;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::error::{IoError, Result};

/// WAV audio format codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WavFormat {
    /// PCM format (1)
    Pcm = 1,
    /// IEEE float format (3)
    Float = 3,
    /// A-law format (6)
    Alaw = 6,
    /// Mu-law format (7)
    Mulaw = 7,
}

impl TryFrom<u16> for WavFormat {
    type Error = IoError;

    fn try_from(value: u16) -> std::result::Result<Self, Self::Error> {
        match value {
            1 => Ok(WavFormat::Pcm),
            3 => Ok(WavFormat::Float),
            6 => Ok(WavFormat::Alaw),
            7 => Ok(WavFormat::Mulaw),
            _ => Err(IoError::FormatError(format!(
                "Unknown WAV format code: {}",
                value
            ))),
        }
    }
}

/// WAV file header information
#[derive(Debug, Clone)]
pub struct WavHeader {
    /// WAV format type
    pub format: WavFormat,
    /// Number of channels
    pub channels: u16,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Bits per sample
    pub bits_per_sample: u16,
    /// Total number of samples per channel
    pub samples_per_channel: usize,
}

/// Output format for writing WAV files
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WavOutputFormat {
    /// 8-bit unsigned PCM
    Pcm8,
    /// 16-bit signed PCM
    Pcm16,
    /// 24-bit signed PCM
    Pcm24,
    /// 32-bit signed PCM
    Pcm32,
    /// 32-bit IEEE float
    #[default]
    Float32,
    /// 64-bit IEEE float
    Float64,
}

impl WavOutputFormat {
    /// WAV format code for this output format
    fn format_code(&self) -> u16 {
        match self {
            WavOutputFormat::Pcm8
            | WavOutputFormat::Pcm16
            | WavOutputFormat::Pcm24
            | WavOutputFormat::Pcm32 => 1,
            WavOutputFormat::Float32 | WavOutputFormat::Float64 => 3,
        }
    }

    /// Bits per sample for this format
    fn bits_per_sample(&self) -> u16 {
        match self {
            WavOutputFormat::Pcm8 => 8,
            WavOutputFormat::Pcm16 => 16,
            WavOutputFormat::Pcm24 => 24,
            WavOutputFormat::Pcm32 => 32,
            WavOutputFormat::Float32 => 32,
            WavOutputFormat::Float64 => 64,
        }
    }
}

/// Configuration for writing WAV files
#[derive(Debug, Clone)]
pub struct WavWriteConfig {
    /// Output format
    pub format: WavOutputFormat,
}

impl Default for WavWriteConfig {
    fn default() -> Self {
        WavWriteConfig {
            format: WavOutputFormat::Float32,
        }
    }
}

/// RIFF chunk type
#[derive(Debug, Clone, PartialEq, Eq)]
struct RiffChunk {
    /// Chunk ID (4 bytes)
    id: [u8; 4],
    /// Chunk size
    size: u32,
}

impl RiffChunk {
    /// Read a RIFF chunk from a reader
    fn read<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut id = [0u8; 4];
        reader.read_exact(&mut id)?;
        let size = reader.read_u32::<LittleEndian>()?;
        Ok(RiffChunk { id, size })
    }

    /// Check if the chunk ID matches a given string
    fn is_id(&self, id: &str) -> bool {
        id.as_bytes() == self.id
    }
}

/// Reads a WAV file
///
/// Returns (header, data) where data is a 2D array [channels, samples] of f32 values
/// normalized to the range [-1.0, 1.0].
///
/// # Arguments
///
/// * `path` - Path to the WAV file
///
/// # Supported formats
///
/// - PCM 8-bit (unsigned)
/// - PCM 16-bit (signed)
/// - PCM 24-bit (signed)
/// - PCM 32-bit (signed)
/// - IEEE float 32-bit
/// - IEEE float 64-bit
pub fn read_wav<P: AsRef<Path>>(path: P) -> Result<(WavHeader, ArrayD<f32>)> {
    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut reader = BufReader::new(file);

    // Read RIFF chunk
    let riff_chunk = RiffChunk::read(&mut reader)
        .map_err(|e| IoError::FormatError(format!("Failed to read RIFF chunk: {}", e)))?;

    if !riff_chunk.is_id("RIFF") {
        return Err(IoError::FormatError("Not a RIFF file".to_string()));
    }

    // Read format (should be "WAVE")
    let mut format = [0u8; 4];
    reader
        .read_exact(&mut format)
        .map_err(|e| IoError::FormatError(format!("Failed to read WAVE format: {}", e)))?;

    if format != *b"WAVE" {
        return Err(IoError::FormatError("Not a WAVE file".to_string()));
    }

    // Find and read fmt chunk (may not be first)
    let (audio_format, channels, sample_rate, bits_per_sample) = loop {
        let chunk = RiffChunk::read(&mut reader)
            .map_err(|e| IoError::FormatError(format!("Failed to read chunk: {}", e)))?;

        if chunk.is_id("fmt ") {
            let af = reader
                .read_u16::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read audio format: {}", e)))?;

            let ch = reader.read_u16::<LittleEndian>().map_err(|e| {
                IoError::FormatError(format!("Failed to read channel count: {}", e))
            })?;

            let sr = reader
                .read_u32::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read sample rate: {}", e)))?;

            let _byte_rate = reader
                .read_u32::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read byte rate: {}", e)))?;

            let _block_align = reader
                .read_u16::<LittleEndian>()
                .map_err(|e| IoError::FormatError(format!("Failed to read block align: {}", e)))?;

            let bps = reader.read_u16::<LittleEndian>().map_err(|e| {
                IoError::FormatError(format!("Failed to read bits per sample: {}", e))
            })?;

            // Skip any extra fmt data
            if chunk.size > 16 {
                let extra_bytes = chunk.size as usize - 16;
                let mut extra_data = vec![0u8; extra_bytes];
                reader.read_exact(&mut extra_data).map_err(|e| {
                    IoError::FormatError(format!("Failed to read extra fmt data: {}", e))
                })?;
            }

            break (af, ch, sr, bps);
        }

        // Skip this chunk
        reader
            .seek(SeekFrom::Current(chunk.size as i64))
            .map_err(|e| IoError::FormatError(format!("Failed to skip chunk: {}", e)))?;
    };

    let wav_format = WavFormat::try_from(audio_format)?;

    // Find data chunk
    let data_size;
    loop {
        let chunk = RiffChunk::read(&mut reader)
            .map_err(|e| IoError::FormatError(format!("Failed to read chunk: {}", e)))?;

        if chunk.is_id("data") {
            data_size = chunk.size;
            break;
        }

        // Skip this chunk
        reader
            .seek(SeekFrom::Current(chunk.size as i64))
            .map_err(|e| IoError::FormatError(format!("Failed to skip chunk: {}", e)))?;
    }

    // Calculate number of samples
    let bytes_per_sample = bits_per_sample / 8;
    let samples_per_channel = (data_size / (channels as u32 * bytes_per_sample as u32)) as usize;

    // Read audio data
    let shape = scirs2_core::ndarray::IxDyn(&[channels as usize, samples_per_channel]);
    let mut data: scirs2_core::ndarray::ArrayD<f32> = scirs2_core::ndarray::Array::zeros(shape);

    read_samples(
        &mut reader,
        &mut data,
        wav_format,
        channels,
        samples_per_channel,
        bits_per_sample,
    )?;

    let header = WavHeader {
        format: wav_format,
        channels,
        sample_rate,
        bits_per_sample,
        samples_per_channel,
    };

    Ok((header, data))
}

/// Read samples from the WAV data section
fn read_samples<R: Read>(
    reader: &mut R,
    data: &mut ArrayD<f32>,
    format: WavFormat,
    channels: u16,
    samples_per_channel: usize,
    bits_per_sample: u16,
) -> Result<()> {
    match bits_per_sample {
        8 => {
            for sample_idx in 0..samples_per_channel {
                for ch in 0..channels as usize {
                    let byte = reader.read_u8().map_err(|e| {
                        IoError::FormatError(format!("Failed to read 8-bit sample: {}", e))
                    })?;
                    data[[ch, sample_idx]] = (byte as f32 - 128.0) / 127.0;
                }
            }
        }
        16 => {
            for sample_idx in 0..samples_per_channel {
                for ch in 0..channels as usize {
                    let sample = reader.read_i16::<LittleEndian>().map_err(|e| {
                        IoError::FormatError(format!("Failed to read 16-bit sample: {}", e))
                    })?;
                    data[[ch, sample_idx]] = sample as f32 / 32767.0;
                }
            }
        }
        24 => {
            for sample_idx in 0..samples_per_channel {
                for ch in 0..channels as usize {
                    let mut bytes = [0u8; 3];
                    reader.read_exact(&mut bytes).map_err(|e| {
                        IoError::FormatError(format!("Failed to read 24-bit sample: {}", e))
                    })?;
                    let sample = if bytes[2] & 0x80 != 0 {
                        ((bytes[2] as i32) << 16)
                            | ((bytes[1] as i32) << 8)
                            | (bytes[0] as i32)
                            | 0xFF000000u32 as i32
                    } else {
                        ((bytes[2] as i32) << 16) | ((bytes[1] as i32) << 8) | (bytes[0] as i32)
                    };
                    data[[ch, sample_idx]] = sample as f32 / 8388607.0;
                }
            }
        }
        32 => {
            if format == WavFormat::Float {
                for sample_idx in 0..samples_per_channel {
                    for ch in 0..channels as usize {
                        let sample = reader.read_f32::<LittleEndian>().map_err(|e| {
                            IoError::FormatError(format!("Failed to read float32 sample: {}", e))
                        })?;
                        data[[ch, sample_idx]] = sample;
                    }
                }
            } else {
                for sample_idx in 0..samples_per_channel {
                    for ch in 0..channels as usize {
                        let sample = reader.read_i32::<LittleEndian>().map_err(|e| {
                            IoError::FormatError(format!("Failed to read 32-bit sample: {}", e))
                        })?;
                        data[[ch, sample_idx]] = sample as f32 / 2147483647.0;
                    }
                }
            }
        }
        64 => {
            if format == WavFormat::Float {
                for sample_idx in 0..samples_per_channel {
                    for ch in 0..channels as usize {
                        let sample = reader.read_f64::<LittleEndian>().map_err(|e| {
                            IoError::FormatError(format!("Failed to read float64 sample: {}", e))
                        })?;
                        data[[ch, sample_idx]] = sample as f32;
                    }
                }
            } else {
                return Err(IoError::FormatError(
                    "64-bit PCM is not supported, only 64-bit float".to_string(),
                ));
            }
        }
        _ => {
            return Err(IoError::FormatError(format!(
                "Unsupported bits per sample: {}",
                bits_per_sample
            )));
        }
    }
    Ok(())
}

/// Writes audio data to a WAV file using default format (32-bit float).
///
/// Data is expected to be in format [channels, samples] with values in [-1.0, 1.0].
pub fn write_wav<P: AsRef<Path>>(path: P, samplerate: u32, data: &ArrayD<f32>) -> Result<()> {
    write_wav_config(path, samplerate, data, WavWriteConfig::default())
}

/// Writes audio data to a WAV file with configurable output format.
///
/// # Arguments
///
/// * `path` - Output file path
/// * `samplerate` - Sample rate in Hz
/// * `data` - Audio data [channels, samples] with values in [-1.0, 1.0]
/// * `config` - Write configuration (format selection)
pub fn write_wav_config<P: AsRef<Path>>(
    path: P,
    samplerate: u32,
    data: &ArrayD<f32>,
    config: WavWriteConfig,
) -> Result<()> {
    if data.ndim() < 2 {
        return Err(IoError::FormatError(
            "Audio data must be at least 2D (channels, samples)".to_string(),
        ));
    }

    let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut writer = BufWriter::new(file);

    let shape = data.shape();
    let channels = shape[0] as u16;
    let samples_per_channel = shape[1];

    let bits_per_sample = config.format.bits_per_sample();
    let bytes_per_sample = bits_per_sample / 8;
    let block_align = channels * bytes_per_sample;
    let byte_rate = samplerate * block_align as u32;
    let data_size = samples_per_channel * channels as usize * bytes_per_sample as usize;

    let format_code = config.format.format_code();

    // For float formats or >16-bit, we need an extended fmt chunk (size 18)
    let fmt_chunk_size: u32 = if format_code == 3 || bits_per_sample > 16 {
        18
    } else {
        16
    };

    // File size = 4 (WAVE) + 8 (fmt hdr) + fmt_chunk_size + 8 (data hdr) + data_size
    let file_size = 4 + 8 + fmt_chunk_size + 8 + data_size as u32;

    // Write RIFF header
    writer
        .write_all(b"RIFF")
        .map_err(|e| IoError::FileError(format!("Failed to write RIFF: {}", e)))?;
    writer
        .write_u32::<LittleEndian>(file_size)
        .map_err(|e| IoError::FileError(format!("Failed to write file size: {}", e)))?;
    writer
        .write_all(b"WAVE")
        .map_err(|e| IoError::FileError(format!("Failed to write WAVE: {}", e)))?;

    // Write fmt chunk
    writer
        .write_all(b"fmt ")
        .map_err(|e| IoError::FileError(format!("Failed to write fmt: {}", e)))?;
    writer
        .write_u32::<LittleEndian>(fmt_chunk_size)
        .map_err(|e| IoError::FileError(format!("Failed to write fmt size: {}", e)))?;
    writer
        .write_u16::<LittleEndian>(format_code)
        .map_err(|e| IoError::FileError(format!("Failed to write format code: {}", e)))?;
    writer
        .write_u16::<LittleEndian>(channels)
        .map_err(|e| IoError::FileError(format!("Failed to write channels: {}", e)))?;
    writer
        .write_u32::<LittleEndian>(samplerate)
        .map_err(|e| IoError::FileError(format!("Failed to write sample rate: {}", e)))?;
    writer
        .write_u32::<LittleEndian>(byte_rate)
        .map_err(|e| IoError::FileError(format!("Failed to write byte rate: {}", e)))?;
    writer
        .write_u16::<LittleEndian>(block_align)
        .map_err(|e| IoError::FileError(format!("Failed to write block align: {}", e)))?;
    writer
        .write_u16::<LittleEndian>(bits_per_sample)
        .map_err(|e| IoError::FileError(format!("Failed to write bits per sample: {}", e)))?;

    // Extended format chunk (cbSize = 0)
    if fmt_chunk_size > 16 {
        writer
            .write_u16::<LittleEndian>(0)
            .map_err(|e| IoError::FileError(format!("Failed to write cbSize: {}", e)))?;
    }

    // Write data chunk header
    writer
        .write_all(b"data")
        .map_err(|e| IoError::FileError(format!("Failed to write data chunk: {}", e)))?;
    writer
        .write_u32::<LittleEndian>(data_size as u32)
        .map_err(|e| IoError::FileError(format!("Failed to write data size: {}", e)))?;

    // Write samples
    write_samples(&mut writer, data, channels, samples_per_channel, &config)?;

    writer
        .flush()
        .map_err(|e| IoError::FileError(format!("Failed to flush: {}", e)))?;

    Ok(())
}

/// Write audio samples in the configured format
fn write_samples<W: Write>(
    writer: &mut W,
    data: &ArrayD<f32>,
    channels: u16,
    samples_per_channel: usize,
    config: &WavWriteConfig,
) -> Result<()> {
    match config.format {
        WavOutputFormat::Pcm8 => {
            for sample_idx in 0..samples_per_channel {
                for ch in 0..channels as usize {
                    let sample = data[[ch, sample_idx]].clamp(-1.0, 1.0);
                    let byte = ((sample * 127.0) + 128.0) as u8;
                    writer.write_u8(byte).map_err(|e| {
                        IoError::FileError(format!("Failed to write PCM8 sample: {}", e))
                    })?;
                }
            }
        }
        WavOutputFormat::Pcm16 => {
            for sample_idx in 0..samples_per_channel {
                for ch in 0..channels as usize {
                    let sample = data[[ch, sample_idx]].clamp(-1.0, 1.0);
                    let val = (sample * 32767.0) as i16;
                    writer.write_i16::<LittleEndian>(val).map_err(|e| {
                        IoError::FileError(format!("Failed to write PCM16 sample: {}", e))
                    })?;
                }
            }
        }
        WavOutputFormat::Pcm24 => {
            for sample_idx in 0..samples_per_channel {
                for ch in 0..channels as usize {
                    let sample = data[[ch, sample_idx]].clamp(-1.0, 1.0);
                    let val = (sample * 8388607.0) as i32;
                    let bytes = [
                        (val & 0xFF) as u8,
                        ((val >> 8) & 0xFF) as u8,
                        ((val >> 16) & 0xFF) as u8,
                    ];
                    writer.write_all(&bytes).map_err(|e| {
                        IoError::FileError(format!("Failed to write PCM24 sample: {}", e))
                    })?;
                }
            }
        }
        WavOutputFormat::Pcm32 => {
            for sample_idx in 0..samples_per_channel {
                for ch in 0..channels as usize {
                    let sample = data[[ch, sample_idx]].clamp(-1.0, 1.0);
                    let val = (sample as f64 * 2147483647.0) as i32;
                    writer.write_i32::<LittleEndian>(val).map_err(|e| {
                        IoError::FileError(format!("Failed to write PCM32 sample: {}", e))
                    })?;
                }
            }
        }
        WavOutputFormat::Float32 => {
            for sample_idx in 0..samples_per_channel {
                for ch in 0..channels as usize {
                    let sample = data[[ch, sample_idx]];
                    writer.write_f32::<LittleEndian>(sample).map_err(|e| {
                        IoError::FileError(format!("Failed to write float32 sample: {}", e))
                    })?;
                }
            }
        }
        WavOutputFormat::Float64 => {
            for sample_idx in 0..samples_per_channel {
                for ch in 0..channels as usize {
                    let sample = data[[ch, sample_idx]] as f64;
                    writer.write_f64::<LittleEndian>(sample).map_err(|e| {
                        IoError::FileError(format!("Failed to write float64 sample: {}", e))
                    })?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Helper to create a simple sine wave
    fn create_sine_wave(
        freq: f64,
        sample_rate: u32,
        duration_secs: f64,
        channels: usize,
    ) -> ArrayD<f32> {
        let num_samples = (sample_rate as f64 * duration_secs) as usize;
        let mut samples = Array2::zeros((channels, num_samples));
        for ch in 0..channels {
            for i in 0..num_samples {
                let t = i as f64 / sample_rate as f64;
                let phase_offset = ch as f64 * std::f64::consts::PI * 0.25;
                samples[[ch, i]] =
                    (2.0 * std::f64::consts::PI * freq * t + phase_offset).sin() as f32;
            }
        }
        samples.into_dyn()
    }

    #[test]
    fn test_wav_roundtrip_float32() {
        let dir = std::env::temp_dir().join("scirs2_wav_test_f32");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_f32.wav");

        let data = create_sine_wave(440.0, 44100, 0.1, 1);
        write_wav(&path, 44100, &data).expect("Write failed");

        let (header, loaded) = read_wav(&path).expect("Read failed");
        assert_eq!(header.sample_rate, 44100);
        assert_eq!(header.channels, 1);
        assert_eq!(header.format, WavFormat::Float);
        assert_eq!(header.bits_per_sample, 32);

        // Verify data
        let orig_slice = data.as_slice().expect("Not contiguous");
        let load_slice = loaded.as_slice().expect("Not contiguous");
        assert_eq!(orig_slice.len(), load_slice.len());
        for (a, b) in orig_slice.iter().zip(load_slice.iter()) {
            assert!((a - b).abs() < 1e-6, "Mismatch: {} vs {}", a, b);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_wav_roundtrip_pcm16() {
        let dir = std::env::temp_dir().join("scirs2_wav_test_pcm16");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_pcm16.wav");

        let data = create_sine_wave(440.0, 44100, 0.1, 1);
        let config = WavWriteConfig {
            format: WavOutputFormat::Pcm16,
        };
        write_wav_config(&path, 44100, &data, config).expect("Write failed");

        let (header, loaded) = read_wav(&path).expect("Read failed");
        assert_eq!(header.format, WavFormat::Pcm);
        assert_eq!(header.bits_per_sample, 16);

        // 16-bit PCM has quantization error
        let orig_slice = data.as_slice().expect("Not contiguous");
        let load_slice = loaded.as_slice().expect("Not contiguous");
        for (a, b) in orig_slice.iter().zip(load_slice.iter()) {
            assert!((a - b).abs() < 0.001, "Too much error: {} vs {}", a, b);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_wav_roundtrip_pcm8() {
        let dir = std::env::temp_dir().join("scirs2_wav_test_pcm8");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_pcm8.wav");

        let data = create_sine_wave(440.0, 44100, 0.1, 1);
        let config = WavWriteConfig {
            format: WavOutputFormat::Pcm8,
        };
        write_wav_config(&path, 44100, &data, config).expect("Write failed");

        let (header, loaded) = read_wav(&path).expect("Read failed");
        assert_eq!(header.format, WavFormat::Pcm);
        assert_eq!(header.bits_per_sample, 8);

        // 8-bit has large quantization error
        let orig_slice = data.as_slice().expect("Not contiguous");
        let load_slice = loaded.as_slice().expect("Not contiguous");
        for (a, b) in orig_slice.iter().zip(load_slice.iter()) {
            assert!((a - b).abs() < 0.02, "Too much error: {} vs {}", a, b);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_wav_roundtrip_pcm24() {
        let dir = std::env::temp_dir().join("scirs2_wav_test_pcm24");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_pcm24.wav");

        let data = create_sine_wave(440.0, 44100, 0.1, 1);
        let config = WavWriteConfig {
            format: WavOutputFormat::Pcm24,
        };
        write_wav_config(&path, 44100, &data, config).expect("Write failed");

        let (header, loaded) = read_wav(&path).expect("Read failed");
        assert_eq!(header.bits_per_sample, 24);

        let orig_slice = data.as_slice().expect("Not contiguous");
        let load_slice = loaded.as_slice().expect("Not contiguous");
        for (a, b) in orig_slice.iter().zip(load_slice.iter()) {
            assert!((a - b).abs() < 1e-4, "Too much error: {} vs {}", a, b);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_wav_roundtrip_pcm32() {
        let dir = std::env::temp_dir().join("scirs2_wav_test_pcm32");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_pcm32.wav");

        let data = create_sine_wave(440.0, 44100, 0.1, 1);
        let config = WavWriteConfig {
            format: WavOutputFormat::Pcm32,
        };
        write_wav_config(&path, 44100, &data, config).expect("Write failed");

        let (header, loaded) = read_wav(&path).expect("Read failed");
        assert_eq!(header.bits_per_sample, 32);
        assert_eq!(header.format, WavFormat::Pcm);

        let orig_slice = data.as_slice().expect("Not contiguous");
        let load_slice = loaded.as_slice().expect("Not contiguous");
        for (a, b) in orig_slice.iter().zip(load_slice.iter()) {
            assert!((a - b).abs() < 1e-5, "Too much error: {} vs {}", a, b);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_wav_roundtrip_float64() {
        let dir = std::env::temp_dir().join("scirs2_wav_test_f64");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_f64.wav");

        let data = create_sine_wave(440.0, 44100, 0.1, 1);
        let config = WavWriteConfig {
            format: WavOutputFormat::Float64,
        };
        write_wav_config(&path, 44100, &data, config).expect("Write failed");

        let (header, loaded) = read_wav(&path).expect("Read failed");
        assert_eq!(header.format, WavFormat::Float);
        assert_eq!(header.bits_per_sample, 64);

        // float64->float32 roundtrip has f32 precision
        let orig_slice = data.as_slice().expect("Not contiguous");
        let load_slice = loaded.as_slice().expect("Not contiguous");
        for (a, b) in orig_slice.iter().zip(load_slice.iter()) {
            assert!((a - b).abs() < 1e-6, "Mismatch: {} vs {}", a, b);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_wav_stereo() {
        let dir = std::env::temp_dir().join("scirs2_wav_test_stereo");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("stereo.wav");

        let data = create_sine_wave(440.0, 44100, 0.1, 2);
        write_wav(&path, 44100, &data).expect("Write failed");

        let (header, loaded) = read_wav(&path).expect("Read failed");
        assert_eq!(header.channels, 2);
        assert_eq!(loaded.shape(), data.shape());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_wav_invalid_dimensions() {
        let dir = std::env::temp_dir().join("scirs2_wav_test_invalid");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("invalid.wav");

        // 1D array should fail
        let data = scirs2_core::ndarray::arr1(&[1.0f32, 2.0, 3.0]).into_dyn();
        let result = write_wav(&path, 44100, &data);
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_wav_nonexistent_file() {
        let result = read_wav(Path::new("/nonexistent/path/audio.wav"));
        assert!(result.is_err());
    }

    #[test]
    fn test_wav_sample_rate_preserved() {
        let dir = std::env::temp_dir().join("scirs2_wav_test_sr");
        let _ = std::fs::create_dir_all(&dir);

        for &sr in &[8000u32, 22050, 44100, 48000, 96000] {
            let path = dir.join(format!("sr_{}.wav", sr));
            let data = create_sine_wave(440.0, sr, 0.05, 1);
            write_wav(&path, sr, &data).expect("Write failed");
            let (header, _) = read_wav(&path).expect("Read failed");
            assert_eq!(header.sample_rate, sr);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
