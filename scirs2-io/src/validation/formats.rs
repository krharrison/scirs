//! Format-specific validation utilities
//!
//! This module provides validators for specific file formats used in scientific computing.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use super::{FormatValidatorRegistry, ValidationSource};
use crate::error::{IoError, Result};

/// Common scientific data format types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataFormat {
    /// Comma-Separated Values
    CSV,
    /// Tab-Separated Values
    TSV,
    /// JavaScript Object Notation
    JSON,
    /// MATLAB .mat file
    MATLAB,
    /// Attribute-Relation File Format (ARFF)
    ARFF,
    /// HDF5 format
    HDF5,
    /// NetCDF format
    NetCDF,
    /// PNG image format
    PNG,
    /// JPEG image format
    JPEG,
    /// TIFF image format
    TIFF,
    /// WAV audio format
    WAV,
}

impl DataFormat {
    /// Get a string representation of the format
    pub fn as_str(&self) -> &'static str {
        match self {
            DataFormat::CSV => "CSV",
            DataFormat::TSV => "TSV",
            DataFormat::JSON => "JSON",
            DataFormat::MATLAB => "MATLAB",
            DataFormat::ARFF => "ARFF",
            DataFormat::HDF5 => "HDF5",
            DataFormat::NetCDF => "NetCDF",
            DataFormat::PNG => "PNG",
            DataFormat::JPEG => "JPEG",
            DataFormat::TIFF => "TIFF",
            DataFormat::WAV => "WAV",
        }
    }

    /// Parse format name from a string
    pub fn from_str(name: &str) -> Option<Self> {
        match name.to_uppercase().as_str() {
            "CSV" => Some(DataFormat::CSV),
            "TSV" => Some(DataFormat::TSV),
            "JSON" => Some(DataFormat::JSON),
            "MAT" | "MATLAB" => Some(DataFormat::MATLAB),
            "ARFF" => Some(DataFormat::ARFF),
            "HDF5" | "H5" => Some(DataFormat::HDF5),
            "NETCDF" | "NC" => Some(DataFormat::NetCDF),
            "PNG" => Some(DataFormat::PNG),
            "JPEG" | "JPG" => Some(DataFormat::JPEG),
            "TIFF" | "TIF" => Some(DataFormat::TIFF),
            "WAV" => Some(DataFormat::WAV),
            _ => None,
        }
    }
}

/// Get a registry with all scientific data format validators
#[allow(dead_code)]
pub fn get_scientific_format_validators() -> FormatValidatorRegistry {
    let mut registry = FormatValidatorRegistry::new();

    // Add format validators

    // PNG validator
    registry.add_validator("PNG", |data| {
        data.len() >= 8 && data[0..8] == [137, 80, 78, 71, 13, 10, 26, 10]
    });

    // JPEG validator
    registry.add_validator("JPEG", |data| {
        data.len() >= 3 && data[0..3] == [0xFF, 0xD8, 0xFF]
    });

    // TIFF validator
    registry.add_validator("TIFF", |data| {
        data.len() >= 4
            && (
                data[0..4] == [0x49, 0x49, 0x2A, 0x00] || // Little endian
            data[0..4] == [0x4D, 0x4D, 0x00, 0x2A]
                // Big endian
            )
    });

    // WAV validator
    registry.add_validator("WAV", |data| {
        data.len() >= 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WAVE"
    });

    // JSON validator
    registry.add_validator("JSON", |data| {
        if data.is_empty() {
            return false;
        }

        // Find the first non-whitespace character
        for (i, &byte) in data.iter().enumerate() {
            if !byte.is_ascii_whitespace() {
                // Check if it's { or [
                return byte == b'{' || byte == b'[' ||
                       // Or allow "key": value format in case it's a fragment
                       (byte == b'"' && data.len() > i + 2 && data[i+1..].contains(&b':'));
            }
        }

        false
    });

    // CSV validator
    registry.add_validator("CSV", |data| {
        // Basic validation: contains commas and has consistent structure
        if data.is_empty() || !data.contains(&b',') {
            return false;
        }

        // Check for newlines (files should have more than one line)
        if !data.contains(&b'\n') && !data.contains(&b'\r') {
            return false;
        }

        // Check for structure consistency by counting commas in first few lines
        let mut lines = data.split(|&b| b == b'\n');

        // Get the first line (skipping empty lines)
        let first_line = lines.find(|line| !line.is_empty()).unwrap_or(&[]);

        // Count commas in first line
        let comma_count = first_line.iter().filter(|&&b| b == b',').count();

        // Check that other lines have similar comma counts
        // (allow some variation for quoted fields)
        for line in lines.take(5) {
            if line.is_empty() {
                continue;
            }

            let line_comma_count = line.iter().filter(|&&b| b == b',').count();

            // Allow some variation, but not too much
            if (line_comma_count as isize - comma_count as isize).abs() > 2 {
                return false;
            }
        }

        true
    });

    // TSV validator
    registry.add_validator("TSV", |data| {
        // Similar to CSV but with tabs
        if data.is_empty() || !data.contains(&b'\t') {
            return false;
        }

        if !data.contains(&b'\n') && !data.contains(&b'\r') {
            return false;
        }

        // Check for structure consistency
        let mut lines = data.split(|&b| b == b'\n');

        let first_line = lines.find(|line| !line.is_empty()).unwrap_or(&[]);

        let tab_count = first_line.iter().filter(|&&b| b == b'\t').count();

        for line in lines.take(5) {
            if line.is_empty() {
                continue;
            }

            let line_tab_count = line.iter().filter(|&&b| b == b'\t').count();

            if (line_tab_count as isize - tab_count as isize).abs() > 2 {
                return false;
            }
        }

        true
    });

    // MATLAB .mat file validator
    registry.add_validator("MATLAB", |data| {
        // Check for MATLAB Level 5 MAT-file format
        if data.len() >= 128
            && (data[0..4] == [0x00, 0x01, 0x00, 0x00] || // Header for MATLAB versions < 7.3
            data[0..4] == [0x00, 0x01, 0x4D, 0x49])
        // Header for compressed MATLAB >= 7.3
        {
            // Check for "MATLAB" text in header
            return data[124..128].windows(6).any(|window| window == b"MATLAB");
        }

        false
    });

    // ARFF validator
    registry.add_validator("ARFF", |data| {
        if data.is_empty() {
            return false;
        }

        // Convert to string for easier parsing
        let mut buffer = Vec::new();
        buffer.extend_from_slice(data);

        // Try to parse as UTF-8, fall back to Latin-1
        let content = String::from_utf8(buffer).unwrap_or_else(|_| {
            // Fall back to Latin-1 encoding
            data.iter().map(|&b| b as char).collect()
        });

        // Check for ARFF header
        content.to_uppercase().contains("@RELATION")
            && content.to_uppercase().contains("@ATTRIBUTE")
            && content.to_uppercase().contains("@DATA")
    });

    // HDF5 validator
    registry.add_validator("HDF5", |data| {
        data.len() >= 8 && data[0..8] == [137, 72, 68, 70, 13, 10, 26, 10]
    });

    // NetCDF validator (basic signature check)
    registry.add_validator("NetCDF", |data| {
        data.len() >= 4 && &data[0..4] == b"CDF\x01" || &data[0..4] == b"CDF\x02"
    });

    registry
}

/// Validate a file against a specific format
#[allow(dead_code)]
pub fn validate_format<P: AsRef<Path>>(path: P, format: DataFormat) -> Result<bool> {
    let _path = path.as_ref();

    // Open file
    let file =
        File::open(_path).map_err(|e| IoError::FileError(format!("Failed to open file: {e}")))?;

    // Read first 8192 bytes for format detection
    let mut buffer = Vec::with_capacity(8192);
    file.take(8192)
        .read_to_end(&mut buffer)
        .map_err(|e| IoError::FileError(format!("Failed to read file: {e}")))?;

    // Get validators
    let registry = get_scientific_format_validators();

    // Find validator for the format
    for validator in registry.validators {
        if validator.format_name.eq_ignore_ascii_case(format.as_str()) {
            return Ok(validator.validate(&buffer));
        }
    }

    Err(IoError::ValidationError(format!(
        "No validator found for format: {}",
        format.as_str()
    )))
}

/// Detect the format of a file
#[allow(dead_code)]
pub fn detect_file_format<P: AsRef<Path>>(path: P) -> Result<Option<String>> {
    let _path = path.as_ref();

    // Use registry to validate format
    let registry = get_scientific_format_validators();
    registry.validate_format(ValidationSource::FilePath(_path))
}

/// Structure for validation result details
#[derive(Debug, Clone)]
pub struct FormatValidationResult {
    /// Whether the validation passed
    pub valid: bool,
    /// The format that was validated
    pub format: String,
    /// Path to the validated file
    pub file_path: String,
    /// Additional validation details
    pub details: Option<String>,
}

/// Perform comprehensive format validation on a file
///
/// This function performs format-specific validation beyond
/// just the basic format detection.
#[allow(dead_code)]
pub fn validate_file_format<P: AsRef<Path>>(
    path: P,
    format: DataFormat,
) -> Result<FormatValidationResult> {
    let path = path.as_ref();

    // First check basic format signature
    let basic_valid = validate_format(path, format)?;

    if !basic_valid {
        return Ok(FormatValidationResult {
            valid: false,
            format: format.as_str().to_string(),
            file_path: path.to_string_lossy().to_string(),
            details: Some("File does not have the correct format signature".to_string()),
        });
    }

    // For some formats, perform more detailed validation
    match format {
        DataFormat::CSV => validate_csv_format(path),
        DataFormat::JSON => validate_json_format(path),
        DataFormat::ARFF => validate_arff_format(path),
        DataFormat::WAV => validate_wav_format(path),
        _ => {
            // For other formats, basic validation is sufficient for now
            Ok(FormatValidationResult {
                valid: true,
                format: format.as_str().to_string(),
                file_path: path.to_string_lossy().to_string(),
                details: None,
            })
        }
    }
}

/// Validate CSV file structure in detail
#[allow(dead_code)]
fn validate_csv_format<P: AsRef<Path>>(path: P) -> Result<FormatValidationResult> {
    let _path = path.as_ref();

    // Open file
    let file =
        File::open(_path).map_err(|e| IoError::FileError(format!("Failed to open file: {e}")))?;

    let mut reader = BufReader::new(file);
    let mut content = Vec::new();
    reader
        .read_to_end(&mut content)
        .map_err(|e| IoError::FileError(format!("Failed to read file: {e}")))?;

    if content.is_empty() {
        return Ok(FormatValidationResult {
            valid: false,
            format: "CSV".to_string(),
            file_path: path.as_ref().to_string_lossy().to_string(),
            details: Some("File is empty".to_string()),
        });
    }

    // Check for consistent number of fields
    let mut lines = content
        .split(|&b| b == b'\n' || b == b'\r')
        .filter(|line| !line.is_empty());

    // Get field count from first line
    let first_line = match lines.next() {
        Some(line) => line,
        None => {
            return Ok(FormatValidationResult {
                valid: false,
                format: "CSV".to_string(),
                file_path: path.as_ref().to_string_lossy().to_string(),
                details: Some("File has no content".to_string()),
            });
        }
    };

    // Count fields in first line (accounting for quoted fields)
    let first_field_count = count_csv_fields(first_line);

    // Check remaining lines for consistency
    let mut inconsistent_lines = Vec::new();

    for (i, line) in lines.enumerate() {
        let field_count = count_csv_fields(line);
        let line_number = i + 2;

        if field_count != first_field_count {
            inconsistent_lines.push(line_number);
        }
    }

    if inconsistent_lines.is_empty() {
        Ok(FormatValidationResult {
            valid: true,
            format: "CSV".to_string(),
            file_path: path.as_ref().to_string_lossy().to_string(),
            details: Some(format!(
                "CSV file with {} fields per line",
                first_field_count
            )),
        })
    } else {
        // Report up to 5 inconsistent lines
        let inconsistent_report = if inconsistent_lines.len() <= 5 {
            format!(
                "Lines with inconsistent field counts: {}",
                inconsistent_lines
                    .iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        } else {
            format!(
                "Lines with inconsistent field counts: {} (and {} more)",
                inconsistent_lines
                    .iter()
                    .take(5)
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                inconsistent_lines.len() - 5
            )
        };

        Ok(FormatValidationResult {
            valid: false,
            format: "CSV".to_string(),
            file_path: path.as_ref().to_string_lossy().to_string(),
            details: Some(format!(
                "Inconsistent field counts. First line has {} fields. {}",
                first_field_count, inconsistent_report
            )),
        })
    }
}

/// Count fields in a CSV line, accounting for quoted fields
#[allow(dead_code)]
fn count_csv_fields(line: &[u8]) -> usize {
    let mut count = 1; // Start at 1 because field count = comma count + 1
    let mut in_quotes = false;

    for &b in line {
        match b {
            b'"' => {
                // Toggle quote state
                in_quotes = !in_quotes;
            }
            b','
                // Only count commas outside quotes
                if !in_quotes => {
                    count += 1;
                }
            _ => {}
        }
    }

    count
}

/// Validate JSON file structure in detail
#[allow(dead_code)]
fn validate_json_format<P: AsRef<Path>>(path: P) -> Result<FormatValidationResult> {
    let _path = path.as_ref();

    // Open and attempt to parse as JSON
    let file =
        File::open(_path).map_err(|e| IoError::FileError(format!("Failed to open file: {e}")))?;

    let reader = BufReader::new(file);

    match serde_json::from_reader::<_, serde_json::Value>(reader) {
        Ok(_) => Ok(FormatValidationResult {
            valid: true,
            format: "JSON".to_string(),
            file_path: path.as_ref().to_string_lossy().to_string(),
            details: Some("Valid JSON structure".to_string()),
        }),
        Err(e) => Ok(FormatValidationResult {
            valid: false,
            format: "JSON".to_string(),
            file_path: path.as_ref().to_string_lossy().to_string(),
            details: Some(format!("Invalid JSON: {}", e)),
        }),
    }
}

/// Validate ARFF file structure in detail
#[allow(dead_code)]
fn validate_arff_format<P: AsRef<Path>>(path: P) -> Result<FormatValidationResult> {
    let _path = path.as_ref();

    // Open file
    let file =
        File::open(_path).map_err(|e| IoError::FileError(format!("Failed to open file: {e}")))?;

    let mut reader = BufReader::new(file);
    let mut content = String::new();
    reader
        .read_to_string(&mut content)
        .map_err(|e| IoError::FileError(format!("Failed to read file: {e}")))?;

    // Check for required sections
    let has_relation = content.to_uppercase().contains("@RELATION");
    let has_attribute = content.to_uppercase().contains("@ATTRIBUTE");
    let has_data = content.to_uppercase().contains("@DATA");

    let mut details = Vec::new();

    if !has_relation {
        details.push("Missing @RELATION section".to_string());
    }

    if !has_attribute {
        details.push("Missing @ATTRIBUTE section".to_string());
    }

    if !has_data {
        details.push("Missing @DATA section".to_string());
    }

    if details.is_empty() {
        // Count attributes
        let attribute_count = content
            .to_uppercase()
            .lines()
            .filter(|line| line.trim().starts_with("@ATTRIBUTE"))
            .count();

        Ok(FormatValidationResult {
            valid: true,
            format: "ARFF".to_string(),
            file_path: path.as_ref().to_string_lossy().to_string(),
            details: Some(format!(
                "Valid ARFF file with {} attributes",
                attribute_count
            )),
        })
    } else {
        Ok(FormatValidationResult {
            valid: false,
            format: "ARFF".to_string(),
            file_path: path.as_ref().to_string_lossy().to_string(),
            details: Some(details.join(", ")),
        })
    }
}

/// Validate WAV file structure in detail
#[allow(dead_code)]
fn validate_wav_format<P: AsRef<Path>>(path: P) -> Result<FormatValidationResult> {
    let _path = path.as_ref();

    // Open file
    let file =
        File::open(_path).map_err(|e| IoError::FileError(format!("Failed to open file: {e}")))?;

    let mut reader = BufReader::new(file);
    let mut header = [0u8; 44]; // Standard WAV header size

    // Try to read header
    if let Err(e) = reader.read_exact(&mut header) {
        return Ok(FormatValidationResult {
            valid: false,
            format: "WAV".to_string(),
            file_path: path.as_ref().to_string_lossy().to_string(),
            details: Some(format!("Failed to read WAV header: {}", e)),
        });
    }

    // Check for RIFF header
    if &header[0..4] != b"RIFF" {
        return Ok(FormatValidationResult {
            valid: false,
            format: "WAV".to_string(),
            file_path: path.as_ref().to_string_lossy().to_string(),
            details: Some("Missing RIFF header".to_string()),
        });
    }

    // Check for WAVE format
    if &header[8..12] != b"WAVE" {
        return Ok(FormatValidationResult {
            valid: false,
            format: "WAV".to_string(),
            file_path: path.as_ref().to_string_lossy().to_string(),
            details: Some("Missing WAVE format identifier".to_string()),
        });
    }

    // Check for fmt chunk
    if &header[12..16] != b"fmt " {
        return Ok(FormatValidationResult {
            valid: false,
            format: "WAV".to_string(),
            file_path: path.as_ref().to_string_lossy().to_string(),
            details: Some("Missing fmt chunk".to_string()),
        });
    }

    // Extract audio format (PCM = 1)
    let audio_format = header[20] as u16 | ((header[21] as u16) << 8);
    let channels = header[22] as u16 | ((header[23] as u16) << 8);
    let sample_rate = header[24] as u32
        | ((header[25] as u32) << 8)
        | ((header[26] as u32) << 16)
        | ((header[27] as u32) << 24);
    let bits_per_sample = header[34] as u16 | ((header[35] as u16) << 8);

    Ok(FormatValidationResult {
        valid: true,
        format: "WAV".to_string(),
        file_path: _path.to_string_lossy().to_string(),
        details: Some(format!(
            "Valid WAV file: {} channels, {}Hz, {}-bit, {}",
            channels,
            sample_rate,
            bits_per_sample,
            if audio_format == 1 { "PCM" } else { "non-PCM" }
        )),
    })
}
