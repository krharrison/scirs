//! ARFF (Attribute-Relation File Format) handling module
//!
//! This module provides functionality for reading and writing ARFF files,
//! commonly used in machine learning applications like WEKA.
//!
//! # Features
//!
//! - Read and write standard dense ARFF files
//! - **Sparse ARFF support**: Read and write sparse ARFF format
//! - Support for numeric, nominal, string, and date attributes
//! - Conversion between ARFF data and numeric matrices
//!
//! # Sparse ARFF Format
//!
//! Sparse ARFF uses curly braces with index-value pairs:
//! ```text
//! @data
//! {0 1.0, 3 "hello", 5 yes}
//! {1 2.5, 4 no}
//! ```
//! Omitted indices are assumed to be 0 (numeric), "" (string), or missing.

pub mod sparse;

use scirs2_core::ndarray::Array2;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::error::{IoError, Result};

pub use sparse::{read_sparse_arff, write_sparse_arff, SparseArffData, SparseInstance};

/// ARFF attribute types
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeType {
    /// Numeric attribute (real or integer)
    Numeric,
    /// String attribute
    String,
    /// Date attribute
    Date(String),
    /// Nominal attribute with possible values
    Nominal(Vec<String>),
}

/// ARFF dataset representation
#[derive(Debug, Clone)]
pub struct ArffData {
    /// Name of the relation
    pub relation: String,
    /// Attributes with their names and types
    pub attributes: Vec<(String, AttributeType)>,
    /// Data as a 2D array where rows are instances and columns are attributes
    pub data: Array2<ArffValue>,
}

/// ARFF data value
#[derive(Debug, Clone, PartialEq)]
pub enum ArffValue {
    /// Numeric value
    Numeric(f64),
    /// String value
    String(String),
    /// Date value as string
    Date(String),
    /// Nominal value
    Nominal(String),
    /// Missing value
    Missing,
}

impl ArffValue {
    /// Try to convert the value to a float if possible
    pub fn to_f64(&self) -> Option<f64> {
        match self {
            ArffValue::Numeric(val) => Some(*val),
            _ => None,
        }
    }

    /// Try to convert the value to a string
    pub fn as_string(&self) -> String {
        match self {
            ArffValue::Numeric(val) => val.to_string(),
            ArffValue::String(val) => val.clone(),
            ArffValue::Date(val) => val.clone(),
            ArffValue::Nominal(val) => val.clone(),
            ArffValue::Missing => "?".to_string(),
        }
    }

    /// Check if this is a missing value
    pub fn is_missing(&self) -> bool {
        matches!(self, ArffValue::Missing)
    }

    /// Check if this is a numeric zero value
    pub fn is_numeric_zero(&self) -> bool {
        matches!(self, ArffValue::Numeric(v) if *v == 0.0)
    }
}

/// Parse attribute definition from ARFF file
fn parse_attribute(line: &str) -> Result<(String, AttributeType)> {
    let trimmed = line.trim();
    if !trimmed.to_lowercase().starts_with("@attribute") {
        return Err(IoError::FormatError("Invalid attribute format".to_string()));
    }

    // Remove @attribute prefix
    let rest = trimmed["@attribute".len()..].trim_start();

    // Parse name - may be quoted
    let (name, type_part) = if rest.starts_with('\'') || rest.starts_with('"') {
        let quote = rest.as_bytes()[0];
        let end = rest[1..]
            .find(|c: char| c as u8 == quote)
            .ok_or_else(|| IoError::FormatError("Unterminated attribute name quote".to_string()))?;
        let name = rest[1..end + 1].to_string();
        let remaining = rest[end + 2..].trim_start();
        (name, remaining)
    } else {
        let parts: Vec<&str> = rest.splitn(2, ' ').collect();
        if parts.len() < 2 {
            return Err(IoError::FormatError("Invalid attribute format".to_string()));
        }
        (parts[0].trim().to_string(), parts[1].trim())
    };

    // Parse attribute type
    let attr_type = if type_part.eq_ignore_ascii_case("numeric")
        || type_part.eq_ignore_ascii_case("real")
        || type_part.eq_ignore_ascii_case("integer")
    {
        AttributeType::Numeric
    } else if type_part.eq_ignore_ascii_case("string") {
        AttributeType::String
    } else if type_part.to_lowercase().starts_with("date") {
        let format = if type_part.len() > 4 && type_part.contains(' ') {
            let format_str = type_part.split_once(' ').map(|x| x.1).unwrap_or("").trim();
            if (format_str.starts_with('"') && format_str.ends_with('"'))
                || (format_str.starts_with('\'') && format_str.ends_with('\''))
            {
                format_str[1..format_str.len() - 1].to_string()
            } else {
                format_str.to_string()
            }
        } else {
            "yyyy-MM-dd'T'HH:mm:ss".to_string()
        };
        AttributeType::Date(format)
    } else if type_part.starts_with('{') && type_part.ends_with('}') {
        let values_str = &type_part[1..type_part.len() - 1];
        let values: Vec<String> = values_str
            .split(',')
            .map(|s| {
                let s = s.trim();
                if (s.starts_with('"') && s.ends_with('"'))
                    || (s.starts_with('\'') && s.ends_with('\''))
                {
                    s[1..s.len() - 1].to_string()
                } else {
                    s.to_string()
                }
            })
            .collect();
        AttributeType::Nominal(values)
    } else {
        return Err(IoError::FormatError(format!(
            "Unknown attribute type: {type_part}"
        )));
    };

    Ok((name, attr_type))
}

/// Parse an ARFF data line into ArffValue instances
fn parse_data_line(line: &str, attributes: &[(String, AttributeType)]) -> Result<Vec<ArffValue>> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Err(IoError::FormatError("Empty data line".to_string()));
    }

    // Check if this is sparse format
    if trimmed.starts_with('{') {
        return parse_sparse_data_line(trimmed, attributes);
    }

    let mut values = Vec::new();
    let parts: Vec<&str> = trimmed.split(',').collect();

    if parts.len() != attributes.len() {
        return Err(IoError::FormatError(format!(
            "Data line has {} values but expected {}",
            parts.len(),
            attributes.len()
        )));
    }

    for (i, part) in parts.iter().enumerate() {
        let part = part.trim();
        if part == "?" {
            values.push(ArffValue::Missing);
            continue;
        }

        let attr_type = &attributes[i].1;
        let value = parse_value(part, attr_type)?;
        values.push(value);
    }

    Ok(values)
}

/// Parse a sparse ARFF data line: {idx val, idx val, ...}
fn parse_sparse_data_line(
    line: &str,
    attributes: &[(String, AttributeType)],
) -> Result<Vec<ArffValue>> {
    let mut values: Vec<ArffValue> = Vec::new();
    // Initialize all values with defaults
    for (_, attr_type) in attributes {
        let default = match attr_type {
            AttributeType::Numeric => ArffValue::Numeric(0.0),
            AttributeType::String => ArffValue::String(String::new()),
            AttributeType::Date(_) => ArffValue::Missing,
            AttributeType::Nominal(_) => ArffValue::Missing,
        };
        values.push(default);
    }

    // Strip braces
    let inner = line
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .trim();

    if inner.is_empty() {
        return Ok(values);
    }

    // Parse index-value pairs
    for pair in inner.split(',') {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }

        let space_pos = pair
            .find(' ')
            .ok_or_else(|| IoError::FormatError(format!("Invalid sparse pair: '{}'", pair)))?;

        let idx_str = &pair[..space_pos];
        let val_str = pair[space_pos + 1..].trim();

        let idx: usize = idx_str
            .parse()
            .map_err(|_| IoError::FormatError(format!("Invalid sparse index: '{}'", idx_str)))?;

        if idx >= attributes.len() {
            return Err(IoError::FormatError(format!(
                "Sparse index {} out of range (max {})",
                idx,
                attributes.len() - 1
            )));
        }

        if val_str == "?" {
            values[idx] = ArffValue::Missing;
        } else {
            values[idx] = parse_value(val_str, &attributes[idx].1)?;
        }
    }

    Ok(values)
}

/// Parse a single value based on attribute type
fn parse_value(part: &str, attr_type: &AttributeType) -> Result<ArffValue> {
    match attr_type {
        AttributeType::Numeric => {
            let num = part
                .parse::<f64>()
                .map_err(|_| IoError::FormatError(format!("Invalid numeric value: {part}")))?;
            Ok(ArffValue::Numeric(num))
        }
        AttributeType::String => {
            let s = strip_quotes(part);
            Ok(ArffValue::String(s))
        }
        AttributeType::Date(_) => {
            let s = strip_quotes(part);
            Ok(ArffValue::Date(s))
        }
        AttributeType::Nominal(allowed_values) => {
            let s = strip_quotes(part);
            if !allowed_values.contains(&s) {
                return Err(IoError::FormatError(format!(
                    "Invalid nominal value: {s}, expected one of {allowed_values:?}"
                )));
            }
            Ok(ArffValue::Nominal(s))
        }
    }
}

/// Strip surrounding quotes from a string
fn strip_quotes(s: &str) -> String {
    let s = s.trim();
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

/// Reads an ARFF file (supports both dense and sparse formats)
pub fn read_arff<P: AsRef<Path>>(path: P) -> Result<ArffData> {
    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let reader = BufReader::new(file);

    let mut relation = String::new();
    let mut attributes = Vec::new();
    let mut data_lines = Vec::new();
    let mut in_data_section = false;

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result
            .map_err(|e| IoError::FileError(format!("Error reading line {}: {e}", line_num + 1)))?;

        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }

        if in_data_section {
            data_lines.push(trimmed.to_string());
        } else {
            let lower = trimmed.to_lowercase();
            if lower.starts_with("@relation") {
                let parts: Vec<&str> = trimmed.splitn(2, ' ').collect();
                if parts.len() < 2 {
                    return Err(IoError::FormatError("Invalid relation format".to_string()));
                }
                relation = strip_quotes(parts[1].trim());
            } else if lower.starts_with("@attribute") {
                let (name, attr_type) = parse_attribute(trimmed)?;
                attributes.push((name, attr_type));
            } else if lower.starts_with("@data") {
                in_data_section = true;
            } else {
                return Err(IoError::FormatError(format!(
                    "Unexpected line in header section: {trimmed}"
                )));
            }
        }
    }

    if !in_data_section {
        return Err(IoError::FormatError("No @data section found".to_string()));
    }

    if attributes.is_empty() {
        return Err(IoError::FormatError("No attributes defined".to_string()));
    }

    // Parse data lines
    let mut data_values = Vec::new();
    for (i, line) in data_lines.iter().enumerate() {
        let values = parse_data_line(line, &attributes)
            .map_err(|e| IoError::FormatError(format!("Error parsing data line {}: {e}", i + 1)))?;
        data_values.push(values);
    }

    // Create data array
    let num_instances = data_values.len();
    let num_attributes = attributes.len();
    let mut data = Array2::from_elem((num_instances, num_attributes), ArffValue::Missing);

    for (i, row) in data_values.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            data[[i, j]] = value.clone();
        }
    }

    Ok(ArffData {
        relation,
        attributes,
        data,
    })
}

/// Extracts a numeric matrix from ARFF data
pub fn get_numeric_matrix(
    arff_data: &ArffData,
    numeric_attributes: &[String],
) -> Result<Array2<f64>> {
    let mut indices = Vec::new();
    let mut attr_names = Vec::new();

    for attr_name in numeric_attributes {
        let mut found = false;
        for (i, (name, attr_type)) in arff_data.attributes.iter().enumerate() {
            if name == attr_name {
                match attr_type {
                    AttributeType::Numeric => {
                        indices.push(i);
                        attr_names.push(name.clone());
                        found = true;
                        break;
                    }
                    _ => {
                        return Err(IoError::FormatError(format!(
                            "Attribute '{name}' is not numeric"
                        )));
                    }
                }
            }
        }

        if !found {
            return Err(IoError::FormatError(format!(
                "Attribute '{attr_name}' not found"
            )));
        }
    }

    let num_instances = arff_data.data.shape()[0];
    let num_selected = indices.len();
    let mut output = Array2::from_elem((num_instances, num_selected), f64::NAN);

    for (out_col, &in_col) in indices.iter().enumerate() {
        for row in 0..num_instances {
            match &arff_data.data[[row, in_col]] {
                ArffValue::Numeric(val) => {
                    output[[row, out_col]] = *val;
                }
                ArffValue::Missing => {} // leave as NaN
                _ => {
                    return Err(IoError::FormatError(format!(
                        "Non-numeric value found in numeric attribute '{}' at row {}",
                        attr_names[out_col], row
                    )));
                }
            }
        }
    }

    Ok(output)
}

/// Writes data to an ARFF file
pub fn write_arff<P: AsRef<Path>>(path: P, arff_data: &ArffData) -> Result<()> {
    let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "@relation {}",
        format_arff_string(&arff_data.relation)
    )
    .map_err(|e| IoError::FileError(format!("Failed to write relation: {e}")))?;

    writeln!(writer).map_err(|e| IoError::FileError(format!("Failed to write newline: {e}")))?;

    for (name, attr_type) in &arff_data.attributes {
        let type_str = match attr_type {
            AttributeType::Numeric => "numeric".to_string(),
            AttributeType::String => "string".to_string(),
            AttributeType::Date(format) => {
                if format.is_empty() {
                    "date".to_string()
                } else {
                    format!("date {}", format_arff_string(format))
                }
            }
            AttributeType::Nominal(values) => {
                let values_str: Vec<String> =
                    values.iter().map(|v| format_arff_string(v)).collect();
                format!("{{{}}}", values_str.join(", "))
            }
        };

        writeln!(
            writer,
            "@attribute {} {}",
            format_arff_string(name),
            type_str
        )
        .map_err(|e| IoError::FileError(format!("Failed to write attribute: {e}")))?;
    }

    writeln!(writer, "\n@data")
        .map_err(|e| IoError::FileError(format!("Failed to write data header: {e}")))?;

    let shape = arff_data.data.shape();
    let num_instances = shape[0];
    let num_attributes = shape[1];

    for i in 0..num_instances {
        let mut line = String::new();
        for j in 0..num_attributes {
            let value = &arff_data.data[[i, j]];
            let value_str = match value {
                ArffValue::Missing => "?".to_string(),
                ArffValue::Numeric(val) => val.to_string(),
                ArffValue::String(val) => format_arff_string(val),
                ArffValue::Date(val) => format_arff_string(val),
                ArffValue::Nominal(val) => format_arff_string(val),
            };
            if j > 0 {
                line.push(',');
            }
            line.push_str(&value_str);
        }
        writeln!(writer, "{line}")
            .map_err(|e| IoError::FileError(format!("Failed to write data line: {e}")))?;
    }

    Ok(())
}

/// Creates an ARFF data structure from a numeric matrix
pub fn numeric_matrix_to_arff(
    relation: &str,
    attribute_names: &[String],
    data: &Array2<f64>,
) -> ArffData {
    let shape = data.shape();
    let num_instances = shape[0];
    let num_attributes = shape[1];

    let mut attributes = Vec::with_capacity(num_attributes);
    for name in attribute_names {
        attributes.push((name.clone(), AttributeType::Numeric));
    }

    let mut arff_data = Array2::from_elem((num_instances, num_attributes), ArffValue::Missing);

    for i in 0..num_instances {
        for j in 0..num_attributes {
            let val = data[[i, j]];
            arff_data[[i, j]] = if val.is_nan() {
                ArffValue::Missing
            } else {
                ArffValue::Numeric(val)
            };
        }
    }

    ArffData {
        relation: relation.to_string(),
        attributes,
        data: arff_data,
    }
}

/// Format a string for ARFF output, adding quotes if needed
fn format_arff_string(s: &str) -> String {
    if s.contains(' ')
        || s.contains(',')
        || s.contains('\'')
        || s.contains('"')
        || s.contains('{')
        || s.contains('}')
    {
        format!("\"{}\"", s.replace('"', "\\\""))
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arff_roundtrip_dense() {
        let dir = std::env::temp_dir().join("scirs2_arff_test_dense");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.arff");

        let arff_data = ArffData {
            relation: "test_relation".to_string(),
            attributes: vec![
                ("temp".to_string(), AttributeType::Numeric),
                (
                    "outlook".to_string(),
                    AttributeType::Nominal(vec![
                        "sunny".to_string(),
                        "overcast".to_string(),
                        "rainy".to_string(),
                    ]),
                ),
            ],
            data: Array2::from_shape_vec(
                (2, 2),
                vec![
                    ArffValue::Numeric(85.0),
                    ArffValue::Nominal("sunny".to_string()),
                    ArffValue::Numeric(72.0),
                    ArffValue::Nominal("overcast".to_string()),
                ],
            )
            .expect("Array creation failed"),
        };

        write_arff(&path, &arff_data).expect("Write failed");
        let loaded = read_arff(&path).expect("Read failed");

        assert_eq!(loaded.relation, "test_relation");
        assert_eq!(loaded.attributes.len(), 2);
        assert_eq!(loaded.data.shape(), &[2, 2]);
        assert_eq!(loaded.data[[0, 0]], ArffValue::Numeric(85.0));
        assert_eq!(loaded.data[[0, 1]], ArffValue::Nominal("sunny".to_string()));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_arff_missing_values() {
        let dir = std::env::temp_dir().join("scirs2_arff_test_missing");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("missing.arff");

        let arff_data = ArffData {
            relation: "test".to_string(),
            attributes: vec![
                ("x".to_string(), AttributeType::Numeric),
                ("y".to_string(), AttributeType::Numeric),
            ],
            data: Array2::from_shape_vec(
                (2, 2),
                vec![
                    ArffValue::Numeric(1.0),
                    ArffValue::Missing,
                    ArffValue::Missing,
                    ArffValue::Numeric(2.0),
                ],
            )
            .expect("Array creation failed"),
        };

        write_arff(&path, &arff_data).expect("Write failed");
        let loaded = read_arff(&path).expect("Read failed");

        assert!(loaded.data[[0, 1]].is_missing());
        assert!(loaded.data[[1, 0]].is_missing());
        assert_eq!(loaded.data[[0, 0]], ArffValue::Numeric(1.0));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_arff_with_date_and_string() {
        let dir = std::env::temp_dir().join("scirs2_arff_test_mixed");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("mixed.arff");

        let arff_data = ArffData {
            relation: "mixed_types".to_string(),
            attributes: vec![
                ("name".to_string(), AttributeType::String),
                (
                    "timestamp".to_string(),
                    AttributeType::Date("yyyy-MM-dd".to_string()),
                ),
                ("value".to_string(), AttributeType::Numeric),
            ],
            data: Array2::from_shape_vec(
                (1, 3),
                vec![
                    ArffValue::String("sensor_1".to_string()),
                    ArffValue::Date("2025-01-15".to_string()),
                    ArffValue::Numeric(42.5),
                ],
            )
            .expect("Array creation failed"),
        };

        write_arff(&path, &arff_data).expect("Write failed");
        let loaded = read_arff(&path).expect("Read failed");

        assert_eq!(loaded.attributes.len(), 3);
        assert_eq!(
            loaded.data[[0, 0]],
            ArffValue::String("sensor_1".to_string())
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_arff_numeric_matrix_conversion() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0])
            .expect("Array creation failed");

        let names = vec!["a".to_string(), "b".to_string()];
        let arff = numeric_matrix_to_arff("test", &names, &data);

        assert_eq!(arff.data[[0, 0]], ArffValue::Numeric(1.0));
        assert!(arff.data[[1, 1]].is_missing()); // NaN -> Missing

        let matrix = get_numeric_matrix(&arff, &names).expect("Conversion failed");
        assert!((matrix[[0, 0]] - 1.0).abs() < 1e-10);
        assert!(matrix[[1, 1]].is_nan()); // Missing -> NaN
    }

    #[test]
    fn test_arff_sparse_read() {
        let dir = std::env::temp_dir().join("scirs2_arff_test_sparse_read");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("sparse.arff");

        // Manually create a sparse ARFF file
        let content = "\
@relation sparse_test

@attribute x numeric
@attribute y numeric
@attribute z numeric

@data
{0 1.0, 2 3.0}
{1 2.5}
{}
";
        std::fs::write(&path, content).expect("Write failed");

        let loaded = read_arff(&path).expect("Read failed");
        assert_eq!(loaded.data.shape(), &[3, 3]);

        // First instance: x=1.0, y=0.0, z=3.0
        assert_eq!(loaded.data[[0, 0]], ArffValue::Numeric(1.0));
        assert_eq!(loaded.data[[0, 1]], ArffValue::Numeric(0.0));
        assert_eq!(loaded.data[[0, 2]], ArffValue::Numeric(3.0));

        // Second instance: x=0.0, y=2.5, z=0.0
        assert_eq!(loaded.data[[1, 0]], ArffValue::Numeric(0.0));
        assert_eq!(loaded.data[[1, 1]], ArffValue::Numeric(2.5));

        // Third instance: all zeros
        assert_eq!(loaded.data[[2, 0]], ArffValue::Numeric(0.0));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_arff_parse_attribute_types() {
        let (name, attr) = parse_attribute("@attribute temp numeric").expect("Parse failed");
        assert_eq!(name, "temp");
        assert_eq!(attr, AttributeType::Numeric);

        let (name, attr) = parse_attribute("@attribute name string").expect("Parse failed");
        assert_eq!(name, "name");
        assert_eq!(attr, AttributeType::String);

        let (name, attr) =
            parse_attribute("@attribute class {yes, no, maybe}").expect("Parse failed");
        assert_eq!(name, "class");
        assert!(matches!(attr, AttributeType::Nominal(_)));
    }

    #[test]
    fn test_arff_no_data_section() {
        let dir = std::env::temp_dir().join("scirs2_arff_test_nodata");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("nodata.arff");

        let content = "@relation test\n@attribute x numeric\n";
        std::fs::write(&path, content).expect("Write failed");

        let result = read_arff(&path);
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_arff_no_attributes() {
        let dir = std::env::temp_dir().join("scirs2_arff_test_noattr");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("noattr.arff");

        let content = "@relation test\n@data\n";
        std::fs::write(&path, content).expect("Write failed");

        let result = read_arff(&path);
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
