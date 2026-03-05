//! Sparse ARFF format support
//!
//! Provides reading and writing of sparse ARFF files where most values are zero/missing.
//! Sparse format uses curly braces with index-value pairs:
//!
//! ```text
//! {0 1.0, 3 "hello", 5 yes}
//! ```
//!
//! This is much more efficient for high-dimensional sparse datasets
//! commonly found in text classification, recommender systems, etc.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::error::{IoError, Result};

use super::{parse_attribute, ArffValue, AttributeType};

/// A sparse instance (row) storing only non-default values
#[derive(Debug, Clone)]
pub struct SparseInstance {
    /// Attribute index -> value mapping
    pub values: BTreeMap<usize, ArffValue>,
}

impl SparseInstance {
    /// Create a new empty sparse instance
    pub fn new() -> Self {
        SparseInstance {
            values: BTreeMap::new(),
        }
    }

    /// Set a value at the given attribute index
    pub fn set(&mut self, index: usize, value: ArffValue) {
        self.values.insert(index, value);
    }

    /// Get a value at the given attribute index
    pub fn get(&self, index: usize) -> Option<&ArffValue> {
        self.values.get(&index)
    }

    /// Get value or return default for the attribute type
    pub fn get_or_default(&self, index: usize, attr_type: &AttributeType) -> ArffValue {
        if let Some(val) = self.values.get(&index) {
            val.clone()
        } else {
            match attr_type {
                AttributeType::Numeric => ArffValue::Numeric(0.0),
                AttributeType::String => ArffValue::String(String::new()),
                AttributeType::Date(_) => ArffValue::Missing,
                AttributeType::Nominal(_) => ArffValue::Missing,
            }
        }
    }

    /// Number of explicitly stored values
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

impl Default for SparseInstance {
    fn default() -> Self {
        Self::new()
    }
}

/// Sparse ARFF dataset
#[derive(Debug, Clone)]
pub struct SparseArffData {
    /// Relation name
    pub relation: String,
    /// Attributes
    pub attributes: Vec<(String, AttributeType)>,
    /// Sparse instances
    pub instances: Vec<SparseInstance>,
}

impl SparseArffData {
    /// Create a new empty sparse ARFF dataset
    pub fn new(relation: impl Into<String>, attributes: Vec<(String, AttributeType)>) -> Self {
        SparseArffData {
            relation: relation.into(),
            attributes,
            instances: Vec::new(),
        }
    }

    /// Add a sparse instance
    pub fn add_instance(&mut self, instance: SparseInstance) {
        self.instances.push(instance);
    }

    /// Number of instances
    pub fn num_instances(&self) -> usize {
        self.instances.len()
    }

    /// Number of attributes
    pub fn num_attributes(&self) -> usize {
        self.attributes.len()
    }

    /// Get the total number of non-default values across all instances
    pub fn total_nnz(&self) -> usize {
        self.instances.iter().map(|inst| inst.nnz()).sum()
    }

    /// Get the sparsity ratio (fraction of default values)
    pub fn sparsity(&self) -> f64 {
        if self.instances.is_empty() || self.attributes.is_empty() {
            return 1.0;
        }
        let total_cells = self.instances.len() * self.attributes.len();
        let nnz = self.total_nnz();
        1.0 - (nnz as f64 / total_cells as f64)
    }

    /// Convert to dense ArffData
    pub fn to_dense(&self) -> super::ArffData {
        use scirs2_core::ndarray::Array2;

        let num_instances = self.instances.len();
        let num_attributes = self.attributes.len();

        let mut data = Array2::from_elem((num_instances, num_attributes), ArffValue::Missing);

        for (i, instance) in self.instances.iter().enumerate() {
            for j in 0..num_attributes {
                data[[i, j]] = instance.get_or_default(j, &self.attributes[j].1);
            }
        }

        super::ArffData {
            relation: self.relation.clone(),
            attributes: self.attributes.clone(),
            data,
        }
    }

    /// Convert from dense ArffData to sparse
    pub fn from_dense(dense: &super::ArffData) -> Self {
        let num_instances = dense.data.shape()[0];
        let mut instances = Vec::with_capacity(num_instances);

        for i in 0..num_instances {
            let mut inst = SparseInstance::new();
            for (j, (_, attr_type)) in dense.attributes.iter().enumerate() {
                let value = &dense.data[[i, j]];
                let is_default = match (value, attr_type) {
                    (ArffValue::Numeric(v), AttributeType::Numeric) => *v == 0.0,
                    (ArffValue::String(s), AttributeType::String) => s.is_empty(),
                    (ArffValue::Missing, _) => true,
                    _ => false,
                };
                if !is_default {
                    inst.set(j, value.clone());
                }
            }
            instances.push(inst);
        }

        SparseArffData {
            relation: dense.relation.clone(),
            attributes: dense.attributes.clone(),
            instances,
        }
    }
}

/// Read a sparse ARFF file
///
/// This reads an ARFF file and stores data in sparse format.
/// Works with both dense and sparse data sections.
pub fn read_sparse_arff<P: AsRef<Path>>(path: P) -> Result<SparseArffData> {
    let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let reader = BufReader::new(file);

    let mut relation = String::new();
    let mut attributes = Vec::new();
    let mut instances = Vec::new();
    let mut in_data_section = false;

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result
            .map_err(|e| IoError::FileError(format!("Error reading line {}: {e}", line_num + 1)))?;

        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }

        if in_data_section {
            let instance = parse_sparse_line(trimmed, &attributes)?;
            instances.push(instance);
        } else {
            let lower = trimmed.to_lowercase();
            if lower.starts_with("@relation") {
                let parts: Vec<&str> = trimmed.splitn(2, ' ').collect();
                if parts.len() < 2 {
                    return Err(IoError::FormatError("Invalid relation format".to_string()));
                }
                relation = strip_quotes_local(parts[1].trim());
            } else if lower.starts_with("@attribute") {
                let (name, attr_type) = parse_attribute(trimmed)?;
                attributes.push((name, attr_type));
            } else if lower.starts_with("@data") {
                in_data_section = true;
            } else {
                return Err(IoError::FormatError(format!(
                    "Unexpected header line: {trimmed}"
                )));
            }
        }
    }

    if !in_data_section {
        return Err(IoError::FormatError("No @data section found".to_string()));
    }

    Ok(SparseArffData {
        relation,
        attributes,
        instances,
    })
}

/// Parse a sparse or dense data line into a SparseInstance
fn parse_sparse_line(line: &str, attributes: &[(String, AttributeType)]) -> Result<SparseInstance> {
    let trimmed = line.trim();

    if trimmed.starts_with('{') {
        // Sparse format: {idx val, idx val, ...}
        let inner = trimmed.trim_start_matches('{').trim_end_matches('}').trim();

        let mut inst = SparseInstance::new();

        if inner.is_empty() {
            return Ok(inst);
        }

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

            let idx: usize = idx_str.parse().map_err(|_| {
                IoError::FormatError(format!("Invalid sparse index: '{}'", idx_str))
            })?;

            if idx >= attributes.len() {
                return Err(IoError::FormatError(format!(
                    "Sparse index {} out of range (max {})",
                    idx,
                    attributes.len() - 1
                )));
            }

            if val_str != "?" {
                let value = super::parse_value(val_str, &attributes[idx].1)?;
                inst.set(idx, value);
            }
        }

        Ok(inst)
    } else {
        // Dense format: val,val,val,...
        let parts: Vec<&str> = trimmed.split(',').collect();
        if parts.len() != attributes.len() {
            return Err(IoError::FormatError(format!(
                "Data line has {} values, expected {}",
                parts.len(),
                attributes.len()
            )));
        }

        let mut inst = SparseInstance::new();
        for (i, part) in parts.iter().enumerate() {
            let part = part.trim();
            if part == "?" {
                continue; // Missing = default
            }

            let value = super::parse_value(part, &attributes[i].1)?;

            // Only store non-default values
            let is_default = match (&value, &attributes[i].1) {
                (ArffValue::Numeric(v), AttributeType::Numeric) => *v == 0.0,
                (ArffValue::String(s), AttributeType::String) => s.is_empty(),
                _ => false,
            };

            if !is_default {
                inst.set(i, value);
            }
        }

        Ok(inst)
    }
}

/// Write a sparse ARFF file
pub fn write_sparse_arff<P: AsRef<Path>>(path: P, data: &SparseArffData) -> Result<()> {
    let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "@relation {}", format_arff_str(&data.relation))
        .map_err(|e| IoError::FileError(format!("Write error: {}", e)))?;
    writeln!(writer).map_err(|e| IoError::FileError(format!("Write error: {}", e)))?;

    for (name, attr_type) in &data.attributes {
        let type_str = match attr_type {
            AttributeType::Numeric => "numeric".to_string(),
            AttributeType::String => "string".to_string(),
            AttributeType::Date(fmt) => {
                if fmt.is_empty() {
                    "date".to_string()
                } else {
                    format!("date {}", format_arff_str(fmt))
                }
            }
            AttributeType::Nominal(values) => {
                let vals: Vec<String> = values.iter().map(|v| format_arff_str(v)).collect();
                format!("{{{}}}", vals.join(", "))
            }
        };
        writeln!(writer, "@attribute {} {}", format_arff_str(name), type_str)
            .map_err(|e| IoError::FileError(format!("Write error: {}", e)))?;
    }

    writeln!(writer, "\n@data").map_err(|e| IoError::FileError(format!("Write error: {}", e)))?;

    // Write sparse instances
    for instance in &data.instances {
        let mut pairs = Vec::new();
        for (&idx, value) in &instance.values {
            let val_str = match value {
                ArffValue::Missing => "?".to_string(),
                ArffValue::Numeric(v) => v.to_string(),
                ArffValue::String(s) => format_arff_str(s),
                ArffValue::Date(s) => format_arff_str(s),
                ArffValue::Nominal(s) => format_arff_str(s),
            };
            pairs.push(format!("{} {}", idx, val_str));
        }
        writeln!(writer, "{{{}}}", pairs.join(", "))
            .map_err(|e| IoError::FileError(format!("Write error: {}", e)))?;
    }

    writer
        .flush()
        .map_err(|e| IoError::FileError(format!("Flush error: {}", e)))?;

    Ok(())
}

fn format_arff_str(s: &str) -> String {
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

fn strip_quotes_local(s: &str) -> String {
    let s = s.trim();
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_arff_roundtrip() {
        let dir = std::env::temp_dir().join("scirs2_arff_sparse_rt");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("sparse.arff");

        let mut data = SparseArffData::new(
            "sparse_test",
            vec![
                ("x".to_string(), AttributeType::Numeric),
                ("y".to_string(), AttributeType::Numeric),
                ("z".to_string(), AttributeType::Numeric),
                ("w".to_string(), AttributeType::Numeric),
            ],
        );

        let mut inst1 = SparseInstance::new();
        inst1.set(0, ArffValue::Numeric(1.0));
        inst1.set(3, ArffValue::Numeric(4.0));
        data.add_instance(inst1);

        let mut inst2 = SparseInstance::new();
        inst2.set(1, ArffValue::Numeric(2.5));
        data.add_instance(inst2);

        data.add_instance(SparseInstance::new()); // all zeros

        write_sparse_arff(&path, &data).expect("Write failed");
        let loaded = read_sparse_arff(&path).expect("Read failed");

        assert_eq!(loaded.num_instances(), 3);
        assert_eq!(loaded.num_attributes(), 4);

        // Check first instance
        let inst0 = &loaded.instances[0];
        assert_eq!(inst0.get(0), Some(&ArffValue::Numeric(1.0)));
        assert_eq!(inst0.get(1), None); // not stored (default 0)
        assert_eq!(inst0.get(3), Some(&ArffValue::Numeric(4.0)));

        // Check empty instance
        assert_eq!(loaded.instances[2].nnz(), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_sparse_to_dense_conversion() {
        let mut data = SparseArffData::new(
            "test",
            vec![
                ("a".to_string(), AttributeType::Numeric),
                ("b".to_string(), AttributeType::Numeric),
            ],
        );

        let mut inst = SparseInstance::new();
        inst.set(0, ArffValue::Numeric(5.0));
        data.add_instance(inst);

        let dense = data.to_dense();
        assert_eq!(dense.data[[0, 0]], ArffValue::Numeric(5.0));
        assert_eq!(dense.data[[0, 1]], ArffValue::Numeric(0.0)); // default
    }

    #[test]
    fn test_dense_to_sparse_conversion() {
        use scirs2_core::ndarray::Array2;

        let dense = super::super::ArffData {
            relation: "test".to_string(),
            attributes: vec![
                ("a".to_string(), AttributeType::Numeric),
                ("b".to_string(), AttributeType::Numeric),
                ("c".to_string(), AttributeType::Numeric),
            ],
            data: Array2::from_shape_vec(
                (2, 3),
                vec![
                    ArffValue::Numeric(1.0),
                    ArffValue::Numeric(0.0),
                    ArffValue::Numeric(3.0),
                    ArffValue::Numeric(0.0),
                    ArffValue::Numeric(0.0),
                    ArffValue::Numeric(0.0),
                ],
            )
            .expect("Array creation failed"),
        };

        let sparse = SparseArffData::from_dense(&dense);
        assert_eq!(sparse.instances[0].nnz(), 2); // 1.0 and 3.0
        assert_eq!(sparse.instances[1].nnz(), 0); // all zeros
    }

    #[test]
    fn test_sparsity_calculation() {
        let mut data = SparseArffData::new(
            "test",
            vec![
                ("a".to_string(), AttributeType::Numeric),
                ("b".to_string(), AttributeType::Numeric),
                ("c".to_string(), AttributeType::Numeric),
                ("d".to_string(), AttributeType::Numeric),
            ],
        );

        // 10 instances, 1 non-zero each = 10/40 = 25% non-zero = 75% sparse
        for i in 0..10 {
            let mut inst = SparseInstance::new();
            inst.set(i % 4, ArffValue::Numeric(1.0));
            data.add_instance(inst);
        }

        let sparsity = data.sparsity();
        assert!((sparsity - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_with_nominal() {
        let dir = std::env::temp_dir().join("scirs2_arff_sparse_nom");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("sparse_nominal.arff");

        let mut data = SparseArffData::new(
            "nominal_test",
            vec![
                ("x".to_string(), AttributeType::Numeric),
                (
                    "class".to_string(),
                    AttributeType::Nominal(vec!["a".to_string(), "b".to_string()]),
                ),
            ],
        );

        let mut inst = SparseInstance::new();
        inst.set(0, ArffValue::Numeric(42.0));
        inst.set(1, ArffValue::Nominal("a".to_string()));
        data.add_instance(inst);

        write_sparse_arff(&path, &data).expect("Write failed");
        let loaded = read_sparse_arff(&path).expect("Read failed");

        let inst0 = &loaded.instances[0];
        assert_eq!(inst0.get(0), Some(&ArffValue::Numeric(42.0)));
        assert_eq!(inst0.get(1), Some(&ArffValue::Nominal("a".to_string())));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_sparse_high_dimensional() {
        let dir = std::env::temp_dir().join("scirs2_arff_sparse_hd");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("high_dim.arff");

        // 100 attributes, very sparse
        let attrs: Vec<(String, AttributeType)> = (0..100)
            .map(|i| (format!("feat_{}", i), AttributeType::Numeric))
            .collect();

        let mut data = SparseArffData::new("high_dim", attrs);

        for i in 0..50 {
            let mut inst = SparseInstance::new();
            // Only 3 non-zero features per instance
            inst.set(i % 100, ArffValue::Numeric(1.0));
            inst.set((i * 7) % 100, ArffValue::Numeric(2.0));
            inst.set((i * 13) % 100, ArffValue::Numeric(3.0));
            data.add_instance(inst);
        }

        write_sparse_arff(&path, &data).expect("Write failed");
        let loaded = read_sparse_arff(&path).expect("Read failed");

        assert_eq!(loaded.num_instances(), 50);
        assert_eq!(loaded.num_attributes(), 100);

        // Check sparsity is high
        assert!(loaded.sparsity() > 0.9);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
