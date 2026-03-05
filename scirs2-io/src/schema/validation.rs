//! Schema validation, inference, and type coercion.
//!
//! This module provides:
//! - [`SchemaValidator`] — validates rows/columns against a [`Schema`]
//! - [`SchemaInference`] — infers a [`Schema`] from sample JSON/CSV data
//! - [`TypeCoercion`] — coerces JSON values to target types
//! - [`ValidationReport`] — collected violations from a validation pass

#![allow(missing_docs)]

use crate::error::{IoError, Result};
use crate::schema::{Constraint, FieldType, Schema, SchemaField, TypedValue};
use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;

// ─── Coercion mode ───────────────────────────────────────────────────────────

/// How aggressively the validator should coerce values to their declared types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CoercionMode {
    /// No coercion — values must match the declared type exactly.
    Strict,
    /// Coerce safely: e.g. integer → float, numeric-string → number.
    #[default]
    Lenient,
    /// Coerce aggressively: e.g. any value to string, boolean from 0/1.
    Aggressive,
}

// ─── ValidationViolation ────────────────────────────────────────────────────

/// A single schema violation found during validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationViolation {
    /// Row index (0-based) where the violation occurred, if applicable.
    pub row: Option<usize>,
    /// Column name where the violation occurred.
    pub column: String,
    /// Human-readable description of the violation.
    pub message: String,
    /// The constraint that was violated, if applicable.
    pub constraint: Option<String>,
    /// The raw value that caused the violation, serialized as a string.
    pub raw_value: Option<String>,
}

impl ValidationViolation {
    /// Create a new violation at a specific row.
    pub fn at_row(
        row: usize,
        column: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            row: Some(row),
            column: column.into(),
            message: message.into(),
            constraint: None,
            raw_value: None,
        }
    }

    /// Set the constraint name.
    pub fn with_constraint(mut self, c: impl Into<String>) -> Self {
        self.constraint = Some(c.into());
        self
    }

    /// Set the raw value representation.
    pub fn with_raw(mut self, raw: impl Into<String>) -> Self {
        self.raw_value = Some(raw.into());
        self
    }
}

// ─── ValidationReport ───────────────────────────────────────────────────────

/// The result of validating a dataset against a schema.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValidationReport {
    /// All violations found during validation.
    pub violations: Vec<ValidationViolation>,
    /// Number of rows validated.
    pub rows_validated: usize,
    /// Number of columns validated.
    pub columns_validated: usize,
    /// Whether the data is considered valid (no violations).
    pub is_valid: bool,
    /// Summary statistics per column.
    pub column_stats: HashMap<String, ColumnValidationStats>,
}

/// Per-column validation statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ColumnValidationStats {
    /// Number of null/missing values seen.
    pub null_count: usize,
    /// Number of type errors.
    pub type_errors: usize,
    /// Number of constraint violations.
    pub constraint_violations: usize,
    /// Number of coercions applied (lenient/aggressive mode).
    pub coercions_applied: usize,
}

impl ValidationReport {
    fn new(rows: usize, cols: usize) -> Self {
        Self {
            violations: Vec::new(),
            rows_validated: rows,
            columns_validated: cols,
            is_valid: true,
            column_stats: HashMap::new(),
        }
    }

    fn add_violation(&mut self, v: ValidationViolation) {
        let stats = self
            .column_stats
            .entry(v.column.clone())
            .or_default();
        if v.constraint.is_some() {
            stats.constraint_violations += 1;
        } else {
            stats.type_errors += 1;
        }
        self.violations.push(v);
        self.is_valid = false;
    }

    fn record_null(&mut self, column: &str) {
        self.column_stats.entry(column.to_string()).or_default().null_count += 1;
    }

    fn record_coercion(&mut self, column: &str) {
        self.column_stats
            .entry(column.to_string())
            .or_default()
            .coercions_applied += 1;
    }

    /// Returns only violations for the given column.
    pub fn violations_for(&self, column: &str) -> Vec<&ValidationViolation> {
        self.violations.iter().filter(|v| v.column == column).collect()
    }

    /// Merge another report into this one.
    pub fn merge(&mut self, other: ValidationReport) {
        self.violations.extend(other.violations);
        self.rows_validated += other.rows_validated;
        for (col, stats) in other.column_stats {
            let s = self.column_stats.entry(col).or_default();
            s.null_count += stats.null_count;
            s.type_errors += stats.type_errors;
            s.constraint_violations += stats.constraint_violations;
            s.coercions_applied += stats.coercions_applied;
        }
        if !other.is_valid {
            self.is_valid = false;
        }
    }
}

// ─── TypeCoercion ────────────────────────────────────────────────────────────

/// Configurable type coercion engine.
pub struct TypeCoercion {
    mode: CoercionMode,
}

impl TypeCoercion {
    /// Create a new `TypeCoercion` engine with the given mode.
    pub fn new(mode: CoercionMode) -> Self {
        Self { mode }
    }

    /// Try to coerce `value` to `target_type`.
    ///
    /// Returns `Ok(TypedValue)` on success, `Err(IoError)` if coercion is
    /// not possible under the current mode.
    pub fn coerce(
        &self,
        value: &serde_json::Value,
        target_type: &FieldType,
    ) -> Result<TypedValue> {
        match self.mode {
            CoercionMode::Strict => TypedValue::from_json_strict(value, target_type),
            CoercionMode::Lenient => self.coerce_lenient(value, target_type),
            CoercionMode::Aggressive => self.coerce_aggressive(value, target_type),
        }
    }

    fn coerce_lenient(
        &self,
        value: &serde_json::Value,
        target: &FieldType,
    ) -> Result<TypedValue> {
        if value.is_null() {
            return Ok(TypedValue::Null);
        }

        match target {
            FieldType::Boolean => {
                if let Some(b) = value.as_bool() {
                    return Ok(TypedValue::Boolean(b));
                }
                if let Some(n) = value.as_i64() {
                    return Ok(TypedValue::Boolean(n != 0));
                }
                if let Some(s) = value.as_str() {
                    match s.trim().to_lowercase().as_str() {
                        "true" | "1" | "yes" | "y" => return Ok(TypedValue::Boolean(true)),
                        "false" | "0" | "no" | "n" => return Ok(TypedValue::Boolean(false)),
                        _ => {}
                    }
                }
            }
            ft if ft.is_integer() => {
                if let Some(i) = value.as_i64() {
                    return Ok(TypedValue::Int(i));
                }
                if let Some(f) = value.as_f64() {
                    if f.fract() == 0.0 {
                        return Ok(TypedValue::Int(f as i64));
                    }
                }
                if let Some(s) = value.as_str() {
                    if let Ok(i) = s.trim().parse::<i64>() {
                        return Ok(TypedValue::Int(i));
                    }
                    if let Ok(f) = s.trim().parse::<f64>() {
                        if f.fract() == 0.0 {
                            return Ok(TypedValue::Int(f as i64));
                        }
                    }
                }
            }
            ft if ft.is_float() => {
                if let Some(f) = value.as_f64() {
                    return Ok(TypedValue::Float(f));
                }
                if let Some(s) = value.as_str() {
                    if let Ok(f) = s.trim().parse::<f64>() {
                        return Ok(TypedValue::Float(f));
                    }
                }
            }
            FieldType::Utf8 => {
                if let Some(s) = value.as_str() {
                    return Ok(TypedValue::Utf8(s.to_string()));
                }
                // Lenient: convert numbers to string
                if let Some(i) = value.as_i64() {
                    return Ok(TypedValue::Utf8(i.to_string()));
                }
                if let Some(f) = value.as_f64() {
                    return Ok(TypedValue::Utf8(f.to_string()));
                }
            }
            FieldType::Date => {
                if let Some(s) = value.as_str() {
                    if let Ok(d) = NaiveDate::from_str(s.trim()) {
                        return Ok(TypedValue::Date(d));
                    }
                }
            }
            FieldType::Timestamp => {
                if let Some(s) = value.as_str() {
                    if let Ok(ts) = DateTime::<Utc>::from_str(s.trim()) {
                        return Ok(TypedValue::Timestamp(ts));
                    }
                }
            }
            FieldType::Json => return Ok(TypedValue::Json(value.clone())),
            _ => {}
        }

        // Fallback to strict
        TypedValue::from_json_strict(value, target)
    }

    fn coerce_aggressive(
        &self,
        value: &serde_json::Value,
        target: &FieldType,
    ) -> Result<TypedValue> {
        // First try lenient
        if let Ok(tv) = self.coerce_lenient(value, target) {
            return Ok(tv);
        }

        // Aggressive: toString everything
        match target {
            FieldType::Utf8 => {
                let s = match value {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                return Ok(TypedValue::Utf8(s));
            }
            FieldType::Boolean => {
                // Any truthy JSON value
                let b = match value {
                    serde_json::Value::Bool(b) => *b,
                    serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0) != 0.0,
                    serde_json::Value::String(s) => {
                        !s.is_empty() && s != "0" && s.to_lowercase() != "false"
                    }
                    serde_json::Value::Null => false,
                    _ => true,
                };
                return Ok(TypedValue::Boolean(b));
            }
            _ => {}
        }

        Err(IoError::ConversionError(format!(
            "Cannot aggressively coerce {} to {}",
            value,
            target.type_name()
        )))
    }
}

// ─── Constraint validation ───────────────────────────────────────────────────

fn check_constraint(
    constraint: &Constraint,
    value: &serde_json::Value,
    field_name: &str,
    row: usize,
) -> Option<ValidationViolation> {
    match constraint {
        Constraint::Range { min, max } => {
            if let (Some(v_f), Some(min_f), Some(max_f)) = (
                value.as_f64(),
                min.as_f64(),
                max.as_f64(),
            ) {
                if v_f < min_f || v_f > max_f {
                    return Some(
                        ValidationViolation::at_row(row, field_name, format!(
                            "Value {v_f} is outside range [{min_f}, {max_f}]"
                        ))
                        .with_constraint("range")
                        .with_raw(value.to_string()),
                    );
                }
            }
        }
        Constraint::MaxLength(max_len) => {
            if let Some(s) = value.as_str() {
                if s.len() > *max_len {
                    return Some(
                        ValidationViolation::at_row(row, field_name, format!(
                            "String length {} exceeds max_length {max_len}",
                            s.len()
                        ))
                        .with_constraint("max_length")
                        .with_raw(s.to_string()),
                    );
                }
            }
        }
        Constraint::MinLength(min_len) => {
            if let Some(s) = value.as_str() {
                if s.len() < *min_len {
                    return Some(
                        ValidationViolation::at_row(row, field_name, format!(
                            "String length {} is below min_length {min_len}",
                            s.len()
                        ))
                        .with_constraint("min_length")
                        .with_raw(s.to_string()),
                    );
                }
            }
        }
        Constraint::OneOf(options) => {
            if !options.contains(value) {
                let opts: Vec<String> = options.iter().map(|v| v.to_string()).collect();
                return Some(
                    ValidationViolation::at_row(row, field_name, format!(
                        "Value {} not in allowed set: [{}]",
                        value,
                        opts.join(", ")
                    ))
                    .with_constraint("one_of")
                    .with_raw(value.to_string()),
                );
            }
        }
        Constraint::Regex(pattern) => {
            if let Some(s) = value.as_str() {
                // Use a simple substring/prefix check without regex crate
                // (regex crate is available in scirs2-io Cargo.toml)
                use regex::Regex;
                match Regex::new(pattern) {
                    Ok(re) => {
                        if !re.is_match(s) {
                            return Some(
                                ValidationViolation::at_row(row, field_name, format!(
                                    "Value '{}' does not match regex '{}'",
                                    s, pattern
                                ))
                                .with_constraint("regex")
                                .with_raw(s.to_string()),
                            );
                        }
                    }
                    Err(e) => {
                        return Some(
                            ValidationViolation::at_row(row, field_name, format!(
                                "Invalid regex '{}': {}",
                                pattern, e
                            ))
                            .with_constraint("regex"),
                        );
                    }
                }
            }
        }
        // Unique is checked at the dataset level, not per-row
        Constraint::Unique | Constraint::Custom { .. } => {}
    }
    None
}

// ─── SchemaValidator ────────────────────────────────────────────────────────

/// Validates rows (as `HashMap<String, serde_json::Value>`) against a [`Schema`].
pub struct SchemaValidator {
    schema: Schema,
    coercion: TypeCoercion,
}

impl SchemaValidator {
    /// Create a new validator.
    pub fn new(schema: Schema, mode: CoercionMode) -> Self {
        Self {
            schema,
            coercion: TypeCoercion::new(mode),
        }
    }

    /// Validate a single row (a map from column name to JSON value).
    /// Returns a `ValidationReport` with at most one pass.
    pub fn validate_row(
        &self,
        row: &HashMap<String, serde_json::Value>,
        row_index: usize,
    ) -> ValidationReport {
        let mut report = ValidationReport::new(1, self.schema.len());

        for field in &self.schema.fields {
            match row.get(&field.name) {
                None | Some(serde_json::Value::Null) => {
                    report.record_null(&field.name);
                    if !field.nullable && field.default.is_none() {
                        report.add_violation(
                            ValidationViolation::at_row(
                                row_index,
                                &field.name,
                                format!(
                                    "Column '{}' is NOT NULL but value is missing",
                                    field.name
                                ),
                            )
                            .with_constraint("not_null"),
                        );
                    }
                }
                Some(value) => {
                    // Type check / coercion
                    match self.coercion.coerce(value, &field.field_type) {
                        Ok(_coerced) => {
                            // Record coercion if the raw JSON type differs from target
                            if !json_matches_type(value, &field.field_type) {
                                report.record_coercion(&field.name);
                            }
                        }
                        Err(e) => {
                            report.add_violation(
                                ValidationViolation::at_row(
                                    row_index,
                                    &field.name,
                                    format!(
                                        "Type error for column '{}': {}",
                                        field.name, e
                                    ),
                                )
                                .with_raw(value.to_string()),
                            );
                        }
                    }

                    // Constraint checks
                    for constraint in &field.constraints {
                        if let Some(violation) =
                            check_constraint(constraint, value, &field.name, row_index)
                        {
                            report.add_violation(violation);
                        }
                    }
                }
            }
        }

        report
    }

    /// Validate a sequence of rows and merge results.
    pub fn validate_rows(
        &self,
        rows: &[HashMap<String, serde_json::Value>],
    ) -> ValidationReport {
        let mut merged = ValidationReport::new(0, self.schema.len());

        // Check Unique constraints across all rows
        let unique_columns: Vec<&SchemaField> = self
            .schema
            .fields
            .iter()
            .filter(|f| f.constraints.iter().any(|c| matches!(c, Constraint::Unique)))
            .collect();

        let mut seen: HashMap<String, std::collections::HashSet<String>> = HashMap::new();

        for (i, row) in rows.iter().enumerate() {
            let row_report = self.validate_row(row, i);
            merged.merge(row_report);

            for field in &unique_columns {
                if let Some(val) = row.get(&field.name) {
                    let key = val.to_string();
                    let set = seen.entry(field.name.clone()).or_default();
                    if !set.insert(key.clone()) {
                        merged.add_violation(
                            ValidationViolation::at_row(
                                i,
                                &field.name,
                                format!(
                                    "Duplicate value '{}' violates unique constraint on '{}'",
                                    key, field.name
                                ),
                            )
                            .with_constraint("unique")
                            .with_raw(key),
                        );
                    }
                }
            }
        }

        merged
    }

    /// Return the schema this validator uses.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}

/// Returns true if the JSON value naturally matches the target type without coercion.
fn json_matches_type(value: &serde_json::Value, ft: &FieldType) -> bool {
    match (ft, value) {
        (FieldType::Boolean, serde_json::Value::Bool(_)) => true,
        (ft, serde_json::Value::Number(_)) if ft.is_numeric() => true,
        (FieldType::Utf8, serde_json::Value::String(_)) => true,
        (FieldType::Date, serde_json::Value::String(_)) => true,
        (FieldType::Timestamp, serde_json::Value::String(_)) => true,
        (FieldType::Json, _) => true,
        _ => false,
    }
}

// ─── SchemaInference ─────────────────────────────────────────────────────────

/// Infer a [`Schema`] from sample data.
pub struct SchemaInference {
    /// Minimum fraction of non-null values for a field to be inferred as NOT NULL
    pub nullable_threshold: f64,
    /// If true, try to parse strings as dates/timestamps
    pub try_temporal: bool,
    /// If true, try to parse strings as numbers
    pub try_numeric_strings: bool,
}

impl Default for SchemaInference {
    fn default() -> Self {
        Self {
            nullable_threshold: 0.05,
            try_temporal: true,
            try_numeric_strings: true,
        }
    }
}

impl SchemaInference {
    /// Infer a schema from a slice of JSON rows (each row is a JSON object).
    ///
    /// Scans up to `max_rows` rows. Returns an error if `rows` is empty.
    pub fn infer_from_json_rows(
        &self,
        rows: &[serde_json::Value],
        max_rows: Option<usize>,
    ) -> Result<Schema> {
        if rows.is_empty() {
            return Err(IoError::ValidationError(
                "Cannot infer schema from empty row set".to_string(),
            ));
        }

        let limit = max_rows.unwrap_or(rows.len()).min(rows.len());
        let sample = &rows[..limit];

        // Collect column names in order of first appearance
        let mut columns: Vec<String> = Vec::new();
        for row in sample {
            if let Some(obj) = row.as_object() {
                for key in obj.keys() {
                    if !columns.contains(key) {
                        columns.push(key.clone());
                    }
                }
            }
        }

        let mut fields = Vec::new();
        for col in &columns {
            let field = self.infer_field(col, sample);
            fields.push(field);
        }

        Ok(Schema {
            fields,
            name: None,
            description: Some("Inferred schema".to_string()),
            version: None,
            metadata: HashMap::new(),
        })
    }

    /// Infer a schema from CSV-style data: header row + data rows (all strings).
    pub fn infer_from_csv_rows(
        &self,
        headers: &[String],
        rows: &[Vec<String>],
        max_rows: Option<usize>,
    ) -> Schema {
        let limit = max_rows.unwrap_or(rows.len()).min(rows.len());
        let sample = &rows[..limit];

        let fields: Vec<SchemaField> = headers
            .iter()
            .enumerate()
            .map(|(col_idx, header)| {
                let values: Vec<&str> = sample
                    .iter()
                    .filter_map(|row| row.get(col_idx).map(|s| s.as_str()))
                    .collect();
                self.infer_field_from_strings(header, &values)
            })
            .collect();

        Schema {
            fields,
            name: None,
            description: Some("Inferred schema from CSV".to_string()),
            version: None,
            metadata: HashMap::new(),
        }
    }

    fn infer_field(&self, name: &str, rows: &[serde_json::Value]) -> SchemaField {
        let mut null_count = 0usize;
        let total = rows.len();
        let mut seen_types: Vec<FieldType> = Vec::new();

        for row in rows {
            let val = match row.as_object().and_then(|o| o.get(name)) {
                Some(v) => v,
                None => {
                    null_count += 1;
                    continue;
                }
            };

            match val {
                serde_json::Value::Null => {
                    null_count += 1;
                }
                serde_json::Value::Bool(_) => {
                    seen_types.push(FieldType::Boolean);
                }
                serde_json::Value::Number(n) => {
                    if n.is_i64() || n.is_u64() {
                        seen_types.push(FieldType::Int64);
                    } else {
                        seen_types.push(FieldType::Float64);
                    }
                }
                serde_json::Value::String(s) => {
                    seen_types.push(self.infer_string_type(s));
                }
                serde_json::Value::Array(_) => {
                    seen_types.push(FieldType::Json);
                }
                serde_json::Value::Object(_) => {
                    seen_types.push(FieldType::Json);
                }
            }
        }

        let null_fraction = if total > 0 {
            null_count as f64 / total as f64
        } else {
            1.0
        };
        let nullable = null_fraction > self.nullable_threshold;
        let inferred_type = coalesce_types(&seen_types);

        SchemaField::new(name, inferred_type).nullable_if(nullable)
    }

    fn infer_field_from_strings(&self, name: &str, values: &[&str]) -> SchemaField {
        let total = values.len();
        let mut null_count = 0usize;
        let mut seen_types: Vec<FieldType> = Vec::new();

        for s in values {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                null_count += 1;
                continue;
            }
            seen_types.push(self.infer_string_type(trimmed));
        }

        let null_fraction = if total > 0 {
            null_count as f64 / total as f64
        } else {
            1.0
        };
        let nullable = null_fraction > self.nullable_threshold;
        let inferred_type = coalesce_types(&seen_types);

        SchemaField::new(name, inferred_type).nullable_if(nullable)
    }

    fn infer_string_type(&self, s: &str) -> FieldType {
        if self.try_numeric_strings {
            if s.parse::<i64>().is_ok() {
                return FieldType::Int64;
            }
            if s.parse::<f64>().is_ok() {
                return FieldType::Float64;
            }
        }

        if self.try_temporal {
            if s.parse::<NaiveDate>().is_ok() {
                return FieldType::Date;
            }
            if s.parse::<DateTime<Utc>>().is_ok() {
                return FieldType::Timestamp;
            }
        }

        match s.to_lowercase().as_str() {
            "true" | "false" | "yes" | "no" | "1" | "0" => return FieldType::Boolean,
            _ => {}
        }

        FieldType::Utf8
    }
}

/// Helper method for conditional nullability.
trait NullableIf: Sized {
    fn nullable_if(self, nullable: bool) -> Self;
}

impl NullableIf for SchemaField {
    fn nullable_if(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }
}

/// Coalesce a collection of observed field types into the most general type.
fn coalesce_types(types: &[FieldType]) -> FieldType {
    if types.is_empty() {
        return FieldType::Utf8;
    }

    let has_float = types.iter().any(|t| matches!(t, FieldType::Float64 | FieldType::Float32));
    let has_int = types.iter().any(|t| matches!(t, FieldType::Int64 | FieldType::Int32));
    let has_bool = types.iter().any(|t| matches!(t, FieldType::Boolean));
    let has_string = types.iter().any(|t| matches!(t, FieldType::Utf8));
    let has_date = types.iter().any(|t| matches!(t, FieldType::Date));
    let has_ts = types.iter().any(|t| matches!(t, FieldType::Timestamp));
    let has_json = types.iter().any(|t| matches!(t, FieldType::Json));

    if has_json {
        return FieldType::Json;
    }
    if has_string {
        return FieldType::Utf8;
    }
    if has_ts || (has_date && has_ts) {
        return FieldType::Timestamp;
    }
    if has_date {
        return FieldType::Date;
    }
    if has_float {
        return FieldType::Float64;
    }
    if has_int {
        return FieldType::Int64;
    }
    if has_bool {
        return FieldType::Boolean;
    }

    FieldType::Utf8
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Constraint, FieldType, Schema, SchemaField};
    use std::collections::HashMap;

    fn make_row(pairs: &[(&str, serde_json::Value)]) -> HashMap<String, serde_json::Value> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn test_validate_valid_row() {
        let schema = Schema::builder()
            .field(SchemaField::new("id", FieldType::Int64).not_nullable())
            .field(SchemaField::new("name", FieldType::Utf8).nullable())
            .build();

        let validator = SchemaValidator::new(schema, CoercionMode::Lenient);
        let row = make_row(&[
            ("id", serde_json::json!(1i64)),
            ("name", serde_json::json!("Alice")),
        ]);
        let report = validator.validate_row(&row, 0);
        assert!(report.is_valid, "Expected no violations: {:?}", report.violations);
    }

    #[test]
    fn test_validate_missing_required_field() {
        let schema = Schema::builder()
            .field(SchemaField::new("id", FieldType::Int64).not_nullable())
            .build();

        let validator = SchemaValidator::new(schema, CoercionMode::Strict);
        let row = make_row(&[]);
        let report = validator.validate_row(&row, 0);
        assert!(!report.is_valid);
        assert_eq!(report.violations.len(), 1);
        assert!(report.violations[0].message.contains("NOT NULL"));
    }

    #[test]
    fn test_type_coercion_lenient_string_to_int() {
        let coercion = TypeCoercion::new(CoercionMode::Lenient);
        let value = serde_json::json!("42");
        let result = coercion.coerce(&value, &FieldType::Int64).unwrap();
        assert_eq!(result, TypedValue::Int(42));
    }

    #[test]
    fn test_type_coercion_strict_failure() {
        let coercion = TypeCoercion::new(CoercionMode::Strict);
        let value = serde_json::json!("not_a_number");
        assert!(coercion.coerce(&value, &FieldType::Int64).is_err());
    }

    #[test]
    fn test_constraint_range_violation() {
        let schema = Schema::builder()
            .field(
                SchemaField::new("score", FieldType::Float64)
                    .with_constraint(Constraint::Range {
                        min: serde_json::json!(0.0),
                        max: serde_json::json!(100.0),
                    }),
            )
            .build();

        let validator = SchemaValidator::new(schema, CoercionMode::Lenient);
        let row = make_row(&[("score", serde_json::json!(150.0))]);
        let report = validator.validate_row(&row, 0);
        assert!(!report.is_valid);
        assert!(report.violations[0].constraint.as_deref() == Some("range"));
    }

    #[test]
    fn test_unique_constraint() {
        let schema = Schema::builder()
            .field(
                SchemaField::new("id", FieldType::Int64)
                    .with_constraint(Constraint::Unique),
            )
            .build();

        let validator = SchemaValidator::new(schema, CoercionMode::Lenient);
        let rows = vec![
            make_row(&[("id", serde_json::json!(1))]),
            make_row(&[("id", serde_json::json!(1))]), // duplicate
            make_row(&[("id", serde_json::json!(2))]),
        ];
        let report = validator.validate_rows(&rows);
        assert!(!report.is_valid);
        let unique_violations: Vec<_> = report
            .violations
            .iter()
            .filter(|v| v.constraint.as_deref() == Some("unique"))
            .collect();
        assert_eq!(unique_violations.len(), 1);
    }

    #[test]
    fn test_schema_inference_from_json() {
        let rows = vec![
            serde_json::json!({"id": 1, "name": "Alice", "score": 99.5}),
            serde_json::json!({"id": 2, "name": "Bob",   "score": 88.0}),
            serde_json::json!({"id": 3, "name": null,    "score": 77.3}),
        ];
        let inference = SchemaInference::default();
        let schema = inference.infer_from_json_rows(&rows, None).unwrap();

        let id_field = schema.field("id").expect("id field missing");
        assert!(id_field.field_type.is_integer() || id_field.field_type.is_float());

        let score_field = schema.field("score").expect("score field missing");
        assert!(score_field.field_type.is_float());
    }

    #[test]
    fn test_schema_inference_from_csv() {
        let headers = vec!["id".to_string(), "value".to_string(), "label".to_string()];
        let rows = vec![
            vec!["1".to_string(), "3.14".to_string(), "hello".to_string()],
            vec!["2".to_string(), "2.71".to_string(), "world".to_string()],
        ];
        let inference = SchemaInference::default();
        let schema = inference.infer_from_csv_rows(&headers, &rows, None);
        assert_eq!(schema.len(), 3);
        assert_eq!(schema.fields[0].field_type, FieldType::Int64);
        assert_eq!(schema.fields[1].field_type, FieldType::Float64);
        assert_eq!(schema.fields[2].field_type, FieldType::Utf8);
    }

    #[test]
    fn test_aggressive_coercion_to_string() {
        let coercion = TypeCoercion::new(CoercionMode::Aggressive);
        let value = serde_json::json!({"key": "nested"});
        let result = coercion.coerce(&value, &FieldType::Utf8).unwrap();
        if let TypedValue::Utf8(s) = result {
            assert!(s.contains("key"));
        } else {
            panic!("Expected Utf8");
        }
    }
}
