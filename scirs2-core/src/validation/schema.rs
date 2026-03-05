//! Schema validation for arrays and tabular data
//!
//! This module provides high-level schema types for validating array shapes,
//! dtypes, value ranges, and tabular data layouts. It supplements the existing
//! `data` sub-module with ndarray-aware, type-parameterized schemas.
//!
//! ## Features
//!
//! - `ArraySchema`: shape, dtype tag, value-range, and custom constraint checks
//! - `DataFrameSchema`: column-oriented tabular validation
//! - `ValidationResult`: rich error collection with field-path tracking
//! - `SchemaBuilder`: fluent builder API for composing schemas
//! - `Constraint` enum: `NotNull`, `Range`, `OneOf`, `Regex`, `Custom`
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::validation::schema::{ArraySchema, Constraint, SchemaValidationResult};
//!
//! let schema = ArraySchema::new()
//!     .with_shape(vec![3, 4])
//!     .with_dtype_tag("f64")
//!     .with_constraint(Constraint::Range { min: 0.0, max: 1.0 });
//!
//! // Validation returns a rich result even on failure
//! let result: SchemaValidationResult = schema.validate_values(&[0.5_f64, 0.2, 0.9]);
//! assert!(result.is_valid());
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constraint enum
// ---------------------------------------------------------------------------

/// A single validation constraint that can be applied to a field or array.
#[derive(Clone)]
pub enum Constraint {
    /// Value must not be null / NaN. For floats this checks `is_finite()`.
    NotNull,
    /// Numeric range (inclusive on both ends).
    Range {
        /// Minimum allowed value (inclusive).
        min: f64,
        /// Maximum allowed value (inclusive).
        max: f64,
    },
    /// The value (converted to String via Display) must be in the given set.
    OneOf(Vec<String>),
    /// The string value must match the given regex pattern.
    #[cfg(feature = "validation")]
    Regex(String),
    /// A custom predicate applied to the stringified value.
    Custom {
        /// Human-readable description used in error messages.
        description: String,
        /// Predicate: receives the field value as a string; returns `Ok(())`
        /// on success or `Err(reason)` on failure.
        predicate: std::sync::Arc<dyn Fn(&str) -> Result<(), String> + Send + Sync>,
    },
}

impl std::fmt::Debug for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Constraint::NotNull => write!(f, "Constraint::NotNull"),
            Constraint::Range { min, max } => write!(f, "Constraint::Range {{ min: {}, max: {} }}", min, max),
            Constraint::OneOf(vals) => write!(f, "Constraint::OneOf({:?})", vals),
            #[cfg(feature = "validation")]
            Constraint::Regex(r) => write!(f, "Constraint::Regex({:?})", r),
            Constraint::Custom { description, .. } => write!(f, "Constraint::Custom {{ description: {:?} }}", description),
        }
    }
}

impl Constraint {
    /// Validate a stringified value against this constraint.
    ///
    /// Returns `Ok(())` when the constraint is satisfied and `Err(message)` otherwise.
    pub fn check(&self, value: &str) -> Result<(), String> {
        match self {
            Constraint::NotNull => {
                if value.is_empty() || value == "null" || value == "NaN" || value == "nan" {
                    Err(format!("Value '{value}' violates NotNull constraint"))
                } else {
                    Ok(())
                }
            }
            Constraint::Range { min, max } => {
                let parsed: f64 = value
                    .parse()
                    .map_err(|_| format!("Cannot parse '{value}' as a number for Range check"))?;
                if parsed < *min || parsed > *max {
                    Err(format!(
                        "Value {parsed} is outside range [{min}, {max}]"
                    ))
                } else {
                    Ok(())
                }
            }
            Constraint::OneOf(options) => {
                if options.iter().any(|opt| opt == value) {
                    Ok(())
                } else {
                    Err(format!(
                        "Value '{value}' is not one of: {}",
                        options.join(", ")
                    ))
                }
            }
            #[cfg(feature = "validation")]
            Constraint::Regex(pattern) => {
                let re = regex::Regex::new(pattern).map_err(|e| {
                    format!("Invalid regex pattern '{pattern}': {e}")
                })?;
                if re.is_match(value) {
                    Ok(())
                } else {
                    Err(format!("Value '{value}' does not match pattern '{pattern}'"))
                }
            }
            Constraint::Custom { description, predicate } => {
                predicate(value).map_err(|reason| {
                    format!("Custom constraint '{description}' failed: {reason}")
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Validation issue & result
// ---------------------------------------------------------------------------

/// A single validation issue with a location path and human-readable message.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Dot-separated path to the field or array element where the issue occurred.
    pub field_path: String,
    /// Human-readable error message.
    pub message: String,
    /// Whether this issue is fatal (error) or advisory (warning).
    pub is_error: bool,
}

impl ValidationIssue {
    /// Create a new error-level issue.
    pub fn error(field_path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field_path: field_path.into(),
            message: message.into(),
            is_error: true,
        }
    }

    /// Create a new warning-level issue.
    pub fn warning(field_path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field_path: field_path.into(),
            message: message.into(),
            is_error: false,
        }
    }
}

/// Rich result of a schema validation pass.
///
/// Collects all issues rather than stopping at the first error, enabling
/// callers to present comprehensive diagnostics.
#[derive(Debug, Clone, Default)]
pub struct SchemaValidationResult {
    issues: Vec<ValidationIssue>,
}

impl SchemaValidationResult {
    /// Create an empty (passing) result.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an error-level issue.
    pub fn add_error(&mut self, field_path: impl Into<String>, message: impl Into<String>) {
        self.issues.push(ValidationIssue::error(field_path, message));
    }

    /// Record a warning-level issue.
    pub fn add_warning(&mut self, field_path: impl Into<String>, message: impl Into<String>) {
        self.issues.push(ValidationIssue::warning(field_path, message));
    }

    /// Merge another result into this one.
    pub fn merge(&mut self, other: SchemaValidationResult) {
        self.issues.extend(other.issues);
    }

    /// Returns `true` when no error-level issues have been recorded.
    pub fn is_valid(&self) -> bool {
        !self.issues.iter().any(|i| i.is_error)
    }

    /// All issues (errors and warnings).
    pub fn issues(&self) -> &[ValidationIssue] {
        &self.issues
    }

    /// Only error-level issues.
    pub fn errors(&self) -> impl Iterator<Item = &ValidationIssue> {
        self.issues.iter().filter(|i| i.is_error)
    }

    /// Only warning-level issues.
    pub fn warnings(&self) -> impl Iterator<Item = &ValidationIssue> {
        self.issues.iter().filter(|i| !i.is_error)
    }

    /// Issues at a specific field path (exact match).
    pub fn issues_at(&self, field_path: &str) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.field_path == field_path)
            .collect()
    }

    /// Number of error-level issues.
    pub fn error_count(&self) -> usize {
        self.issues.iter().filter(|i| i.is_error).count()
    }

    /// Number of warning-level issues.
    pub fn warning_count(&self) -> usize {
        self.issues.iter().filter(|i| !i.is_error).count()
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        let ec = self.error_count();
        let wc = self.warning_count();
        if ec == 0 && wc == 0 {
            "Validation passed with no issues.".to_string()
        } else {
            format!("Validation: {ec} error(s), {wc} warning(s).")
        }
    }
}

// ---------------------------------------------------------------------------
// ArraySchema
// ---------------------------------------------------------------------------

/// Schema for validating an ndarray-like array.
///
/// Checks shape, an optional dtype tag, value ranges, and arbitrary constraints
/// applied element-wise.
#[derive(Debug, Clone, Default)]
pub struct ArraySchema {
    /// Expected shape, e.g. `[3, 4]`. `None` means any shape is accepted.
    pub expected_shape: Option<Vec<usize>>,
    /// Human-readable dtype tag such as `"f32"`, `"f64"`, `"i32"`.
    pub dtype_tag: Option<String>,
    /// Minimum allowed value (applied to each element via `f64` coercion).
    pub value_min: Option<f64>,
    /// Maximum allowed value.
    pub value_max: Option<f64>,
    /// Whether all elements must be finite (no NaN or infinity).
    pub require_finite: bool,
    /// Whether all elements must be non-negative.
    pub require_non_negative: bool,
    /// Whether all elements must lie in [0, 1] (probability check).
    pub require_probability: bool,
    /// Extra element-level constraints applied to the stringified value.
    pub constraints: Vec<Constraint>,
    /// Optional human-readable name used in error messages.
    pub name: Option<String>,
}

impl ArraySchema {
    /// Create a new, empty `ArraySchema` (accepts everything by default).
    pub fn new() -> Self {
        Self::default()
    }

    /// Restrict the schema to a specific shape.
    pub fn with_shape(mut self, shape: Vec<usize>) -> Self {
        self.expected_shape = Some(shape);
        self
    }

    /// Attach a dtype tag for documentation / error messages.
    pub fn with_dtype_tag(mut self, tag: impl Into<String>) -> Self {
        self.dtype_tag = Some(tag.into());
        self
    }

    /// Require all elements to be ≥ `min`.
    pub fn with_value_min(mut self, min: f64) -> Self {
        self.value_min = Some(min);
        self
    }

    /// Require all elements to be ≤ `max`.
    pub fn with_value_max(mut self, max: f64) -> Self {
        self.value_max = Some(max);
        self
    }

    /// Require all elements to be finite.
    pub fn require_finite(mut self) -> Self {
        self.require_finite = true;
        self
    }

    /// Require all elements to be non-negative.
    pub fn require_non_negative(mut self) -> Self {
        self.require_non_negative = true;
        self
    }

    /// Require all elements to lie in [0, 1].
    pub fn require_probability(mut self) -> Self {
        self.require_probability = true;
        self
    }

    /// Add an element-level constraint.
    pub fn with_constraint(mut self, constraint: Constraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Set a human-readable name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Validate the actual shape against the expected shape.
    pub fn validate_shape(&self, actual: &[usize]) -> SchemaValidationResult {
        let mut result = SchemaValidationResult::new();
        let field = self.name.as_deref().unwrap_or("array");
        if let Some(expected) = &self.expected_shape {
            if actual != expected.as_slice() {
                result.add_error(
                    field,
                    format!(
                        "Shape mismatch: expected {expected:?}, got {actual:?}"
                    ),
                );
            }
        }
        result
    }

    /// Validate a slice of `f64` values element-wise (range, finite, probability,
    /// and extra constraints).
    ///
    /// This is the generic validation path; callers can convert their typed array
    /// to `f64` before calling.
    pub fn validate_values(&self, values: &[f64]) -> SchemaValidationResult {
        let mut result = SchemaValidationResult::new();
        let field = self.name.as_deref().unwrap_or("array");

        for (idx, &v) in values.iter().enumerate() {
            let path = format!("{field}[{idx}]");

            if self.require_finite && !v.is_finite() {
                result.add_error(&path, format!("Value {v} is not finite"));
            }
            if self.require_non_negative && v < 0.0 {
                result.add_error(&path, format!("Value {v} is negative"));
            }
            if self.require_probability && (v < 0.0 || v > 1.0) {
                result.add_error(
                    &path,
                    format!("Value {v} is not a probability (must be in [0, 1])"),
                );
            }
            if let Some(min) = self.value_min {
                if v < min {
                    result.add_error(
                        &path,
                        format!("Value {v} is below minimum {min}"),
                    );
                }
            }
            if let Some(max) = self.value_max {
                if v > max {
                    result.add_error(
                        &path,
                        format!("Value {v} exceeds maximum {max}"),
                    );
                }
            }

            // Apply extra constraints via string representation
            let value_str = v.to_string();
            for constraint in &self.constraints {
                if let Err(msg) = constraint.check(&value_str) {
                    result.add_error(&path, msg);
                }
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// ColumnSchema
// ---------------------------------------------------------------------------

/// Schema for a single column in a `DataFrameSchema`.
#[derive(Debug, Clone)]
pub struct ColumnSchema {
    /// Column name.
    pub name: String,
    /// Human-readable dtype tag.
    pub dtype_tag: String,
    /// Whether the column is required (must be present).
    pub required: bool,
    /// Whether null / NaN values are forbidden.
    pub not_null: bool,
    /// Optional minimum value.
    pub value_min: Option<f64>,
    /// Optional maximum value.
    pub value_max: Option<f64>,
    /// Optional set of allowed string values.
    pub one_of: Option<Vec<String>>,
    /// Extra constraints applied to each cell's string representation.
    pub constraints: Vec<Constraint>,
}

impl ColumnSchema {
    /// Create a minimal column schema.
    pub fn new(name: impl Into<String>, dtype_tag: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dtype_tag: dtype_tag.into(),
            required: true,
            not_null: false,
            value_min: None,
            value_max: None,
            one_of: None,
            constraints: Vec::new(),
        }
    }

    /// Mark column as optional (may be absent).
    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    /// Forbid null/NaN values in this column.
    pub fn not_null(mut self) -> Self {
        self.not_null = true;
        self
    }

    /// Set minimum value.
    pub fn with_min(mut self, min: f64) -> Self {
        self.value_min = Some(min);
        self
    }

    /// Set maximum value.
    pub fn with_max(mut self, max: f64) -> Self {
        self.value_max = Some(max);
        self
    }

    /// Restrict to an enumerated set of string values.
    pub fn one_of(mut self, options: Vec<String>) -> Self {
        self.one_of = Some(options);
        self
    }

    /// Add an extra constraint.
    pub fn with_constraint(mut self, c: Constraint) -> Self {
        self.constraints.push(c);
        self
    }

    /// Validate a column of string cells.
    pub fn validate_cells(&self, cells: &[Option<String>]) -> SchemaValidationResult {
        let mut result = SchemaValidationResult::new();
        for (row, cell) in cells.iter().enumerate() {
            let path = format!("{}[{}]", self.name, row);
            match cell {
                None => {
                    if self.not_null {
                        result.add_error(&path, "Null value violates not_null constraint");
                    }
                }
                Some(value) => {
                    if let Some(ref options) = self.one_of {
                        if !options.iter().any(|o| o == value) {
                            result.add_error(
                                &path,
                                format!(
                                    "Value '{value}' is not one of: {}",
                                    options.join(", ")
                                ),
                            );
                        }
                    }
                    // Numeric checks (best-effort parse)
                    if self.value_min.is_some() || self.value_max.is_some() {
                        if let Ok(v) = value.parse::<f64>() {
                            if let Some(min) = self.value_min {
                                if v < min {
                                    result.add_error(
                                        &path,
                                        format!("Value {v} is below minimum {min}"),
                                    );
                                }
                            }
                            if let Some(max) = self.value_max {
                                if v > max {
                                    result.add_error(
                                        &path,
                                        format!("Value {v} exceeds maximum {max}"),
                                    );
                                }
                            }
                        } else {
                            result.add_warning(
                                &path,
                                format!(
                                    "Cannot parse '{value}' as numeric for range check (dtype={})",
                                    self.dtype_tag
                                ),
                            );
                        }
                    }
                    for constraint in &self.constraints {
                        if let Err(msg) = constraint.check(value) {
                            result.add_error(&path, msg);
                        }
                    }
                }
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// DataFrameSchema
// ---------------------------------------------------------------------------

/// Schema for validating tabular (DataFrame-like) data.
///
/// A `DataFrameSchema` holds a map of column names to `ColumnSchema`s
/// and validates presence, row-counts, and per-cell constraints.
#[derive(Debug, Clone, Default)]
pub struct DataFrameSchema {
    /// Column schemas, keyed by column name.
    pub columns: HashMap<String, ColumnSchema>,
    /// If `Some(n)`, all columns must have exactly `n` rows.
    pub expected_rows: Option<usize>,
    /// Allow extra columns not listed in `columns`.
    pub allow_extra_columns: bool,
    /// Optional name for this schema.
    pub name: Option<String>,
}

impl DataFrameSchema {
    /// Create an empty schema.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a human-readable name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Require a specific number of rows.
    pub fn with_expected_rows(mut self, n: usize) -> Self {
        self.expected_rows = Some(n);
        self
    }

    /// Allow columns not declared in this schema.
    pub fn allow_extra_columns(mut self) -> Self {
        self.allow_extra_columns = true;
        self
    }

    /// Add (or replace) a column schema.
    pub fn add_column(mut self, col: ColumnSchema) -> Self {
        self.columns.insert(col.name.clone(), col);
        self
    }

    /// Validate a map of column-name → cell-vector.
    ///
    /// `data` maps each column name to a vector of optional string values.
    pub fn validate(
        &self,
        data: &HashMap<String, Vec<Option<String>>>,
    ) -> SchemaValidationResult {
        let mut result = SchemaValidationResult::new();
        let schema_name = self.name.as_deref().unwrap_or("dataframe");

        // Check for required columns that are missing
        for (col_name, col_schema) in &self.columns {
            if col_schema.required && !data.contains_key(col_name) {
                result.add_error(
                    schema_name,
                    format!("Required column '{col_name}' is missing"),
                );
            }
        }

        // Check for extra columns (if not allowed)
        if !self.allow_extra_columns {
            for col_name in data.keys() {
                if !self.columns.contains_key(col_name) {
                    result.add_error(
                        schema_name,
                        format!("Unexpected column '{col_name}' is not declared in schema"),
                    );
                }
            }
        }

        // Row count check
        if let Some(expected_rows) = self.expected_rows {
            for (col_name, cells) in data {
                if cells.len() != expected_rows {
                    result.add_error(
                        col_name.as_str(),
                        format!(
                            "Column '{col_name}' has {} rows, expected {expected_rows}",
                            cells.len()
                        ),
                    );
                }
            }
        }

        // Per-column cell validation
        for (col_name, cells) in data {
            if let Some(col_schema) = self.columns.get(col_name) {
                let col_result = col_schema.validate_cells(cells);
                result.merge(col_result);
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// SchemaBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing `ArraySchema` objects.
///
/// ```rust
/// use scirs2_core::validation::schema::{SchemaBuilder, Constraint};
///
/// let schema = SchemaBuilder::new()
///     .shape(vec![10, 5])
///     .dtype("f64")
///     .min(0.0)
///     .max(1.0)
///     .finite()
///     .build_array();
/// ```
#[derive(Debug, Clone, Default)]
pub struct SchemaBuilder {
    inner: ArraySchema,
}

impl SchemaBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the expected shape.
    pub fn shape(mut self, shape: Vec<usize>) -> Self {
        self.inner.expected_shape = Some(shape);
        self
    }

    /// Set the dtype tag.
    pub fn dtype(mut self, tag: impl Into<String>) -> Self {
        self.inner.dtype_tag = Some(tag.into());
        self
    }

    /// Set minimum value.
    pub fn min(mut self, min: f64) -> Self {
        self.inner.value_min = Some(min);
        self
    }

    /// Set maximum value.
    pub fn max(mut self, max: f64) -> Self {
        self.inner.value_max = Some(max);
        self
    }

    /// Require finite values.
    pub fn finite(mut self) -> Self {
        self.inner.require_finite = true;
        self
    }

    /// Require non-negative values.
    pub fn non_negative(mut self) -> Self {
        self.inner.require_non_negative = true;
        self
    }

    /// Require probability values (in [0,1]).
    pub fn probability(mut self) -> Self {
        self.inner.require_probability = true;
        self
    }

    /// Add an element-level constraint.
    pub fn constraint(mut self, c: Constraint) -> Self {
        self.inner.constraints.push(c);
        self
    }

    /// Set a human-readable name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.inner.name = Some(name.into());
        self
    }

    /// Consume the builder and produce an `ArraySchema`.
    pub fn build_array(self) -> ArraySchema {
        self.inner
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_not_null_ok() {
        assert!(Constraint::NotNull.check("hello").is_ok());
        assert!(Constraint::NotNull.check("42").is_ok());
    }

    #[test]
    fn test_constraint_not_null_fail() {
        assert!(Constraint::NotNull.check("").is_err());
        assert!(Constraint::NotNull.check("null").is_err());
        assert!(Constraint::NotNull.check("NaN").is_err());
    }

    #[test]
    fn test_constraint_range_ok() {
        let c = Constraint::Range { min: 0.0, max: 1.0 };
        assert!(c.check("0.5").is_ok());
        assert!(c.check("0.0").is_ok());
        assert!(c.check("1.0").is_ok());
    }

    #[test]
    fn test_constraint_range_fail() {
        let c = Constraint::Range { min: 0.0, max: 1.0 };
        assert!(c.check("-0.1").is_err());
        assert!(c.check("1.1").is_err());
        assert!(c.check("not_a_number").is_err());
    }

    #[test]
    fn test_constraint_one_of_ok() {
        let c = Constraint::OneOf(vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        assert!(c.check("a").is_ok());
        assert!(c.check("c").is_ok());
    }

    #[test]
    fn test_constraint_one_of_fail() {
        let c = Constraint::OneOf(vec!["a".to_string(), "b".to_string()]);
        assert!(c.check("d").is_err());
    }

    #[test]
    fn test_constraint_custom() {
        let c = Constraint::Custom {
            description: "must be positive integer".to_string(),
            predicate: std::sync::Arc::new(|v: &str| {
                v.parse::<i64>()
                    .map_err(|_| "not an integer".to_string())
                    .and_then(|n| {
                        if n > 0 {
                            Ok(())
                        } else {
                            Err("not positive".to_string())
                        }
                    })
            }),
        };
        assert!(c.check("42").is_ok());
        assert!(c.check("0").is_err());
        assert!(c.check("abc").is_err());
    }

    #[test]
    fn test_schema_validation_result_merge() {
        let mut r1 = SchemaValidationResult::new();
        r1.add_error("field_a", "error 1");
        let mut r2 = SchemaValidationResult::new();
        r2.add_warning("field_b", "warning 1");
        r1.merge(r2);

        assert_eq!(r1.error_count(), 1);
        assert_eq!(r1.warning_count(), 1);
        assert!(!r1.is_valid());
    }

    #[test]
    fn test_array_schema_shape_ok() {
        let schema = ArraySchema::new().with_shape(vec![2, 3]);
        let result = schema.validate_shape(&[2, 3]);
        assert!(result.is_valid());
    }

    #[test]
    fn test_array_schema_shape_fail() {
        let schema = ArraySchema::new().with_shape(vec![2, 3]);
        let result = schema.validate_shape(&[2, 4]);
        assert!(!result.is_valid());
        assert_eq!(result.error_count(), 1);
    }

    #[test]
    fn test_array_schema_values_finite() {
        let schema = ArraySchema::new().require_finite();
        let ok = schema.validate_values(&[1.0, 2.0, 3.0]);
        assert!(ok.is_valid());
        let fail = schema.validate_values(&[1.0, f64::NAN, 3.0]);
        assert!(!fail.is_valid());
    }

    #[test]
    fn test_array_schema_values_non_negative() {
        let schema = ArraySchema::new().require_non_negative();
        assert!(schema.validate_values(&[0.0, 1.0, 2.0]).is_valid());
        assert!(!schema.validate_values(&[1.0, -0.5]).is_valid());
    }

    #[test]
    fn test_array_schema_values_probability() {
        let schema = ArraySchema::new().require_probability();
        assert!(schema.validate_values(&[0.0, 0.5, 1.0]).is_valid());
        assert!(!schema.validate_values(&[0.5, 1.1]).is_valid());
        assert!(!schema.validate_values(&[-0.1, 0.5]).is_valid());
    }

    #[test]
    fn test_array_schema_range_constraint() {
        let schema = ArraySchema::new()
            .with_constraint(Constraint::Range { min: 10.0, max: 20.0 });
        assert!(schema.validate_values(&[10.0, 15.0, 20.0]).is_valid());
        assert!(!schema.validate_values(&[10.0, 25.0]).is_valid());
    }

    #[test]
    fn test_schema_builder() {
        let schema = SchemaBuilder::new()
            .shape(vec![3])
            .dtype("f64")
            .min(0.0)
            .max(1.0)
            .finite()
            .name("probabilities")
            .build_array();

        assert_eq!(schema.expected_shape, Some(vec![3]));
        assert_eq!(schema.dtype_tag, Some("f64".to_string()));
        assert_eq!(schema.value_min, Some(0.0));
        assert_eq!(schema.value_max, Some(1.0));
        assert!(schema.require_finite);
        assert_eq!(schema.name, Some("probabilities".to_string()));
    }

    #[test]
    fn test_column_schema_not_null() {
        let col = ColumnSchema::new("age", "i32").not_null();
        let cells: Vec<Option<String>> = vec![Some("25".to_string()), None];
        let result = col.validate_cells(&cells);
        assert!(!result.is_valid());
        assert_eq!(result.error_count(), 1);
    }

    #[test]
    fn test_column_schema_one_of() {
        let col = ColumnSchema::new("status", "str")
            .one_of(vec!["active".to_string(), "inactive".to_string()]);
        let ok_cells = vec![Some("active".to_string()), Some("inactive".to_string())];
        assert!(col.validate_cells(&ok_cells).is_valid());
        let bad_cells = vec![Some("pending".to_string())];
        assert!(!col.validate_cells(&bad_cells).is_valid());
    }

    #[test]
    fn test_column_schema_range() {
        let col = ColumnSchema::new("score", "f64")
            .with_min(0.0)
            .with_max(100.0);
        let ok = vec![Some("50.0".to_string()), Some("0.0".to_string())];
        assert!(col.validate_cells(&ok).is_valid());
        let bad = vec![Some("150.0".to_string())];
        assert!(!col.validate_cells(&bad).is_valid());
    }

    #[test]
    fn test_dataframe_schema_missing_required_column() {
        let schema = DataFrameSchema::new()
            .add_column(ColumnSchema::new("name", "str"))
            .add_column(ColumnSchema::new("age", "i32"));

        let mut data: HashMap<String, Vec<Option<String>>> = HashMap::new();
        data.insert(
            "name".to_string(),
            vec![Some("Alice".to_string()), Some("Bob".to_string())],
        );
        // "age" is missing

        let result = schema.validate(&data);
        assert!(!result.is_valid());
    }

    #[test]
    fn test_dataframe_schema_extra_column_rejected() {
        let schema = DataFrameSchema::new()
            .add_column(ColumnSchema::new("name", "str"));

        let mut data: HashMap<String, Vec<Option<String>>> = HashMap::new();
        data.insert("name".to_string(), vec![Some("Alice".to_string())]);
        data.insert("extra".to_string(), vec![Some("x".to_string())]);

        let result = schema.validate(&data);
        assert!(!result.is_valid());
    }

    #[test]
    fn test_dataframe_schema_extra_column_allowed() {
        let schema = DataFrameSchema::new()
            .add_column(ColumnSchema::new("name", "str"))
            .allow_extra_columns();

        let mut data: HashMap<String, Vec<Option<String>>> = HashMap::new();
        data.insert("name".to_string(), vec![Some("Alice".to_string())]);
        data.insert("extra".to_string(), vec![Some("x".to_string())]);

        let result = schema.validate(&data);
        assert!(result.is_valid());
    }

    #[test]
    fn test_dataframe_schema_row_count_mismatch() {
        let schema = DataFrameSchema::new()
            .with_expected_rows(3)
            .add_column(ColumnSchema::new("id", "i32").optional());

        let mut data: HashMap<String, Vec<Option<String>>> = HashMap::new();
        data.insert(
            "id".to_string(),
            vec![Some("1".to_string()), Some("2".to_string())],
        );

        let result = schema.validate(&data);
        assert!(!result.is_valid());
    }

    #[test]
    fn test_schema_validation_result_summary() {
        let mut r = SchemaValidationResult::new();
        assert!(r.summary().contains("no issues"));
        r.add_error("f", "oops");
        assert!(r.summary().contains("1 error"));
    }

    #[test]
    fn test_issues_at_field_path() {
        let mut r = SchemaValidationResult::new();
        r.add_error("col_a", "problem 1");
        r.add_error("col_a", "problem 2");
        r.add_error("col_b", "problem 3");

        let at_a = r.issues_at("col_a");
        assert_eq!(at_a.len(), 2);
        let at_b = r.issues_at("col_b");
        assert_eq!(at_b.len(), 1);
    }
}
