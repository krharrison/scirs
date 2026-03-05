//! Configuration validation utilities
//!
//! This module provides types and functions for validating
//! `HashMap<String, serde_json::Value>` configuration maps, with support for:
//!
//! - Required and optional field declarations
//! - Type checking per field
//! - Range/pattern constraints on values
//! - Nested config validation (via recursive delegation)
//! - Default value injection into validated configs
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::validation::config::{ConfigSchema, FieldSpec, ValueType, ConfigValidator};
//! use std::collections::HashMap;
//! use serde_json::Value;
//!
//! let schema = ConfigSchema::new()
//!     .field("host", FieldSpec::required(ValueType::String))
//!     .field("port", FieldSpec::required(ValueType::Integer)
//!         .with_range(1.0, 65535.0))
//!     .field("timeout_ms", FieldSpec::optional(ValueType::Integer)
//!         .with_default(Value::from(5000)));
//!
//! let mut cfg: HashMap<String, Value> = HashMap::new();
//! cfg.insert("host".to_string(), Value::String("localhost".to_string()));
//! cfg.insert("port".to_string(), Value::from(8080));
//!
//! let validator = ConfigValidator::new(schema);
//! let result = validator.validate_and_fill(&mut cfg).expect("valid config");
//! assert!(result.is_valid());
//! // Default was injected
//! assert_eq!(cfg.get("timeout_ms"), Some(&Value::from(5000)));
//! ```

use std::collections::HashMap;
use serde_json::Value;

// ---------------------------------------------------------------------------
// ValueType
// ---------------------------------------------------------------------------

/// The expected JSON/value type for a configuration field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueType {
    /// `true` / `false`
    Boolean,
    /// Integer (JSON number with no fractional part)
    Integer,
    /// Floating-point number (any JSON number)
    Float,
    /// UTF-8 string
    String,
    /// A JSON array
    Array,
    /// A JSON object (nested config)
    Object,
    /// Any value is accepted (no type check)
    Any,
}

impl ValueType {
    /// Returns `true` when `value` satisfies this type constraint.
    pub fn matches(&self, value: &Value) -> bool {
        match self {
            ValueType::Boolean => value.is_boolean(),
            ValueType::Integer => value.is_i64() || value.is_u64()
                || value.as_f64().map_or(false, |f| f.fract() == 0.0),
            ValueType::Float => value.is_number(),
            ValueType::String => value.is_string(),
            ValueType::Array => value.is_array(),
            ValueType::Object => value.is_object(),
            ValueType::Any => true,
        }
    }

    /// Human-readable name of the type.
    pub fn name(&self) -> &'static str {
        match self {
            ValueType::Boolean => "boolean",
            ValueType::Integer => "integer",
            ValueType::Float => "float",
            ValueType::String => "string",
            ValueType::Array => "array",
            ValueType::Object => "object",
            ValueType::Any => "any",
        }
    }
}

// ---------------------------------------------------------------------------
// FieldSpec
// ---------------------------------------------------------------------------

/// Specification for a single configuration field.
#[derive(Debug, Clone)]
pub struct FieldSpec {
    /// Whether the field must be present.
    pub required: bool,
    /// Expected value type.
    pub value_type: ValueType,
    /// Optional minimum numeric value.
    pub min_value: Option<f64>,
    /// Optional maximum numeric value.
    pub max_value: Option<f64>,
    /// Optional allowed string values (enum).
    pub one_of: Option<Vec<String>>,
    /// Optional default value injected when the field is absent.
    pub default_value: Option<Value>,
    /// Human-readable description.
    pub description: Option<String>,
    /// Sub-schema for nested object fields.
    pub nested_schema: Option<Box<ConfigSchema>>,
}

impl FieldSpec {
    /// Create a required field with the given type.
    pub fn required(value_type: ValueType) -> Self {
        Self {
            required: true,
            value_type,
            min_value: None,
            max_value: None,
            one_of: None,
            default_value: None,
            description: None,
            nested_schema: None,
        }
    }

    /// Create an optional field with the given type.
    pub fn optional(value_type: ValueType) -> Self {
        Self {
            required: false,
            value_type,
            min_value: None,
            max_value: None,
            one_of: None,
            default_value: None,
            description: None,
            nested_schema: None,
        }
    }

    /// Constrain the field to a numeric range [min, max].
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.min_value = Some(min);
        self.max_value = Some(max);
        self
    }

    /// Restrict the field to one of the provided string values.
    pub fn with_one_of(mut self, options: Vec<String>) -> Self {
        self.one_of = Some(options);
        self
    }

    /// Set a default value that is injected when the field is absent.
    pub fn with_default(mut self, default: Value) -> Self {
        self.default_value = Some(default);
        self
    }

    /// Add a human-readable description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Attach a nested schema for `ValueType::Object` fields.
    pub fn with_nested(mut self, schema: ConfigSchema) -> Self {
        self.nested_schema = Some(Box::new(schema));
        self
    }
}

// ---------------------------------------------------------------------------
// ConfigSchema
// ---------------------------------------------------------------------------

/// Schema describing the expected structure of a configuration map.
#[derive(Debug, Clone, Default)]
pub struct ConfigSchema {
    /// Field specifications, keyed by field name.
    pub fields: HashMap<String, FieldSpec>,
    /// Whether to reject fields not declared in the schema.
    pub strict: bool,
    /// Optional human-readable schema name.
    pub name: Option<String>,
}

impl ConfigSchema {
    /// Create a new empty schema.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the schema name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Reject unknown fields (strict mode).
    pub fn strict(mut self) -> Self {
        self.strict = true;
        self
    }

    /// Add (or replace) a field specification.
    pub fn field(mut self, name: impl Into<String>, spec: FieldSpec) -> Self {
        self.fields.insert(name.into(), spec);
        self
    }
}

// ---------------------------------------------------------------------------
// ConfigValidationError
// ---------------------------------------------------------------------------

/// A single validation error in a configuration map.
#[derive(Debug, Clone)]
pub struct ConfigValidationError {
    /// Dot-separated path to the field that failed.
    pub field_path: String,
    /// Human-readable description of the failure.
    pub message: String,
}

impl ConfigValidationError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field_path: path.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ConfigValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.field_path, self.message)
    }
}

// ---------------------------------------------------------------------------
// ConfigValidationResult
// ---------------------------------------------------------------------------

/// Result of validating a configuration map.
#[derive(Debug, Clone, Default)]
pub struct ConfigValidationResult {
    errors: Vec<ConfigValidationError>,
    warnings: Vec<ConfigValidationError>,
}

impl ConfigValidationResult {
    /// Create an empty (passing) result.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an error.
    pub fn add_error(&mut self, path: impl Into<String>, message: impl Into<String>) {
        self.errors.push(ConfigValidationError::new(path, message));
    }

    /// Add a warning.
    pub fn add_warning(&mut self, path: impl Into<String>, message: impl Into<String>) {
        self.warnings.push(ConfigValidationError::new(path, message));
    }

    /// Merge another result into this one, prepending `prefix` to all paths.
    pub fn merge_with_prefix(&mut self, other: ConfigValidationResult, prefix: &str) {
        for mut e in other.errors {
            e.field_path = format!("{prefix}.{}", e.field_path);
            self.errors.push(e);
        }
        for mut w in other.warnings {
            w.field_path = format!("{prefix}.{}", w.field_path);
            self.warnings.push(w);
        }
    }

    /// `true` when no errors are present.
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// All error-level issues.
    pub fn errors(&self) -> &[ConfigValidationError] {
        &self.errors
    }

    /// All warning-level issues.
    pub fn warnings(&self) -> &[ConfigValidationError] {
        &self.warnings
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        if self.errors.is_empty() && self.warnings.is_empty() {
            return "Config is valid.".to_string();
        }
        let mut s = String::new();
        if !self.errors.is_empty() {
            s.push_str(&format!("{} error(s):\n", self.errors.len()));
            for e in &self.errors {
                s.push_str(&format!("  - {e}\n"));
            }
        }
        if !self.warnings.is_empty() {
            s.push_str(&format!("{} warning(s):\n", self.warnings.len()));
            for w in &self.warnings {
                s.push_str(&format!("  - {w}\n"));
            }
        }
        s
    }
}

// ---------------------------------------------------------------------------
// ConfigValidator
// ---------------------------------------------------------------------------

/// Validates and optionally fills defaults into a `HashMap<String, Value>`.
pub struct ConfigValidator {
    schema: ConfigSchema,
}

impl ConfigValidator {
    /// Create a new validator for the given schema.
    pub fn new(schema: ConfigSchema) -> Self {
        Self { schema }
    }

    /// Validate `config` against the schema and inject default values for
    /// absent optional fields that have a `default_value`.
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` only for internal logic failures (currently
    /// unused; all validation errors are captured in the returned
    /// `ConfigValidationResult`).
    pub fn validate_and_fill(
        &self,
        config: &mut HashMap<String, Value>,
    ) -> Result<ConfigValidationResult, String> {
        let mut result = ConfigValidationResult::new();

        // Check for unknown fields in strict mode
        if self.schema.strict {
            for key in config.keys() {
                if !self.schema.fields.contains_key(key) {
                    result.add_error(key, format!("Unknown field '{key}' not in schema"));
                }
            }
        }

        // Validate declared fields
        for (field_name, spec) in &self.schema.fields {
            match config.get(field_name) {
                None => {
                    if spec.required && spec.default_value.is_none() {
                        result.add_error(
                            field_name,
                            format!("Required field '{field_name}' is missing"),
                        );
                    } else if let Some(default) = &spec.default_value {
                        // Inject the default
                        config.insert(field_name.clone(), default.clone());
                    }
                }
                Some(value) => {
                    Self::validate_value(field_name, value, spec, &mut result);
                }
            }
        }

        Ok(result)
    }

    /// Validate `config` without modifying it (read-only pass).
    pub fn validate(&self, config: &HashMap<String, Value>) -> ConfigValidationResult {
        let mut cloned = config.clone();
        self.validate_and_fill(&mut cloned)
            .unwrap_or_else(|_| ConfigValidationResult::new())
    }

    // -- internal helpers --

    fn validate_value(
        field_name: &str,
        value: &Value,
        spec: &FieldSpec,
        result: &mut ConfigValidationResult,
    ) {
        // Type check
        if !spec.value_type.matches(value) {
            result.add_error(
                field_name,
                format!(
                    "Field '{field_name}' has wrong type: expected {}, got {}",
                    spec.value_type.name(),
                    Self::value_type_name(value)
                ),
            );
            // No point checking further constraints when the type is wrong
            return;
        }

        // Numeric range check
        if let Some(num) = value.as_f64() {
            if let Some(min) = spec.min_value {
                if num < min {
                    result.add_error(
                        field_name,
                        format!("Field '{field_name}' value {num} is below minimum {min}"),
                    );
                }
            }
            if let Some(max) = spec.max_value {
                if num > max {
                    result.add_error(
                        field_name,
                        format!("Field '{field_name}' value {num} exceeds maximum {max}"),
                    );
                }
            }
        }

        // OneOf check (string fields)
        if let Some(options) = &spec.one_of {
            if let Some(s) = value.as_str() {
                if !options.iter().any(|o| o == s) {
                    result.add_error(
                        field_name,
                        format!(
                            "Field '{field_name}' value '{s}' is not one of: {}",
                            options.join(", ")
                        ),
                    );
                }
            }
        }

        // Nested object schema
        if let (Some(nested_schema), Some(obj)) = (&spec.nested_schema, value.as_object()) {
            let mut nested_map: HashMap<String, Value> = obj
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            let nested_validator = ConfigValidator::new(*nested_schema.clone());
            if let Ok(nested_result) = nested_validator.validate_and_fill(&mut nested_map) {
                result.merge_with_prefix(nested_result, field_name);
            }
        }
    }

    fn value_type_name(value: &Value) -> &'static str {
        match value {
            Value::Null => "null",
            Value::Bool(_) => "boolean",
            Value::Number(n) => {
                if n.is_f64() {
                    "float"
                } else {
                    "integer"
                }
            }
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_schema() -> ConfigSchema {
        ConfigSchema::new()
            .with_name("test_config")
            .field("host", FieldSpec::required(ValueType::String))
            .field(
                "port",
                FieldSpec::required(ValueType::Integer).with_range(1.0, 65535.0),
            )
            .field(
                "timeout_ms",
                FieldSpec::optional(ValueType::Integer)
                    .with_default(Value::from(5000)),
            )
            .field(
                "mode",
                FieldSpec::optional(ValueType::String).with_one_of(vec![
                    "dev".to_string(),
                    "prod".to_string(),
                    "test".to_string(),
                ]),
            )
    }

    #[test]
    fn test_valid_config_with_defaults_injected() {
        let schema = make_schema();
        let validator = ConfigValidator::new(schema);

        let mut cfg: HashMap<String, Value> = HashMap::new();
        cfg.insert("host".to_string(), json!("localhost"));
        cfg.insert("port".to_string(), json!(8080));

        let result = validator.validate_and_fill(&mut cfg).expect("no logic error");
        assert!(result.is_valid(), "{}", result.summary());

        // Default injected
        assert_eq!(cfg.get("timeout_ms"), Some(&json!(5000)));
    }

    #[test]
    fn test_missing_required_field() {
        let schema = make_schema();
        let validator = ConfigValidator::new(schema);

        let mut cfg: HashMap<String, Value> = HashMap::new();
        cfg.insert("host".to_string(), json!("localhost"));
        // "port" is missing

        let result = validator.validate_and_fill(&mut cfg).expect("no logic error");
        assert!(!result.is_valid());
        let port_errors: Vec<_> = result
            .errors()
            .iter()
            .filter(|e| e.field_path.contains("port"))
            .collect();
        assert!(!port_errors.is_empty());
    }

    #[test]
    fn test_wrong_type() {
        let schema = make_schema();
        let validator = ConfigValidator::new(schema);

        let mut cfg: HashMap<String, Value> = HashMap::new();
        cfg.insert("host".to_string(), json!(42)); // should be String
        cfg.insert("port".to_string(), json!(8080));

        let result = validator.validate_and_fill(&mut cfg).expect("no logic error");
        assert!(!result.is_valid());
    }

    #[test]
    fn test_out_of_range_integer() {
        let schema = make_schema();
        let validator = ConfigValidator::new(schema);

        let mut cfg: HashMap<String, Value> = HashMap::new();
        cfg.insert("host".to_string(), json!("localhost"));
        cfg.insert("port".to_string(), json!(99999)); // > 65535

        let result = validator.validate_and_fill(&mut cfg).expect("no logic error");
        assert!(!result.is_valid());
    }

    #[test]
    fn test_one_of_valid() {
        let schema = make_schema();
        let validator = ConfigValidator::new(schema);

        let mut cfg: HashMap<String, Value> = HashMap::new();
        cfg.insert("host".to_string(), json!("localhost"));
        cfg.insert("port".to_string(), json!(8080));
        cfg.insert("mode".to_string(), json!("prod"));

        let result = validator.validate_and_fill(&mut cfg).expect("no logic error");
        assert!(result.is_valid(), "{}", result.summary());
    }

    #[test]
    fn test_one_of_invalid() {
        let schema = make_schema();
        let validator = ConfigValidator::new(schema);

        let mut cfg: HashMap<String, Value> = HashMap::new();
        cfg.insert("host".to_string(), json!("localhost"));
        cfg.insert("port".to_string(), json!(8080));
        cfg.insert("mode".to_string(), json!("staging")); // not in one_of

        let result = validator.validate_and_fill(&mut cfg).expect("no logic error");
        assert!(!result.is_valid());
    }

    #[test]
    fn test_strict_mode_rejects_unknown_fields() {
        let schema = ConfigSchema::new()
            .strict()
            .field("x", FieldSpec::required(ValueType::Integer));

        let validator = ConfigValidator::new(schema);

        let mut cfg: HashMap<String, Value> = HashMap::new();
        cfg.insert("x".to_string(), json!(1));
        cfg.insert("y".to_string(), json!(2)); // unknown

        let result = validator.validate_and_fill(&mut cfg).expect("no logic error");
        assert!(!result.is_valid());
    }

    #[test]
    fn test_strict_mode_allows_declared_fields() {
        let schema = ConfigSchema::new()
            .strict()
            .field("x", FieldSpec::required(ValueType::Integer));

        let validator = ConfigValidator::new(schema);

        let mut cfg: HashMap<String, Value> = HashMap::new();
        cfg.insert("x".to_string(), json!(1));

        let result = validator.validate_and_fill(&mut cfg).expect("no logic error");
        assert!(result.is_valid(), "{}", result.summary());
    }

    #[test]
    fn test_nested_schema() {
        let nested = ConfigSchema::new()
            .field("db_name", FieldSpec::required(ValueType::String))
            .field(
                "db_port",
                FieldSpec::required(ValueType::Integer).with_range(1.0, 65535.0),
            );

        let schema = ConfigSchema::new()
            .field("app_name", FieldSpec::required(ValueType::String))
            .field(
                "database",
                FieldSpec::required(ValueType::Object).with_nested(nested),
            );

        let validator = ConfigValidator::new(schema);

        let mut cfg: HashMap<String, Value> = HashMap::new();
        cfg.insert("app_name".to_string(), json!("my_app"));
        cfg.insert(
            "database".to_string(),
            json!({ "db_name": "mydb", "db_port": 5432 }),
        );

        let result = validator.validate_and_fill(&mut cfg).expect("no logic error");
        assert!(result.is_valid(), "{}", result.summary());
    }

    #[test]
    fn test_nested_schema_invalid_port() {
        let nested = ConfigSchema::new()
            .field("db_name", FieldSpec::required(ValueType::String))
            .field(
                "db_port",
                FieldSpec::required(ValueType::Integer).with_range(1.0, 65535.0),
            );

        let schema = ConfigSchema::new()
            .field("app_name", FieldSpec::required(ValueType::String))
            .field(
                "database",
                FieldSpec::required(ValueType::Object).with_nested(nested),
            );

        let validator = ConfigValidator::new(schema);

        let mut cfg: HashMap<String, Value> = HashMap::new();
        cfg.insert("app_name".to_string(), json!("my_app"));
        cfg.insert(
            "database".to_string(),
            json!({ "db_name": "mydb", "db_port": 99999 }), // invalid port
        );

        let result = validator.validate_and_fill(&mut cfg).expect("no logic error");
        assert!(!result.is_valid());
    }

    #[test]
    fn test_validate_readonly() {
        let schema = ConfigSchema::new()
            .field("x", FieldSpec::required(ValueType::Integer));
        let validator = ConfigValidator::new(schema);

        let cfg: HashMap<String, Value> = {
            let mut m = HashMap::new();
            m.insert("x".to_string(), json!(42));
            m
        };

        let result = validator.validate(&cfg);
        assert!(result.is_valid());
    }

    #[test]
    fn test_config_validation_result_summary_no_issues() {
        let r = ConfigValidationResult::new();
        assert!(r.summary().contains("valid"));
    }

    #[test]
    fn test_config_validation_result_summary_with_errors() {
        let mut r = ConfigValidationResult::new();
        r.add_error("field_a", "something went wrong");
        let s = r.summary();
        assert!(s.contains("1 error"));
        assert!(s.contains("field_a"));
    }

    #[test]
    fn test_value_type_matches() {
        assert!(ValueType::Boolean.matches(&json!(true)));
        assert!(!ValueType::Boolean.matches(&json!(1)));
        assert!(ValueType::Integer.matches(&json!(42)));
        assert!(ValueType::Integer.matches(&json!(42.0)));
        assert!(!ValueType::Integer.matches(&json!(42.5)));
        assert!(ValueType::Float.matches(&json!(3.14)));
        assert!(ValueType::String.matches(&json!("hello")));
        assert!(!ValueType::String.matches(&json!(42)));
        assert!(ValueType::Array.matches(&json!([1, 2, 3])));
        assert!(ValueType::Object.matches(&json!({ "a": 1 })));
        assert!(ValueType::Any.matches(&json!(null)));
    }

    #[test]
    fn test_merge_with_prefix() {
        let mut parent = ConfigValidationResult::new();
        let mut child = ConfigValidationResult::new();
        child.add_error("field", "child error");
        parent.merge_with_prefix(child, "parent");

        assert!(!parent.is_valid());
        assert_eq!(parent.errors()[0].field_path, "parent.field");
    }
}
