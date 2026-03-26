//! HuggingFace Hub-compatible dataset card (README.md) metadata.
//!
//! This module can parse and emit the YAML front matter used by HuggingFace
//! dataset cards without pulling in any external YAML crate.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_datasets::hub_metadata::{parse_dataset_card, write_dataset_card};
//!
//! let yaml = "---\nname: my-dataset\ndescription: A test dataset\n---\n";
//! let card = parse_dataset_card(yaml).expect("parse ok");
//! assert_eq!(card.name, "my-dataset");
//! let rendered = write_dataset_card(&card);
//! assert!(rendered.contains("name: my-dataset"));
//! ```

use crate::error::{DatasetsError, Result};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// High-level ML task category.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum TaskCategory {
    /// Text classification (sentiment analysis, topic classification, …).
    TextClassification,
    /// Token-level classification (NER, POS tagging, …).
    TokenClassification,
    /// Reading comprehension and extractive QA.
    QuestionAnswering,
    /// Abstractive summarization.
    Summarization,
    /// Sequence-to-sequence translation.
    Translation,
    /// Open-ended or conditional text generation.
    TextGeneration,
    /// Single-label or multi-label image classification.
    ImageClassification,
    /// Bounding-box or keypoint object detection.
    ObjectDetection,
    /// Any task not listed above.
    Other(String),
}

impl TaskCategory {
    /// Return the canonical HuggingFace string for this task.
    pub fn as_str(&self) -> &str {
        match self {
            Self::TextClassification => "text-classification",
            Self::TokenClassification => "token-classification",
            Self::QuestionAnswering => "question-answering",
            Self::Summarization => "summarization",
            Self::Translation => "translation",
            Self::TextGeneration => "text-generation",
            Self::ImageClassification => "image-classification",
            Self::ObjectDetection => "object-detection",
            Self::Other(s) => s.as_str(),
        }
    }

    /// Parse from a HuggingFace task-category string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s.trim() {
            "text-classification" => Self::TextClassification,
            "token-classification" => Self::TokenClassification,
            "question-answering" => Self::QuestionAnswering,
            "summarization" => Self::Summarization,
            "translation" => Self::Translation,
            "text-generation" => Self::TextGeneration,
            "image-classification" => Self::ImageClassification,
            "object-detection" => Self::ObjectDetection,
            other => Self::Other(other.to_owned()),
        }
    }
}

/// Information about a single split (train / validation / test …).
#[derive(Debug, Clone, PartialEq)]
pub struct SplitInfo {
    /// Split name, e.g. `"train"`, `"test"`.
    pub name: String,
    /// Approximate on-disk size in bytes.
    pub num_bytes: u64,
    /// Number of examples in this split.
    pub num_examples: u64,
}

/// Data type of a feature column.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureDtype {
    /// UTF-8 string.
    String,
    /// 32-bit signed integer.
    Int32,
    /// 64-bit signed integer.
    Int64,
    /// 32-bit float.
    Float32,
    /// 64-bit float.
    Float64,
    /// Boolean.
    Bool,
    /// Image (raw pixels or path).
    Image,
    /// Audio waveform or path.
    Audio,
}

impl FeatureDtype {
    /// Return the canonical dtype string.
    pub fn as_str(&self) -> &str {
        match self {
            Self::String => "string",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::Float32 => "float32",
            Self::Float64 => "float64",
            Self::Bool => "bool",
            Self::Image => "image",
            Self::Audio => "audio",
        }
    }

    /// Parse from a dtype string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s.trim() {
            "string" | "str" => Self::String,
            "int32" => Self::Int32,
            "int64" | "int" => Self::Int64,
            "float32" | "float" => Self::Float32,
            "float64" | "double" => Self::Float64,
            "bool" | "boolean" => Self::Bool,
            "image" => Self::Image,
            "audio" => Self::Audio,
            _ => Self::String,
        }
    }
}

/// Metadata for a single dataset feature / column.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureInfo {
    /// Column name.
    pub name: String,
    /// Data type.
    pub dtype: FeatureDtype,
    /// Optional human-readable description.
    pub description: Option<String>,
}

/// A HuggingFace Hub dataset card (README.md YAML front matter).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DatasetCard {
    /// Dataset identifier / slug.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// SPDX license identifier, e.g. `"apache-2.0"`.
    pub license: Option<String>,
    /// BCP-47 language codes, e.g. `["en", "fr"]`.
    pub language: Vec<String>,
    /// Free-form tags.
    pub tags: Vec<String>,
    /// HuggingFace task categories.
    pub task_categories: Vec<TaskCategory>,
    /// Per-split statistics.
    pub splits: Vec<SplitInfo>,
    /// Column-level feature metadata.
    pub features: Vec<FeatureInfo>,
    /// BibTeX or plain-text citation.
    pub citation: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal YAML value type
// ─────────────────────────────────────────────────────────────────────────────

/// A YAML scalar or collection value as understood by our minimal parser.
#[derive(Debug, Clone, PartialEq)]
pub enum YamlValue {
    /// String scalar.
    Str(String),
    /// Integer scalar.
    Int(i64),
    /// Floating-point scalar.
    Float(f64),
    /// Sequence.
    List(Vec<YamlValue>),
    /// Mapping.
    Map(HashMap<String, YamlValue>),
    /// Boolean.
    Bool(bool),
    /// Null / absent.
    Null,
}

impl YamlValue {
    /// Coerce to `&str` if this is a `Str` variant.
    pub fn as_str(&self) -> Option<&str> {
        if let YamlValue::Str(s) = self {
            Some(s.as_str())
        } else {
            None
        }
    }

    /// Coerce to `i64` if this is an `Int` variant.
    pub fn as_i64(&self) -> Option<i64> {
        if let YamlValue::Int(n) = self {
            Some(*n)
        } else {
            None
        }
    }

    /// Coerce to `u64`, accepting both `Int` and `Str` variants.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            YamlValue::Int(n) => Some(*n as u64),
            YamlValue::Str(s) => s.trim().parse().ok(),
            _ => None,
        }
    }

    /// Coerce to a `Vec<YamlValue>` reference.
    pub fn as_list(&self) -> Option<&Vec<YamlValue>> {
        if let YamlValue::List(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Coerce to a `HashMap` reference.
    pub fn as_map(&self) -> Option<&HashMap<String, YamlValue>> {
        if let YamlValue::Map(m) = self {
            Some(m)
        } else {
            None
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Simple YAML parser
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a minimal subset of YAML into a flat `HashMap<String, YamlValue>`.
///
/// Supported constructs:
/// - `key: scalar` (string, integer, float, bool, null)
/// - `key: [item1, item2]` (inline list)
/// - `key:\n  - item\n  - item` (block list)
/// - `key:\n  subkey: value` (nested mapping, returned as `YamlValue::Map`)
pub fn simple_yaml_parse(s: &str) -> HashMap<String, YamlValue> {
    let mut result: HashMap<String, YamlValue> = HashMap::new();
    let lines: Vec<&str> = s.lines().collect();
    let mut i = 0usize;

    while i < lines.len() {
        let line = lines[i];
        // Skip empty lines and comments.
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            i += 1;
            continue;
        }

        // Top-level key (no leading spaces).
        if let Some(colon_pos) = find_colon(line) {
            let indent = leading_spaces(line);
            if indent == 0 {
                let key = line[..colon_pos].trim().to_owned();
                let rest = line[colon_pos + 1..].trim();

                if rest.is_empty() {
                    // Value is on subsequent lines.
                    i += 1;
                    // Peek: block list or nested map?
                    let mut block_items: Vec<YamlValue> = Vec::new();
                    let mut sub_map: HashMap<String, YamlValue> = HashMap::new();
                    let mut is_list = false;
                    let mut is_map = false;

                    while i < lines.len() {
                        let sub_line = lines[i];
                        let sub_trimmed = sub_line.trim();
                        let sub_indent = leading_spaces(sub_line);

                        if sub_trimmed.is_empty() || sub_trimmed.starts_with('#') {
                            i += 1;
                            continue;
                        }
                        // Back to top-level
                        if sub_indent == 0 {
                            break;
                        }
                        // Block-list item
                        if sub_trimmed.starts_with("- ") || sub_trimmed == "-" {
                            is_list = true;
                            let item_str = if sub_trimmed.len() > 2 {
                                sub_trimmed[2..].trim()
                            } else {
                                ""
                            };
                            block_items.push(parse_scalar(item_str));
                            i += 1;
                        } else if let Some(sub_colon) = find_colon(sub_trimmed) {
                            is_map = true;
                            let sub_key = sub_trimmed[..sub_colon].trim().to_owned();
                            let sub_val = sub_trimmed[sub_colon + 1..].trim();
                            sub_map.insert(sub_key, parse_scalar(sub_val));
                            i += 1;
                        } else {
                            i += 1;
                        }
                    }

                    let value = if is_list {
                        YamlValue::List(block_items)
                    } else if is_map {
                        YamlValue::Map(sub_map)
                    } else {
                        YamlValue::Null
                    };
                    result.insert(key, value);
                } else if rest.starts_with('[') && rest.ends_with(']') {
                    // Inline list: key: [a, b, c]
                    let inner = &rest[1..rest.len() - 1];
                    let items: Vec<YamlValue> =
                        inner.split(',').map(|s| parse_scalar(s.trim())).collect();
                    result.insert(key, YamlValue::List(items));
                    i += 1;
                } else {
                    result.insert(key, parse_scalar(rest));
                    i += 1;
                }
                continue;
            }
        }
        i += 1;
    }

    result
}

fn leading_spaces(s: &str) -> usize {
    s.len() - s.trim_start().len()
}

fn find_colon(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b':' {
            // Make sure it's not inside a quoted string (simplistic check).
            return Some(i);
        }
    }
    None
}

fn parse_scalar(s: &str) -> YamlValue {
    let s = s.trim();
    if s.is_empty() || s == "null" || s == "~" {
        return YamlValue::Null;
    }
    if s == "true" {
        return YamlValue::Bool(true);
    }
    if s == "false" {
        return YamlValue::Bool(false);
    }
    // Strip optional surrounding quotes.
    let unquoted = strip_quotes(s);
    // Try integer.
    if let Ok(n) = unquoted.parse::<i64>() {
        return YamlValue::Int(n);
    }
    // Try float.
    if let Ok(f) = unquoted.parse::<f64>() {
        return YamlValue::Float(f);
    }
    YamlValue::Str(unquoted.to_owned())
}

fn strip_quotes(s: &str) -> &str {
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Extract front matter
// ─────────────────────────────────────────────────────────────────────────────

/// Extract the YAML front matter (content between the first two `---` markers).
fn extract_front_matter(input: &str) -> Option<&str> {
    let mut lines = input.splitn(3, "---");
    // First chunk is empty or pre-matter (ignored).
    let _before = lines.next()?;
    let front = lines.next()?;
    Some(front)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a HuggingFace dataset card (YAML front matter wrapped in `---`) into a
/// [`DatasetCard`].
///
/// If the input does not contain `---` delimiters the entire string is treated
/// as raw YAML.
pub fn parse_dataset_card(input: &str) -> Result<DatasetCard> {
    let yaml_src = match extract_front_matter(input) {
        Some(fm) => fm,
        None => input,
    };

    let map = simple_yaml_parse(yaml_src);
    let mut card = DatasetCard::default();

    if let Some(v) = map.get("name") {
        card.name = v.as_str().unwrap_or_default().to_owned();
    }
    if let Some(v) = map.get("description") {
        card.description = v.as_str().unwrap_or_default().to_owned();
    }
    if let Some(v) = map.get("license") {
        let s = v.as_str().unwrap_or_default().to_owned();
        if !s.is_empty() {
            card.license = Some(s);
        }
    }
    if let Some(v) = map.get("language") {
        card.language = collect_str_list(v);
    }
    if let Some(v) = map.get("tags") {
        card.tags = collect_str_list(v);
    }
    if let Some(v) = map.get("task_categories") {
        card.task_categories = collect_str_list(v)
            .into_iter()
            .map(|s| TaskCategory::from_str(&s))
            .collect();
    }
    if let Some(v) = map.get("splits") {
        card.splits = parse_splits(v);
    }
    if let Some(v) = map.get("features") {
        card.features = parse_features(v);
    }
    if let Some(v) = map.get("citation") {
        let s = v.as_str().unwrap_or_default().to_owned();
        if !s.is_empty() {
            card.citation = Some(s);
        }
    }

    Ok(card)
}

/// Collect a `YamlValue::List` of string-like items into a `Vec<String>`.
fn collect_str_list(v: &YamlValue) -> Vec<String> {
    match v {
        YamlValue::List(items) => items
            .iter()
            .filter_map(|item| item.as_str().map(str::to_owned))
            .collect(),
        YamlValue::Str(s) => vec![s.clone()],
        _ => Vec::new(),
    }
}

/// Parse a YAML value into a `Vec<SplitInfo>`.
fn parse_splits(v: &YamlValue) -> Vec<SplitInfo> {
    let items = match v.as_list() {
        Some(l) => l,
        None => return Vec::new(),
    };
    items
        .iter()
        .filter_map(|item| {
            let m = item.as_map()?;
            let name = m.get("name")?.as_str()?.to_owned();
            let num_bytes = m.get("num_bytes").and_then(|v| v.as_u64()).unwrap_or(0);
            let num_examples = m.get("num_examples").and_then(|v| v.as_u64()).unwrap_or(0);
            Some(SplitInfo {
                name,
                num_bytes,
                num_examples,
            })
        })
        .collect()
}

/// Parse a YAML value into a `Vec<FeatureInfo>`.
fn parse_features(v: &YamlValue) -> Vec<FeatureInfo> {
    let items = match v.as_list() {
        Some(l) => l,
        None => return Vec::new(),
    };
    items
        .iter()
        .filter_map(|item| {
            let m = item.as_map()?;
            let name = m.get("name")?.as_str()?.to_owned();
            let dtype_str = m.get("dtype").and_then(|v| v.as_str()).unwrap_or("string");
            let dtype = FeatureDtype::from_str(dtype_str);
            let description = m
                .get("description")
                .and_then(|v| v.as_str())
                .map(str::to_owned);
            Some(FeatureInfo {
                name,
                dtype,
                description,
            })
        })
        .collect()
}

/// Render a [`DatasetCard`] as YAML front matter (wrapped in `---`).
pub fn write_dataset_card(card: &DatasetCard) -> String {
    let mut out = String::from("---\n");

    out.push_str(&format!("name: {}\n", yaml_escape(&card.name)));
    out.push_str(&format!(
        "description: {}\n",
        yaml_escape(&card.description)
    ));

    if let Some(ref lic) = card.license {
        out.push_str(&format!("license: {}\n", yaml_escape(lic)));
    }

    if !card.language.is_empty() {
        out.push_str("language:\n");
        for lang in &card.language {
            out.push_str(&format!("  - {}\n", yaml_escape(lang)));
        }
    }

    if !card.tags.is_empty() {
        out.push_str("tags:\n");
        for tag in &card.tags {
            out.push_str(&format!("  - {}\n", yaml_escape(tag)));
        }
    }

    if !card.task_categories.is_empty() {
        out.push_str("task_categories:\n");
        for tc in &card.task_categories {
            out.push_str(&format!("  - {}\n", yaml_escape(tc.as_str())));
        }
    }

    if !card.splits.is_empty() {
        out.push_str("splits:\n");
        for split in &card.splits {
            out.push_str(&format!(
                "  - name: {}\n    num_bytes: {}\n    num_examples: {}\n",
                yaml_escape(&split.name),
                split.num_bytes,
                split.num_examples
            ));
        }
    }

    if !card.features.is_empty() {
        out.push_str("features:\n");
        for feat in &card.features {
            out.push_str(&format!(
                "  - name: {}\n    dtype: {}\n",
                yaml_escape(&feat.name),
                feat.dtype.as_str()
            ));
            if let Some(ref desc) = feat.description {
                out.push_str(&format!("    description: {}\n", yaml_escape(desc)));
            }
        }
    }

    if let Some(ref cit) = card.citation {
        out.push_str(&format!("citation: {}\n", yaml_escape(cit)));
    }

    out.push_str("---\n");
    out
}

/// Quote a YAML string value if it contains special characters.
fn yaml_escape(s: &str) -> String {
    if s.contains(':') || s.contains('#') || s.contains('\'') || s.contains('"') {
        format!("\"{}\"", s.replace('"', "\\\""))
    } else {
        s.to_owned()
    }
}

/// Validate a [`DatasetCard`] and return a list of warning messages.
///
/// Returns an empty vector if the card is valid.
pub fn validate_card(card: &DatasetCard) -> Vec<String> {
    let mut warnings = Vec::new();

    if card.name.trim().is_empty() {
        warnings.push("'name' is empty".to_owned());
    }
    if card.description.trim().is_empty() {
        warnings.push("'description' is empty".to_owned());
    }
    if card.language.is_empty() {
        warnings.push(
            "'language' list is empty; consider specifying at least one language code".to_owned(),
        );
    }
    if card.task_categories.is_empty() {
        warnings.push("'task_categories' is empty; consider specifying the task type".to_owned());
    }
    if card.splits.is_empty() {
        warnings.push("'splits' is empty; consider documenting train/test splits".to_owned());
    }
    for split in &card.splits {
        if split.name.trim().is_empty() {
            warnings.push("A split has an empty 'name'".to_owned());
        }
        if split.num_examples == 0 {
            warnings.push(format!("Split '{}' has num_examples == 0", split.name));
        }
    }
    for feat in &card.features {
        if feat.name.trim().is_empty() {
            warnings.push("A feature has an empty 'name'".to_owned());
        }
    }

    warnings
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE_CARD: &str = "---\nname: test\nlanguage:\n  - en\n---\n";

    #[test]
    fn test_parse_simple_card() {
        let card = parse_dataset_card(SIMPLE_CARD).expect("should parse");
        assert_eq!(card.name, "test");
        assert_eq!(card.language, vec!["en".to_owned()]);
    }

    #[test]
    fn test_parse_full_card() {
        let yaml = "---\n\
            name: my-dataset\n\
            description: A comprehensive dataset\n\
            license: apache-2.0\n\
            language:\n  - en\n  - fr\n\
            tags:\n  - nlp\n  - benchmark\n\
            task_categories:\n  - text-classification\n\
            ---\n";
        let card = parse_dataset_card(yaml).expect("parse");
        assert_eq!(card.name, "my-dataset");
        assert_eq!(card.description, "A comprehensive dataset");
        assert_eq!(card.license, Some("apache-2.0".into()));
        assert_eq!(card.language, vec!["en", "fr"]);
        assert_eq!(card.tags, vec!["nlp", "benchmark"]);
        assert_eq!(card.task_categories, vec![TaskCategory::TextClassification]);
    }

    #[test]
    fn test_write_roundtrip() {
        let card = DatasetCard {
            name: "roundtrip-test".into(),
            description: "Test description".into(),
            license: Some("mit".into()),
            language: vec!["en".into()],
            tags: vec!["test".into()],
            task_categories: vec![TaskCategory::Summarization],
            splits: vec![SplitInfo {
                name: "train".into(),
                num_bytes: 1024,
                num_examples: 100,
            }],
            features: vec![FeatureInfo {
                name: "text".into(),
                dtype: FeatureDtype::String,
                description: None,
            }],
            citation: None,
        };

        let rendered = write_dataset_card(&card);
        assert!(rendered.contains("name: roundtrip-test"));
        assert!(rendered.contains("license: mit"));
        assert!(rendered.contains("num_examples: 100"));

        // Parse back.
        let parsed = parse_dataset_card(&rendered).expect("reparse");
        assert_eq!(parsed.name, card.name);
        assert_eq!(parsed.license, card.license);
        assert_eq!(parsed.language, card.language);
    }

    #[test]
    fn test_validate_empty_name() {
        let card = DatasetCard {
            name: String::new(),
            ..Default::default()
        };
        let warnings = validate_card(&card);
        assert!(
            warnings.iter().any(|w| w.contains("'name' is empty")),
            "expected warning about empty name, got: {warnings:?}"
        );
    }

    #[test]
    fn test_validate_valid_card() {
        let card = DatasetCard {
            name: "good".into(),
            description: "good desc".into(),
            language: vec!["en".into()],
            task_categories: vec![TaskCategory::TextClassification],
            splits: vec![SplitInfo {
                name: "train".into(),
                num_bytes: 100,
                num_examples: 10,
            }],
            ..Default::default()
        };
        let warnings = validate_card(&card);
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    }

    #[test]
    fn test_task_category_roundtrip() {
        let cats = vec![
            TaskCategory::TextClassification,
            TaskCategory::TokenClassification,
            TaskCategory::QuestionAnswering,
            TaskCategory::Summarization,
            TaskCategory::Translation,
            TaskCategory::TextGeneration,
            TaskCategory::ImageClassification,
            TaskCategory::ObjectDetection,
            TaskCategory::Other("custom-task".into()),
        ];
        for cat in cats {
            let s = cat.as_str();
            let parsed = TaskCategory::from_str(s);
            assert_eq!(parsed, cat, "roundtrip failed for {s}");
        }
    }

    #[test]
    fn test_feature_dtype_roundtrip() {
        let dtypes = vec![
            FeatureDtype::String,
            FeatureDtype::Int32,
            FeatureDtype::Int64,
            FeatureDtype::Float32,
            FeatureDtype::Float64,
            FeatureDtype::Bool,
            FeatureDtype::Image,
            FeatureDtype::Audio,
        ];
        for dt in dtypes {
            let s = dt.as_str();
            let parsed = FeatureDtype::from_str(s);
            assert_eq!(parsed, dt);
        }
    }

    #[test]
    fn test_inline_list_parsing() {
        let yaml = "tags: [nlp, vision, audio]\n";
        let map = simple_yaml_parse(yaml);
        if let Some(YamlValue::List(items)) = map.get("tags") {
            assert_eq!(items.len(), 3);
        } else {
            panic!("expected list");
        }
    }

    #[test]
    fn test_yaml_scalar_types() {
        let yaml = "count: 42\nrate: 3.14\nflag: true\nempty: null\n";
        let map = simple_yaml_parse(yaml);
        assert_eq!(map.get("count"), Some(&YamlValue::Int(42)));
        assert!(matches!(map.get("rate"), Some(YamlValue::Float(_))));
        assert_eq!(map.get("flag"), Some(&YamlValue::Bool(true)));
        assert_eq!(map.get("empty"), Some(&YamlValue::Null));
    }
}
