//! Text Annotation
//!
//! This module provides span-based annotation infrastructure for NLP pipelines,
//! including named layers (NER, POS, dependency, coreference), BIO/BIOES
//! encoding/decoding, an annotator trait, and utilities for merging overlapping
//! or adjacent annotations.
//!
//! ## Overview
//!
//! - [`TextAnnotation`]: Span-based annotation over text
//! - [`AnnotationLayer`]: Named annotation layers (NER, POS, dep, coref)
//! - [`BIO`]: BIO/BIOES annotation encoding/decoding
//! - [`SpanAnnotator`]: Trait: `annotate(text: &str) -> Vec<Span>`
//! - [`AnnotationMerger`]: Merge overlapping/adjacent annotations
//!
//! ## Example
//!
//! ```rust
//! use scirs2_text::annotation::{
//!     TextAnnotation, AnnotationLayer, Span, SpanAnnotator,
//!     BIO, BIOScheme, AnnotationMerger,
//! };
//!
//! // Build a text annotation
//! let text = "John Smith lives in New York.";
//! let mut ann = TextAnnotation::new(text);
//! ann.add_span(Span::new(0, 10, "PER", 1.0));
//! ann.add_span(Span::new(20, 28, "LOC", 1.0));
//!
//! // BIO encoding
//! let tokens = vec!["John", "Smith", "lives", "in", "New", "York"];
//! let spans = vec![
//!     Span::new(0, 2, "PER", 1.0),  // token span
//!     Span::new(4, 6, "LOC", 1.0),
//! ];
//! let bio_tags = BIO::encode_tokens(&tokens, &spans, BIOScheme::BIO);
//! println!("{:?}", bio_tags);
//! ```

use crate::error::{Result, TextError};
use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// Span
// ────────────────────────────────────────────────────────────────────────────

/// A span annotation over a piece of text
#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    /// Start index (character offset or token index depending on context)
    pub start: usize,
    /// End index (exclusive)
    pub end: usize,
    /// Label / tag (e.g. "PER", "LOC", "ORG", "VERB")
    pub label: String,
    /// Confidence score in [0.0, 1.0]
    pub score: f64,
    /// Optional metadata key-value pairs
    pub metadata: HashMap<String, String>,
}

impl Span {
    /// Create a new span
    pub fn new(start: usize, end: usize, label: impl Into<String>, score: f64) -> Self {
        assert!(start <= end, "span start must be <= end");
        Self {
            start,
            end,
            label: label.into(),
            score,
            metadata: HashMap::new(),
        }
    }

    /// Length of the span
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Returns true if the span has zero length
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    /// Check if this span overlaps with another
    pub fn overlaps(&self, other: &Span) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Check if this span is adjacent to another (no gap)
    pub fn adjacent(&self, other: &Span) -> bool {
        self.end == other.start || other.end == self.start
    }

    /// Check if this span contains another
    pub fn contains(&self, other: &Span) -> bool {
        self.start <= other.start && other.end <= self.end
    }

    /// Add metadata
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

impl std::fmt::Display for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{},{}) {} ({:.2})", self.start, self.end, self.label, self.score)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// TextAnnotation
// ────────────────────────────────────────────────────────────────────────────

/// A text with its associated span annotations
#[derive(Debug, Clone)]
pub struct TextAnnotation {
    /// The original text
    pub text: String,
    /// Named annotation layers
    layers: HashMap<String, Vec<Span>>,
    /// Default layer name
    default_layer: String,
}

impl TextAnnotation {
    /// Create a new empty text annotation
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            layers: HashMap::new(),
            default_layer: "default".to_string(),
        }
    }

    /// Set the default layer name
    pub fn with_default_layer(mut self, name: impl Into<String>) -> Self {
        self.default_layer = name.into();
        self
    }

    /// Add a span to the default layer
    pub fn add_span(&mut self, span: Span) {
        self.add_span_to_layer(span, &self.default_layer.clone());
    }

    /// Add a span to a named layer
    pub fn add_span_to_layer(&mut self, span: Span, layer: &str) {
        self.layers.entry(layer.to_string()).or_default().push(span);
    }

    /// Get all spans in the default layer
    pub fn spans(&self) -> &[Span] {
        self.layers
            .get(&self.default_layer)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get all spans in a named layer
    pub fn spans_in_layer(&self, layer: &str) -> &[Span] {
        self.layers.get(layer).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get spans filtered by label in the default layer
    pub fn spans_by_label(&self, label: &str) -> Vec<&Span> {
        self.spans().iter().filter(|s| s.label == label).collect()
    }

    /// Get all layer names
    pub fn layer_names(&self) -> Vec<&str> {
        self.layers.keys().map(|s| s.as_str()).collect()
    }

    /// Sort all layers by span start position
    pub fn sort_spans(&mut self) {
        for spans in self.layers.values_mut() {
            spans.sort_by_key(|s| s.start);
        }
    }

    /// Extract the text covered by a span (character offsets)
    pub fn span_text(&self, span: &Span) -> Option<&str> {
        let chars: Vec<char> = self.text.chars().collect();
        let n = chars.len();
        if span.end > n {
            return None;
        }
        // Find byte offsets
        let byte_start: usize = self.text
            .char_indices()
            .nth(span.start)
            .map(|(b, _)| b)
            .unwrap_or(0);
        let byte_end: usize = if span.end >= n {
            self.text.len()
        } else {
            self.text
                .char_indices()
                .nth(span.end)
                .map(|(b, _)| b)
                .unwrap_or(self.text.len())
        };
        self.text.get(byte_start..byte_end)
    }

    /// Total number of annotations across all layers
    pub fn total_span_count(&self) -> usize {
        self.layers.values().map(|v| v.len()).sum()
    }

    /// Merge another TextAnnotation's layers into this one
    pub fn merge_from(&mut self, other: &TextAnnotation) -> Result<()> {
        if self.text != other.text {
            return Err(TextError::InvalidInput(
                "Cannot merge annotations for different texts".to_string(),
            ));
        }
        for (layer, spans) in &other.layers {
            self.layers.entry(layer.clone()).or_default().extend(spans.iter().cloned());
        }
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AnnotationLayer
// ────────────────────────────────────────────────────────────────────────────

/// Predefined annotation layer kinds
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LayerKind {
    /// Named Entity Recognition
    NER,
    /// Part-of-Speech tagging
    POS,
    /// Dependency relations
    Dependency,
    /// Coreference chains
    Coreference,
    /// Sentiment
    Sentiment,
    /// Custom user-defined layer
    Custom(String),
}

impl LayerKind {
    /// Canonical string name for the layer
    pub fn name(&self) -> &str {
        match self {
            LayerKind::NER => "ner",
            LayerKind::POS => "pos",
            LayerKind::Dependency => "dep",
            LayerKind::Coreference => "coref",
            LayerKind::Sentiment => "sentiment",
            LayerKind::Custom(s) => s.as_str(),
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Self {
        match s {
            "ner" => LayerKind::NER,
            "pos" => LayerKind::POS,
            "dep" | "dependency" => LayerKind::Dependency,
            "coref" | "coreference" => LayerKind::Coreference,
            "sentiment" => LayerKind::Sentiment,
            other => LayerKind::Custom(other.to_string()),
        }
    }
}

/// A named annotation layer with typed spans
#[derive(Debug, Clone)]
pub struct AnnotationLayer {
    /// Layer kind
    pub kind: LayerKind,
    /// All spans in this layer
    pub spans: Vec<Span>,
    /// Description
    pub description: String,
}

impl AnnotationLayer {
    /// Create a new annotation layer
    pub fn new(kind: LayerKind) -> Self {
        let desc = kind.name().to_string();
        Self { kind, spans: Vec::new(), description: desc }
    }

    /// Add a span to this layer
    pub fn add(&mut self, span: Span) {
        self.spans.push(span);
    }

    /// Get spans by label
    pub fn by_label(&self, label: &str) -> Vec<&Span> {
        self.spans.iter().filter(|s| s.label == label).collect()
    }

    /// Sort spans by start
    pub fn sort(&mut self) {
        self.spans.sort_by_key(|s| s.start);
    }

    /// Unique labels present in this layer
    pub fn unique_labels(&self) -> Vec<String> {
        let mut labels: Vec<String> = self
            .spans
            .iter()
            .map(|s| s.label.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        labels.sort();
        labels
    }

    /// Number of spans
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Whether layer is empty
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// BIO / BIOES encoding
// ────────────────────────────────────────────────────────────────────────────

/// BIO annotation scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BIOScheme {
    /// B-X, I-X, O
    BIO,
    /// B-X, I-X, O, E-X, S-X (BIOES / BILOU)
    BIOES,
}

/// BIO tag
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BIOTag {
    /// Beginning of entity
    B(String),
    /// Inside entity
    I(String),
    /// Outside any entity
    O,
    /// End of entity (BIOES only)
    E(String),
    /// Single-token entity (BIOES only)
    S(String),
    /// Last token of entity (BILOU)
    L(String),
    /// Unit / single-token (BILOU)
    U(String),
}

impl BIOTag {
    /// Parse a tag string like "B-PER" into a BIOTag
    pub fn from_str(s: &str) -> Self {
        if s == "O" {
            return BIOTag::O;
        }
        let parts: Vec<&str> = s.splitn(2, '-').collect();
        if parts.len() == 2 {
            let label = parts[1].to_string();
            match parts[0] {
                "B" => BIOTag::B(label),
                "I" => BIOTag::I(label),
                "E" => BIOTag::E(label),
                "S" => BIOTag::S(label),
                "L" => BIOTag::L(label),
                "U" => BIOTag::U(label),
                _ => BIOTag::O,
            }
        } else {
            BIOTag::O
        }
    }

    /// Convert to string representation
    pub fn to_string_repr(&self) -> String {
        match self {
            BIOTag::B(l) => format!("B-{}", l),
            BIOTag::I(l) => format!("I-{}", l),
            BIOTag::O => "O".to_string(),
            BIOTag::E(l) => format!("E-{}", l),
            BIOTag::S(l) => format!("S-{}", l),
            BIOTag::L(l) => format!("L-{}", l),
            BIOTag::U(l) => format!("U-{}", l),
        }
    }

    /// Get the entity label (None for O)
    pub fn entity_label(&self) -> Option<&str> {
        match self {
            BIOTag::B(l) | BIOTag::I(l) | BIOTag::E(l) | BIOTag::S(l) |
            BIOTag::L(l) | BIOTag::U(l) => Some(l.as_str()),
            BIOTag::O => None,
        }
    }

    /// Whether this tag starts an entity
    pub fn is_start(&self) -> bool {
        matches!(self, BIOTag::B(_) | BIOTag::S(_) | BIOTag::U(_))
    }

    /// Whether this tag ends an entity
    pub fn is_end(&self) -> bool {
        matches!(self, BIOTag::E(_) | BIOTag::S(_) | BIOTag::L(_) | BIOTag::U(_))
    }
}

impl std::fmt::Display for BIOTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string_repr())
    }
}

/// BIO/BIOES encoding and decoding utilities
pub struct BIO;

impl BIO {
    /// Encode token-level spans into BIO tags
    ///
    /// # Arguments
    ///
    /// * `tokens` – the token list
    /// * `spans` – token-indexed spans (start/end are token indices, exclusive end)
    /// * `scheme` – BIO or BIOES
    pub fn encode_tokens(tokens: &[&str], spans: &[Span], scheme: BIOScheme) -> Vec<BIOTag> {
        let n = tokens.len();
        let mut tags = vec![BIOTag::O; n];

        for span in spans {
            if span.start >= n || span.end > n || span.is_empty() {
                continue;
            }
            let label = span.label.clone();
            let len = span.end - span.start;

            match scheme {
                BIOScheme::BIO => {
                    tags[span.start] = BIOTag::B(label.clone());
                    for i in (span.start + 1)..span.end {
                        tags[i] = BIOTag::I(label.clone());
                    }
                }
                BIOScheme::BIOES => {
                    if len == 1 {
                        tags[span.start] = BIOTag::S(label);
                    } else {
                        tags[span.start] = BIOTag::B(label.clone());
                        for i in (span.start + 1)..(span.end - 1) {
                            tags[i] = BIOTag::I(label.clone());
                        }
                        tags[span.end - 1] = BIOTag::E(label);
                    }
                }
            }
        }

        tags
    }

    /// Encode a tag sequence as strings
    pub fn tags_to_strings(tags: &[BIOTag]) -> Vec<String> {
        tags.iter().map(|t| t.to_string_repr()).collect()
    }

    /// Decode BIO/BIOES tag strings into spans
    pub fn decode(tag_strings: &[&str]) -> Vec<Span> {
        let tags: Vec<BIOTag> = tag_strings.iter().map(|s| BIOTag::from_str(s)).collect();
        Self::decode_tags(&tags)
    }

    /// Decode BIOTag sequence into spans
    pub fn decode_tags(tags: &[BIOTag]) -> Vec<Span> {
        let mut spans = Vec::new();
        let mut current_start: Option<usize> = None;
        let mut current_label: Option<String> = None;

        for (i, tag) in tags.iter().enumerate() {
            match tag {
                BIOTag::B(label) => {
                    // Close previous span if any
                    if let (Some(start), Some(lbl)) = (current_start, current_label.take()) {
                        spans.push(Span::new(start, i, lbl, 1.0));
                    }
                    current_start = Some(i);
                    current_label = Some(label.clone());
                }
                BIOTag::I(label) => {
                    // Continue current span; if mismatch, start new
                    if current_label.as_deref() != Some(label.as_str()) {
                        if let (Some(start), Some(lbl)) = (current_start, current_label.take()) {
                            spans.push(Span::new(start, i, lbl, 1.0));
                        }
                        current_start = Some(i);
                        current_label = Some(label.clone());
                    }
                }
                BIOTag::E(label) => {
                    let start = current_start.unwrap_or(i);
                    spans.push(Span::new(start, i + 1, label.clone(), 1.0));
                    current_start = None;
                    current_label = None;
                }
                BIOTag::S(label) | BIOTag::U(label) => {
                    if let (Some(start), Some(lbl)) = (current_start, current_label.take()) {
                        spans.push(Span::new(start, i, lbl, 1.0));
                    }
                    spans.push(Span::new(i, i + 1, label.clone(), 1.0));
                    current_start = None;
                    current_label = None;
                }
                BIOTag::L(label) => {
                    let start = current_start.unwrap_or(i);
                    spans.push(Span::new(start, i + 1, label.clone(), 1.0));
                    current_start = None;
                    current_label = None;
                }
                BIOTag::O => {
                    if let (Some(start), Some(lbl)) = (current_start, current_label.take()) {
                        spans.push(Span::new(start, i, lbl, 1.0));
                    }
                    current_start = None;
                }
            }
        }

        // Close any trailing open span
        if let (Some(start), Some(lbl)) = (current_start, current_label) {
            spans.push(Span::new(start, tags.len(), lbl, 1.0));
        }

        spans
    }

    /// Compute per-label F1, precision, recall given predicted and gold tag sequences
    pub fn evaluation_metrics(
        gold: &[BIOTag],
        pred: &[BIOTag],
    ) -> HashMap<String, (f64, f64, f64)> {
        let gold_spans = Self::decode_tags(gold);
        let pred_spans = Self::decode_tags(pred);

        // Collect unique labels
        let mut labels: std::collections::HashSet<String> = std::collections::HashSet::new();
        for s in gold_spans.iter().chain(pred_spans.iter()) {
            labels.insert(s.label.clone());
        }

        let mut metrics = HashMap::new();
        for label in &labels {
            let gold_set: std::collections::HashSet<(usize, usize)> = gold_spans
                .iter()
                .filter(|s| &s.label == label)
                .map(|s| (s.start, s.end))
                .collect();
            let pred_set: std::collections::HashSet<(usize, usize)> = pred_spans
                .iter()
                .filter(|s| &s.label == label)
                .map(|s| (s.start, s.end))
                .collect();

            let tp = gold_set.intersection(&pred_set).count() as f64;
            let fp = pred_set.difference(&gold_set).count() as f64;
            let fn_ = gold_set.difference(&pred_set).count() as f64;

            let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };
            metrics.insert(label.clone(), (precision, recall, f1));
        }
        metrics
    }
}

// ────────────────────────────────────────────────────────────────────────────
// SpanAnnotator trait
// ────────────────────────────────────────────────────────────────────────────

/// Trait for anything that can annotate text with spans
pub trait SpanAnnotator: Send + Sync {
    /// Annotate text and return a vector of spans
    fn annotate(&self, text: &str) -> Result<Vec<Span>>;

    /// Name of this annotator
    fn name(&self) -> &str;
}

/// A simple keyword-based annotator (implements SpanAnnotator for testing)
pub struct KeywordAnnotator {
    /// Map: keyword -> label
    keywords: HashMap<String, String>,
    /// Annotator name
    annotator_name: String,
}

impl KeywordAnnotator {
    /// Create a new keyword annotator
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            keywords: HashMap::new(),
            annotator_name: name.into(),
        }
    }

    /// Add a keyword → label mapping
    pub fn add_keyword(&mut self, keyword: impl Into<String>, label: impl Into<String>) {
        self.keywords.insert(keyword.into(), label.into());
    }
}

impl SpanAnnotator for KeywordAnnotator {
    fn annotate(&self, text: &str) -> Result<Vec<Span>> {
        let lower = text.to_lowercase();
        let mut spans = Vec::new();
        for (kw, label) in &self.keywords {
            let kw_lower = kw.to_lowercase();
            let mut start = 0;
            while let Some(pos) = lower[start..].find(kw_lower.as_str()) {
                let abs_start = start + pos;
                let abs_end = abs_start + kw.len();
                spans.push(Span::new(abs_start, abs_end, label.clone(), 1.0));
                start = abs_start + 1;
            }
        }
        spans.sort_by_key(|s| s.start);
        Ok(spans)
    }

    fn name(&self) -> &str {
        &self.annotator_name
    }
}

/// A regex-based span annotator
pub struct RegexAnnotator {
    /// Patterns: (regex_pattern, label)
    patterns: Vec<(regex::Regex, String)>,
    /// Annotator name
    annotator_name: String,
}

impl RegexAnnotator {
    /// Create a new regex annotator
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            patterns: Vec::new(),
            annotator_name: name.into(),
        }
    }

    /// Add a pattern → label mapping
    pub fn add_pattern(
        &mut self,
        pattern: &str,
        label: impl Into<String>,
    ) -> Result<()> {
        let re = regex::Regex::new(pattern)
            .map_err(|e| TextError::InvalidInput(format!("Invalid regex: {}", e)))?;
        self.patterns.push((re, label.into()));
        Ok(())
    }
}

impl SpanAnnotator for RegexAnnotator {
    fn annotate(&self, text: &str) -> Result<Vec<Span>> {
        let mut spans = Vec::new();
        for (re, label) in &self.patterns {
            for m in re.find_iter(text) {
                spans.push(Span::new(m.start(), m.end(), label.clone(), 1.0));
            }
        }
        spans.sort_by_key(|s| s.start);
        Ok(spans)
    }

    fn name(&self) -> &str {
        &self.annotator_name
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AnnotationMerger
// ────────────────────────────────────────────────────────────────────────────

/// Strategy for resolving overlapping annotations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Keep the longer span
    KeepLonger,
    /// Keep the span with higher score
    KeepHigherScore,
    /// Keep the first span (leftmost start)
    KeepFirst,
    /// Keep the last span
    KeepLast,
    /// Merge adjacent spans with the same label into one
    MergeAdjacent,
    /// Keep all spans (no conflict resolution)
    KeepAll,
}

/// Utilities for merging overlapping or adjacent annotations
pub struct AnnotationMerger {
    /// How to resolve conflicts
    pub strategy: MergeStrategy,
    /// Whether to merge adjacent same-label spans
    pub merge_adjacent: bool,
}

impl Default for AnnotationMerger {
    fn default() -> Self {
        Self::new()
    }
}

impl AnnotationMerger {
    /// Create a merger with default settings (keep higher-score span on overlap)
    pub fn new() -> Self {
        Self {
            strategy: MergeStrategy::KeepHigherScore,
            merge_adjacent: false,
        }
    }

    /// Set the merge strategy
    pub fn with_strategy(mut self, strategy: MergeStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Whether to merge adjacent same-label spans
    pub fn merge_adjacent_spans(mut self, v: bool) -> Self {
        self.merge_adjacent = v;
        self
    }

    /// Merge spans according to the configured strategy
    ///
    /// Input spans need not be sorted; output spans are sorted by start.
    pub fn merge(&self, spans: Vec<Span>) -> Vec<Span> {
        if spans.is_empty() {
            return spans;
        }

        let mut sorted = spans;
        sorted.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));

        if self.strategy == MergeStrategy::KeepAll && !self.merge_adjacent {
            return sorted;
        }

        // Optional: merge adjacent same-label spans first
        let sorted = if self.merge_adjacent || self.strategy == MergeStrategy::MergeAdjacent {
            self.merge_adjacent_impl(sorted)
        } else {
            sorted
        };

        if self.strategy == MergeStrategy::KeepAll {
            return sorted;
        }

        // Greedily resolve overlaps
        let mut result: Vec<Span> = Vec::new();
        for span in sorted {
            if let Some(last) = result.last() {
                if span.overlaps(last) {
                    let winner = self.resolve_conflict(last, &span);
                    let last_idx = result.len() - 1;
                    result[last_idx] = winner;
                    continue;
                }
            }
            result.push(span);
        }

        result
    }

    /// Merge adjacent spans that have the same label
    fn merge_adjacent_impl(&self, mut spans: Vec<Span>) -> Vec<Span> {
        if spans.is_empty() {
            return spans;
        }
        let mut result: Vec<Span> = Vec::new();
        spans.sort_by_key(|s| s.start);

        for span in spans {
            if let Some(last) = result.last_mut() {
                if last.label == span.label && last.adjacent(&span) {
                    // Extend the last span
                    let new_end = span.end.max(last.end);
                    let new_score = (last.score + span.score) / 2.0;
                    *last = Span::new(last.start, new_end, last.label.clone(), new_score);
                    continue;
                }
            }
            result.push(span);
        }
        result
    }

    /// Resolve a conflict between two overlapping spans
    fn resolve_conflict(&self, existing: &Span, new_span: &Span) -> Span {
        match self.strategy {
            MergeStrategy::KeepLonger => {
                if new_span.len() >= existing.len() { new_span.clone() } else { existing.clone() }
            }
            MergeStrategy::KeepHigherScore => {
                if new_span.score >= existing.score { new_span.clone() } else { existing.clone() }
            }
            MergeStrategy::KeepFirst => existing.clone(),
            MergeStrategy::KeepLast => new_span.clone(),
            MergeStrategy::MergeAdjacent | MergeStrategy::KeepAll => existing.clone(),
        }
    }

    /// Merge spans from multiple annotators (ensemble)
    ///
    /// If `min_agreement` > 1, only spans that are agreed upon by at least
    /// that many annotators (exact start/end/label match) are kept.
    pub fn ensemble_merge(
        &self,
        all_spans: Vec<Vec<Span>>,
        min_agreement: usize,
    ) -> Vec<Span> {
        if all_spans.is_empty() {
            return Vec::new();
        }
        if min_agreement <= 1 {
            let merged: Vec<Span> = all_spans.into_iter().flatten().collect();
            return self.merge(merged);
        }

        // Count agreements: (start, end, label) -> count
        let mut vote_map: HashMap<(usize, usize, String), usize> = HashMap::new();
        let mut score_map: HashMap<(usize, usize, String), f64> = HashMap::new();
        for spans in &all_spans {
            for span in spans {
                let key = (span.start, span.end, span.label.clone());
                *vote_map.entry(key.clone()).or_insert(0) += 1;
                *score_map.entry(key).or_insert(0.0) += span.score;
            }
        }

        let n_annotators = all_spans.len() as f64;
        let mut result: Vec<Span> = vote_map
            .into_iter()
            .filter(|(_, count)| *count >= min_agreement)
            .map(|((start, end, label), count)| {
                let avg_score = score_map
                    .get(&(start, end, label.clone()))
                    .copied()
                    .unwrap_or(1.0)
                    / n_annotators;
                Span::new(start, end, label, avg_score)
            })
            .collect();

        result.sort_by_key(|s| s.start);
        result
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AnnotationDocument – high-level document with multiple layers
// ────────────────────────────────────────────────────────────────────────────

/// A complete annotated document with multiple named layers
#[derive(Debug, Clone)]
pub struct AnnotationDocument {
    /// Document identifier
    pub id: String,
    /// Document text
    pub text: String,
    /// Named annotation layers indexed by kind
    pub layers: HashMap<String, AnnotationLayer>,
    /// Document-level metadata
    pub metadata: HashMap<String, String>,
}

impl AnnotationDocument {
    /// Create a new empty document
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            layers: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add or replace a layer
    pub fn set_layer(&mut self, layer: AnnotationLayer) {
        self.layers.insert(layer.kind.name().to_string(), layer);
    }

    /// Get a named layer
    pub fn get_layer(&self, name: &str) -> Option<&AnnotationLayer> {
        self.layers.get(name)
    }

    /// Add a span to a named layer, creating the layer if needed
    pub fn add_span(&mut self, kind: LayerKind, span: Span) {
        let name = kind.name().to_string();
        let layer = self.layers.entry(name).or_insert_with(|| AnnotationLayer::new(kind));
        layer.add(span);
    }

    /// Annotate using a SpanAnnotator and store into a layer
    pub fn annotate_with(
        &mut self,
        annotator: &dyn SpanAnnotator,
        kind: LayerKind,
    ) -> Result<()> {
        let spans = annotator.annotate(&self.text)?;
        for span in spans {
            self.add_span(kind.clone(), span);
        }
        Ok(())
    }

    /// Add metadata
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_basic_properties() {
        let span = Span::new(0, 5, "PER", 0.9);
        assert_eq!(span.len(), 5);
        assert!(!span.is_empty());
    }

    #[test]
    fn test_span_overlap_and_adjacent() {
        let a = Span::new(0, 5, "X", 1.0);
        let b = Span::new(3, 8, "Y", 1.0);
        let c = Span::new(5, 10, "Z", 1.0);
        assert!(a.overlaps(&b), "a and b should overlap");
        assert!(!a.overlaps(&c), "a and c should not overlap");
        assert!(a.adjacent(&c), "a and c should be adjacent");
        assert!(!a.adjacent(&b), "a and b are overlapping, not adjacent");
    }

    #[test]
    fn test_span_contains() {
        let outer = Span::new(0, 10, "X", 1.0);
        let inner = Span::new(2, 7, "Y", 1.0);
        assert!(outer.contains(&inner));
        assert!(!inner.contains(&outer));
    }

    #[test]
    fn test_text_annotation_add_and_retrieve() {
        let text = "John Smith is in New York.";
        let mut ann = TextAnnotation::new(text);
        ann.add_span(Span::new(0, 10, "PER", 1.0));
        ann.add_span(Span::new(17, 25, "LOC", 0.95));
        assert_eq!(ann.spans().len(), 2);
        assert_eq!(ann.spans_by_label("PER").len(), 1);
        assert_eq!(ann.spans_by_label("LOC").len(), 1);
    }

    #[test]
    fn test_text_annotation_layers() {
        let mut ann = TextAnnotation::new("hello world");
        ann.add_span_to_layer(Span::new(0, 5, "HELLO", 1.0), "ner");
        ann.add_span_to_layer(Span::new(6, 11, "WORLD", 1.0), "ner");
        ann.add_span_to_layer(Span::new(0, 5, "NOUN", 1.0), "pos");
        assert_eq!(ann.spans_in_layer("ner").len(), 2);
        assert_eq!(ann.spans_in_layer("pos").len(), 1);
        let names = ann.layer_names();
        assert!(names.contains(&"ner"));
        assert!(names.contains(&"pos"));
    }

    #[test]
    fn test_bio_encode_decode_roundtrip_bio() {
        let tokens = vec!["John", "Smith", "lives", "in", "New", "York"];
        let spans = vec![
            Span::new(0, 2, "PER", 1.0),
            Span::new(4, 6, "LOC", 1.0),
        ];
        let tags = BIO::encode_tokens(&tokens, &spans, BIOScheme::BIO);
        assert_eq!(tags[0], BIOTag::B("PER".to_string()));
        assert_eq!(tags[1], BIOTag::I("PER".to_string()));
        assert_eq!(tags[2], BIOTag::O);
        assert_eq!(tags[3], BIOTag::O);
        assert_eq!(tags[4], BIOTag::B("LOC".to_string()));
        assert_eq!(tags[5], BIOTag::I("LOC".to_string()));

        let decoded = BIO::decode_tags(&tags);
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].label, "PER");
        assert_eq!(decoded[0].start, 0);
        assert_eq!(decoded[0].end, 2);
        assert_eq!(decoded[1].label, "LOC");
    }

    #[test]
    fn test_bio_encode_decode_bioes() {
        let tokens = vec!["New", "York", "is", "great"];
        let spans = vec![Span::new(0, 2, "LOC", 1.0)];
        let tags = BIO::encode_tokens(&tokens, &spans, BIOScheme::BIOES);
        assert_eq!(tags[0], BIOTag::B("LOC".to_string()));
        assert_eq!(tags[1], BIOTag::E("LOC".to_string()));
        assert_eq!(tags[2], BIOTag::O);

        let decoded = BIO::decode_tags(&tags);
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].label, "LOC");
        assert_eq!((decoded[0].start, decoded[0].end), (0, 2));
    }

    #[test]
    fn test_bio_single_token_bioes() {
        let tokens = vec!["Paris"];
        let spans = vec![Span::new(0, 1, "LOC", 1.0)];
        let tags = BIO::encode_tokens(&tokens, &spans, BIOScheme::BIOES);
        assert_eq!(tags[0], BIOTag::S("LOC".to_string()));
    }

    #[test]
    fn test_bio_from_str() {
        assert_eq!(BIOTag::from_str("B-PER"), BIOTag::B("PER".to_string()));
        assert_eq!(BIOTag::from_str("I-LOC"), BIOTag::I("LOC".to_string()));
        assert_eq!(BIOTag::from_str("O"), BIOTag::O);
        assert_eq!(BIOTag::from_str("E-ORG"), BIOTag::E("ORG".to_string()));
        assert_eq!(BIOTag::from_str("S-DATE"), BIOTag::S("DATE".to_string()));
    }

    #[test]
    fn test_annotation_merger_keep_longer() {
        let spans = vec![
            Span::new(0, 5, "PER", 0.8),
            Span::new(2, 8, "ORG", 0.6), // overlaps with first; longer
        ];
        let merger = AnnotationMerger::new().with_strategy(MergeStrategy::KeepLonger);
        let result = merger.merge(spans);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].label, "ORG"); // ORG is longer (6 > 5)
    }

    #[test]
    fn test_annotation_merger_keep_higher_score() {
        let spans = vec![
            Span::new(0, 5, "PER", 0.9),
            Span::new(2, 8, "ORG", 0.6),
        ];
        let merger = AnnotationMerger::new().with_strategy(MergeStrategy::KeepHigherScore);
        let result = merger.merge(spans);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].label, "PER");
    }

    #[test]
    fn test_annotation_merger_adjacent() {
        let spans = vec![
            Span::new(0, 3, "PER", 0.9),
            Span::new(3, 6, "PER", 0.8), // adjacent, same label
        ];
        let merger = AnnotationMerger::new()
            .with_strategy(MergeStrategy::MergeAdjacent)
            .merge_adjacent_spans(true);
        let result = merger.merge(spans);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].start, 0);
        assert_eq!(result[0].end, 6);
    }

    #[test]
    fn test_ensemble_merge_agreement() {
        let a1 = vec![
            Span::new(0, 5, "PER", 1.0),
            Span::new(10, 15, "LOC", 1.0),
        ];
        let a2 = vec![
            Span::new(0, 5, "PER", 0.9),  // agreed
            Span::new(20, 25, "ORG", 1.0), // unique to a2
        ];
        let merger = AnnotationMerger::new();
        let result = merger.ensemble_merge(vec![a1, a2], 2);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].label, "PER");
    }

    #[test]
    fn test_keyword_annotator() {
        let mut ann = KeywordAnnotator::new("ner");
        ann.add_keyword("New York", "LOC");
        ann.add_keyword("John", "PER");
        let spans = ann.annotate("John lives in New York today").expect("ok");
        assert!(!spans.is_empty());
        assert!(spans.iter().any(|s| s.label == "PER"));
        assert!(spans.iter().any(|s| s.label == "LOC"));
    }

    #[test]
    fn test_annotation_layer_basic() {
        let mut layer = AnnotationLayer::new(LayerKind::NER);
        layer.add(Span::new(0, 5, "PER", 0.9));
        layer.add(Span::new(10, 15, "LOC", 0.8));
        assert_eq!(layer.len(), 2);
        assert_eq!(layer.by_label("PER").len(), 1);
        let labels = layer.unique_labels();
        assert!(labels.contains(&"PER".to_string()));
        assert!(labels.contains(&"LOC".to_string()));
    }

    #[test]
    fn test_annotation_document() {
        let mut doc = AnnotationDocument::new("doc1", "Alice lives in Wonderland.");
        doc.add_span(LayerKind::NER, Span::new(0, 5, "PER", 1.0));
        doc.add_span(LayerKind::NER, Span::new(15, 25, "LOC", 1.0));
        let ner_layer = doc.get_layer("ner").expect("ner layer exists");
        assert_eq!(ner_layer.len(), 2);
    }

    #[test]
    fn test_bio_evaluation_metrics() {
        let gold = vec![
            BIOTag::B("PER".to_string()),
            BIOTag::I("PER".to_string()),
            BIOTag::O,
            BIOTag::B("LOC".to_string()),
        ];
        let pred = vec![
            BIOTag::B("PER".to_string()),
            BIOTag::I("PER".to_string()),
            BIOTag::O,
            BIOTag::O, // missed LOC
        ];
        let metrics = BIO::evaluation_metrics(&gold, &pred);
        let per_metrics = metrics.get("PER").copied().unwrap_or((0.0, 0.0, 0.0));
        assert!((per_metrics.2 - 1.0).abs() < 1e-9, "PER F1 should be 1.0");
    }

    #[test]
    fn test_layer_kind_roundtrip() {
        let kinds = ["ner", "pos", "dep", "coref", "sentiment", "custom_layer"];
        for k in &kinds {
            let kind = LayerKind::from_str(k);
            assert_eq!(kind.name(), *k);
        }
    }

    #[test]
    fn test_regex_annotator() {
        let mut ann = RegexAnnotator::new("date_finder");
        ann.add_pattern(r"\d{4}-\d{2}-\d{2}", "DATE").expect("regex ok");
        let spans = ann.annotate("Event on 2024-03-15 and 2025-01-01.").expect("ok");
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].label, "DATE");
    }
}
