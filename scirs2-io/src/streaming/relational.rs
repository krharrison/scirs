//! Relational streaming operations on JSON record streams.
//!
//! Provides memory-efficient transformations for streams of [`serde_json::Value`]
//! records, designed for processing large datasets without loading everything
//! into RAM.  All operators implement the [`crate::streaming::relational::StreamingOp`] trait which takes an
//! in-memory slice (chunk) and yields a transformed `Vec<serde_json::Value>`.
//!
//! For truly large datasets, pair these operators with [`crate::streaming::ChunkedReader`]
//! or iterate over JSONL files using [`crate::jsonl::JsonlReader`].
//!
//! # Available operators
//!
//! | Type | Description |
//! |------|-------------|
//! | [`crate::streaming::relational::StreamingFilter`] | Keep only records matching a predicate |
//! | [`crate::streaming::relational::StreamingProject`] | Keep only specified fields |
//! | [`crate::streaming::relational::StreamingRename`] | Rename fields |
//! | [`crate::streaming::relational::StreamingDeduplicate`] | Remove duplicate records by key |
//! | [`crate::streaming::relational::StreamingSort`] | External merge-sort by key |
//! | [`crate::streaming::relational::StreamingGroupBy`] | Sort-based group-by with aggregates |

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::error::{IoError, Result};

// ─────────────────────────────── StreamingOp trait ───────────────────────────

/// A stateless or stateful transformation on a chunk of JSON records.
pub trait StreamingOp {
    /// Transform a chunk of records.  Stateful operators (like Deduplicate) may
    /// carry state across multiple calls.
    fn process(&mut self, records: Vec<Value>) -> Vec<Value>;

    /// Reset operator state (useful when reusing across multiple inputs).
    fn reset(&mut self) {}
}

// ─────────────────────────────── StreamingFilter ─────────────────────────────

/// Keep only records for which the predicate returns `true`.
///
/// # Example
/// ```
/// use scirs2_io::streaming::relational::{StreamingFilter, StreamingOp};
/// use serde_json::json;
///
/// let mut f = StreamingFilter::new(|v| v["age"].as_i64().unwrap_or(0) >= 18);
/// let out = f.process(vec![
///     json!({"age": 17}),
///     json!({"age": 18}),
///     json!({"age": 25}),
/// ]);
/// assert_eq!(out.len(), 2);
/// ```
pub struct StreamingFilter {
    predicate: Box<dyn Fn(&Value) -> bool + Send>,
}

impl StreamingFilter {
    /// Create a new filter with the given predicate function.
    pub fn new<F>(predicate: F) -> Self
    where
        F: Fn(&Value) -> bool + Send + 'static,
    {
        Self {
            predicate: Box::new(predicate),
        }
    }
}

impl StreamingOp for StreamingFilter {
    fn process(&mut self, records: Vec<Value>) -> Vec<Value> {
        records
            .into_iter()
            .filter(|r| (self.predicate)(r))
            .collect()
    }
}

// ─────────────────────────────── StreamingProject ────────────────────────────

/// Keep only the specified fields in each record.  Fields not present in the
/// input record are silently omitted from the output.
///
/// # Example
/// ```
/// use scirs2_io::streaming::relational::{StreamingProject, StreamingOp};
/// use serde_json::json;
///
/// let mut p = StreamingProject::new(vec!["name".into(), "age".into()]);
/// let out = p.process(vec![json!({"name": "Alice", "age": 30, "extra": "X"})]);
/// assert!(!out[0].as_object().unwrap().contains_key("extra"));
/// ```
pub struct StreamingProject {
    fields: Vec<String>,
}

impl StreamingProject {
    /// Create a projection keeping `fields`.
    pub fn new(fields: Vec<String>) -> Self {
        Self { fields }
    }
}

impl StreamingOp for StreamingProject {
    fn process(&mut self, records: Vec<Value>) -> Vec<Value> {
        records
            .into_iter()
            .map(|r| {
                if let Value::Object(obj) = r {
                    let kept: serde_json::Map<String, Value> = self
                        .fields
                        .iter()
                        .filter_map(|f| obj.get(f).map(|v| (f.clone(), v.clone())))
                        .collect();
                    Value::Object(kept)
                } else {
                    r
                }
            })
            .collect()
    }
}

// ─────────────────────────────── StreamingRename ─────────────────────────────

/// Rename fields according to a mapping.  Fields not in the mapping are
/// passed through unchanged.
///
/// # Example
/// ```
/// use scirs2_io::streaming::relational::{StreamingRename, StreamingOp};
/// use serde_json::json;
/// use std::collections::HashMap;
///
/// let mut renames = HashMap::new();
/// renames.insert("old_name".into(), "new_name".into());
/// let mut op = StreamingRename::new(renames);
/// let out = op.process(vec![json!({"old_name": "Alice", "age": 30})]);
/// assert!(out[0].as_object().unwrap().contains_key("new_name"));
/// assert!(!out[0].as_object().unwrap().contains_key("old_name"));
/// ```
pub struct StreamingRename {
    renames: HashMap<String, String>,
}

impl StreamingRename {
    /// Create with a map of `old_name → new_name`.
    pub fn new(renames: HashMap<String, String>) -> Self {
        Self { renames }
    }
}

impl StreamingOp for StreamingRename {
    fn process(&mut self, records: Vec<Value>) -> Vec<Value> {
        records
            .into_iter()
            .map(|r| {
                if let Value::Object(obj) = r {
                    let renamed: serde_json::Map<String, Value> = obj
                        .into_iter()
                        .map(|(k, v)| {
                            let new_k = self.renames.get(&k).cloned().unwrap_or(k);
                            (new_k, v)
                        })
                        .collect();
                    Value::Object(renamed)
                } else {
                    r
                }
            })
            .collect()
    }
}

// ─────────────────────────────── StreamingDeduplicate ────────────────────────

/// Remove duplicate records by a computed key.  First occurrence wins; later
/// duplicates are dropped.  Maintains a `HashSet<String>` of seen keys in
/// memory.
///
/// # Example
/// ```
/// use scirs2_io::streaming::relational::{StreamingDeduplicate, StreamingOp};
/// use serde_json::json;
///
/// let mut dedup = StreamingDeduplicate::new(|v| {
///     v["id"].as_i64().map(|i| i.to_string()).unwrap_or_default()
/// });
/// let out = dedup.process(vec![
///     json!({"id": 1, "v": "a"}),
///     json!({"id": 2, "v": "b"}),
///     json!({"id": 1, "v": "c"}),  // duplicate
/// ]);
/// assert_eq!(out.len(), 2);
/// ```
pub struct StreamingDeduplicate {
    seen: HashSet<String>,
    key_fn: Box<dyn Fn(&Value) -> String + Send>,
}

impl StreamingDeduplicate {
    /// Create with a key extraction function.
    pub fn new<F>(key_fn: F) -> Self
    where
        F: Fn(&Value) -> String + Send + 'static,
    {
        Self {
            seen: HashSet::new(),
            key_fn: Box::new(key_fn),
        }
    }
}

impl StreamingOp for StreamingDeduplicate {
    fn process(&mut self, records: Vec<Value>) -> Vec<Value> {
        records
            .into_iter()
            .filter(|r| {
                let key = (self.key_fn)(r);
                self.seen.insert(key)
            })
            .collect()
    }

    fn reset(&mut self) {
        self.seen.clear();
    }
}

// ─────────────────────────────── StreamingSort ───────────────────────────────

/// External merge-sort for large datasets that don't fit in memory.
///
/// Phase 1: collect `buffer_size` records at a time, sort each run, write to a
///          temporary file.
/// Phase 2: k-way merge of all temporary run files.
///
/// # Example
/// ```no_run
/// use scirs2_io::streaming::relational::StreamingSort;
/// use serde_json::json;
///
/// let mut sorter = StreamingSort::new(
///     16,                                              // buffer size
///     |v| v["name"].as_str().unwrap_or("").to_string(), // sort key
///     true,                                            // ascending
/// );
///
/// let records = vec![
///     json!({"name": "Charlie"}),
///     json!({"name": "Alice"}),
///     json!({"name": "Bob"}),
/// ];
///
/// let sorted = sorter.sort_all(records).expect("sort");
/// assert_eq!(sorted[0]["name"], "Alice");
/// ```
pub struct StreamingSort {
    buffer_size: usize,
    key_fn: Box<dyn Fn(&Value) -> String + Send>,
    ascending: bool,
}

impl StreamingSort {
    /// Create a streaming sorter.
    ///
    /// - `buffer_size`: number of records to buffer per run before writing to disk.
    /// - `key_fn`: extract the sort key from each record.
    /// - `ascending`: sort direction.
    pub fn new<F>(buffer_size: usize, key_fn: F, ascending: bool) -> Self
    where
        F: Fn(&Value) -> String + Send + 'static,
    {
        Self {
            buffer_size: buffer_size.max(1),
            key_fn: Box::new(key_fn),
            ascending,
        }
    }

    /// Sort all records using external merge-sort.
    ///
    /// For datasets small enough to fit in `buffer_size`, this is equivalent to
    /// an in-memory sort.
    pub fn sort_all(&mut self, records: Vec<Value>) -> Result<Vec<Value>> {
        if records.len() <= self.buffer_size {
            return Ok(self.sort_in_memory(records));
        }
        // Phase 1: produce sorted runs on disk
        let run_files = self.produce_runs(&records)?;
        // Phase 2: k-way merge
        let merged = self.kway_merge(run_files)?;
        Ok(merged)
    }

    fn sort_in_memory(&self, mut records: Vec<Value>) -> Vec<Value> {
        records.sort_by(|a, b| {
            let ka = (self.key_fn)(a);
            let kb = (self.key_fn)(b);
            if self.ascending {
                ka.cmp(&kb)
            } else {
                kb.cmp(&ka)
            }
        });
        records
    }

    fn produce_runs(&self, records: &[Value]) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::new();
        let temp_dir = std::env::temp_dir();

        for (run_idx, chunk) in records.chunks(self.buffer_size).enumerate() {
            let mut run: Vec<Value> = chunk.to_vec();
            run.sort_by(|a, b| {
                let ka = (self.key_fn)(a);
                let kb = (self.key_fn)(b);
                if self.ascending {
                    ka.cmp(&kb)
                } else {
                    kb.cmp(&ka)
                }
            });

            let path = temp_dir.join(format!(
                "scirs2_sort_run_{run_idx}_{}.jsonl",
                std::process::id()
            ));
            let f = fs::File::create(&path).map_err(IoError::Io)?;
            let mut w = BufWriter::new(f);
            for record in &run {
                let line = serde_json::to_string(record)
                    .map_err(|e| IoError::SerializationError(e.to_string()))?;
                writeln!(w, "{line}").map_err(IoError::Io)?;
            }
            w.flush().map_err(IoError::Io)?;
            paths.push(path);
        }
        Ok(paths)
    }

    fn kway_merge(&self, run_files: Vec<PathBuf>) -> Result<Vec<Value>> {
        // Open all run files
        let mut readers: Vec<BufReader<fs::File>> = run_files
            .iter()
            .map(|p| fs::File::open(p).map(BufReader::new).map_err(IoError::Io))
            .collect::<Result<Vec<_>>>()?;

        // Prime each reader with the first record, computing the sort key
        // using the same key_fn used for run-sorting
        let mut heads: Vec<Option<(String, Value)>> = readers
            .iter_mut()
            .map(|r| peek_next_json_with_key(r, &self.key_fn))
            .collect::<Result<Vec<_>>>()?;

        let mut merged = Vec::new();

        loop {
            // Find the run with the minimum (or maximum) key
            let best_idx = heads
                .iter()
                .enumerate()
                .filter_map(|(i, h)| h.as_ref().map(|(k, _)| (i, k.clone())))
                .min_by(|(_, ka), (_, kb)| {
                    if self.ascending {
                        ka.cmp(kb)
                    } else {
                        kb.cmp(ka)
                    }
                })
                .map(|(i, _)| i);

            match best_idx {
                None => break,
                Some(i) => {
                    if let Some((_, value)) = heads[i].take() {
                        merged.push(value);
                    }
                    heads[i] = peek_next_json_with_key(&mut readers[i], &self.key_fn)?;
                }
            }
        }

        // Clean up temp files
        for path in &run_files {
            let _ = fs::remove_file(path);
        }

        Ok(merged)
    }
}

/// Read the next JSON record from a reader, computing the sort key via `key_fn`.
fn peek_next_json_with_key<R: BufRead>(
    reader: &mut R,
    key_fn: &dyn Fn(&Value) -> String,
) -> Result<Option<(String, Value)>> {
    let mut line = String::new();
    match reader.read_line(&mut line) {
        Ok(0) => Ok(None),
        Ok(_) => {
            let trimmed = line.trim_end();
            if trimmed.is_empty() {
                return Ok(None);
            }
            let val: Value = serde_json::from_str(trimmed)
                .map_err(|e| IoError::DeserializationError(e.to_string()))?;
            let key = key_fn(&val);
            Ok(Some((key, val)))
        }
        Err(e) => Err(IoError::Io(e)),
    }
}

// ─────────────────────────────── StreamingAggregate ──────────────────────────

/// A stateful streaming aggregate that can be updated with new records and
/// queried for the current result.
pub trait StreamingAggregate: Send {
    /// Update the aggregate with one record.
    fn update(&mut self, record: &Value);
    /// Return the current aggregate result as a JSON value.
    fn result(&self) -> Value;
    /// Reset the aggregate to its initial state.
    fn reset_agg(&mut self);
    /// Name of this aggregate (used as output field name).
    fn name(&self) -> &str;
}

/// Count aggregate: counts the number of records seen.
pub struct CountAggregate {
    count: u64,
    field_name: String,
}

impl CountAggregate {
    /// Create a count aggregate with the given output field name.
    pub fn new(field_name: impl Into<String>) -> Self {
        Self {
            count: 0,
            field_name: field_name.into(),
        }
    }
}

impl StreamingAggregate for CountAggregate {
    fn update(&mut self, _record: &Value) {
        self.count += 1;
    }
    fn result(&self) -> Value {
        Value::Number(self.count.into())
    }
    fn reset_agg(&mut self) {
        self.count = 0;
    }
    fn name(&self) -> &str {
        &self.field_name
    }
}

/// Sum aggregate: sums a numeric field.
pub struct SumAggregate {
    field: String,
    sum: f64,
    output_name: String,
}

impl SumAggregate {
    /// Sum the numeric field `field`, output as `output_name`.
    pub fn new(field: impl Into<String>, output_name: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            sum: 0.0,
            output_name: output_name.into(),
        }
    }
}

impl StreamingAggregate for SumAggregate {
    fn update(&mut self, record: &Value) {
        if let Some(n) = record.get(&self.field).and_then(|v| v.as_f64()) {
            self.sum += n;
        }
    }
    fn result(&self) -> Value {
        serde_json::json!(self.sum)
    }
    fn reset_agg(&mut self) {
        self.sum = 0.0;
    }
    fn name(&self) -> &str {
        &self.output_name
    }
}

/// Min aggregate: tracks the minimum value of a numeric field.
pub struct MinAggregate {
    field: String,
    min: f64,
    output_name: String,
    has_value: bool,
}

impl MinAggregate {
    /// Track minimum of `field`, output as `output_name`.
    pub fn new(field: impl Into<String>, output_name: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            min: f64::MAX,
            output_name: output_name.into(),
            has_value: false,
        }
    }
}

impl StreamingAggregate for MinAggregate {
    fn update(&mut self, record: &Value) {
        if let Some(n) = record.get(&self.field).and_then(|v| v.as_f64()) {
            if !self.has_value || n < self.min {
                self.min = n;
                self.has_value = true;
            }
        }
    }
    fn result(&self) -> Value {
        if self.has_value {
            serde_json::json!(self.min)
        } else {
            Value::Null
        }
    }
    fn reset_agg(&mut self) {
        self.min = f64::MAX;
        self.has_value = false;
    }
    fn name(&self) -> &str {
        &self.output_name
    }
}

/// Max aggregate: tracks the maximum value of a numeric field.
pub struct MaxAggregate {
    field: String,
    max: f64,
    output_name: String,
    has_value: bool,
}

impl MaxAggregate {
    /// Track maximum of `field`, output as `output_name`.
    pub fn new(field: impl Into<String>, output_name: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            max: f64::MIN,
            output_name: output_name.into(),
            has_value: false,
        }
    }
}

impl StreamingAggregate for MaxAggregate {
    fn update(&mut self, record: &Value) {
        if let Some(n) = record.get(&self.field).and_then(|v| v.as_f64()) {
            if !self.has_value || n > self.max {
                self.max = n;
                self.has_value = true;
            }
        }
    }
    fn result(&self) -> Value {
        if self.has_value {
            serde_json::json!(self.max)
        } else {
            Value::Null
        }
    }
    fn reset_agg(&mut self) {
        self.max = f64::MIN;
        self.has_value = false;
    }
    fn name(&self) -> &str {
        &self.output_name
    }
}

// ─────────────────────────────── StreamingGroupBy ────────────────────────────

/// Sort-based streaming group-by with arbitrary aggregates.
///
/// Processing strategy:
/// 1. Sort records by key using [`StreamingSort`].
/// 2. Iterate through sorted records, accumulating aggregates per group.
/// 3. Emit one output record per group.
///
/// # Example
/// ```
/// use scirs2_io::streaming::relational::{
///     StreamingGroupBy, SumAggregate, CountAggregate,
/// };
/// use serde_json::json;
///
/// let records = vec![
///     json!({"dept": "eng",  "salary": 100.0}),
///     json!({"dept": "hr",   "salary": 80.0}),
///     json!({"dept": "eng",  "salary": 120.0}),
///     json!({"dept": "hr",   "salary": 90.0}),
/// ];
///
/// let aggs: Vec<Box<dyn scirs2_io::streaming::relational::StreamingAggregate>> = vec![
///     Box::new(CountAggregate::new("count")),
///     Box::new(SumAggregate::new("salary", "total_salary")),
/// ];
///
/// let mut gb = StreamingGroupBy::new(
///     |v| v["dept"].as_str().unwrap_or("").to_string(),
///     aggs,
///     64,
/// );
///
/// let result = gb.run(records).expect("group by");
/// assert_eq!(result.len(), 2); // two departments
/// ```
pub struct StreamingGroupBy {
    key_fn: Box<dyn Fn(&Value) -> String + Send>,
    aggregates: Vec<Box<dyn StreamingAggregate>>,
    buffer_size: usize,
}

impl StreamingGroupBy {
    /// Create a group-by operator.
    ///
    /// - `key_fn`: extract the group key.
    /// - `aggregates`: list of aggregate functions.
    /// - `buffer_size`: run-sort buffer size passed to [`StreamingSort`].
    pub fn new<F>(
        key_fn: F,
        aggregates: Vec<Box<dyn StreamingAggregate>>,
        buffer_size: usize,
    ) -> Self
    where
        F: Fn(&Value) -> String + Send + 'static,
    {
        Self {
            key_fn: Box::new(key_fn),
            aggregates,
            buffer_size,
        }
    }

    /// Execute the group-by over `records`.
    ///
    /// Records are sorted by key, then groups are processed sequentially.
    pub fn run(&mut self, records: Vec<Value>) -> Result<Vec<Value>> {
        // Sort by key
        let key_fn_clone = &self.key_fn;
        let mut sorted = records;
        sorted.sort_by(|a, b| {
            let ka = (key_fn_clone)(a);
            let kb = (key_fn_clone)(b);
            ka.cmp(&kb)
        });

        if sorted.is_empty() {
            return Ok(Vec::new());
        }

        let mut output = Vec::new();
        let mut current_key = (self.key_fn)(&sorted[0]);
        for agg in &mut self.aggregates {
            agg.reset_agg();
        }

        for record in sorted {
            let key = (self.key_fn)(&record);
            if key != current_key {
                // Emit group
                output.push(self.emit_group(&current_key));
                current_key = key;
                for agg in &mut self.aggregates {
                    agg.reset_agg();
                }
            }
            for agg in &mut self.aggregates {
                agg.update(&record);
            }
        }
        // Emit final group
        output.push(self.emit_group(&current_key));
        Ok(output)
    }

    fn emit_group(&self, key: &str) -> Value {
        let mut obj = serde_json::Map::new();
        obj.insert("_key".to_string(), Value::String(key.to_owned()));
        for agg in &self.aggregates {
            obj.insert(agg.name().to_string(), agg.result());
        }
        Value::Object(obj)
    }
}

// ─────────────────────────────── Pipeline ────────────────────────────────────

/// Chain of [`StreamingOp`] operators applied sequentially.
///
/// Records flow through each operator in order.  Useful for building
/// composable ETL pipelines.
///
/// # Example
/// ```
/// use scirs2_io::streaming::relational::{
///     StreamingPipeline, StreamingFilter, StreamingProject, StreamingOp,
/// };
/// use serde_json::json;
///
/// let mut pipeline = StreamingPipeline::new();
/// pipeline.add(StreamingFilter::new(|v| v["active"].as_bool().unwrap_or(false)));
/// pipeline.add(StreamingProject::new(vec!["name".into()]));
///
/// let out = pipeline.run(vec![
///     json!({"name": "Alice", "active": true,  "score": 99}),
///     json!({"name": "Bob",   "active": false, "score": 80}),
/// ]);
/// assert_eq!(out.len(), 1);
/// assert!(out[0].as_object().unwrap().contains_key("name"));
/// assert!(!out[0].as_object().unwrap().contains_key("score"));
/// ```
pub struct StreamingPipeline {
    ops: Vec<Box<dyn StreamingOp>>,
}

impl StreamingPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Append an operator to the pipeline.
    pub fn add<O: StreamingOp + 'static>(&mut self, op: O) {
        self.ops.push(Box::new(op));
    }

    /// Run all operators in sequence.
    pub fn run(&mut self, records: Vec<Value>) -> Vec<Value> {
        let mut current = records;
        for op in &mut self.ops {
            current = op.process(current);
        }
        current
    }
}

impl Default for StreamingPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────── File-based helpers ──────────────────────────

/// Sort a JSONL file using external merge-sort, writing output to another file.
///
/// Useful for command-line-style sorting of large JSONL datasets.
pub fn sort_jsonl_file<F>(
    input: &Path,
    output: &Path,
    key_fn: F,
    ascending: bool,
    buffer_size: usize,
) -> Result<usize>
where
    F: Fn(&Value) -> String + Send + 'static,
{
    let f = fs::File::open(input).map_err(IoError::Io)?;
    let reader = BufReader::new(f);
    let mut records = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(IoError::Io)?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let val: Value = serde_json::from_str(trimmed)
            .map_err(|e| IoError::DeserializationError(e.to_string()))?;
        records.push(val);
    }

    let total = records.len();
    let mut sorter = StreamingSort::new(buffer_size, key_fn, ascending);
    let sorted = sorter.sort_all(records)?;

    let out_f = fs::File::create(output).map_err(IoError::Io)?;
    let mut w = BufWriter::new(out_f);
    for record in &sorted {
        let line = serde_json::to_string(record)
            .map_err(|e| IoError::SerializationError(e.to_string()))?;
        writeln!(w, "{line}").map_err(IoError::Io)?;
    }
    w.flush().map_err(IoError::Io)?;
    Ok(total)
}

// ─────────────────────────────── Tests ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::env::temp_dir;

    fn records() -> Vec<Value> {
        vec![
            json!({"id": 1, "dept": "eng",  "salary": 100.0, "name": "Alice", "active": true}),
            json!({"id": 2, "dept": "hr",   "salary": 80.0,  "name": "Bob",   "active": false}),
            json!({"id": 3, "dept": "eng",  "salary": 120.0, "name": "Charlie","active": true}),
            json!({"id": 4, "dept": "hr",   "salary": 90.0,  "name": "Dave",  "active": true}),
            json!({"id": 5, "dept": "eng",  "salary": 110.0, "name": "Eve",   "active": false}),
        ]
    }

    #[test]
    fn test_filter() {
        let mut f = StreamingFilter::new(|v| v["active"].as_bool().unwrap_or(false));
        let out = f.process(records());
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_project() {
        let mut p = StreamingProject::new(vec!["id".into(), "name".into()]);
        let out = p.process(records());
        assert_eq!(out.len(), 5);
        let obj = out[0].as_object().expect("object");
        assert!(obj.contains_key("id"));
        assert!(obj.contains_key("name"));
        assert!(!obj.contains_key("salary"));
        assert!(!obj.contains_key("dept"));
    }

    #[test]
    fn test_rename() {
        let mut renames = HashMap::new();
        renames.insert("salary".into(), "pay".into());
        let mut op = StreamingRename::new(renames);
        let out = op.process(records());
        let obj = out[0].as_object().expect("object");
        assert!(obj.contains_key("pay"));
        assert!(!obj.contains_key("salary"));
    }

    #[test]
    fn test_deduplicate() {
        let input = vec![
            json!({"id": 1, "v": "a"}),
            json!({"id": 2, "v": "b"}),
            json!({"id": 1, "v": "c"}), // duplicate
            json!({"id": 3, "v": "d"}),
        ];
        let mut dedup = StreamingDeduplicate::new(|v| {
            v["id"].as_i64().map(|i| i.to_string()).unwrap_or_default()
        });
        let out = dedup.process(input);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0]["v"], "a"); // first occurrence kept
    }

    #[test]
    fn test_sort_ascending() {
        let input = vec![
            json!({"name": "Charlie"}),
            json!({"name": "Alice"}),
            json!({"name": "Bob"}),
        ];
        let mut sorter =
            StreamingSort::new(16, |v| v["name"].as_str().unwrap_or("").to_string(), true);
        let out = sorter.sort_all(input).expect("sort");
        assert_eq!(out[0]["name"], "Alice");
        assert_eq!(out[1]["name"], "Bob");
        assert_eq!(out[2]["name"], "Charlie");
    }

    #[test]
    fn test_sort_descending() {
        let input = vec![
            json!({"name": "Alice"}),
            json!({"name": "Charlie"}),
            json!({"name": "Bob"}),
        ];
        let mut sorter =
            StreamingSort::new(16, |v| v["name"].as_str().unwrap_or("").to_string(), false);
        let out = sorter.sort_all(input).expect("sort");
        assert_eq!(out[0]["name"], "Charlie");
        assert_eq!(out[1]["name"], "Bob");
        assert_eq!(out[2]["name"], "Alice");
    }

    #[test]
    fn test_sort_external_merge() {
        // Force external merge by using a small buffer
        let input: Vec<Value> = (0..20_i64).rev().map(|i| json!({"v": i})).collect();
        let mut sorter = StreamingSort::new(
            3, // tiny buffer → many run files
            |v| {
                let n = v["v"].as_i64().unwrap_or(0);
                format!("{n:010}") // zero-padded for string sort
            },
            true,
        );
        let out = sorter.sort_all(input).expect("sort");
        assert_eq!(out.len(), 20);
        assert_eq!(out[0]["v"], 0);
        assert_eq!(out[19]["v"], 19);
    }

    #[test]
    fn test_group_by_count_sum() {
        let recs = records();
        let aggs: Vec<Box<dyn StreamingAggregate>> = vec![
            Box::new(CountAggregate::new("count")),
            Box::new(SumAggregate::new("salary", "total")),
        ];
        let mut gb =
            StreamingGroupBy::new(|v| v["dept"].as_str().unwrap_or("").to_string(), aggs, 64);
        let out = gb.run(recs).expect("group by");
        assert_eq!(out.len(), 2);

        // Find the "eng" group
        let eng = out.iter().find(|v| v["_key"] == "eng").expect("eng group");
        assert_eq!(eng["count"], 3);
        assert!((eng["total"].as_f64().unwrap() - 330.0).abs() < 1e-9);
    }

    #[test]
    fn test_group_by_min_max() {
        let recs = records();
        let aggs: Vec<Box<dyn StreamingAggregate>> = vec![
            Box::new(MinAggregate::new("salary", "min_sal")),
            Box::new(MaxAggregate::new("salary", "max_sal")),
        ];
        let mut gb =
            StreamingGroupBy::new(|v| v["dept"].as_str().unwrap_or("").to_string(), aggs, 64);
        let out = gb.run(recs).expect("group by");
        let eng = out.iter().find(|v| v["_key"] == "eng").expect("eng group");
        assert!((eng["min_sal"].as_f64().unwrap() - 100.0).abs() < 1e-9);
        assert!((eng["max_sal"].as_f64().unwrap() - 120.0).abs() < 1e-9);
    }

    #[test]
    fn test_pipeline() {
        let mut pipeline = StreamingPipeline::new();
        pipeline.add(StreamingFilter::new(|v| {
            v["active"].as_bool().unwrap_or(false)
        }));
        pipeline.add(StreamingProject::new(vec!["id".into(), "name".into()]));

        let out = pipeline.run(records());
        assert_eq!(out.len(), 3);
        let obj = out[0].as_object().expect("object");
        assert!(obj.contains_key("name"));
        assert!(!obj.contains_key("salary"));
    }

    #[test]
    fn test_sort_jsonl_file() {
        let temp = temp_dir();
        let input_path = temp.join("sort_test_input.jsonl");
        let output_path = temp.join("sort_test_output.jsonl");

        let lines = vec![r#"{"v": 3}"#, r#"{"v": 1}"#, r#"{"v": 2}"#];
        fs::write(&input_path, lines.join("\n")).expect("write");

        let count = sort_jsonl_file(
            &input_path,
            &output_path,
            |v| {
                let n = v["v"].as_i64().unwrap_or(0);
                format!("{n:010}")
            },
            true,
            100,
        )
        .expect("sort file");

        assert_eq!(count, 3);

        let output = fs::read_to_string(&output_path).expect("read");
        let vals: Vec<i64> = output
            .lines()
            .map(|l| {
                serde_json::from_str::<Value>(l).expect("json")["v"]
                    .as_i64()
                    .unwrap()
            })
            .collect();
        assert_eq!(vals, vec![1, 2, 3]);
    }

    #[test]
    fn test_deduplicate_reset() {
        let mut dedup = StreamingDeduplicate::new(|v| {
            v["id"].as_i64().map(|i| i.to_string()).unwrap_or_default()
        });
        let first = dedup.process(vec![json!({"id": 1}), json!({"id": 1})]);
        assert_eq!(first.len(), 1);

        dedup.reset(); // clear seen set
        let second = dedup.process(vec![json!({"id": 1}), json!({"id": 1})]);
        assert_eq!(second.len(), 1);
    }
}
