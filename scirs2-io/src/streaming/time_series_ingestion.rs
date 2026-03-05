//! High-throughput time-series data ingestion
//!
//! Provides:
//! - `TimeSeriesIngester` — buffered deduplication with configurable flush policy
//! - `ingest_csv_stream()` — streaming CSV with automatic timestamp parsing
//! - `downsampling_writer()` — emit downsampled series using LTTB or min-max
//! - `gap_detection()` — identify irregular gaps in a time series
//! - `monotonic_check()` — validate timestamp monotonicity

use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::error::{IoError, Result};
use crate::streaming::log_parsing::parse_timestamp;

// ──────────────────────────────────────────────────────────────────────────────
// Core data structures
// ──────────────────────────────────────────────────────────────────────────────

/// A single time-stamped data point.
#[derive(Debug, Clone, PartialEq)]
pub struct DataPoint {
    /// Unix milliseconds.
    pub timestamp_ms: i64,
    /// Named values at this timestamp.
    pub values: HashMap<String, f64>,
}

impl DataPoint {
    /// Create a data point with a single named value.
    pub fn single(timestamp_ms: i64, name: impl Into<String>, value: f64) -> Self {
        let mut values = HashMap::new();
        values.insert(name.into(), value);
        DataPoint {
            timestamp_ms,
            values,
        }
    }

    /// Create a data point from a list of (name, value) pairs.
    pub fn from_pairs(timestamp_ms: i64, pairs: &[(&str, f64)]) -> Self {
        let values = pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        DataPoint {
            timestamp_ms,
            values,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// TimeSeriesIngester
// ──────────────────────────────────────────────────────────────────────────────

/// Policy for handling duplicate timestamps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeduplicationPolicy {
    /// Keep the first value seen for a given timestamp.
    KeepFirst,
    /// Replace with the latest value seen.
    KeepLast,
    /// Average values for the same timestamp.
    Average,
    /// Error on duplicate timestamps.
    Reject,
}

/// Flush trigger conditions for the ingester buffer.
#[derive(Debug, Clone)]
pub struct FlushPolicy {
    /// Flush when the buffer reaches this many points.
    pub max_buffer_size: usize,
    /// Flush when the age of the oldest buffered point exceeds this (millis).
    pub max_age_ms: i64,
}

impl Default for FlushPolicy {
    fn default() -> Self {
        FlushPolicy {
            max_buffer_size: 4096,
            max_age_ms: 60_000,
        }
    }
}

/// Buffered time-series ingester with deduplication.
///
/// Points are buffered in a `BTreeMap` keyed by timestamp to maintain order.
/// When a flush is triggered (manually or by policy), the sorted buffer is
/// drained to the configured sink.
pub struct TimeSeriesIngester {
    /// Deduplication policy.
    pub policy: DeduplicationPolicy,
    /// Flush configuration.
    pub flush_policy: FlushPolicy,
    /// Internal buffer: timestamp_ms → accumulated point.
    buffer: BTreeMap<i64, DataPoint>,
    /// Count of values accumulated per timestamp (used for Average policy).
    count_map: HashMap<i64, usize>,
    /// Total points ingested (before deduplication).
    pub ingested_total: u64,
    /// Duplicate timestamps encountered.
    pub duplicate_count: u64,
    /// Points flushed.
    pub flushed_total: u64,
}

impl TimeSeriesIngester {
    /// Create a new ingester.
    pub fn new(policy: DeduplicationPolicy, flush_policy: FlushPolicy) -> Self {
        TimeSeriesIngester {
            policy,
            flush_policy,
            buffer: BTreeMap::new(),
            count_map: HashMap::new(),
            ingested_total: 0,
            duplicate_count: 0,
            flushed_total: 0,
        }
    }

    /// Ingest a single data point.
    ///
    /// Returns `Err` only when `DeduplicationPolicy::Reject` is used and a
    /// duplicate is encountered.
    pub fn ingest(&mut self, point: DataPoint) -> Result<()> {
        self.ingested_total += 1;
        let ts = point.timestamp_ms;

        if self.buffer.contains_key(&ts) {
            self.duplicate_count += 1;
            match self.policy {
                DeduplicationPolicy::KeepFirst => {
                    // do nothing — keep existing
                }
                DeduplicationPolicy::KeepLast => {
                    self.buffer.insert(ts, point);
                }
                DeduplicationPolicy::Average => {
                    let entry = self.buffer.entry(ts).or_insert_with(|| DataPoint {
                        timestamp_ms: ts,
                        values: HashMap::new(),
                    });
                    let count = *self.count_map.get(&ts).unwrap_or(&1);
                    for (name, val) in &point.values {
                        let current = entry.values.entry(name.clone()).or_insert(0.0);
                        *current = (*current * count as f64 + val) / (count + 1) as f64;
                    }
                    self.count_map.insert(ts, count + 1);
                }
                DeduplicationPolicy::Reject => {
                    return Err(IoError::ValidationError(format!(
                        "duplicate timestamp: {}",
                        ts
                    )));
                }
            }
        } else {
            self.count_map.insert(ts, 1);
            self.buffer.insert(ts, point);
        }

        Ok(())
    }

    /// Ingest multiple points.
    pub fn ingest_batch(&mut self, points: impl IntoIterator<Item = DataPoint>) -> Result<()> {
        for p in points {
            self.ingest(p)?;
        }
        Ok(())
    }

    /// Return true if the flush policy threshold has been reached.
    pub fn should_flush(&self) -> bool {
        if self.buffer.len() >= self.flush_policy.max_buffer_size {
            return true;
        }
        if let Some((&oldest_ts, _)) = self.buffer.iter().next() {
            if let Some((&newest_ts, _)) = self.buffer.iter().next_back() {
                let age = newest_ts - oldest_ts;
                if age >= self.flush_policy.max_age_ms {
                    return true;
                }
            }
        }
        false
    }

    /// Drain the buffer and return all points in timestamp order.
    pub fn flush(&mut self) -> Vec<DataPoint> {
        let points: Vec<DataPoint> = self.buffer.values().cloned().collect();
        self.flushed_total += points.len() as u64;
        self.buffer.clear();
        self.count_map.clear();
        points
    }

    /// Return a reference to buffered points without flushing.
    pub fn peek(&self) -> Vec<&DataPoint> {
        self.buffer.values().collect()
    }

    /// Number of distinct timestamps currently buffered.
    pub fn buffered_count(&self) -> usize {
        self.buffer.len()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Streaming CSV ingestion
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for streaming CSV ingestion.
#[derive(Debug, Clone)]
pub struct CsvIngestionConfig {
    /// Column delimiter (default: `,`).
    pub delimiter: char,
    /// Name of the column containing the timestamp.
    pub timestamp_column: String,
    /// Columns to ingest as values (empty = all non-timestamp columns).
    pub value_columns: Vec<String>,
    /// Whether the first row is a header.
    pub has_header: bool,
    /// Maximum number of rows to ingest (0 = unlimited).
    pub max_rows: usize,
}

impl Default for CsvIngestionConfig {
    fn default() -> Self {
        CsvIngestionConfig {
            delimiter: ',',
            timestamp_column: "timestamp".into(),
            value_columns: Vec::new(),
            has_header: true,
            max_rows: 0,
        }
    }
}

/// Result of a streaming CSV ingestion run.
#[derive(Debug, Default)]
pub struct CsvIngestionResult {
    /// Total rows read.
    pub rows_read: usize,
    /// Rows successfully ingested.
    pub rows_ingested: usize,
    /// Rows skipped due to parse errors.
    pub rows_skipped: usize,
    /// All ingested points in timestamp order.
    pub points: Vec<DataPoint>,
}

/// Ingest a CSV file as a time series stream.
///
/// Reads the file line by line; each data row becomes a `DataPoint`.
pub fn ingest_csv_stream<P: AsRef<Path>>(
    path: P,
    config: &CsvIngestionConfig,
) -> Result<CsvIngestionResult> {
    let file = std::fs::File::open(path.as_ref()).map_err(IoError::Io)?;
    let reader = BufReader::new(file);
    ingest_csv_reader(reader, config)
}

/// Ingest CSV from any `BufRead` source.
pub fn ingest_csv_reader<R: BufRead>(
    reader: R,
    config: &CsvIngestionConfig,
) -> Result<CsvIngestionResult> {
    let mut result = CsvIngestionResult::default();
    let mut ingester =
        TimeSeriesIngester::new(DeduplicationPolicy::KeepLast, FlushPolicy::default());

    let mut lines = reader.lines();
    let mut ts_col_idx: Option<usize> = None;
    let mut val_col_indices: Vec<(String, usize)> = Vec::new();

    // Parse header
    if config.has_header {
        if let Some(Ok(header_line)) = lines.next() {
            let headers = split_csv_row(&header_line, config.delimiter);
            ts_col_idx = headers.iter().position(|h| h == &config.timestamp_column);
            if config.value_columns.is_empty() {
                // Use all non-timestamp columns
                let ts_idx = ts_col_idx.unwrap_or(usize::MAX);
                for (i, h) in headers.iter().enumerate() {
                    if i != ts_idx {
                        val_col_indices.push((h.clone(), i));
                    }
                }
            } else {
                for col_name in &config.value_columns {
                    if let Some(idx) = headers.iter().position(|h| h == col_name) {
                        val_col_indices.push((col_name.clone(), idx));
                    }
                }
            }
        }
    }

    for line_res in lines {
        let line = line_res.map_err(IoError::Io)?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        result.rows_read += 1;
        if config.max_rows > 0 && result.rows_read > config.max_rows {
            break;
        }

        let cells = split_csv_row(trimmed, config.delimiter);

        // Determine timestamp
        let ts_ms = if let Some(idx) = ts_col_idx {
            match cells.get(idx) {
                Some(s) => match parse_timestamp(s) {
                    Ok(ms) => ms,
                    Err(_) => {
                        result.rows_skipped += 1;
                        continue;
                    }
                },
                None => {
                    result.rows_skipped += 1;
                    continue;
                }
            }
        } else {
            // No timestamp column — use row index as ms
            result.rows_read as i64
        };

        // Build values map
        let mut values: HashMap<String, f64> = HashMap::new();
        for (col_name, idx) in &val_col_indices {
            if let Some(cell) = cells.get(*idx) {
                if let Ok(v) = cell.trim().parse::<f64>() {
                    values.insert(col_name.clone(), v);
                }
            }
        }

        if values.is_empty() && !val_col_indices.is_empty() {
            result.rows_skipped += 1;
            continue;
        }

        let point = DataPoint {
            timestamp_ms: ts_ms,
            values,
        };
        if ingester.ingest(point).is_ok() {
            result.rows_ingested += 1;
        } else {
            result.rows_skipped += 1;
        }
    }

    result.points = ingester.flush();
    Ok(result)
}

/// Split a CSV row respecting double-quoted fields.
fn split_csv_row(line: &str, delim: char) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '"' {
            if in_quotes {
                // Peek for escaped quote
                if chars.peek() == Some(&'"') {
                    chars.next();
                    current.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                in_quotes = true;
            }
        } else if c == delim && !in_quotes {
            fields.push(current.trim().to_string());
            current = String::new();
        } else {
            current.push(c);
        }
    }
    fields.push(current.trim().to_string());
    fields
}

// ──────────────────────────────────────────────────────────────────────────────
// Downsampling
// ──────────────────────────────────────────────────────────────────────────────

/// Downsampling algorithm selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DownsamplingMethod {
    /// Largest-Triangle-Three-Buckets — visually faithful downsampling.
    Lttb,
    /// Retain the min and max of each bucket.
    MinMax,
    /// Simple uniform sampling (every N-th point).
    Uniform,
}

/// Downsample a single-channel time series to at most `target_count` points.
///
/// `timestamps` and `values` must have the same length and the timestamps
/// must be in non-decreasing order.
///
/// Returns `(downsampled_timestamps, downsampled_values)`.
pub fn downsample(
    timestamps: &[i64],
    values: &[f64],
    target_count: usize,
    method: DownsamplingMethod,
) -> Result<(Vec<i64>, Vec<f64>)> {
    if timestamps.len() != values.len() {
        return Err(IoError::ValidationError(
            "timestamps and values must have equal length".into(),
        ));
    }
    if target_count == 0 {
        return Err(IoError::ValidationError("target_count must be > 0".into()));
    }

    let n = timestamps.len();

    if n <= target_count {
        return Ok((timestamps.to_vec(), values.to_vec()));
    }

    match method {
        DownsamplingMethod::Lttb => Ok(lttb(timestamps, values, target_count)),
        DownsamplingMethod::MinMax => Ok(min_max_downsample(timestamps, values, target_count)),
        DownsamplingMethod::Uniform => {
            let step = n as f64 / target_count as f64;
            let ts: Vec<i64> = (0..target_count)
                .map(|i| timestamps[(i as f64 * step) as usize])
                .collect();
            let vs: Vec<f64> = (0..target_count)
                .map(|i| values[(i as f64 * step) as usize])
                .collect();
            Ok((ts, vs))
        }
    }
}

/// LTTB (Largest Triangle Three Buckets) downsampling.
///
/// Reference: Sveinn Steinarsson, "Downsampling Time Series for Visual
/// Representation", MSc thesis, 2013.
fn lttb(timestamps: &[i64], values: &[f64], target_count: usize) -> (Vec<i64>, Vec<f64>) {
    let n = timestamps.len();
    if target_count >= n {
        return (timestamps.to_vec(), values.to_vec());
    }

    let mut out_ts = Vec::with_capacity(target_count);
    let mut out_vs = Vec::with_capacity(target_count);

    // Always keep first point
    out_ts.push(timestamps[0]);
    out_vs.push(values[0]);

    // Bucket size for middle portion
    let bucket_size = (n - 2) as f64 / (target_count - 2) as f64;
    let mut a = 0usize; // previous selected point

    for i in 0..target_count - 2 {
        // Current bucket
        let start = ((i as f64 * bucket_size) as usize + 1).min(n - 1);
        let end = (((i + 1) as f64 * bucket_size) as usize + 1).min(n);

        // Next bucket average (used as the "third" point for area calculation)
        let next_start = end;
        let next_end = (((i + 2) as f64 * bucket_size) as usize + 1).min(n);
        let (avg_ts, avg_v) = if next_end > next_start {
            let slice_ts = &timestamps[next_start..next_end];
            let slice_vs = &values[next_start..next_end];
            let len = slice_ts.len() as f64;
            (
                slice_ts.iter().sum::<i64>() as f64 / len,
                slice_vs.iter().sum::<f64>() / len,
            )
        } else {
            (timestamps[n - 1] as f64, values[n - 1])
        };

        // Select the point in the current bucket that maximises triangle area
        let ax = timestamps[a] as f64;
        let ay = values[a];
        let mut max_area = -1.0f64;
        let mut max_idx = start;

        for j in start..end {
            let bx = timestamps[j] as f64;
            let by = values[j];
            // Area of triangle (a, b, avg_next)
            let area = ((ax - avg_ts) * (by - ay) - (ax - bx) * (avg_v - ay)).abs() * 0.5;
            if area > max_area {
                max_area = area;
                max_idx = j;
            }
        }

        out_ts.push(timestamps[max_idx]);
        out_vs.push(values[max_idx]);
        a = max_idx;
    }

    // Always keep last point
    out_ts.push(*timestamps.last().expect("non-empty slice"));
    out_vs.push(*values.last().expect("non-empty slice"));

    (out_ts, out_vs)
}

/// Min-max downsampling: within each bucket retain the min and max value.
fn min_max_downsample(
    timestamps: &[i64],
    values: &[f64],
    target_count: usize,
) -> (Vec<i64>, Vec<f64>) {
    let n = timestamps.len();
    // Number of buckets (each contributes 2 points: min and max)
    let buckets = target_count / 2;
    if buckets == 0 {
        return (vec![timestamps[0]], vec![values[0]]);
    }

    let bucket_size = n as f64 / buckets as f64;
    let mut out_ts = Vec::with_capacity(target_count);
    let mut out_vs = Vec::with_capacity(target_count);

    for b in 0..buckets {
        let start = (b as f64 * bucket_size) as usize;
        let end = ((b + 1) as f64 * bucket_size) as usize;
        let end = end.min(n);
        if start >= end {
            continue;
        }

        let slice_vs = &values[start..end];
        let slice_ts = &timestamps[start..end];

        let mut min_val = slice_vs[0];
        let mut min_ts = slice_ts[0];
        let mut min_idx = 0usize;
        let mut max_val = slice_vs[0];
        let mut max_ts = slice_ts[0];
        let mut max_idx = 0usize;

        for (i, (&v, &ts)) in slice_vs.iter().zip(slice_ts).enumerate() {
            if v < min_val {
                min_val = v;
                min_ts = ts;
                min_idx = i;
            }
            if v > max_val {
                max_val = v;
                max_ts = ts;
                max_idx = i;
            }
        }

        // Emit in chronological order
        if min_idx <= max_idx {
            out_ts.push(min_ts);
            out_vs.push(min_val);
            if min_idx != max_idx {
                out_ts.push(max_ts);
                out_vs.push(max_val);
            }
        } else {
            out_ts.push(max_ts);
            out_vs.push(max_val);
            out_ts.push(min_ts);
            out_vs.push(min_val);
        }
    }

    (out_ts, out_vs)
}

/// Write a downsampled time series to a CSV writer.
///
/// Output format: `timestamp_ms,value\n`
pub fn downsampling_writer<W: Write>(
    timestamps: &[i64],
    values: &[f64],
    target_count: usize,
    method: DownsamplingMethod,
    mut writer: W,
) -> Result<()> {
    let (ts_out, vs_out) = downsample(timestamps, values, target_count, method)?;
    for (ts, v) in ts_out.iter().zip(vs_out.iter()) {
        writeln!(writer, "{},{}", ts, v).map_err(IoError::Io)?;
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Gap detection
// ──────────────────────────────────────────────────────────────────────────────

/// A gap detected in a time series.
#[derive(Debug, Clone, PartialEq)]
pub struct Gap {
    /// Timestamp of the last point before the gap (ms).
    pub start_ms: i64,
    /// Timestamp of the first point after the gap (ms).
    pub end_ms: i64,
    /// Duration of the gap (ms).
    pub duration_ms: i64,
    /// Index of the point just before the gap.
    pub start_index: usize,
}

/// Configuration for gap detection.
#[derive(Debug, Clone)]
pub struct GapDetectionConfig {
    /// Expected interval between consecutive points (ms).
    pub expected_interval_ms: i64,
    /// A gap is reported when the actual interval exceeds this multiplier × expected.
    pub gap_multiplier: f64,
}

impl Default for GapDetectionConfig {
    fn default() -> Self {
        GapDetectionConfig {
            expected_interval_ms: 1000,
            gap_multiplier: 2.0,
        }
    }
}

/// Detect gaps in a time series.
///
/// A gap is defined as a consecutive-point interval that exceeds
/// `config.expected_interval_ms * config.gap_multiplier`.
pub fn gap_detection(timestamps: &[i64], config: &GapDetectionConfig) -> Vec<Gap> {
    if timestamps.len() < 2 {
        return Vec::new();
    }

    let threshold = (config.expected_interval_ms as f64 * config.gap_multiplier) as i64;
    let mut gaps = Vec::new();

    for i in 0..timestamps.len() - 1 {
        let delta = timestamps[i + 1] - timestamps[i];
        if delta > threshold {
            gaps.push(Gap {
                start_ms: timestamps[i],
                end_ms: timestamps[i + 1],
                duration_ms: delta,
                start_index: i,
            });
        }
    }

    gaps
}

// ──────────────────────────────────────────────────────────────────────────────
// Monotonicity check
// ──────────────────────────────────────────────────────────────────────────────

/// Result of a monotonicity check.
#[derive(Debug, Clone)]
pub struct MonotonicityReport {
    /// True if all timestamps are strictly increasing.
    pub is_strictly_monotonic: bool,
    /// True if all timestamps are non-decreasing (allows duplicates).
    pub is_non_decreasing: bool,
    /// Indices of out-of-order timestamps.
    pub violations: Vec<MonotonicityViolation>,
    /// Number of duplicate timestamps.
    pub duplicate_count: usize,
}

/// A single monotonicity violation.
#[derive(Debug, Clone)]
pub struct MonotonicityViolation {
    /// Index of the violating point.
    pub index: usize,
    /// The out-of-order timestamp.
    pub timestamp_ms: i64,
    /// The timestamp immediately preceding it.
    pub previous_ms: i64,
}

/// Validate the monotonicity of a timestamp sequence.
///
/// Returns a `MonotonicityReport` describing any ordering violations.
pub fn monotonic_check(timestamps: &[i64]) -> MonotonicityReport {
    let mut violations = Vec::new();
    let mut duplicate_count = 0usize;

    for i in 1..timestamps.len() {
        let prev = timestamps[i - 1];
        let curr = timestamps[i];
        if curr < prev {
            violations.push(MonotonicityViolation {
                index: i,
                timestamp_ms: curr,
                previous_ms: prev,
            });
        } else if curr == prev {
            duplicate_count += 1;
        }
    }

    let is_strictly_monotonic = violations.is_empty() && duplicate_count == 0;
    let is_non_decreasing = violations.is_empty();

    MonotonicityReport {
        is_strictly_monotonic,
        is_non_decreasing,
        violations,
        duplicate_count,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_ingester_keep_last() {
        let mut ing =
            TimeSeriesIngester::new(DeduplicationPolicy::KeepLast, FlushPolicy::default());
        let p1 = DataPoint::single(1000, "v", 1.0);
        let p2 = DataPoint::single(1000, "v", 2.0); // duplicate ts
        ing.ingest(p1).expect("ingest p1");
        ing.ingest(p2).expect("ingest p2");
        let pts = ing.flush();
        assert_eq!(pts.len(), 1);
        assert_eq!(pts[0].values["v"], 2.0);
    }

    #[test]
    fn test_ingester_keep_first() {
        let mut ing =
            TimeSeriesIngester::new(DeduplicationPolicy::KeepFirst, FlushPolicy::default());
        ing.ingest(DataPoint::single(1, "x", 10.0)).expect("ok");
        ing.ingest(DataPoint::single(1, "x", 20.0)).expect("ok");
        let pts = ing.flush();
        assert_eq!(pts[0].values["x"], 10.0);
    }

    #[test]
    fn test_ingester_reject_duplicate() {
        let mut ing = TimeSeriesIngester::new(DeduplicationPolicy::Reject, FlushPolicy::default());
        ing.ingest(DataPoint::single(1, "x", 1.0)).expect("ok");
        let err = ing.ingest(DataPoint::single(1, "x", 2.0));
        assert!(err.is_err());
    }

    #[test]
    fn test_ingester_average() {
        let mut ing = TimeSeriesIngester::new(DeduplicationPolicy::Average, FlushPolicy::default());
        ing.ingest(DataPoint::single(1000, "v", 10.0)).expect("ok");
        ing.ingest(DataPoint::single(1000, "v", 20.0)).expect("ok");
        let pts = ing.flush();
        assert!((pts[0].values["v"] - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_ingest_csv_stream() {
        // Use 13-digit timestamps so parse_timestamp treats them as milliseconds directly
        let csv = "timestamp,value\n1000000000001,1.5\n1000000000002,2.5\n1000000000003,3.5\n";
        let config = CsvIngestionConfig {
            timestamp_column: "timestamp".into(),
            ..Default::default()
        };
        let result = ingest_csv_reader(Cursor::new(csv.as_bytes()), &config).expect("ingest CSV");
        assert_eq!(result.rows_ingested, 3);
        assert_eq!(result.points.len(), 3);
        assert_eq!(result.points[0].timestamp_ms, 1000000000001);
    }

    #[test]
    fn test_gap_detection() {
        let timestamps = vec![1000i64, 2000, 3000, 10_000, 11_000];
        let config = GapDetectionConfig {
            expected_interval_ms: 1000,
            gap_multiplier: 2.0,
        };
        let gaps = gap_detection(&timestamps, &config);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].start_ms, 3000);
        assert_eq!(gaps[0].end_ms, 10_000);
    }

    #[test]
    fn test_monotonic_check_clean() {
        let ts = vec![100i64, 200, 300, 400];
        let report = monotonic_check(&ts);
        assert!(report.is_strictly_monotonic);
        assert_eq!(report.duplicate_count, 0);
        assert!(report.violations.is_empty());
    }

    #[test]
    fn test_monotonic_check_violation() {
        let ts = vec![100i64, 200, 150, 400];
        let report = monotonic_check(&ts);
        assert!(!report.is_strictly_monotonic);
        assert!(!report.is_non_decreasing);
        assert_eq!(report.violations.len(), 1);
        assert_eq!(report.violations[0].index, 2);
    }

    #[test]
    fn test_monotonic_check_duplicates() {
        let ts = vec![100i64, 200, 200, 300];
        let report = monotonic_check(&ts);
        assert!(!report.is_strictly_monotonic);
        assert!(report.is_non_decreasing);
        assert_eq!(report.duplicate_count, 1);
    }

    #[test]
    fn test_lttb_downsample() {
        let ts: Vec<i64> = (0..100).map(|i| i * 1000).collect();
        let vs: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let (ds_ts, ds_vs) =
            downsample(&ts, &vs, 20, DownsamplingMethod::Lttb).expect("downsample");
        assert_eq!(ds_ts.len(), 20);
        assert_eq!(ds_vs.len(), 20);
        // First and last must be preserved
        assert_eq!(ds_ts[0], ts[0]);
        assert_eq!(*ds_ts.last().expect("last"), *ts.last().expect("last"));
    }

    #[test]
    fn test_min_max_downsample() {
        let ts: Vec<i64> = (0..50).map(|i| i * 100).collect();
        let vs: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let (ds_ts, ds_vs) =
            downsample(&ts, &vs, 10, DownsamplingMethod::MinMax).expect("downsample");
        assert!(ds_ts.len() <= 10);
        assert_eq!(ds_ts.len(), ds_vs.len());
    }

    #[test]
    fn test_downsampling_writer() {
        let ts: Vec<i64> = (0..50).map(|i| i * 1000).collect();
        let vs: Vec<f64> = ts.iter().map(|&t| t as f64 / 1000.0).collect();
        let mut buf = Vec::new();
        downsampling_writer(&ts, &vs, 10, DownsamplingMethod::Uniform, &mut buf)
            .expect("write downsampled");
        let s = String::from_utf8(buf).expect("utf8");
        let line_count = s.lines().count();
        assert_eq!(line_count, 10);
    }
}
