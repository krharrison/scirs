//! Chunked CSV streaming with single-pass statistics.
//!
//! This module provides a low-memory CSV reader that delivers data in
//! configurable chunks rather than loading the entire file at once.
//!
//! ## Example
//!
//! ```no_run
//! use scirs2_datasets::streaming_csv::{CsvStreamConfig, CsvStreamReader, streaming_statistics};
//!
//! let config = CsvStreamConfig::default();
//! let mut reader = CsvStreamReader::open("data.csv", config).expect("open");
//! while let Ok(Some(chunk)) = reader.next_chunk() {
//!     println!("chunk {}: {} rows", chunk.chunk_id, chunk.rows.len());
//!     if chunk.is_last { break; }
//! }
//! ```

use crate::error::{DatasetsError, Result};
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for chunked CSV streaming.
#[derive(Debug, Clone)]
pub struct CsvStreamConfig {
    /// Number of data rows per chunk (header not counted).
    pub chunk_size: usize,
    /// Whether the first non-empty line is a header.
    pub has_header: bool,
    /// Field delimiter byte (default `b','`).
    pub delimiter: u8,
    /// Number of parallel worker threads (reserved for future use; currently unused).
    pub n_workers: usize,
}

impl Default for CsvStreamConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            has_header: true,
            delimiter: b',',
            n_workers: 1,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Output types
// ─────────────────────────────────────────────────────────────────────────────

/// A chunk of rows read from a CSV file.
#[derive(Debug, Clone)]
pub struct CsvChunk {
    /// Column headers (empty if `has_header` is `false`).
    pub headers: Vec<String>,
    /// Data rows; each row is a `Vec<String>` of field values.
    pub rows: Vec<Vec<String>>,
    /// Zero-based index of this chunk.
    pub chunk_id: usize,
    /// `true` if no more data follows this chunk.
    pub is_last: bool,
}

/// Statistics computed via a single streaming pass over a CSV column.
#[derive(Debug, Clone)]
pub struct CsvStreamStats {
    /// Arithmetic mean (Welford online algorithm).
    pub mean: f64,
    /// Sample variance (Welford online algorithm).
    pub variance: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Total number of successfully parsed rows.
    pub n_rows: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// CsvStreamReader
// ─────────────────────────────────────────────────────────────────────────────

/// A streaming CSV reader that yields data in fixed-size chunks.
pub struct CsvStreamReader {
    /// Path to the CSV file.
    pub path: String,
    /// Configuration used by this reader.
    pub config: CsvStreamConfig,
    /// Byte offset of the next chunk start (after the header).
    pub position: u64,
    /// Identifier of the next chunk to be returned.
    pub chunk_id: usize,
    /// Column headers (populated on open when `has_header` is `true`).
    headers: Vec<String>,
    /// Byte offset immediately after the header line.
    data_start: u64,
    /// Number of columns (derived from header or first row).
    n_columns: Option<usize>,
    /// Set to `true` once we reach EOF.
    exhausted: bool,
}

impl CsvStreamReader {
    /// Open a CSV file for streaming.
    pub fn open(path: &str, config: CsvStreamConfig) -> Result<Self> {
        let file = File::open(path).map_err(DatasetsError::IoError)?;
        let mut reader = BufReader::new(file);

        let mut headers = Vec::new();
        let mut data_start = 0u64;

        if config.has_header {
            let mut header_line = String::new();
            let bytes_read = reader
                .read_line(&mut header_line)
                .map_err(DatasetsError::IoError)?;
            if bytes_read == 0 {
                return Err(DatasetsError::InvalidFormat(
                    "CSV file is empty — cannot read header".into(),
                ));
            }
            let line = header_line.trim_end_matches(['\n', '\r']);
            headers = split_csv_line(line, config.delimiter);
            data_start = bytes_read as u64;
        }

        let n_columns = if headers.is_empty() {
            None
        } else {
            Some(headers.len())
        };

        Ok(Self {
            path: path.to_owned(),
            config,
            position: data_start,
            chunk_id: 0,
            headers,
            data_start,
            n_columns,
            exhausted: false,
        })
    }

    /// Number of columns (known only after the header has been read or the
    /// first row has been parsed).
    pub fn n_columns(&self) -> Option<usize> {
        self.n_columns
    }

    /// Read the next chunk of rows.
    ///
    /// Returns `Ok(None)` when the file is exhausted.
    pub fn next_chunk(&mut self) -> Result<Option<CsvChunk>> {
        if self.exhausted {
            return Ok(None);
        }

        let file = File::open(&self.path).map_err(DatasetsError::IoError)?;
        let mut reader = BufReader::new(file);
        reader
            .seek(SeekFrom::Start(self.position))
            .map_err(DatasetsError::IoError)?;

        let mut rows: Vec<Vec<String>> = Vec::with_capacity(self.config.chunk_size);
        let mut bytes_consumed = 0u64;

        for _ in 0..self.config.chunk_size {
            let mut line = String::new();
            let n = reader
                .read_line(&mut line)
                .map_err(DatasetsError::IoError)?;
            if n == 0 {
                self.exhausted = true;
                break;
            }
            bytes_consumed += n as u64;
            let trimmed = line.trim_end_matches(['\n', '\r']);
            if trimmed.is_empty() {
                // Skip blank lines.
                continue;
            }
            let fields = split_csv_line(trimmed, self.config.delimiter);

            // Infer column count from the first row we see.
            if self.n_columns.is_none() {
                self.n_columns = Some(fields.len());
            }
            rows.push(fields);
        }

        if rows.is_empty() {
            return Ok(None);
        }

        self.position += bytes_consumed;
        let chunk_id = self.chunk_id;
        self.chunk_id += 1;

        // We need to peek ahead to know whether this is the last chunk.
        let is_last = self.exhausted || {
            let mut peek_file = File::open(&self.path).map_err(DatasetsError::IoError)?;
            let mut peek_reader = BufReader::new(&mut peek_file);
            peek_reader
                .seek(SeekFrom::Start(self.position))
                .map_err(DatasetsError::IoError)?;
            let mut tmp = String::new();
            let peek_n = peek_reader
                .read_line(&mut tmp)
                .map_err(DatasetsError::IoError)?;
            peek_n == 0
        };
        self.exhausted = is_last;

        Ok(Some(CsvChunk {
            headers: self.headers.clone(),
            rows,
            chunk_id,
            is_last,
        }))
    }

    /// Reset the reader back to the beginning of the data (after the header).
    pub fn reset(&mut self) -> Result<()> {
        self.position = self.data_start;
        self.chunk_id = 0;
        self.exhausted = false;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Split a single CSV line respecting double-quoted fields.
fn split_csv_line(line: &str, delimiter: u8) -> Vec<String> {
    let delim = delimiter as char;
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;

    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                // Handle escaped double-quote ("")
                if in_quotes && chars.peek() == Some(&'"') {
                    chars.next();
                    current.push('"');
                } else {
                    in_quotes = !in_quotes;
                }
            }
            c if c == delim && !in_quotes => {
                fields.push(current.clone());
                current.clear();
            }
            other => {
                current.push(other);
            }
        }
    }
    fields.push(current);
    fields
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience functions
// ─────────────────────────────────────────────────────────────────────────────

/// Stream a CSV file and extract the specified columns as `f64`.
///
/// Rows where any requested column cannot be parsed as `f64` are silently
/// skipped.  Column indices are 0-based.
pub fn stream_csv_as_f64(
    path: &str,
    config: &CsvStreamConfig,
    column_indices: &[usize],
) -> Result<Vec<Vec<f64>>> {
    let mut reader = CsvStreamReader::open(path, config.clone())?;
    let mut result: Vec<Vec<f64>> = Vec::new();

    loop {
        match reader.next_chunk()? {
            None => break,
            Some(chunk) => {
                for row in &chunk.rows {
                    let mut vals = Vec::with_capacity(column_indices.len());
                    let mut ok = true;
                    for &col_idx in column_indices {
                        match row.get(col_idx).and_then(|s| s.trim().parse::<f64>().ok()) {
                            Some(v) => vals.push(v),
                            None => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if ok {
                        result.push(vals);
                    }
                }
                if chunk.is_last {
                    break;
                }
            }
        }
    }

    Ok(result)
}

/// Compute streaming statistics for a single column using the Welford algorithm.
///
/// Rows where the specified column cannot be parsed as `f64` are skipped.
pub fn streaming_statistics(
    path: &str,
    config: &CsvStreamConfig,
    column: usize,
) -> Result<CsvStreamStats> {
    let mut reader = CsvStreamReader::open(path, config.clone())?;

    let mut n: usize = 0;
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;

    loop {
        match reader.next_chunk()? {
            None => break,
            Some(chunk) => {
                for row in &chunk.rows {
                    if let Some(val_str) = row.get(column) {
                        if let Ok(x) = val_str.trim().parse::<f64>() {
                            n += 1;
                            let delta = x - mean;
                            mean += delta / n as f64;
                            let delta2 = x - mean;
                            m2 += delta * delta2;
                            if x < min_val {
                                min_val = x;
                            }
                            if x > max_val {
                                max_val = x;
                            }
                        }
                    }
                }
                if chunk.is_last {
                    break;
                }
            }
        }
    }

    if n == 0 {
        return Err(DatasetsError::InvalidFormat(
            "No parseable values found in the specified column".into(),
        ));
    }

    let variance = if n > 1 { m2 / (n - 1) as f64 } else { 0.0 };

    Ok(CsvStreamStats {
        mean,
        variance,
        min: min_val,
        max: max_val,
        n_rows: n,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // Helper: create a temp CSV and get its path.  Returns (content, path).
    fn make_temp_csv(content: &str) -> String {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "scirs2_csv_test_{}.csv",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let mut f = File::create(&path).expect("create");
        f.write_all(content.as_bytes()).expect("write");
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn test_open_and_read_header() {
        let csv = "a,b,c\n1,2,3\n4,5,6\n";
        let path = make_temp_csv(csv);
        let config = CsvStreamConfig {
            chunk_size: 10,
            ..Default::default()
        };
        let reader = CsvStreamReader::open(&path, config).expect("open");
        assert_eq!(reader.headers, vec!["a", "b", "c"]);
        assert_eq!(reader.n_columns(), Some(3));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_next_chunk_returns_rows() {
        let csv = "x,y\n1.0,2.0\n3.0,4.0\n5.0,6.0\n";
        let path = make_temp_csv(csv);
        let config = CsvStreamConfig {
            chunk_size: 10,
            ..Default::default()
        };
        let mut reader = CsvStreamReader::open(&path, config).expect("open");
        let chunk = reader.next_chunk().expect("read").expect("some");
        assert_eq!(chunk.rows.len(), 3);
        assert_eq!(chunk.chunk_id, 0);
        assert!(chunk.is_last);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_chunked_reading() {
        // 10 rows, chunk_size = 4 → chunks of 4, 4, 2
        let mut csv = "value\n".to_owned();
        for i in 0..10u32 {
            csv.push_str(&format!("{i}\n"));
        }
        let path = make_temp_csv(&csv);
        let config = CsvStreamConfig {
            chunk_size: 4,
            ..Default::default()
        };
        let mut reader = CsvStreamReader::open(&path, config).expect("open");

        let mut total_rows = 0;
        let mut n_chunks = 0;
        loop {
            match reader.next_chunk().expect("read") {
                None => break,
                Some(chunk) => {
                    total_rows += chunk.rows.len();
                    n_chunks += 1;
                    if chunk.is_last {
                        break;
                    }
                }
            }
        }
        assert_eq!(total_rows, 10);
        assert_eq!(n_chunks, 3);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_reset() {
        let csv = "val\n1\n2\n3\n";
        let path = make_temp_csv(csv);
        let config = CsvStreamConfig {
            chunk_size: 10,
            ..Default::default()
        };
        let mut reader = CsvStreamReader::open(&path, config).expect("open");
        let _first = reader.next_chunk().expect("read").expect("some");
        reader.reset().expect("reset");
        let second = reader.next_chunk().expect("read").expect("some");
        assert_eq!(second.rows.len(), 3);
        assert_eq!(second.chunk_id, 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_stream_csv_as_f64() {
        let csv = "a,b,c\n1.0,2.0,3.0\n4.0,5.0,6.0\n";
        let path = make_temp_csv(csv);
        let config = CsvStreamConfig::default();
        let data = stream_csv_as_f64(&path, &config, &[0, 2]).expect("ok");
        assert_eq!(data.len(), 2);
        assert!((data[0][0] - 1.0).abs() < 1e-10);
        assert!((data[0][1] - 3.0).abs() < 1e-10);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_streaming_statistics_mean() {
        // 5 values: 1..=5, mean = 3.0
        let csv = "value\n1\n2\n3\n4\n5\n";
        let path = make_temp_csv(csv);
        let config = CsvStreamConfig::default();
        let stats = streaming_statistics(&path, &config, 0).expect("stats");
        assert!((stats.mean - 3.0).abs() < 1e-10, "mean={}", stats.mean);
        assert_eq!(stats.n_rows, 5);
        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.max - 5.0).abs() < 1e-10);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_split_csv_line_basic() {
        let fields = split_csv_line("a,b,c", b',');
        assert_eq!(fields, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_split_csv_line_quoted() {
        let fields = split_csv_line("\"hello, world\",42", b',');
        assert_eq!(fields, vec!["hello, world", "42"]);
    }

    #[test]
    fn test_split_csv_line_escaped_quote() {
        let fields = split_csv_line("\"say \"\"hi\"\"\",val", b',');
        assert_eq!(fields, vec!["say \"hi\"", "val"]);
    }
}
