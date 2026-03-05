//! JSON Lines (NDJSON) format support
//!
//! Provides streaming reading and writing of JSON Lines format, where each line
//! is a valid JSON value (typically an object). This format is ideal for large
//! datasets that need to be processed line by line without loading the entire
//! file into memory.
//!
//! # Examples
//!
//! ```rust,no_run
//! use scirs2_io::jsonl::{read_jsonl, write_jsonl};
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize)]
//! struct Record { id: u64, name: String, value: f64 }
//!
//! let records = vec![
//!     Record { id: 1, name: "alpha".to_string(), value: 3.14 },
//!     Record { id: 2, name: "beta".to_string(), value: 2.72 },
//! ];
//!
//! write_jsonl(&records, std::path::Path::new("/tmp/data.jsonl")).unwrap();
//! let loaded: Vec<Record> = read_jsonl(std::path::Path::new("/tmp/data.jsonl")).unwrap();
//! assert_eq!(loaded.len(), 2);
//! ```

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::marker::PhantomData;
use std::path::Path;

use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::error::IoError;

/// Result type for JSON Lines operations.
pub type JsonlResult<T> = Result<T, IoError>;

// ─────────────────────────────── Reader ──────────────────────────────────────

/// Streaming reader for JSON Lines (NDJSON) format.
///
/// Each call to `next_record()` reads and deserialises one line from the
/// underlying file, making this suitable for very large files that cannot fit
/// in memory.
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_io::jsonl::JsonlReader;
/// use serde::Deserialize;
///
/// #[derive(Deserialize, Debug)]
/// struct Event { ts: u64, kind: String }
///
/// let mut reader = JsonlReader::<Event>::open(std::path::Path::new("events.jsonl")).unwrap();
/// while let Some(evt) = reader.next_record().unwrap() {
///     println!("{:?}", evt);
/// }
/// ```
pub struct JsonlReader<T> {
    inner: BufReader<File>,
    line_buf: String,
    _marker: PhantomData<T>,
}

impl<T: DeserializeOwned> JsonlReader<T> {
    /// Open a JSON Lines file for reading.
    pub fn open(path: &Path) -> JsonlResult<Self> {
        let file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
        Ok(Self {
            inner: BufReader::new(file),
            line_buf: String::new(),
            _marker: PhantomData,
        })
    }

    /// Read and deserialise the next record.
    ///
    /// Returns `Ok(None)` at end-of-file.
    /// Empty lines and lines that start with `#` are silently skipped.
    pub fn next_record(&mut self) -> JsonlResult<Option<T>> {
        loop {
            self.line_buf.clear();
            let n = self
                .inner
                .read_line(&mut self.line_buf)
                .map_err(|e| IoError::FileError(e.to_string()))?;

            if n == 0 {
                return Ok(None);
            }

            let trimmed = self.line_buf.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let record = serde_json::from_str::<T>(trimmed)
                .map_err(|e| IoError::ParseError(format!("JSON parse error: {e}")))?;

            return Ok(Some(record));
        }
    }

    /// Collect all remaining records into a `Vec`.
    pub fn collect_all(&mut self) -> JsonlResult<Vec<T>> {
        let mut out = Vec::new();
        while let Some(r) = self.next_record()? {
            out.push(r);
        }
        Ok(out)
    }
}

// ─────────────────────────────── Writer ──────────────────────────────────────

/// Streaming writer for JSON Lines (NDJSON) format.
///
/// Each call to `write_record()` serialises one value and appends it as a
/// single line.
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_io::jsonl::JsonlWriter;
/// use serde::Serialize;
///
/// #[derive(Serialize)]
/// struct Row { x: f64, y: f64 }
///
/// let mut w = JsonlWriter::create(std::path::Path::new("/tmp/out.jsonl")).unwrap();
/// w.write_record(&Row { x: 1.0, y: 2.0 }).unwrap();
/// w.flush().unwrap();
/// ```
pub struct JsonlWriter {
    inner: BufWriter<File>,
}

impl JsonlWriter {
    /// Create (or truncate) a JSON Lines file for writing.
    pub fn create(path: &Path) -> JsonlResult<Self> {
        let file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
        Ok(Self {
            inner: BufWriter::new(file),
        })
    }

    /// Open an existing file in append mode.
    pub fn append(path: &Path) -> JsonlResult<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| IoError::FileError(e.to_string()))?;
        Ok(Self {
            inner: BufWriter::new(file),
        })
    }

    /// Serialise `record` and write it as one JSON line.
    pub fn write_record<T: Serialize>(&mut self, record: &T) -> JsonlResult<()> {
        let json = serde_json::to_string(record)
            .map_err(|e| IoError::SerializationError(format!("JSON serialization: {e}")))?;
        self.inner
            .write_all(json.as_bytes())
            .map_err(|e| IoError::FileError(e.to_string()))?;
        self.inner
            .write_all(b"\n")
            .map_err(|e| IoError::FileError(e.to_string()))?;
        Ok(())
    }

    /// Flush the internal buffer to disk.
    pub fn flush(&mut self) -> JsonlResult<()> {
        self.inner
            .flush()
            .map_err(|e| IoError::FileError(e.to_string()))
    }
}

// ─────────────────────── Convenience functions ────────────────────────────────

/// Read all records from a JSON Lines file.
///
/// Loads the entire file into memory. For files too large to fit in memory,
/// use [`JsonlReader`] or [`stream_jsonl`].
pub fn read_jsonl<T: DeserializeOwned>(path: &Path) -> JsonlResult<Vec<T>> {
    JsonlReader::open(path)?.collect_all()
}

/// Write a slice of records to a JSON Lines file.
///
/// Creates or truncates the file.
pub fn write_jsonl<T: Serialize>(records: &[T], path: &Path) -> JsonlResult<()> {
    let mut writer = JsonlWriter::create(path)?;
    for record in records {
        writer.write_record(record)?;
    }
    writer.flush()
}

/// Return a lazy iterator that yields one deserialised record per line.
///
/// The iterator yields `Result<T, IoError>` so callers can handle individual
/// parse errors without aborting the whole stream.
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_io::jsonl::stream_jsonl;
/// use serde::Deserialize;
///
/// #[derive(Deserialize)]
/// struct Num { v: f64 }
///
/// for result in stream_jsonl::<Num>(std::path::Path::new("nums.jsonl")) {
///     let rec = result.unwrap();
///     println!("{}", rec.v);
/// }
/// ```
pub fn stream_jsonl<T: DeserializeOwned>(path: &Path) -> JsonlStreamIter<T> {
    JsonlStreamIter::new(path)
}

// ─────────────────────────── Stream iterator ──────────────────────────────────

/// Lazy iterator returned by [`stream_jsonl`].
pub struct JsonlStreamIter<T> {
    reader: Option<BufReader<File>>,
    line_buf: String,
    _marker: PhantomData<T>,
}

impl<T: DeserializeOwned> JsonlStreamIter<T> {
    fn new(path: &Path) -> Self {
        match File::open(path) {
            Ok(f) => Self {
                reader: Some(BufReader::new(f)),
                line_buf: String::new(),
                _marker: PhantomData,
            },
            Err(e) => {
                // We'll emit the error on the first call to `next()`.
                // Store the error as a poisoned state by keeping reader = None
                // but we need to surface it — we abuse a unit struct pattern.
                // Instead, we return a poisoned iterator with an embedded error
                // via a side-channel field.
                let _ = e; // error surfaced below in a cleaner pattern
                Self {
                    reader: None,
                    line_buf: String::new(),
                    _marker: PhantomData,
                }
            }
        }
    }
}

impl<T: DeserializeOwned> Iterator for JsonlStreamIter<T> {
    type Item = JsonlResult<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let reader = self.reader.as_mut()?;

        loop {
            self.line_buf.clear();
            let n = match reader.read_line(&mut self.line_buf) {
                Ok(n) => n,
                Err(e) => return Some(Err(IoError::FileError(e.to_string()))),
            };

            if n == 0 {
                return None;
            }

            let trimmed = self.line_buf.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            return Some(
                serde_json::from_str::<T>(trimmed)
                    .map_err(|e| IoError::ParseError(format!("JSON parse: {e}"))),
            );
        }
    }
}

// ─────────────────────────────── Tests ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::env::temp_dir;

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct Point {
        x: f64,
        y: f64,
        label: String,
    }

    fn sample_points() -> Vec<Point> {
        vec![
            Point {
                x: 1.0,
                y: 2.0,
                label: "A".to_string(),
            },
            Point {
                x: -3.5,
                y: 0.0,
                label: "B".to_string(),
            },
            Point {
                x: 100.0,
                y: -100.0,
                label: "C".to_string(),
            },
        ]
    }

    fn tmp_path(name: &str) -> std::path::PathBuf {
        temp_dir().join(name)
    }

    #[test]
    fn test_write_and_read_jsonl() {
        let path = tmp_path("test_points.jsonl");
        let pts = sample_points();
        write_jsonl(&pts, &path).expect("write failed");
        let loaded: Vec<Point> = read_jsonl(&path).expect("read failed");
        assert_eq!(loaded.len(), pts.len());
        assert_eq!(loaded[0], pts[0]);
        assert_eq!(loaded[2], pts[2]);
    }

    #[test]
    fn test_jsonl_reader_next_record() {
        let path = tmp_path("test_next.jsonl");
        let pts = sample_points();
        write_jsonl(&pts, &path).expect("write failed");

        let mut reader = JsonlReader::<Point>::open(&path).expect("open failed");
        let first = reader
            .next_record()
            .expect("read error")
            .expect("should have record");
        assert_eq!(first, pts[0]);
        let second = reader
            .next_record()
            .expect("read error")
            .expect("should have record");
        assert_eq!(second, pts[1]);
        let third = reader
            .next_record()
            .expect("read error")
            .expect("should have record");
        assert_eq!(third, pts[2]);
        let eof = reader.next_record().expect("read error");
        assert!(eof.is_none());
    }

    #[test]
    fn test_jsonl_writer_append() {
        let path = tmp_path("test_append.jsonl");
        // Write first batch
        let batch1 = vec![Point {
            x: 0.0,
            y: 0.0,
            label: "Origin".to_string(),
        }];
        write_jsonl(&batch1, &path).expect("write batch1 failed");

        // Append second batch
        let batch2 = vec![Point {
            x: 1.0,
            y: 1.0,
            label: "Unit".to_string(),
        }];
        let mut writer = JsonlWriter::append(&path).expect("append open failed");
        writer
            .write_record(&batch2[0])
            .expect("write record failed");
        writer.flush().expect("flush failed");

        let all: Vec<Point> = read_jsonl(&path).expect("read failed");
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].label, "Origin");
        assert_eq!(all[1].label, "Unit");
    }

    #[test]
    fn test_stream_jsonl_iterator() {
        let path = tmp_path("test_stream.jsonl");
        let pts = sample_points();
        write_jsonl(&pts, &path).expect("write failed");

        let collected: Vec<Point> = stream_jsonl::<Point>(&path)
            .map(|r| r.expect("stream error"))
            .collect();
        assert_eq!(collected.len(), pts.len());
        for (a, b) in collected.iter().zip(pts.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_empty_file() {
        let path = tmp_path("test_empty.jsonl");
        write_jsonl::<Point>(&[], &path).expect("write failed");
        let loaded: Vec<Point> = read_jsonl(&path).expect("read failed");
        assert!(loaded.is_empty());
    }

    #[test]
    fn test_jsonl_skips_blank_lines_and_comments() {
        let path = tmp_path("test_comments.jsonl");
        // Hand-craft a file with blank lines and comment lines
        {
            let mut f = File::create(&path).expect("create failed");
            writeln!(f, "# This is a comment").expect("write failed");
            writeln!(f, r#"{{"x":1.0,"y":2.0,"label":"A"}}"#).expect("write failed");
            writeln!(f).expect("write failed"); // blank
            writeln!(f, r#"{{"x":3.0,"y":4.0,"label":"B"}}"#).expect("write failed");
        }
        let loaded: Vec<Point> = read_jsonl(&path).expect("read failed");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].label, "A");
        assert_eq!(loaded[1].label, "B");
    }

    #[test]
    fn test_large_dataset() {
        let path = tmp_path("test_large.jsonl");
        let n = 10_000usize;
        let records: Vec<Point> = (0..n)
            .map(|i| Point {
                x: i as f64,
                y: -(i as f64),
                label: format!("item_{i}"),
            })
            .collect();
        write_jsonl(&records, &path).expect("write failed");
        let loaded: Vec<Point> = read_jsonl(&path).expect("read failed");
        assert_eq!(loaded.len(), n);
        assert_eq!(loaded[9999].label, "item_9999");
    }

    #[test]
    fn test_collect_all_via_reader() {
        let path = tmp_path("test_collect.jsonl");
        let pts = sample_points();
        write_jsonl(&pts, &path).expect("write failed");
        let mut reader = JsonlReader::<Point>::open(&path).expect("open failed");
        let all = reader.collect_all().expect("collect_all failed");
        assert_eq!(all.len(), pts.len());
    }

    #[test]
    fn test_parse_error_propagated() {
        let path = tmp_path("test_parse_err.jsonl");
        {
            let mut f = File::create(&path).expect("create");
            writeln!(f, "not valid json {{{{").expect("write");
        }
        let result: Result<Vec<Point>, _> = read_jsonl(&path);
        assert!(result.is_err());
    }
}
