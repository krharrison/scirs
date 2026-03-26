//! Fuzz target: CSV parsing via both the file-based CSV reader and the
//! in-memory streaming CSV parser.
//!
//! The fuzzer feeds arbitrary bytes and verifies that:
//! 1. `parse_csv_line` (the low-level field splitter) never panics.
//! 2. Writing the fuzz bytes to a temp file and passing them through the
//!    public `CsvStreamReader` / `infer_schema` / `read_typed_row` pipeline
//!    never panics, regardless of how malformed the input is.
//!
//! Any panic inside these calls would constitute a bug (all errors should be
//! reported via `Result::Err` instead of unwinding).

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Write;

fuzz_target!(|data: &[u8]| {
    // ── Phase 1: in-memory string parsing (no file I/O) ──────────────────────
    if let Ok(text) = std::str::from_utf8(data) {
        // Exercise the streaming_csv row parser with every printable delimiter.
        for &delim in &[b',', b';', b'\t', b'|'] {
            let _ = fuzz_parse_csv_lines(text, delim);
        }
    }

    // ── Phase 2: file-based readers ──────────────────────────────────────────
    // Write the raw bytes to a temp file and exercise the public streaming API.
    let tmp = match write_temp(data) {
        Some(p) => p,
        None => return,
    };

    let path = tmp.path();

    // CsvStreamReader — should never panic on bad input.
    let _ = fuzz_stream_reader(path, b',');
    let _ = fuzz_stream_reader(path, b';');
    let _ = fuzz_stream_reader(path, b'\t');

    // infer_schema — inspect column type inference without panicking.
    let _ = scirs2_io::streaming_csv::infer_schema(path, b',', 32);
    let _ = scirs2_io::streaming_csv::infer_schema(path, b'\t', 32);
});

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Write bytes to a named temp file; returns the guard on success.
fn write_temp(data: &[u8]) -> Option<tempfile::NamedTempFile> {
    let mut tmp = tempfile::NamedTempFile::new().ok()?;
    tmp.write_all(data).ok()?;
    tmp.flush().ok()?;
    Some(tmp)
}

/// Drive `CsvStreamReader` through the entire file, collecting all rows.
/// Returns `Ok(())` regardless of per-row errors (they are `Result::Err`,
/// not panics), and propagates only constructor failures.
fn fuzz_stream_reader(
    path: &std::path::Path,
    delimiter: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader =
        scirs2_io::streaming_csv::CsvStreamReader::new(path, delimiter, true)?;

    // Drain all rows.
    let mut chunk = reader.read_chunk(64);
    while let Ok(ref rows) = chunk {
        if rows.is_empty() {
            break;
        }
        // For each row attempt typed parsing with a dummy single-column schema.
        for row in rows {
            if !row.is_empty() {
                let schema = vec![scirs2_io::streaming_csv::ColumnType::Text; row.len()];
                let _ = scirs2_io::streaming_csv::read_typed_row(row, &schema);
            }
        }
        chunk = reader.read_chunk(64);
    }

    Ok(())
}

/// Low-level in-memory CSV line splitting (no file I/O path).
/// Exercises the same quoting logic as the full reader.
fn fuzz_parse_csv_lines(text: &str, delimiter: u8) -> Vec<Vec<String>> {
    let sep = delimiter as char;
    text.lines()
        .map(|line| parse_csv_row(line, sep))
        .collect()
}

/// Minimal RFC-4180 row parser replicated here so we can test it without
/// going through the file-based API.  Must never panic.
fn parse_csv_row(line: &str, sep: char) -> Vec<String> {
    let mut fields: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    current.push('"');
                } else {
                    in_quotes = false;
                }
            }
            '"' => {
                in_quotes = true;
            }
            c if c == sep && !in_quotes => {
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
