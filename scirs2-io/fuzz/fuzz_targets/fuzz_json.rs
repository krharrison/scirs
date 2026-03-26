//! Fuzz target: JSON / JSON Lines parsing.
//!
//! Exercises:
//! 1. The pure-Rust `parse_json` function with arbitrary strings — no panics
//!    allowed; all errors must be `Result::Err`.
//! 2. `JsonLinesReader` draining a temp-file written with the fuzz bytes.
//! 3. `flatten_json` and `extract_field` on every successfully-parsed value.
//! 4. The `serde_json`-based `JsonlReader` (`scirs2_io::jsonl`) for a
//!    round-trip with a `serde_json::Value` target type.
//!
//! Any panic inside these call paths constitutes a confirmed bug.

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Write;

fuzz_target!(|data: &[u8]| {
    // ── Phase 1: pure-Rust in-memory JSON parser ─────────────────────────────
    if let Ok(text) = std::str::from_utf8(data) {
        // Single-document parse.
        if let Ok(value) = scirs2_io::streaming_json::parse_json(text) {
            // Navigate the parsed tree — must not panic.
            let flat = scirs2_io::streaming_json::flatten_json(&value, "");
            for key in flat.keys() {
                let _ = scirs2_io::streaming_json::extract_field(&value, key);
            }
            // Common field-path probes.
            for probe in &["id", "name", "value", "data", "items.0", "nested.key"] {
                let _ = scirs2_io::streaming_json::extract_field(&value, probe);
            }
        }

        // Also attempt to parse each line individually (NDJSON simulation).
        for line in text.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() && !trimmed.starts_with('#') {
                let _ = scirs2_io::streaming_json::parse_json(trimmed);
            }
        }
    }

    // ── Phase 2: file-based streaming readers ────────────────────────────────
    let tmp = match write_temp(data) {
        Some(p) => p,
        None => return,
    };
    let path = tmp.path();

    // streaming_json::JsonLinesReader — must never panic.
    {
        let _ = fuzz_json_lines_reader(path);
    }

    // jsonl::JsonlReader<serde_json::Value> — serde-based path.
    {
        let _ = fuzz_jsonl_reader(path);
    }
});

// ── Helpers ──────────────────────────────────────────────────────────────────

fn write_temp(data: &[u8]) -> Option<tempfile::NamedTempFile> {
    let mut tmp = tempfile::NamedTempFile::new().ok()?;
    tmp.write_all(data).ok()?;
    tmp.flush().ok()?;
    Some(tmp)
}

/// Drive the pure-Rust `JsonLinesReader` to exhaustion.
fn fuzz_json_lines_reader(
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = scirs2_io::streaming_json::JsonLinesReader::open(path)?;

    let mut iterations = 0usize;
    loop {
        match reader.next_record() {
            Ok(Some(value)) => {
                // Exercise helper functions on each parsed value.
                let flat = scirs2_io::streaming_json::flatten_json(&value, "root");
                let _ = flat.len();
                let _ = scirs2_io::streaming_json::extract_field(&value, "id");
                // Limit iterations to keep fuzzing fast.
                iterations += 1;
                if iterations >= 1024 {
                    break;
                }
            }
            Ok(None) => break,
            Err(_) => break,
        }
    }

    Ok(())
}

/// Drive the serde-based `jsonl::JsonlReader` to exhaustion.
fn fuzz_jsonl_reader(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader =
        scirs2_io::jsonl::JsonlReader::<serde_json::Value>::open(path)?;

    let mut iterations = 0usize;
    loop {
        match reader.next_record() {
            Ok(Some(_val)) => {
                iterations += 1;
                if iterations >= 1024 {
                    break;
                }
            }
            Ok(None) => break,
            Err(_) => break,
        }
    }

    Ok(())
}
