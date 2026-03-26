//! Fuzz target: Matrix Market sparse/dense format parsing.
//!
//! Exercises `read_sparse_matrix`, `read_dense_matrix`, and
//! `MMHeader::parse_header` on arbitrary byte inputs written to temp files.
//!
//! Invariants:
//! - No function may panic on any input, including truncated or binary data.
//! - All errors must surface as `Result::Err`.

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Write;

fuzz_target!(|data: &[u8]| {
    // ── In-memory string parsing ──────────────────────────────────────────────
    if let Ok(text) = std::str::from_utf8(data) {
        // Exercise the header-line parser on every line of the fuzz input.
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("%%") {
                let _ = scirs2_io::matrix_market::MMHeader::parse_header(trimmed);
            }
        }
    }

    // ── File-based readers ────────────────────────────────────────────────────
    let tmp = match write_temp(data) {
        Some(p) => p,
        None => return,
    };
    let path = tmp.path();

    // Sparse COO reader — the primary Matrix Market path.
    let sparse_result = scirs2_io::matrix_market::read_sparse_matrix(path);
    if let Ok(matrix) = sparse_result {
        // Decompose to COO arrays — must not panic.
        let (rows, cols, vals) = scirs2_io::matrix_market::sparse_to_coo(&matrix);
        let _ = (rows.len(), cols.len(), vals.len());
    }

    // Dense array reader — separate parsing code path.
    let _ = scirs2_io::matrix_market::read_dense_matrix(path);
});

fn write_temp(data: &[u8]) -> Option<tempfile::NamedTempFile> {
    let mut tmp = tempfile::NamedTempFile::new().ok()?;
    tmp.write_all(data).ok()?;
    tmp.flush().ok()?;
    Some(tmp)
}
