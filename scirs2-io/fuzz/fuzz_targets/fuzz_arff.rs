//! Fuzz target: ARFF (Attribute-Relation File Format) parsing.
//!
//! Writes arbitrary bytes to a temp file and calls `read_arff`, which
//! exercises the header parser, attribute-type parser, and data-section
//! parser.  The sparse ARFF reader is also exercised via `read_sparse_arff`.
//!
//! Invariants:
//! - Neither `read_arff` nor `read_sparse_arff` may panic on any input.
//! - All errors must be reported via `Result::Err`.

#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Write;

fuzz_target!(|data: &[u8]| {
    let tmp = match write_temp(data) {
        Some(p) => p,
        None => return,
    };
    let path = tmp.path();

    // Dense ARFF reader.
    let result = scirs2_io::arff::read_arff(path);
    if let Ok(arff_data) = result {
        // Probe derived helper: collect the names of numeric attributes and pass them.
        let numeric_names: Vec<String> = arff_data
            .attributes
            .iter()
            .filter_map(|(name, attr_type)| {
                if matches!(attr_type, scirs2_io::arff::AttributeType::Numeric) {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();
        let _ = scirs2_io::arff::get_numeric_matrix(&arff_data, &numeric_names);
    }

    // Sparse ARFF reader — a separate code path.
    let _ = scirs2_io::arff::read_sparse_arff(path);
});

fn write_temp(data: &[u8]) -> Option<tempfile::NamedTempFile> {
    let mut tmp = tempfile::NamedTempFile::new().ok()?;
    tmp.write_all(data).ok()?;
    tmp.flush().ok()?;
    Some(tmp)
}
