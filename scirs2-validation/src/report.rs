//! Report generation for validation results.
//!
//! Provides human-readable text tables and optional JSON output for
//! distribution validation results. JSON output is available either via
//! the `serialization` feature (using serde_json) or via a built-in
//! hand-rolled formatter.

use crate::validators::ValidationResult;

/// A collection of validation results with summary statistics.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Name or title of the report
    pub name: String,
    /// Individual distribution validation results
    pub results: Vec<ValidationResult>,
    /// Number of distributions that passed
    pub passed: usize,
    /// Number of distributions that failed
    pub failed: usize,
    /// Total number of distributions tested
    pub total: usize,
}

impl ValidationReport {
    /// Create a new report from a set of validation results.
    pub fn new(name: impl Into<String>, results: Vec<ValidationResult>) -> Self {
        let total = results.len();
        let passed = results.iter().filter(|r| r.passed).count();
        let failed = total - passed;
        Self {
            name: name.into(),
            results,
            passed,
            failed,
            total,
        }
    }

    /// Whether all validations passed.
    pub fn all_passed(&self) -> bool {
        self.failed == 0
    }
}

impl core::fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "=== {} ===", self.name)?;
        writeln!(
            f,
            "Total: {} | Passed: {} | Failed: {}",
            self.total, self.passed, self.failed
        )?;
        writeln!(f)?;
        for result in &self.results {
            writeln!(f, "  {}", result)?;
        }
        Ok(())
    }
}

/// Generate a human-readable text report from validation results.
///
/// The report includes a header, per-distribution status lines, and a summary
/// of any errors found.
///
/// # Arguments
///
/// * `results` - Slice of validation results to include
///
/// # Returns
///
/// A formatted string containing the full report.
pub fn generate_report(results: &[ValidationResult]) -> String {
    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = total - passed;

    let mut out = String::with_capacity(2048);

    out.push_str(
        "+-------------------------------+--------+------+------+------+-----------+-----------+\n",
    );
    out.push_str(
        "| Distribution                  | Status | PDF  | CDF  | PPF  | Mean Err  | Var Err   |\n",
    );
    out.push_str(
        "+-------------------------------+--------+------+------+------+-----------+-----------+\n",
    );

    for r in results {
        let status = if r.passed { "PASS" } else { "FAIL" };
        let pdf_count = r.pdf_errors.len();
        let cdf_count = r.cdf_errors.len();
        let ppf_count = r.ppf_errors.len();

        let mean_str = if r.mean_error.is_nan() {
            "NaN".to_string()
        } else {
            format!("{:.3e}", r.mean_error)
        };

        let var_str = if r.variance_error.is_nan() {
            "NaN".to_string()
        } else {
            format!("{:.3e}", r.variance_error)
        };

        out.push_str(&format!(
            "| {:<29} | {:>6} | {:>4} | {:>4} | {:>4} | {:>9} | {:>9} |\n",
            r.distribution, status, pdf_count, cdf_count, ppf_count, mean_str, var_str
        ));
    }

    out.push_str(
        "+-------------------------------+--------+------+------+------+-----------+-----------+\n",
    );
    out.push_str(&format!(
        "\nSummary: {}/{} passed, {} failed\n",
        passed, total, failed
    ));

    // Detail section for failures
    let failures: Vec<&ValidationResult> = results.iter().filter(|r| !r.passed).collect();
    if !failures.is_empty() {
        out.push_str("\n--- Failure Details ---\n");
        for r in failures {
            out.push_str(&format!("\n[FAIL] {}:\n", r.distribution));
            for (x, expected, actual) in &r.pdf_errors {
                out.push_str(&format!(
                    "  PDF at x={}: expected={:.15e}, got={:.15e}, err={:.3e}\n",
                    x,
                    expected,
                    actual,
                    (expected - actual).abs()
                ));
            }
            for (x, expected, actual) in &r.cdf_errors {
                out.push_str(&format!(
                    "  CDF at x={}: expected={:.15e}, got={:.15e}, err={:.3e}\n",
                    x,
                    expected,
                    actual,
                    (expected - actual).abs()
                ));
            }
            for (p, expected, actual) in &r.ppf_errors {
                out.push_str(&format!(
                    "  PPF at p={}: expected={:.15e}, got={:.15e}, err={:.3e}\n",
                    p,
                    expected,
                    actual,
                    (expected - actual).abs()
                ));
            }
            if r.mean_error > 1e-9 {
                out.push_str(&format!("  Mean error: {:.3e}\n", r.mean_error));
            }
            if r.variance_error > 1e-9 {
                out.push_str(&format!("  Variance error: {:.3e}\n", r.variance_error));
            }
        }
    }

    out
}

/// Generate a JSON-format report from validation results.
///
/// This function does not require serde; it produces valid JSON via string
/// formatting. When the `serialization` feature is enabled, the serde-based
/// path can be used instead for more complex scenarios.
///
/// # Arguments
///
/// * `results` - Slice of validation results to include
///
/// # Returns
///
/// A JSON string representing the results array.
pub fn generate_json_report(results: &[ValidationResult]) -> String {
    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = total - passed;

    let mut out = String::with_capacity(4096);
    out.push_str("{\n");
    out.push_str(&format!("  \"total\": {},\n", total));
    out.push_str(&format!("  \"passed\": {},\n", passed));
    out.push_str(&format!("  \"failed\": {},\n", failed));
    out.push_str("  \"results\": [\n");

    for (i, r) in results.iter().enumerate() {
        out.push_str("    {\n");
        out.push_str(&format!(
            "      \"distribution\": \"{}\",\n",
            escape_json_string(&r.distribution)
        ));
        out.push_str(&format!("      \"passed\": {},\n", r.passed));

        // Mean error
        out.push_str(&format!(
            "      \"mean_error\": {},\n",
            format_json_f64(r.mean_error)
        ));

        // Variance error
        out.push_str(&format!(
            "      \"variance_error\": {},\n",
            format_json_f64(r.variance_error)
        ));

        // PDF errors
        out.push_str("      \"pdf_errors\": [");
        for (j, (x, expected, actual)) in r.pdf_errors.iter().enumerate() {
            if j > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!(
                "{{\"x\": {}, \"expected\": {}, \"actual\": {}}}",
                format_json_f64(*x),
                format_json_f64(*expected),
                format_json_f64(*actual)
            ));
        }
        out.push_str("],\n");

        // CDF errors
        out.push_str("      \"cdf_errors\": [");
        for (j, (x, expected, actual)) in r.cdf_errors.iter().enumerate() {
            if j > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!(
                "{{\"x\": {}, \"expected\": {}, \"actual\": {}}}",
                format_json_f64(*x),
                format_json_f64(*expected),
                format_json_f64(*actual)
            ));
        }
        out.push_str("],\n");

        // PPF errors
        out.push_str("      \"ppf_errors\": [");
        for (j, (p, expected, actual)) in r.ppf_errors.iter().enumerate() {
            if j > 0 {
                out.push_str(", ");
            }
            out.push_str(&format!(
                "{{\"p\": {}, \"expected\": {}, \"actual\": {}}}",
                format_json_f64(*p),
                format_json_f64(*expected),
                format_json_f64(*actual)
            ));
        }
        out.push_str("]\n");

        out.push_str("    }");
        if i < results.len() - 1 {
            out.push(',');
        }
        out.push('\n');
    }

    out.push_str("  ]\n");
    out.push_str("}\n");
    out
}

/// Format an f64 as valid JSON (handling NaN and Infinity as null).
fn format_json_f64(v: f64) -> String {
    if v.is_nan() || v.is_infinite() {
        "null".to_string()
    } else {
        format!("{}", v)
    }
}

/// Escape special characters in a string for JSON embedding.
fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validators::ValidationResult;

    fn make_pass_result(name: &str) -> ValidationResult {
        ValidationResult {
            distribution: name.to_string(),
            pdf_errors: vec![],
            cdf_errors: vec![],
            ppf_errors: vec![],
            mean_error: 1e-14,
            variance_error: 1e-14,
            passed: true,
        }
    }

    fn make_fail_result(name: &str) -> ValidationResult {
        ValidationResult {
            distribution: name.to_string(),
            pdf_errors: vec![(0.0, 1.0, 0.5)],
            cdf_errors: vec![(1.0, 0.9, 0.1)],
            ppf_errors: vec![],
            mean_error: 100.0,
            variance_error: 50.0,
            passed: false,
        }
    }

    #[test]
    fn test_report_generation_not_empty() {
        let results = vec![
            make_pass_result("Normal(0,1)"),
            make_fail_result("Bad(1,2)"),
        ];
        let report = generate_report(&results);
        assert!(!report.is_empty());
        assert!(report.contains("Normal(0,1)"));
        assert!(report.contains("Bad(1,2)"));
        assert!(report.contains("PASS"));
        assert!(report.contains("FAIL"));
        assert!(report.contains("1/2 passed"));
    }

    #[test]
    fn test_json_report_parseable() {
        let results = vec![
            make_pass_result("Normal(0,1)"),
            make_pass_result("Exponential(1)"),
            make_fail_result("Bad(1,2)"),
        ];
        let json = generate_json_report(&results);

        // Verify it's valid JSON structure
        assert!(json.starts_with('{'));
        assert!(json.contains("\"total\": 3"));
        assert!(json.contains("\"passed\": 2"));
        assert!(json.contains("\"failed\": 1"));
        assert!(json.contains("\"distribution\": \"Normal(0,1)\""));
        assert!(json.contains("\"distribution\": \"Bad(1,2)\""));
        assert!(json.contains("\"pdf_errors\""));
        assert!(json.contains("\"cdf_errors\""));
        assert!(json.contains("\"ppf_errors\""));
    }

    #[test]
    fn test_json_report_nan_handling() {
        let result = ValidationResult {
            distribution: "Cauchy(0,1)".to_string(),
            pdf_errors: vec![],
            cdf_errors: vec![],
            ppf_errors: vec![],
            mean_error: f64::NAN,
            variance_error: f64::NAN,
            passed: true,
        };
        let json = generate_json_report(&[result]);
        // NaN should be serialized as null in JSON
        assert!(json.contains("\"mean_error\": null"));
        assert!(json.contains("\"variance_error\": null"));
    }

    #[test]
    fn test_validation_report_struct() {
        let results = vec![make_pass_result("A"), make_fail_result("B")];
        let report = ValidationReport::new("Test Suite", results);
        assert_eq!(report.total, 2);
        assert_eq!(report.passed, 1);
        assert_eq!(report.failed, 1);
        assert!(!report.all_passed());

        let display = format!("{}", report);
        assert!(display.contains("Test Suite"));
        assert!(display.contains("Passed: 1"));
    }

    #[test]
    fn test_validation_report_all_pass() {
        let results = vec![make_pass_result("A"), make_pass_result("B")];
        let report = ValidationReport::new("All Good", results);
        assert!(report.all_passed());
    }

    #[test]
    fn test_escape_json_string() {
        assert_eq!(escape_json_string("hello"), "hello");
        assert_eq!(escape_json_string("he\"llo"), "he\\\"llo");
        assert_eq!(escape_json_string("a\\b"), "a\\\\b");
        assert_eq!(escape_json_string("a\nb"), "a\\nb");
    }

    #[test]
    fn test_format_json_f64() {
        assert_eq!(format_json_f64(1.5), "1.5");
        assert_eq!(format_json_f64(f64::NAN), "null");
        assert_eq!(format_json_f64(f64::INFINITY), "null");
        assert_eq!(format_json_f64(f64::NEG_INFINITY), "null");
    }

    #[test]
    fn test_report_with_failure_details() {
        let results = vec![make_fail_result("BadDist")];
        let report = generate_report(&results);
        assert!(report.contains("Failure Details"));
        assert!(report.contains("PDF at x=0"));
        assert!(report.contains("CDF at x=1"));
        assert!(report.contains("Mean error"));
        assert!(report.contains("Variance error"));
    }
}
