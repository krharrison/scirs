//! Structured policy violation types and formatting.
//!
//! [`PolicyViolation`] is the canonical representation of a single violation
//! emitted by any of the linter checks.  It is distinct from the older
//! [`crate::rules::Violation`] (which is simpler and used by the
//! [`crate::rules::PolicyRule`] trait).
//!
//! Use [`PolicyViolation`] when you need per-line source-location information
//! (e.g., `banned_imports` and `unwrap_check` checks).

use std::fmt;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

/// Severity level for a policy violation.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational note; does not block the build.
    Info,
    /// Should be fixed but is not release-blocking.
    Warning,
    /// Must be fixed before release.
    Error,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[allow(unreachable_patterns)]
        match self {
            Severity::Error => write!(f, "ERROR"),
            Severity::Warning => write!(f, "WARN"),
            Severity::Info => write!(f, "INFO"),
            _ => write!(f, "UNKNOWN"),
        }
    }
}

// ---------------------------------------------------------------------------
// PolicyViolation
// ---------------------------------------------------------------------------

/// A single policy violation emitted by a linter check.
///
/// Unlike [`crate::rules::Violation`], this type carries the full source
/// location (file path + line number) and the owning crate name.
#[derive(Debug, Clone)]
pub struct PolicyViolation {
    /// Name of the workspace crate in which the violation was found.
    pub crate_name: String,
    /// Absolute path to the file containing the violation.
    pub file: PathBuf,
    /// 1-based line number within `file`.  `0` means "unknown/not applicable".
    pub line: usize,
    /// Human-readable description of the violation.
    pub message: String,
    /// Severity of this violation.
    pub severity: Severity,
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/// Format a single violation as a human-readable string.
///
/// Output format: `<severity> [<crate>] <file>:<line>: <message>`
/// When `line` is 0, the `:<line>` part is omitted.
pub fn format_violation(v: &PolicyViolation) -> String {
    let loc = if v.line > 0 {
        format!("{}:{}", v.file.display(), v.line)
    } else {
        v.file.display().to_string()
    };
    format!("[{}] [{}] {}: {}", v.severity, v.crate_name, loc, v.message)
}

// ---------------------------------------------------------------------------
// Exit code
// ---------------------------------------------------------------------------

/// Return `1` if any violation has severity [`Severity::Error`], else `0`.
pub fn exit_code(violations: &[PolicyViolation]) -> i32 {
    if violations.iter().any(|v| v.severity == Severity::Error) {
        1
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_violation(severity: Severity, line: usize) -> PolicyViolation {
        PolicyViolation {
            crate_name: "my-crate".to_string(),
            file: PathBuf::from("/src/lib.rs"),
            line,
            message: "test message".to_string(),
            severity,
        }
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(Severity::Error.to_string(), "ERROR");
        assert_eq!(Severity::Warning.to_string(), "WARN");
        assert_eq!(Severity::Info.to_string(), "INFO");
    }

    #[test]
    fn test_format_violation_with_line() {
        let v = make_violation(Severity::Error, 42);
        let s = format_violation(&v);
        assert!(s.contains("[ERROR]"), "Should contain severity");
        assert!(s.contains("[my-crate]"), "Should contain crate name");
        assert!(s.contains(":42"), "Should contain line number");
        assert!(s.contains("test message"), "Should contain message");
    }

    #[test]
    fn test_format_violation_no_line() {
        let v = make_violation(Severity::Warning, 0);
        let s = format_violation(&v);
        assert!(!s.contains(":0"), "Should omit ':0' when line is 0");
        assert!(s.contains("[WARN]"), "Should contain severity");
    }

    #[test]
    fn test_exit_code_with_errors() {
        let violations = vec![
            make_violation(Severity::Warning, 1),
            make_violation(Severity::Error, 2),
        ];
        assert_eq!(exit_code(&violations), 1);
    }

    #[test]
    fn test_exit_code_no_errors() {
        let violations = vec![
            make_violation(Severity::Warning, 1),
            make_violation(Severity::Info, 2),
        ];
        assert_eq!(exit_code(&violations), 0);
    }

    #[test]
    fn test_exit_code_empty() {
        assert_eq!(exit_code(&[]), 0);
    }
}
