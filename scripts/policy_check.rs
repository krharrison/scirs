#!/usr/bin/env rust-script
//! SciRS2 Policy Compliance Checker — Rust Implementation
//!
//! A precise policy linter for the SciRS2 workspace that detects import
//! violations with better accuracy than the shell script version.
//!
//! Compile and run with:
//!   rustc -o /tmp/policy_check scripts/policy_check.rs && /tmp/policy_check
//!
//! Or with rust-script (if installed):
//!   rust-script scripts/policy_check.rs
//!
//! Or via the shell wrapper:
//!   bash scripts/check_policy.sh
//!
//! # Policy Rules Enforced
//!
//! RULE-01  No `use rand::` in non-core crates
//! RULE-02  No `use ndarray::` in non-core crates
//! RULE-03  No `use ndarray_rand::` in non-core crates
//! RULE-04  No `use scirs2_autograd::ndarray` (use scirs2_core::ndarray)
//! RULE-05  No `extern crate rand/ndarray` in non-core crates
//!
//! # Exemptions
//!
//! The following paths are exempt from all rules:
//!   - `scirs2-core/src/`       — the one crate allowed to use external deps directly
//!   - `scirs2-numpy/src/`      — SciRS2's ndarray-compatible fork, uses ndarray directly
//!   - `scirs2-autograd/src/`   — exempt only for RULE-04 (it re-exports ndarray itself)
//!   - `*/target/*`             — compiled artifacts
//!   - `*/examples_disabled/*`  — disabled examples not part of the build
//!   - `*.backup.*`             — backup dirs from scirs2-policy-refactor.sh
//!
//! # Precision Improvements over Shell Script
//!
//! - Skips lines that are entirely within `//` or `/* */` comments.
//! - Tracks `#[cfg(test)]` attribute scopes so test-only code is
//!   reported separately (as warnings, not errors, in lenient mode).
//! - Reports precise byte column offsets alongside line numbers.
//! - Groups violations by rule and by file.
//! - Provides machine-readable JSON output mode (`--json`).

use std::collections::BTreeMap;
use std::env;
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single policy violation.
#[derive(Debug, Clone)]
struct Violation {
    rule: &'static str,
    description: &'static str,
    suggestion: &'static str,
    file: PathBuf,
    /// 1-based line number.
    line: usize,
    /// 1-based column of the matched pattern.
    col: usize,
    /// The offending source line (trimmed).
    source_snippet: String,
    /// True when the violation occurs inside a `#[cfg(test)]` block.
    in_test_scope: bool,
}

impl Violation {
    #[allow(dead_code)]
    fn location(&self) -> String {
        format!("{}:{}:{}", self.file.display(), self.line, self.col)
    }
}

/// Linter configuration parsed from CLI arguments.
#[derive(Debug, Default)]
struct Config {
    /// Root of the workspace (parent of Cargo.toml).
    workspace_root: PathBuf,
    /// Emit machine-readable JSON instead of human text.
    json_output: bool,
    /// Report test-scoped violations as errors (default: warnings).
    strict: bool,
    /// Suppress color in output.
    no_color: bool,
    /// Show help and exit.
    show_help: bool,
}

/// Summary counts after all checks.
#[derive(Debug, Default)]
struct Summary {
    files_scanned: usize,
    files_with_violations: usize,
    total_violations: usize,
    test_scope_violations: usize,
}

// ---------------------------------------------------------------------------
// Policy rules
// ---------------------------------------------------------------------------

/// A single policy rule.
struct Rule {
    id: &'static str,
    description: &'static str,
    suggestion: &'static str,
    /// Regex-like pattern matched as a plain string prefix on trimmed lines.
    /// We use simple prefix matching for speed; no regex crate is available
    /// in a single-file rustc-compiled script.
    patterns: &'static [&'static str],
    /// Crate directory names (relative to workspace root) that are entirely
    /// exempt from this rule.
    exempt_crate_dirs: &'static [&'static str],
}

const RULES: &[Rule] = &[
    Rule {
        id: "RULE-01",
        description: "Direct 'use rand::' import in non-core crate",
        suggestion: "Replace with 'use scirs2_core::random::*' (scirs2-core provides all rand/rand_distr functionality)",
        patterns: &["use rand::"],
        // scirs2-core is entirely exempt: its src/, tests/, examples/, benches/ all
        // legitimately use rand (it re-exports rand for the whole ecosystem).
        // scirs2-numpy is a direct fork of rust-numpy with native ndarray/rand usage.
        exempt_crate_dirs: &["scirs2-core", "scirs2-numpy"],
    },
    Rule {
        id: "RULE-02",
        description: "Direct 'use ndarray::' import in non-core crate",
        suggestion: "Replace with 'use scirs2_core::ndarray::*' (enable 'array' feature in scirs2-core)",
        patterns: &["use ndarray::"],
        exempt_crate_dirs: &["scirs2-core", "scirs2-numpy"],
    },
    Rule {
        id: "RULE-03",
        description: "Direct 'use ndarray_rand::' import",
        suggestion: "Replace with 'use scirs2_core::ndarray::*' (ndarray-rand re-exported via array feature)",
        patterns: &["use ndarray_rand::"],
        exempt_crate_dirs: &["scirs2-core", "scirs2-numpy"],
    },
    Rule {
        id: "RULE-04",
        description: "'use scirs2_autograd::ndarray' found",
        suggestion: "Replace with 'use scirs2_core::ndarray' to use the canonical ndarray path and avoid type mismatches",
        patterns: &["use scirs2_autograd::ndarray"],
        // scirs2-autograd itself defines the ndarray re-export; exempt its own code.
        exempt_crate_dirs: &["scirs2-autograd"],
    },
    Rule {
        id: "RULE-05",
        description: "'extern crate rand/ndarray/ndarray_rand' in non-core crate",
        suggestion: "Remove extern crate declaration; access through scirs2_core instead",
        patterns: &[
            "extern crate rand",
            "extern crate ndarray",
            "extern crate ndarray_rand",
        ],
        exempt_crate_dirs: &["scirs2-core", "scirs2-numpy"],
    },
];

// ---------------------------------------------------------------------------
// File-system walker
// ---------------------------------------------------------------------------

/// Recursively collect all `.rs` files under `root`, skipping known
/// non-source directories.
fn collect_rust_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_rust_files_inner(root, &mut files);
    files
}

fn collect_rust_files_inner(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_owned(),
            None => continue,
        };

        // Skip directories we never want to scan.
        if path.is_dir() {
            match name.as_str() {
                "target" | ".git" | "examples_disabled" | ".cargo" => continue,
                _ => {}
            }
            // Skip backup directories created by scirs2-policy-refactor.sh
            if name.contains(".backup.") {
                continue;
            }
            collect_rust_files_inner(&path, out);
        } else if name.ends_with(".rs") {
            out.push(path);
        }
    }
}

// ---------------------------------------------------------------------------
// Comment tracking
//
// We perform a simple line-by-line scan that keeps track of whether we are
// inside a block comment `/* ... */`.  This is not a full Rust parser but
// covers the vast majority of real-world cases correctly.
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
struct ScanState {
    /// True when the current scan position is inside a `/* ... */` comment.
    in_block_comment: bool,
    /// Depth of `#[cfg(test)]` attribute scopes we are tracking.
    /// We use a simple brace-counting heuristic.
    cfg_test_depth: usize,
    /// Whether the last non-blank line opened a `#[cfg(test)]` attribute.
    saw_cfg_test_attr: bool,
}

/// Strip the leading whitespace from a line and return the trimmed slice
/// together with the column offset (1-based).
fn trimmed_with_col(line: &str) -> (&str, usize) {
    let trimmed = line.trim_start();
    let spaces = line.len() - trimmed.len();
    (trimmed, spaces + 1)
}

/// Returns true if `trimmed_line` starts a `//` line comment, meaning the
/// entire line is a comment and should be skipped for pattern matching.
fn is_line_comment(trimmed: &str) -> bool {
    trimmed.starts_with("//")
}

/// Update the block-comment tracking state for a single source line.
/// Returns a version of the line with block-comment content replaced by spaces,
/// making it safe to search for import patterns.
fn strip_block_comments(line: &str, state: &mut ScanState) -> String {
    let mut result = String::with_capacity(line.len());
    let bytes = line.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if state.in_block_comment {
            // Look for the end of the block comment.
            if i + 1 < bytes.len() && bytes[i] == b'*' && bytes[i + 1] == b'/' {
                state.in_block_comment = false;
                result.push(' ');
                result.push(' ');
                i += 2;
            } else {
                result.push(' ');
                i += 1;
            }
        } else {
            // Look for the start of a block comment or a line comment.
            if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
                state.in_block_comment = true;
                result.push(' ');
                result.push(' ');
                i += 2;
            } else if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
                // Rest of line is a comment; stop processing.
                break;
            } else {
                result.push(bytes[i] as char);
                i += 1;
            }
        }
    }

    result
}

/// Heuristic to update the `#[cfg(test)]` scope depth.
///
/// The approach is intentionally simple: when we see a line containing
/// `#[cfg(test)]` or `#[cfg(any(..., test, ...))]` we set a flag, and
/// when the next `{` is encountered we increment the depth.  When we
/// see `}` and depth > 0 we decrement.
///
/// This is not a fully accurate Rust parser but handles the common case
/// of `#[cfg(test)] mod tests { ... }`.
fn update_cfg_test_scope(stripped_line: &str, state: &mut ScanState) {
    let trimmed = stripped_line.trim();

    // Detect #[cfg(test)] attribute.
    if trimmed.contains("#[cfg(test)]") || trimmed.contains(", test,") || trimmed.contains(", test)") {
        state.saw_cfg_test_attr = true;
    }

    // Count braces.
    for ch in trimmed.chars() {
        match ch {
            '{' => {
                if state.saw_cfg_test_attr {
                    state.cfg_test_depth += 1;
                    state.saw_cfg_test_attr = false;
                } else if state.cfg_test_depth > 0 {
                    state.cfg_test_depth += 1;
                }
            }
            '}' => {
                if state.cfg_test_depth > 0 {
                    state.cfg_test_depth -= 1;
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Crate path helpers
// ---------------------------------------------------------------------------

/// Extract the top-level crate directory name from a workspace-relative path.
/// E.g. "scirs2-stats/src/mcmc/nuts.rs" -> Some("scirs2-stats")
fn top_level_crate_dir<'a>(workspace_root: &Path, file: &'a Path) -> Option<&'a str> {
    let rel = file.strip_prefix(workspace_root).ok()?;
    rel.components().next()?.as_os_str().to_str()
}

/// Check whether the given file path is in one of the exempt crate directories.
fn is_exempt(workspace_root: &Path, file: &Path, exempt_dirs: &[&str]) -> bool {
    match top_level_crate_dir(workspace_root, file) {
        Some(crate_dir) => exempt_dirs.contains(&crate_dir),
        None => false,
    }
}

// ---------------------------------------------------------------------------
// Line-level checker
// ---------------------------------------------------------------------------

/// Scan a single `.rs` file for violations of all rules.
fn check_file(path: &Path, config: &Config) -> Vec<Violation> {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };

    let mut violations = Vec::new();
    let mut scan_state = ScanState::default();

    for (line_idx, raw_line) in source.lines().enumerate() {
        let line_num = line_idx + 1;

        // Strip block comments; this also updates in_block_comment state.
        let stripped = strip_block_comments(raw_line, &mut scan_state);

        // Update cfg(test) scope tracking.
        update_cfg_test_scope(&stripped, &mut scan_state);

        let (trimmed, col_offset) = trimmed_with_col(&stripped);

        // Skip lines that are pure comments (after stripping block comments,
        // a line comment that remains will start with `//`).
        if is_line_comment(trimmed) || trimmed.is_empty() {
            continue;
        }

        // Check each rule against this line.
        for rule in RULES {
            if is_exempt(config.workspace_root.as_path(), path, rule.exempt_crate_dirs) {
                continue;
            }

            for pattern in rule.patterns {
                if trimmed.starts_with(pattern) {
                    let source_snippet = raw_line.trim().to_owned();
                    violations.push(Violation {
                        rule: rule.id,
                        description: rule.description,
                        suggestion: rule.suggestion,
                        file: path.to_owned(),
                        line: line_num,
                        col: col_offset,
                        source_snippet,
                        in_test_scope: scan_state.cfg_test_depth > 0,
                    });
                    // One violation per (line, rule); move to next rule.
                    break;
                }
            }
        }
    }

    violations
}

// ---------------------------------------------------------------------------
// Output formatters
// ---------------------------------------------------------------------------

struct Colorizer {
    enabled: bool,
}

impl Colorizer {
    fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    fn red<'a>(&self, s: &'a str) -> ColoredStr<'a> {
        ColoredStr { s, code: "\x1b[0;31m", enabled: self.enabled }
    }
    fn green<'a>(&self, s: &'a str) -> ColoredStr<'a> {
        ColoredStr { s, code: "\x1b[0;32m", enabled: self.enabled }
    }
    fn yellow<'a>(&self, s: &'a str) -> ColoredStr<'a> {
        ColoredStr { s, code: "\x1b[1;33m", enabled: self.enabled }
    }
    fn cyan<'a>(&self, s: &'a str) -> ColoredStr<'a> {
        ColoredStr { s, code: "\x1b[0;36m", enabled: self.enabled }
    }
    fn bold<'a>(&self, s: &'a str) -> ColoredStr<'a> {
        ColoredStr { s, code: "\x1b[1m", enabled: self.enabled }
    }
    fn dim<'a>(&self, s: &'a str) -> ColoredStr<'a> {
        ColoredStr { s, code: "\x1b[2m", enabled: self.enabled }
    }
}

struct ColoredStr<'a> {
    s: &'a str,
    code: &'static str,
    enabled: bool,
}

impl<'a> fmt::Display for ColoredStr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.enabled {
            write!(f, "{}{}\x1b[0m", self.code, self.s)
        } else {
            write!(f, "{}", self.s)
        }
    }
}

/// Print human-readable output to stdout.
fn print_text_report(
    all_violations: &[Violation],
    summary: &Summary,
    config: &Config,
    out: &mut impl Write,
) -> io::Result<()> {
    let c = Colorizer::new(!config.no_color);

    writeln!(out, "")?;
    writeln!(out, "{}", c.bold("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))?;
    writeln!(out, "{}", c.cyan("  SciRS2 Policy Linter  (Rust edition)"))?;
    writeln!(out, "{}", c.bold("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))?;
    writeln!(out, "")?;

    // Group violations by file for readability.
    let mut by_file: BTreeMap<&Path, Vec<&Violation>> = BTreeMap::new();
    for v in all_violations {
        by_file.entry(v.file.as_path()).or_default().push(v);
    }

    for (file, viols) in &by_file {
        let rel = file.strip_prefix(&config.workspace_root)
            .unwrap_or(file);
        writeln!(out, "  {} {}", c.bold("FILE:"), c.cyan(&rel.display().to_string()))?;

        for v in viols {
            let severity = if v.in_test_scope && !config.strict {
                c.yellow("warn")
            } else {
                c.red("error")
            };

            writeln!(out, "    [{}] {} {}:{}",
                severity,
                c.bold(v.rule),
                v.line, v.col
            )?;
            writeln!(out, "         {}", c.dim(&format!("→ {}", v.description)))?;
            writeln!(out, "         {} {}",
                c.dim("snippet:"),
                c.dim(&v.source_snippet)
            )?;
            writeln!(out, "         {} {}",
                c.yellow("fix:"),
                v.suggestion
            )?;
            writeln!(out, "")?;
        }
    }

    // Summary
    writeln!(out, "{}", c.bold("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))?;
    writeln!(out, "  {} {} files", c.dim("Scanned:"), summary.files_scanned)?;

    if summary.total_violations == 0 {
        writeln!(out, "  {}", c.green("ALL POLICY CHECKS PASSED - No violations found."))?;
    } else {
        writeln!(out, "  {} {} violation(s) in {} file(s)",
            c.red("VIOLATIONS:"),
            summary.total_violations,
            summary.files_with_violations
        )?;

        if summary.test_scope_violations > 0 && !config.strict {
            writeln!(out, "  {} {} in test scopes (warnings in lenient mode; use --strict to treat as errors)",
                c.yellow("Test-scope:"),
                summary.test_scope_violations
            )?;
        }

        writeln!(out, "")?;
        writeln!(out, "  {} See SCIRS2_POLICY.md for full details.",
            c.dim("Ref:")
        )?;
        writeln!(out, "  {} bash scripts/scirs2-policy-refactor.sh --help",
            c.dim("Fix:")
        )?;
    }

    writeln!(out, "{}", c.bold("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))?;
    writeln!(out, "")?;

    Ok(())
}

/// Print machine-readable JSON output.
/// We hand-roll the JSON to avoid requiring the `serde_json` crate in a
/// single-file script.
fn print_json_report(
    all_violations: &[Violation],
    summary: &Summary,
    config: &Config,
    out: &mut impl Write,
) -> io::Result<()> {
    writeln!(out, "{{")?;
    writeln!(out, "  \"summary\": {{")?;
    writeln!(out, "    \"files_scanned\": {},", summary.files_scanned)?;
    writeln!(out, "    \"files_with_violations\": {},", summary.files_with_violations)?;
    writeln!(out, "    \"total_violations\": {},", summary.total_violations)?;
    writeln!(out, "    \"test_scope_violations\": {},", summary.test_scope_violations)?;
    writeln!(out, "    \"strict_mode\": {}", config.strict)?;
    writeln!(out, "  }},")?;
    writeln!(out, "  \"violations\": [")?;

    for (i, v) in all_violations.iter().enumerate() {
        let rel = v.file.strip_prefix(&config.workspace_root)
            .unwrap_or(v.file.as_path());
        let comma = if i + 1 < all_violations.len() { "," } else { "" };
        writeln!(out, "    {{")?;
        writeln!(out, "      \"rule\": \"{}\",", json_escape(v.rule))?;
        writeln!(out, "      \"file\": \"{}\",", json_escape(&rel.display().to_string()))?;
        writeln!(out, "      \"line\": {},", v.line)?;
        writeln!(out, "      \"col\": {},", v.col)?;
        writeln!(out, "      \"description\": \"{}\",", json_escape(v.description))?;
        writeln!(out, "      \"suggestion\": \"{}\",", json_escape(v.suggestion))?;
        writeln!(out, "      \"snippet\": \"{}\",", json_escape(&v.source_snippet))?;
        writeln!(out, "      \"in_test_scope\": {}", v.in_test_scope)?;
        writeln!(out, "    }}{}", comma)?;
    }

    writeln!(out, "  ]")?;
    writeln!(out, "}}")?;

    Ok(())
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
     .replace('"', "\\\"")
     .replace('\n', "\\n")
     .replace('\r', "\\r")
     .replace('\t', "\\t")
}

// ---------------------------------------------------------------------------
// CLI argument parser
// ---------------------------------------------------------------------------

fn parse_args() -> Result<Config, String> {
    let args: Vec<String> = env::args().collect();
    let mut config = Config::default();

    // Default workspace root: parent of the directory containing this script,
    // or the current directory if we can't determine it.
    config.workspace_root = env::current_dir()
        .map_err(|e| format!("Cannot determine current directory: {e}"))?;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                config.show_help = true;
                return Ok(config);
            }
            "--json" => {
                config.json_output = true;
            }
            "--strict" => {
                config.strict = true;
            }
            "--no-color" => {
                config.no_color = true;
            }
            "--workspace" => {
                i += 1;
                if i >= args.len() {
                    return Err("--workspace requires a path argument".to_owned());
                }
                config.workspace_root = PathBuf::from(&args[i]);
            }
            other => {
                return Err(format!("Unknown argument: {other}. Run with --help for usage."));
            }
        }
        i += 1;
    }

    Ok(config)
}

fn print_help() {
    eprintln!(
        r#"SciRS2 Policy Linter (Rust edition)

USAGE:
    policy_check [OPTIONS]

OPTIONS:
    --workspace <PATH>   Workspace root (default: current directory)
    --json               Emit machine-readable JSON instead of text
    --strict             Treat test-scope violations as errors (default: warnings)
    --no-color           Disable ANSI color codes
    --help, -h           Show this help message

EXIT CODES:
    0   No violations (or only test-scope violations in lenient mode)
    1   One or more policy violations found

EXAMPLES:
    # Run from workspace root:
    rustc -o /tmp/policy_check scripts/policy_check.rs && /tmp/policy_check

    # JSON output for CI integration:
    /tmp/policy_check --json > /tmp/policy_report.json

    # Strict mode (test violations also fail):
    /tmp/policy_check --strict

    # Custom workspace path:
    /tmp/policy_check --workspace /path/to/scirs
"#
    );
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let config = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(2);
        }
    };

    if config.show_help {
        print_help();
        return;
    }

    let workspace_root = config.workspace_root.clone();

    // Verify the workspace root looks right.
    let cargo_toml = workspace_root.join("Cargo.toml");
    if !cargo_toml.exists() {
        eprintln!(
            "Error: '{}' does not contain a Cargo.toml. \
             Run from the workspace root or pass --workspace <path>.",
            workspace_root.display()
        );
        std::process::exit(2);
    }

    // Collect all Rust source files.
    let files = collect_rust_files(&workspace_root);
    let files_scanned = files.len();

    // Run the checker on every file.
    let mut all_violations: Vec<Violation> = Vec::new();
    let mut files_with_violations: usize = 0;

    for file in &files {
        let violations = check_file(file, &config);
        if !violations.is_empty() {
            files_with_violations += 1;
            all_violations.extend(violations);
        }
    }

    // Compute summary.
    let test_scope_violations = all_violations.iter().filter(|v| v.in_test_scope).count();
    let summary = Summary {
        files_scanned,
        files_with_violations,
        total_violations: all_violations.len(),
        test_scope_violations,
    };

    let stdout = io::stdout();
    let mut out = stdout.lock();

    if config.json_output {
        print_json_report(&all_violations, &summary, &config, &mut out)
            .expect("failed to write JSON report to stdout");
    } else {
        print_text_report(&all_violations, &summary, &config, &mut out)
            .expect("failed to write text report to stdout");
    }

    // Determine exit code.
    // In lenient mode, test-scope-only violations do not cause a non-zero exit.
    let hard_violations = if config.strict {
        all_violations.len()
    } else {
        all_violations.iter().filter(|v| !v.in_test_scope).count()
    };

    if hard_violations > 0 {
        std::process::exit(1);
    }
}
