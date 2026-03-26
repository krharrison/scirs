//! Cross-platform compatibility utilities for consistent behavior across
//! Windows, macOS, Linux, and WebAssembly targets.
//!
//! This module provides helper functions that abstract over platform differences
//! so that the rest of the SciRS2 codebase can remain platform-agnostic.
//!
//! # Examples
//!
//! ```
//! use scirs2_core::platform_compat;
//!
//! // Portable temporary directory
//! let tmp = platform_compat::temp_dir();
//! assert!(tmp.is_absolute());
//!
//! // Portable temporary file path
//! let f = platform_compat::temp_file("my_data.bin");
//! assert!(f.ends_with("my_data.bin"));
//!
//! // CPU count
//! let n = platform_compat::num_cpus();
//! assert!(n >= 1);
//! ```

use std::path::PathBuf;

/// Return the platform's temporary directory.
///
/// On Unix this is typically `/tmp`; on Windows it is `%TEMP%` or similar.
/// Always prefer this over hard-coding `/tmp/`.
#[inline]
pub fn temp_dir() -> PathBuf {
    std::env::temp_dir()
}

/// Build a [`PathBuf`] pointing to `<temp_dir>/<name>`.
///
/// Useful for constructing throwaway file paths in tests and transient
/// storage configurations.
#[inline]
pub fn temp_file(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(name);
    p
}

/// Build a [`PathBuf`] pointing to `<temp_dir>/<subdir>/<name>`.
///
/// Creates a namespaced temporary path without actually creating the directory.
#[inline]
pub fn temp_path(subdir: &str, name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(subdir);
    p.push(name);
    p
}

/// Return the number of logical CPUs available on the current machine.
///
/// Falls back to `1` if the value cannot be determined (e.g. under some
/// sandboxed or embedded environments).
#[inline]
pub fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// The native path separator character for the current platform.
///
/// `'/'` on Unix, `'\\'` on Windows.
#[inline]
pub fn path_separator() -> char {
    std::path::MAIN_SEPARATOR
}

/// `true` when compiled for a Windows target.
#[inline]
pub const fn is_windows() -> bool {
    cfg!(target_os = "windows")
}

/// `true` when compiled for a macOS target.
#[inline]
pub const fn is_macos() -> bool {
    cfg!(target_os = "macos")
}

/// `true` when compiled for a Linux target.
#[inline]
pub const fn is_linux() -> bool {
    cfg!(target_os = "linux")
}

/// `true` when compiled for a WebAssembly target.
#[inline]
pub const fn is_wasm() -> bool {
    cfg!(target_family = "wasm")
}

/// `true` when compiled for any Unix-family target (Linux, macOS, BSDs, etc.).
#[inline]
pub const fn is_unix() -> bool {
    cfg!(target_family = "unix")
}

/// Return the default temporary directory as a [`String`].
///
/// Convenience wrapper around [`temp_dir`] for code that stores paths as
/// `String` fields (e.g. configuration structs).
pub fn temp_dir_string() -> String {
    temp_dir()
        .to_str()
        .unwrap_or(if cfg!(target_os = "windows") {
            "C:\\Temp"
        } else {
            "/tmp"
        })
        .to_string()
}

/// Join a subdirectory name onto the platform temporary directory and return
/// the result as a [`String`].
pub fn temp_subdir_string(subdir: &str) -> String {
    let mut p = temp_dir();
    p.push(subdir);
    p.to_str()
        .unwrap_or(if cfg!(target_os = "windows") {
            "C:\\Temp"
        } else {
            "/tmp"
        })
        .to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_dir_is_absolute() {
        assert!(temp_dir().is_absolute());
    }

    #[test]
    fn temp_file_ends_with_name() {
        let p = temp_file("hello.txt");
        assert!(p.ends_with("hello.txt"));
    }

    #[test]
    fn temp_path_contains_subdir() {
        let p = temp_path("scirs2", "data.bin");
        assert!(p.ends_with("data.bin"));
        // The parent should contain "scirs2"
        let parent = p.parent().expect("should have parent");
        assert!(parent.ends_with("scirs2"));
    }

    #[test]
    fn num_cpus_at_least_one() {
        assert!(num_cpus() >= 1);
    }

    #[test]
    fn path_separator_is_correct() {
        let sep = path_separator();
        if cfg!(target_os = "windows") {
            assert_eq!(sep, '\\');
        } else {
            assert_eq!(sep, '/');
        }
    }

    #[test]
    fn platform_detection_consistent() {
        // At least one platform family should be true
        let any = is_windows() || is_unix() || is_wasm();
        assert!(any, "should detect at least one platform family");
    }

    #[test]
    fn temp_dir_string_is_nonempty() {
        assert!(!temp_dir_string().is_empty());
    }

    #[test]
    fn temp_subdir_string_contains_subdir() {
        let s = temp_subdir_string("scirs2_test");
        assert!(s.contains("scirs2_test"));
    }
}
