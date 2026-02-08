//! Cross-platform memory detection and system information.
//!
//! This module provides platform-agnostic access to system memory information
//! across Linux, Windows, macOS, FreeBSD, and OpenBSD platforms.

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use std::io::{BufRead, BufReader};

/// Platform-specific memory information
#[derive(Debug, Clone, Copy)]
pub struct PlatformMemoryInfo {
    /// Total physical memory in bytes
    pub total_memory: usize,
    /// Available memory in bytes (free + buffers + cache on Linux)
    pub available_memory: usize,
}

impl PlatformMemoryInfo {
    /// Detect system memory information on the current platform
    ///
    /// # Returns
    ///
    /// Returns `Some(PlatformMemoryInfo)` if memory detection succeeds,
    /// `None` if the platform is unsupported or detection fails.
    ///
    /// # Platform Support
    ///
    /// - **Linux**: Parses `/proc/meminfo` using BufReader
    /// - **Windows**: Uses `GlobalMemoryStatusEx` from `windows-sys`
    /// - **macOS**: Uses `sysctl` for `hw.memsize` + parses `vm_stat`
    /// - **FreeBSD/OpenBSD**: Uses `sysctl` for `hw.physmem`
    pub fn detect() -> Option<Self> {
        #[cfg(target_os = "linux")]
        {
            Self::detect_linux()
        }

        #[cfg(target_os = "windows")]
        {
            Self::detect_windows()
        }

        #[cfg(target_os = "macos")]
        {
            Self::detect_macos()
        }

        #[cfg(any(target_os = "freebsd", target_os = "openbsd"))]
        {
            Self::detect_bsd()
        }

        #[cfg(not(any(
            target_os = "linux",
            target_os = "windows",
            target_os = "macos",
            target_os = "freebsd",
            target_os = "openbsd"
        )))]
        {
            None
        }
    }

    /// Detect memory on Linux by parsing /proc/meminfo
    #[cfg(target_os = "linux")]
    fn detect_linux() -> Option<Self> {
        use std::fs::File;

        let file = File::open("/proc/meminfo").ok()?;
        let reader = BufReader::new(file);

        let mut total_memory = None;
        let mut available_memory = None;

        for line in reader.lines() {
            let line = line.ok()?;
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.len() < 2 {
                continue;
            }

            match parts[0] {
                "MemTotal:" => {
                    if let Ok(kb) = parts[1].parse::<usize>() {
                        total_memory = Some(kb * 1024); // Convert KB to bytes
                    }
                }
                "MemAvailable:" => {
                    if let Ok(kb) = parts[1].parse::<usize>() {
                        available_memory = Some(kb * 1024); // Convert KB to bytes
                    }
                }
                _ => {}
            }

            // Early exit if we have both values
            if total_memory.is_some() && available_memory.is_some() {
                break;
            }
        }

        // Fallback: If MemAvailable is not present (older kernels), use MemFree
        if available_memory.is_none() {
            let file = File::open("/proc/meminfo").ok()?;
            let reader = BufReader::new(file);

            for line in reader.lines() {
                let line = line.ok()?;
                let parts: Vec<&str> = line.split_whitespace().collect();

                if parts.len() >= 2 && parts[0] == "MemFree:" {
                    if let Ok(kb) = parts[1].parse::<usize>() {
                        available_memory = Some(kb * 1024);
                        break;
                    }
                }
            }
        }

        Some(Self {
            total_memory: total_memory?,
            available_memory: available_memory?,
        })
    }

    /// Detect memory on Windows using GlobalMemoryStatus
    /// Note: Using GlobalMemoryStatus instead of GlobalMemoryStatusEx for windows-sys 0.52 compatibility
    #[cfg(target_os = "windows")]
    fn detect_windows() -> Option<Self> {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatus, MEMORYSTATUS};

        let mut mem_status = MEMORYSTATUS {
            dwLength: std::mem::size_of::<MEMORYSTATUS>() as u32,
            dwMemoryLoad: 0,
            dwTotalPhys: 0,
            dwAvailPhys: 0,
            dwTotalPageFile: 0,
            dwAvailPageFile: 0,
            dwTotalVirtual: 0,
            dwAvailVirtual: 0,
        };

        unsafe {
            GlobalMemoryStatus(&mut mem_status);
            Some(Self {
                total_memory: mem_status.dwTotalPhys as usize,
                available_memory: mem_status.dwAvailPhys as usize,
            })
        }
    }

    /// Detect memory on macOS using sysctl
    #[cfg(target_os = "macos")]
    fn detect_macos() -> Option<Self> {
        use std::process::Command;

        // Get total memory using sysctl hw.memsize
        let total_output = Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;

        let total_memory = String::from_utf8(total_output.stdout)
            .ok()?
            .trim()
            .parse::<usize>()
            .ok()?;

        // Get available memory using vm_stat
        let vm_output = Command::new("vm_stat").output().ok()?;

        let vm_str = String::from_utf8(vm_output.stdout).ok()?;

        // Parse vm_stat output to get free pages
        // Format: "Pages free:                               12345."
        let mut free_pages = 0usize;
        let mut inactive_pages = 0usize;

        for line in vm_str.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }

            match parts[0] {
                "Pages" if parts.len() > 2 => match parts[1] {
                    "free:" => {
                        if let Some(value) = parts.last() {
                            let cleaned = value.trim_end_matches('.');
                            free_pages = cleaned.parse().unwrap_or(0);
                        }
                    }
                    "inactive:" => {
                        if let Some(value) = parts.last() {
                            let cleaned = value.trim_end_matches('.');
                            inactive_pages = cleaned.parse().unwrap_or(0);
                        }
                    }
                    _ => {}
                },
                _ => {}
            }
        }

        // macOS page size is typically 4096 bytes
        let page_size = 4096usize;
        let available_memory = (free_pages + inactive_pages) * page_size;

        Some(Self {
            total_memory,
            available_memory,
        })
    }

    /// Detect memory on FreeBSD/OpenBSD using sysctl
    #[cfg(any(target_os = "freebsd", target_os = "openbsd"))]
    fn detect_bsd() -> Option<Self> {
        use std::process::Command;

        // Get total physical memory
        let total_output = Command::new("sysctl")
            .args(["-n", "hw.physmem"])
            .output()
            .ok()?;

        let total_memory = String::from_utf8(total_output.stdout)
            .ok()?
            .trim()
            .parse::<usize>()
            .ok()?;

        // Get available memory (free + inactive + cache)
        // On FreeBSD: vm.stats.vm.v_free_count * page_size
        let free_output = Command::new("sysctl")
            .args(["-n", "vm.stats.vm.v_free_count"])
            .output()
            .ok();

        let inactive_output = Command::new("sysctl")
            .args(["-n", "vm.stats.vm.v_inactive_count"])
            .output()
            .ok();

        let page_size_output = Command::new("sysctl")
            .args(["-n", "hw.pagesize"])
            .output()
            .ok()?;

        let page_size = String::from_utf8(page_size_output.stdout)
            .ok()?
            .trim()
            .parse::<usize>()
            .ok()?;

        let mut free_pages = 0usize;
        let mut inactive_pages = 0usize;

        if let Some(output) = free_output {
            if let Ok(s) = String::from_utf8(output.stdout) {
                free_pages = s.trim().parse().unwrap_or(0);
            }
        }

        if let Some(output) = inactive_output {
            if let Ok(s) = String::from_utf8(output.stdout) {
                inactive_pages = s.trim().parse().unwrap_or(0);
            }
        }

        let available_memory = (free_pages + inactive_pages) * page_size;

        // Fallback: Use 50% of total if we couldn't get precise info
        let available_memory = if available_memory == 0 {
            total_memory / 2
        } else {
            available_memory
        };

        Some(Self {
            total_memory,
            available_memory,
        })
    }

    /// Get the total physical memory in bytes
    pub const fn total(&self) -> usize {
        self.total_memory
    }

    /// Get the available memory in bytes
    pub const fn available(&self) -> usize {
        self.available_memory
    }

    /// Get the percentage of memory that is available (0.0 to 1.0)
    pub fn available_fraction(&self) -> f64 {
        if self.total_memory == 0 {
            0.0
        } else {
            self.available_memory as f64 / self.total_memory as f64
        }
    }

    /// Validate that the memory info is reasonable
    pub fn validate(&self) -> CoreResult<()> {
        if self.total_memory == 0 {
            return Err(CoreError::ValidationError(
                ErrorContext::new("Total memory cannot be zero".to_string())
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        if self.available_memory > self.total_memory {
            return Err(CoreError::ValidationError(
                ErrorContext::new(format!(
                    "Available memory ({}) exceeds total memory ({})",
                    self.available_memory, self.total_memory
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_memory_detection() {
        // Test that memory detection works on the current platform
        let mem_info = PlatformMemoryInfo::detect();

        if let Some(info) = mem_info {
            println!("Total memory: {} bytes", info.total_memory);
            println!("Available memory: {} bytes", info.available_memory);
            println!(
                "Available fraction: {:.2}%",
                info.available_fraction() * 100.0
            );

            // Validate the detected memory info
            assert!(info.validate().is_ok());

            // Basic sanity checks
            assert!(info.total_memory > 0, "Total memory should be positive");
            assert!(
                info.available_memory <= info.total_memory,
                "Available memory should not exceed total"
            );
            assert!(
                info.available_fraction() >= 0.0 && info.available_fraction() <= 1.0,
                "Available fraction should be between 0 and 1"
            );
        } else {
            println!("Memory detection not supported on this platform (or failed to detect)");
        }
    }

    #[test]
    fn test_memory_info_accessors() {
        let info = PlatformMemoryInfo {
            total_memory: 8 * 1024 * 1024 * 1024,     // 8 GB
            available_memory: 4 * 1024 * 1024 * 1024, // 4 GB
        };

        assert_eq!(info.total(), 8 * 1024 * 1024 * 1024);
        assert_eq!(info.available(), 4 * 1024 * 1024 * 1024);
        assert!((info.available_fraction() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_memory_info_validation() {
        // Valid case
        let valid = PlatformMemoryInfo {
            total_memory: 1024 * 1024,
            available_memory: 512 * 1024,
        };
        assert!(valid.validate().is_ok());

        // Invalid: zero total
        let zero_total = PlatformMemoryInfo {
            total_memory: 0,
            available_memory: 0,
        };
        assert!(zero_total.validate().is_err());

        // Invalid: available > total
        let invalid_available = PlatformMemoryInfo {
            total_memory: 1024,
            available_memory: 2048,
        };
        assert!(invalid_available.validate().is_err());
    }
}
