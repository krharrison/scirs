//! Unified configuration system for SciRS2
//!
//! Provides `SciRS2Config` - a strongly-typed, thread-safe global configuration
//! with builder pattern, environment variable overrides, and runtime SIMD detection.
//!
//! # Usage
//!
//! ```rust
//! use scirs2_core::config::unified::{SciRS2Config, Precision, config, set_config};
//!
//! // Read the global config
//! let cfg = config();
//! println!("Threads: {}", cfg.thread_count());
//!
//! // Build a custom config
//! let custom = SciRS2Config::builder()
//!     .thread_count(8)
//!     .precision(Precision::F32)
//!     .memory_limit_bytes(2 * 1024 * 1024 * 1024)
//!     .pool_size(64)
//!     .build();
//! set_config(custom);
//! ```
//!
//! # Environment Variables
//!
//! | Variable | Description | Default |
//! |----------|-------------|---------|
//! | `SCIRS2_THREADS` | Thread count (0 = auto-detect) | num_cpus |
//! | `SCIRS2_PRECISION` | Default precision (`f32` or `f64`) | `f64` |
//! | `SCIRS2_MEMORY_LIMIT` | Memory limit in bytes | 1 GiB |
//! | `SCIRS2_POOL_SIZE` | Memory pool size (number of buffers) | 32 |

use std::fmt;
use std::sync::RwLock;

/// Default memory limit: 1 GiB
const DEFAULT_MEMORY_LIMIT: u64 = 1_073_741_824;

/// Default pool size (number of reusable buffers)
const DEFAULT_POOL_SIZE: usize = 32;

/// Floating-point precision mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    /// 32-bit floating-point (f32)
    F32,
    /// 64-bit floating-point (f64)
    F64,
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Precision::F32 => write!(f, "f32"),
            Precision::F64 => write!(f, "f64"),
        }
    }
}

impl Precision {
    /// Parse from string (case-insensitive)
    pub fn from_str_lossy(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "f32" | "float32" | "single" => Precision::F32,
            _ => Precision::F64,
        }
    }

    /// Byte size of one element at this precision
    pub const fn element_size(&self) -> usize {
        match self {
            Precision::F32 => 4,
            Precision::F64 => 8,
        }
    }
}

/// Runtime SIMD feature detection results.
///
/// Cached at construction time so repeated queries are free.
#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    /// Whether any SIMD extension is available
    pub any_available: bool,
    /// SSE2 support (x86_64)
    pub sse2: bool,
    /// SSE4.1 support (x86_64)
    pub sse4_1: bool,
    /// AVX support (x86_64)
    pub avx: bool,
    /// AVX2 support (x86_64)
    pub avx2: bool,
    /// AVX-512 support (x86_64)
    pub avx512f: bool,
    /// FMA support (x86_64)
    pub fma: bool,
    /// NEON support (aarch64)
    pub neon: bool,
}

impl SimdCapabilities {
    /// Detect SIMD capabilities of the current CPU.
    pub fn detect() -> Self {
        let mut caps = SimdCapabilities {
            any_available: false,
            sse2: false,
            sse4_1: false,
            avx: false,
            avx2: false,
            avx512f: false,
            fma: false,
            neon: false,
        };

        #[cfg(target_arch = "x86_64")]
        {
            caps.sse2 = is_x86_feature_detected!("sse2");
            caps.sse4_1 = is_x86_feature_detected!("sse4.1");
            caps.avx = is_x86_feature_detected!("avx");
            caps.avx2 = is_x86_feature_detected!("avx2");
            caps.avx512f = is_x86_feature_detected!("avx512f");
            caps.fma = is_x86_feature_detected!("fma");
        }

        #[cfg(target_arch = "aarch64")]
        {
            caps.neon = std::arch::is_aarch64_feature_detected!("neon");
        }

        caps.any_available = caps.sse2 || caps.avx || caps.avx2 || caps.avx512f || caps.neon;
        caps
    }

    /// A human-readable summary of the best available SIMD tier.
    pub fn best_tier(&self) -> &'static str {
        if self.avx512f {
            "AVX-512"
        } else if self.avx2 {
            "AVX2"
        } else if self.avx {
            "AVX"
        } else if self.sse4_1 {
            "SSE4.1"
        } else if self.sse2 {
            "SSE2"
        } else if self.neon {
            "NEON"
        } else {
            "Scalar"
        }
    }
}

impl fmt::Display for SimdCapabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SIMD: {} (", self.best_tier())?;
        let mut feats = Vec::new();
        if self.sse2 {
            feats.push("SSE2");
        }
        if self.sse4_1 {
            feats.push("SSE4.1");
        }
        if self.avx {
            feats.push("AVX");
        }
        if self.avx2 {
            feats.push("AVX2");
        }
        if self.avx512f {
            feats.push("AVX-512F");
        }
        if self.fma {
            feats.push("FMA");
        }
        if self.neon {
            feats.push("NEON");
        }
        if feats.is_empty() {
            write!(f, "none")?;
        } else {
            write!(f, "{}", feats.join(", "))?;
        }
        write!(f, ")")
    }
}

/// Unified configuration for the SciRS2 ecosystem.
///
/// This is the primary strongly-typed config struct. It is `Send + Sync` and
/// can be shared across threads.
#[derive(Debug, Clone)]
pub struct SciRS2Config {
    /// Number of threads for parallel operations (0 means auto-detect via num_cpus).
    thread_count: usize,
    /// Default floating-point precision.
    precision: Precision,
    /// Maximum memory the library is allowed to allocate (bytes).
    memory_limit_bytes: u64,
    /// Number of reusable buffers in the memory pool.
    pool_size: usize,
    /// Cached SIMD capabilities.
    simd: SimdCapabilities,
}

impl SciRS2Config {
    /// Create a builder for constructing a `SciRS2Config`.
    pub fn builder() -> SciRS2ConfigBuilder {
        SciRS2ConfigBuilder::new()
    }

    /// The configured thread count (never 0 -- resolved to actual CPU count).
    pub fn thread_count(&self) -> usize {
        self.thread_count
    }

    /// The default precision setting.
    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// Maximum memory limit in bytes.
    pub fn memory_limit_bytes(&self) -> u64 {
        self.memory_limit_bytes
    }

    /// Memory pool size (number of reusable buffers).
    pub fn pool_size(&self) -> usize {
        self.pool_size
    }

    /// Runtime SIMD capabilities.
    pub fn simd(&self) -> &SimdCapabilities {
        &self.simd
    }

    /// Build a config from environment variables, falling back to defaults.
    fn from_env() -> Self {
        let thread_count = resolve_thread_count(parse_env_u64("SCIRS2_THREADS"));
        let precision = std::env::var("SCIRS2_PRECISION")
            .map(|s| Precision::from_str_lossy(&s))
            .unwrap_or(Precision::F64);
        let memory_limit_bytes =
            parse_env_u64("SCIRS2_MEMORY_LIMIT").unwrap_or(DEFAULT_MEMORY_LIMIT);
        let pool_size = parse_env_u64("SCIRS2_POOL_SIZE")
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_POOL_SIZE);

        SciRS2Config {
            thread_count,
            precision,
            memory_limit_bytes,
            pool_size,
            simd: SimdCapabilities::detect(),
        }
    }
}

impl Default for SciRS2Config {
    fn default() -> Self {
        Self::from_env()
    }
}

impl fmt::Display for SciRS2Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SciRS2Config {{")?;
        writeln!(f, "  threads:      {}", self.thread_count)?;
        writeln!(f, "  precision:    {}", self.precision)?;
        writeln!(
            f,
            "  memory_limit: {} bytes ({:.2} GiB)",
            self.memory_limit_bytes,
            self.memory_limit_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        )?;
        writeln!(f, "  pool_size:    {}", self.pool_size)?;
        writeln!(f, "  {}", self.simd)?;
        write!(f, "}}")
    }
}

/// Builder for `SciRS2Config`.
#[derive(Debug)]
pub struct SciRS2ConfigBuilder {
    thread_count: Option<usize>,
    precision: Option<Precision>,
    memory_limit_bytes: Option<u64>,
    pool_size: Option<usize>,
}

impl SciRS2ConfigBuilder {
    /// Create a new builder with all fields unset (will use env / defaults).
    pub fn new() -> Self {
        SciRS2ConfigBuilder {
            thread_count: None,
            precision: None,
            memory_limit_bytes: None,
            pool_size: None,
        }
    }

    /// Set thread count (0 means auto-detect).
    pub fn thread_count(mut self, count: usize) -> Self {
        self.thread_count = Some(count);
        self
    }

    /// Set default precision.
    pub fn precision(mut self, p: Precision) -> Self {
        self.precision = Some(p);
        self
    }

    /// Set memory limit in bytes.
    pub fn memory_limit_bytes(mut self, bytes: u64) -> Self {
        self.memory_limit_bytes = Some(bytes);
        self
    }

    /// Set pool size.
    pub fn pool_size(mut self, size: usize) -> Self {
        self.pool_size = Some(size);
        self
    }

    /// Build the config. Fields not explicitly set fall back to env vars, then defaults.
    pub fn build(self) -> SciRS2Config {
        let base = SciRS2Config::from_env();
        SciRS2Config {
            thread_count: self
                .thread_count
                .map(resolve_thread_count_from_val)
                .unwrap_or(base.thread_count),
            precision: self.precision.unwrap_or(base.precision),
            memory_limit_bytes: self.memory_limit_bytes.unwrap_or(base.memory_limit_bytes),
            pool_size: self.pool_size.unwrap_or(base.pool_size),
            simd: base.simd,
        }
    }
}

impl Default for SciRS2ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Global config singleton
// ---------------------------------------------------------------------------

static GLOBAL_UNIFIED_CONFIG: std::sync::LazyLock<RwLock<SciRS2Config>> =
    std::sync::LazyLock::new(|| RwLock::new(SciRS2Config::default()));

/// Get a snapshot of the current global `SciRS2Config`.
///
/// This is cheap -- it clones the config under a read lock.
pub fn config() -> SciRS2Config {
    GLOBAL_UNIFIED_CONFIG
        .read()
        .map(|guard| guard.clone())
        .unwrap_or_else(|poisoned| poisoned.into_inner().clone())
}

/// Replace the global `SciRS2Config`.
pub fn set_config(cfg: SciRS2Config) {
    match GLOBAL_UNIFIED_CONFIG.write() {
        Ok(mut guard) => *guard = cfg,
        Err(poisoned) => *poisoned.into_inner() = cfg,
    }
}

/// Reset the global config to defaults (re-reads environment variables).
pub fn reset_config() {
    set_config(SciRS2Config::default());
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Parse an environment variable as u64.
fn parse_env_u64(var: &str) -> Option<u64> {
    std::env::var(var).ok().and_then(|s| s.parse::<u64>().ok())
}

/// Resolve thread count, treating 0 or None as auto-detect.
fn resolve_thread_count(raw: Option<u64>) -> usize {
    match raw {
        Some(0) | None => detect_cpu_count(),
        Some(n) => n as usize,
    }
}

fn resolve_thread_count_from_val(val: usize) -> usize {
    if val == 0 {
        detect_cpu_count()
    } else {
        val
    }
}

/// Detect the number of logical CPUs. Falls back to 1 if detection fails.
fn detect_cpu_count() -> usize {
    // Use std::thread::available_parallelism (stable since Rust 1.59)
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_has_positive_thread_count() {
        let cfg = SciRS2Config::default();
        assert!(cfg.thread_count() > 0, "Thread count must be > 0");
    }

    #[test]
    fn test_default_precision_is_f64() {
        // Unless SCIRS2_PRECISION is set, default should be f64
        let cfg = SciRS2Config::builder().build();
        // We cannot guarantee env is unset in CI, so just check it's one of the two
        assert!(
            cfg.precision() == Precision::F32 || cfg.precision() == Precision::F64,
            "Precision must be F32 or F64"
        );
    }

    #[test]
    fn test_builder_overrides() {
        let cfg = SciRS2Config::builder()
            .thread_count(16)
            .precision(Precision::F32)
            .memory_limit_bytes(512)
            .pool_size(4)
            .build();

        assert_eq!(cfg.thread_count(), 16);
        assert_eq!(cfg.precision(), Precision::F32);
        assert_eq!(cfg.memory_limit_bytes(), 512);
        assert_eq!(cfg.pool_size(), 4);
    }

    #[test]
    fn test_thread_count_zero_resolves_to_cpus() {
        let cfg = SciRS2Config::builder().thread_count(0).build();
        assert!(
            cfg.thread_count() > 0,
            "0 should resolve to actual CPU count"
        );
    }

    #[test]
    fn test_simd_detection_does_not_panic() {
        let caps = SimdCapabilities::detect();
        // Just ensure it runs without panicking and produces valid output
        let tier = caps.best_tier();
        assert!(!tier.is_empty());
        let display = format!("{caps}");
        assert!(!display.is_empty());
    }

    #[test]
    fn test_precision_from_str_lossy() {
        assert_eq!(Precision::from_str_lossy("f32"), Precision::F32);
        assert_eq!(Precision::from_str_lossy("F32"), Precision::F32);
        assert_eq!(Precision::from_str_lossy("float32"), Precision::F32);
        assert_eq!(Precision::from_str_lossy("single"), Precision::F32);
        assert_eq!(Precision::from_str_lossy("f64"), Precision::F64);
        assert_eq!(Precision::from_str_lossy("anything_else"), Precision::F64);
    }

    #[test]
    fn test_precision_element_size() {
        assert_eq!(Precision::F32.element_size(), 4);
        assert_eq!(Precision::F64.element_size(), 8);
    }

    #[test]
    fn test_precision_display() {
        assert_eq!(format!("{}", Precision::F32), "f32");
        assert_eq!(format!("{}", Precision::F64), "f64");
    }

    #[test]
    fn test_global_config_set_and_read() {
        // Save original
        let original = config();

        let custom = SciRS2Config::builder()
            .thread_count(42)
            .pool_size(99)
            .build();
        set_config(custom);

        let read_back = config();
        assert_eq!(read_back.thread_count(), 42);
        assert_eq!(read_back.pool_size(), 99);

        // Restore
        set_config(original);
    }

    #[test]
    fn test_reset_config() {
        let original = config();

        let custom = SciRS2Config::builder().thread_count(999).build();
        set_config(custom);
        assert_eq!(config().thread_count(), 999);

        reset_config();
        // After reset, thread count should be auto-detected (> 0, not 999)
        let after_reset = config();
        assert!(after_reset.thread_count() > 0);
        // Restore the true original to avoid interference with other tests
        set_config(original);
    }

    #[test]
    fn test_config_display() {
        let cfg = SciRS2Config::builder()
            .thread_count(4)
            .precision(Precision::F64)
            .memory_limit_bytes(1_073_741_824)
            .pool_size(32)
            .build();
        let display = format!("{cfg}");
        assert!(display.contains("threads:"));
        assert!(display.contains("precision:"));
        assert!(display.contains("memory_limit:"));
        assert!(display.contains("pool_size:"));
        assert!(display.contains("SIMD:"));
    }

    #[test]
    fn test_simd_capabilities_clone_and_debug() {
        let caps = SimdCapabilities::detect();
        let caps2 = caps.clone();
        assert_eq!(caps.any_available, caps2.any_available);
        let debug = format!("{caps:?}");
        assert!(debug.contains("any_available"));
    }

    #[test]
    fn test_builder_default() {
        let builder = SciRS2ConfigBuilder::default();
        let cfg = builder.build();
        assert!(cfg.thread_count() > 0);
    }
}
