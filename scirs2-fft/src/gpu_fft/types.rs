//! Types for the GPU-accelerated FFT pipeline.
//!
//! Provides configuration, error, and result types used throughout
//! the `gpu_fft` module.  The implementation is a pure-Rust simulation
//! of a tile-based GPU FFT pipeline; no actual GPU calls are made.

use scirs2_core::numeric::Complex64;

// ─────────────────────────────────────────────────────────────────────────────
// Normalization mode
// ─────────────────────────────────────────────────────────────────────────────

/// How to normalise after an FFT.
///
/// * `None`     – raw DFT output (no scaling)
/// * `Forward`  – multiply by `1/N` after the forward transform
/// * `Backward` – multiply by `1/N` after the inverse transform (SciPy default)
/// * `Ortho`    – multiply by `1/√N` in both directions (unitary)
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormalizationMode {
    /// No normalisation.
    #[default]
    None,
    /// Scale by `1/N` (applied to the *forward* transform).
    Forward,
    /// Scale by `1/N` (applied to the *inverse* transform; SciPy default).
    Backward,
    /// Scale by `1/√N` so the DFT matrix is unitary.
    Ortho,
}

// ─────────────────────────────────────────────────────────────────────────────
// Transform direction
// ─────────────────────────────────────────────────────────────────────────────

/// Direction of the FFT.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftDirection {
    /// Forward DFT (negative-exponent convention).
    Forward,
    /// Inverse DFT (positive-exponent convention).
    Inverse,
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the GPU FFT pipeline.
///
/// All fields have sensible defaults via [`Default`].
#[derive(Debug, Clone)]
pub struct GpuFftConfig {
    /// Number of points processed per simulated GPU tile (default: 256).
    pub tile_size: usize,
    /// Number of independent transforms processed simultaneously (default: 8).
    pub batch_size: usize,
    /// Simulate shared-memory tiling optimisation (default: true).
    pub use_shared_memory: bool,
    /// Normalisation applied after each transform (default: `None`).
    pub normalization: NormalizationMode,
}

impl Default for GpuFftConfig {
    fn default() -> Self {
        Self {
            tile_size: 256,
            batch_size: 8,
            use_shared_memory: true,
            normalization: NormalizationMode::None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise in the GPU FFT pipeline.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum GpuFftError {
    /// FFT size is zero or below the minimum supported size.
    #[error("FFT size {0} is too small (minimum is 2)")]
    SizeTooSmall(usize),
    /// Cooley-Tukey path requires a power-of-two input but a non-power-of-two
    /// was given *without* Bluestein fallback.
    #[error("FFT size {0} is not a power of two; use bluestein_gpu for arbitrary sizes")]
    NonPowerOfTwo(usize),
    /// A batch of zero elements was submitted.
    #[error("Batch is empty; provide at least one signal")]
    BatchEmpty,
    /// A memory allocation would have failed.
    #[error("GPU buffer allocation failed for {0} bytes")]
    AllocationFailed(usize),
    /// An internal kernel launch encountered an error.
    #[error("Kernel launch failed: {0}")]
    KernelLaunchFailed(String),
    /// The output length requested for C2R is inconsistent.
    #[error("C2R output length {requested} is inconsistent with input length {input_len}")]
    InvalidOutputLength {
        /// Requested output length.
        requested: usize,
        /// Input (complex) length that constrains valid choices.
        input_len: usize,
    },
}

/// Convenience `Result` alias for the GPU FFT pipeline.
pub type GpuFftResult<T> = Result<T, GpuFftError>;

// ─────────────────────────────────────────────────────────────────────────────
// Plan
// ─────────────────────────────────────────────────────────────────────────────

/// A compiled FFT plan that caches twiddle factors.
///
/// Create plans through [`crate::gpu_fft::pipeline::GpuFftPipeline::plan`]
/// rather than constructing this directly.
#[derive(Debug, Clone)]
pub struct GpuFftPlan {
    /// Transform size (number of complex points).
    pub size: usize,
    /// Direction encoded in the plan.
    pub direction: FftDirection,
    /// Configuration snapshot used when the plan was compiled.
    pub config: GpuFftConfig,
    /// Precomputed twiddle factors `W_N^k = exp(-2πi·k/N)` for `k = 0..N/2`.
    pub twiddle_cache: Vec<Complex64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch result
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a batched GPU FFT execution.
#[derive(Debug, Clone)]
pub struct BatchFftResult {
    /// Per-signal output spectra, each of length equal to the input.
    pub outputs: Vec<Vec<Complex64>>,
    /// Wall-clock duration of the (simulated) kernel in nanoseconds.
    pub elapsed_ns: u64,
}
