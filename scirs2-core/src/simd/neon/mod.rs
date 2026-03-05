//! ARM NEON and SVE SIMD optimisations
//!
//! This module provides ARM-specific SIMD implementations organised into
//! focused sub-modules:
//!
//! ## Core mobile operations
//! - [`basic`]: Element-wise arithmetic (add, sub, mul, div) for f32/f64
//! - [`matrix`]: GEMM / GEMV for f32/f64 (mobile-tuned block sizes)
//! - [`activation`]: Neural network activations (ReLU, LeakyReLU, Sigmoid, Tanh, GELU)
//! - [`mobile`]: Battery / thermal-aware wrappers for mobile devices
//!
//! ## Extended arithmetic
//! - [`fma`]: Fused multiply-add, scalar scale, abs, neg for f32/f64
//! - [`reductions`]: Horizontal sum, max, min for f32/f64
//!
//! ## Server / high-performance operations
//! - [`server`]: Four-way unrolled add and dot, cache-blocked 2-D matmul,
//!   batch normalisation for inference — tuned for server L2/L3 caches and
//!   wide out-of-order pipelines (Ampere Altra, AWS Graviton, Apple Silicon).
//!
//! ## Quantised inference
//! - [`i8_ops`]: i8 dot products with i32 accumulation, f32↔i8 quantisation /
//!   dequantisation, saturating i8 addition.
//!
//! ## ARM Scalable Vector Extension
//! - [`sve`]: Runtime SVE / SVE2 capability detection, VLEN query via `rdvl`,
//!   and element-wise operations that dispatch through SVE → NEON → scalar.
//!
//! ## Architecture Support
//!
//! | Module      | AArch64 (NEON) | AArch64 (SVE) | AArch32 (NEON) | Other |
//! |-------------|:--------------:|:-------------:|:--------------:|:-----:|
//! | basic       | ✓              | —             | ✓              | scalar|
//! | matrix      | ✓              | —             | ✓              | scalar|
//! | activation  | ✓              | —             | ✓              | scalar|
//! | mobile      | ✓              | —             | ✓              | scalar|
//! | fma         | ✓              | —             | fallback       | scalar|
//! | reductions  | ✓              | —             | fallback       | scalar|
//! | server      | ✓ (unrolled)   | —             | fallback       | scalar|
//! | i8_ops      | ✓              | —             | fallback       | scalar|
//! | sve         | NEON fallback  | ✓             | NEON fallback  | scalar|

pub mod activation;
pub mod basic;
pub mod fma;
pub mod i8_ops;
pub mod matrix;
pub mod mobile;
pub mod reductions;
pub mod server;
pub mod sve;

// ============================================================
// Re-exports — basic arithmetic
// ============================================================

pub use basic::{
    neon_add_f32, neon_add_f64, neon_div_f32, neon_dot_f32, neon_dot_f64, neon_mul_f32,
    neon_mul_f64, neon_sub_f32, neon_sub_f64,
};

// ============================================================
// Re-exports — matrix operations
// ============================================================

pub use matrix::{neon_gemm_f32, neon_gemm_f64, neon_gemv_f32, neon_gemv_f64};

// ============================================================
// Re-exports — activation functions
// ============================================================

pub use activation::{
    neon_gelu_f32, neon_leaky_relu_f32, neon_relu_f32, neon_sigmoid_f32, neon_tanh_f32,
};

// ============================================================
// Re-exports — mobile-optimised variants
// ============================================================

pub use mobile::{
    neon_dot_battery_optimized, neon_gemm_battery_optimized, neon_gemm_thermal_aware, BatteryMode,
    MobileOptimizer, ThermalState,
};

// ============================================================
// Re-exports — fused multiply-add and extended arithmetic
// ============================================================

pub use fma::{
    neon_abs_f32, neon_abs_f64, neon_fmadd_f32, neon_fmadd_f64, neon_neg_f32, neon_scale_f32,
    neon_scale_f64,
};

// ============================================================
// Re-exports — horizontal reductions
// ============================================================

pub use reductions::{
    neon_max_f32, neon_max_f64, neon_min_f32, neon_min_f64, neon_sum_f32, neon_sum_f64,
};

// ============================================================
// Re-exports — server / high-throughput operations
// ============================================================

pub use server::{
    neon_add_f32_unrolled, neon_batch_norm_f32_server, neon_dot_f32_unrolled,
    neon_dot_f64_unrolled, neon_matmul_f32,
};

// ============================================================
// Re-exports — quantised i8 inference operations
// ============================================================

pub use i8_ops::{
    neon_add_i8_saturating, neon_dequantize_i8_to_f32, neon_dot_i8, neon_dot_i8_i32acc,
    neon_quantize_f32_to_i8,
};

// ============================================================
// Re-exports — SVE capability detection
// ============================================================

pub use sve::{
    detect_sve_capabilities, has_sve, has_sve2, sve_add_f32, sve_dot_f32, sve_scale_f32,
    sve_sum_f32, SveCapabilities,
};

// ============================================================
// ARM SIMD capability summary
// ============================================================

/// Detailed SIMD capabilities for ARM processors.
///
/// Use [`detect_arm_capabilities`] to populate this at runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArmSimdCapabilities {
    /// NEON (ARM Advanced SIMD) is available.
    ///
    /// Always `true` on AArch64; may be `true` on ARMv7-A with NEON enabled.
    pub has_neon: bool,
    /// ARM Scalable Vector Extension (SVE) is available.
    pub has_sve: bool,
    /// ARM Scalable Vector Extension 2 (SVE2) is available.
    pub has_sve2: bool,
    /// Integer 8-bit dot-product extension (`UDOT`/`SDOT` instructions).
    ///
    /// Present on Cortex-A55, A76, A78, Neoverse N1/V1/V2, Apple M-series.
    pub has_dotprod: bool,
    /// BFloat16 (brain float) arithmetic extension.
    ///
    /// Present on Cortex-A78C, Neoverse V1/V2, Apple M-series (M2+).
    pub has_bf16: bool,
    /// Number of f32 elements processed per SIMD register width.
    ///
    /// - NEON only: 4 (128-bit)
    /// - SVE: `vector_len_bytes / 4`
    pub vector_width_f32: usize,
    /// Number of f64 elements processed per SIMD register width.
    pub vector_width_f64: usize,
    /// SVE hardware vector length in bytes (0 if SVE is unavailable).
    pub sve_vector_len_bytes: usize,
    /// Whether this appears to be an Apple Silicon processor.
    ///
    /// Detected heuristically: AArch64 + BF16 + dotprod + SVE2 (M4+).
    /// This field is a hint for selecting Apple-specific micro-kernel tuning.
    pub is_apple_silicon_hint: bool,
}

impl Default for ArmSimdCapabilities {
    fn default() -> Self {
        detect_arm_capabilities()
    }
}

/// Detect all ARM SIMD capabilities for the current CPU at runtime.
///
/// This function queries the available instruction-set extensions and
/// returns a populated [`ArmSimdCapabilities`] struct.
///
/// The function is inexpensive to call but not free; callers that invoke it
/// in hot loops should cache the result.
///
/// # Example
///
/// ```rust
/// use scirs2_core::simd::neon::detect_arm_capabilities;
///
/// let caps = detect_arm_capabilities();
/// if caps.has_neon {
///     println!("NEON is available ({} f32 lanes)", caps.vector_width_f32);
/// }
/// if caps.has_sve {
///     println!("SVE vector length: {} bytes", caps.sve_vector_len_bytes);
/// }
/// ```
pub fn detect_arm_capabilities() -> ArmSimdCapabilities {
    #[cfg(target_arch = "aarch64")]
    {
        let has_neon = std::arch::is_aarch64_feature_detected!("neon");
        let has_dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
        let has_bf16 = std::arch::is_aarch64_feature_detected!("bf16");

        let sve_caps = detect_sve_capabilities();
        let has_sve = sve_caps.has_sve;
        let has_sve2 = sve_caps.has_sve2;
        let sve_vector_len_bytes = sve_caps.vector_len_bytes;

        // Compute SIMD vector widths.
        let (vector_width_f32, vector_width_f64) = if has_sve && sve_vector_len_bytes > 0 {
            (sve_vector_len_bytes / 4, sve_vector_len_bytes / 8)
        } else if has_neon {
            (4, 2) // 128-bit NEON
        } else {
            (1, 1) // scalar
        };

        // Apple Silicon heuristic: AArch64 + bf16 + dotprod.
        // Actual Apple Silicon detection would require sysctl or OS API;
        // this is a compile-time-free runtime heuristic.
        let is_apple_silicon_hint = has_bf16 && has_dotprod;

        ArmSimdCapabilities {
            has_neon,
            has_sve,
            has_sve2,
            has_dotprod,
            has_bf16,
            vector_width_f32,
            vector_width_f64,
            sve_vector_len_bytes,
            is_apple_silicon_hint,
        }
    }
    #[cfg(target_arch = "arm")]
    {
        // ARMv7-A: NEON may be present (feature "neon" is always detected as true
        // on arm targets compiled with +neon).
        ArmSimdCapabilities {
            has_neon: true, // compile-time assumption for arm target
            has_sve: false,
            has_sve2: false,
            has_dotprod: false,
            has_bf16: false,
            vector_width_f32: 4,
            vector_width_f64: 2,
            sve_vector_len_bytes: 0,
            is_apple_silicon_hint: false,
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
    {
        ArmSimdCapabilities {
            has_neon: false,
            has_sve: false,
            has_sve2: false,
            has_dotprod: false,
            has_bf16: false,
            vector_width_f32: 1,
            vector_width_f64: 1,
            sve_vector_len_bytes: 0,
            is_apple_silicon_hint: false,
        }
    }
}

// ============================================================
// Convenience functions
// ============================================================

/// Returns `true` if NEON instructions are available on the current CPU.
///
/// On AArch64 this is detected at runtime. On non-ARM platforms this always
/// returns `false`.
#[inline]
pub fn has_neon() -> bool {
    is_neon_available()
}

/// Check if NEON is available at runtime.
///
/// This is identical to [`has_neon`]; it is kept for compatibility with
/// existing callers.
#[inline]
pub fn is_neon_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(target_arch = "arm")]
    {
        // On 32-bit ARM we assume the code was compiled with NEON enabled.
        true
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
    {
        false
    }
}

/// Return a summary of available ARM SIMD capabilities.
///
/// This is a thin wrapper around [`detect_arm_capabilities`] provided for
/// ergonomic use at the call site.
#[inline]
pub fn simd_capabilities() -> ArmSimdCapabilities {
    detect_arm_capabilities()
}

// ============================================================
// Module-level tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_detection() {
        let available = is_neon_available();
        #[cfg(target_arch = "aarch64")]
        {
            // On AArch64, NEON should always be available.
            assert!(available, "NEON should be available on AArch64");
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
        {
            assert!(!available, "NEON should not be available on non-ARM");
        }
    }

    #[test]
    fn test_has_neon_alias() {
        // has_neon() and is_neon_available() must agree.
        assert_eq!(has_neon(), is_neon_available());
    }

    #[test]
    fn test_simd_capabilities_returns_valid_struct() {
        let caps = simd_capabilities();
        // Vector widths must be at least 1 (scalar fallback).
        assert!(caps.vector_width_f32 >= 1);
        assert!(caps.vector_width_f64 >= 1);
        // SVE vector length must be 0 when SVE is absent.
        if !caps.has_sve {
            assert_eq!(caps.sve_vector_len_bytes, 0);
        }
        // SVE2 implies SVE.
        if caps.has_sve2 {
            assert!(caps.has_sve, "SVE2 implies SVE");
        }
    }

    #[test]
    fn test_detect_arm_capabilities_consistency() {
        let caps = detect_arm_capabilities();
        // NEON must be present for NEON-width vectors.
        if caps.vector_width_f32 >= 4 && !caps.has_sve {
            assert!(caps.has_neon, "4-wide f32 requires NEON");
        }
        // SVE vector length must be a multiple of 16 bytes when present.
        if caps.has_sve {
            assert!(
                caps.sve_vector_len_bytes >= 16,
                "SVE VLEN must be >= 16 bytes"
            );
            assert_eq!(
                caps.sve_vector_len_bytes % 16,
                0,
                "SVE VLEN must be a multiple of 16 bytes"
            );
        }
    }
}
