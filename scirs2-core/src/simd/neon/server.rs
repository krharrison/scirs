//! ARM server SIMD optimizations — high-throughput workloads on AArch64
//!
//! Unlike the mobile-oriented NEON code, this module targets server-class ARM
//! processors (Ampere Altra, AWS Graviton, Apple Silicon in server mode) where:
//!
//! - Power budget is ample
//! - Cache hierarchy is large (L2 ≥ 1 MB, L3 ≥ 16 MB)
//! - Out-of-order pipelines are wide (≥ 4 issue slots)
//! - NEON throughput benefits from software pipelining and 4-way unrolling
//!
//! ## Design Principles
//!
//! 1. **Four-way unrolling**: 16 f32 elements (or 8 f64) per loop iteration to
//!    keep all NEON functional units busy.
//! 2. **Software prefetch**: explicit `__prefetch` hints to tolerate DRAM latency
//!    on large working sets.
//! 3. **Cache-blocked matrix multiply**: the `neon_matmul_f32` entry point picks
//!    block sizes appropriate for server L2/L3 cache sizes.
//! 4. **Batch normalisation for inference**: computes `(x - mean) / sqrt(var + eps) * gamma + beta`
//!    entirely in NEON registers.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// Distance to prefetch ahead, in elements.  Chosen empirically for typical
// server DRAM latency (~100 ns) at 3 GHz with 128-bit NEON loads.
const PREFETCH_DISTANCE_F32: usize = 64;
const PREFETCH_DISTANCE_F64: usize = 32;

/// Emit a `prfm pldl1keep` hint for the given address.
///
/// Uses inline assembly so it works on stable Rust without the unstable
/// `stdarch_aarch64_prefetch` feature.  The instruction is a hint and is
/// legal to execute on any AArch64 processor.
///
/// # Safety
///
/// `ptr` need not point to valid memory — the processor is free to ignore
/// any prefetch hint.  Must only be called on AArch64.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn prefetch_read_data<T>(ptr: *const T) {
    core::arch::asm!(
        "prfm pldl1keep, [{ptr}]",
        ptr = in(reg) ptr,
        options(nostack, readonly)
    );
}

// ============================================================
// Four-way unrolled vector addition for f32
// ============================================================

/// NEON server-optimised vector addition: processes 16 f32 elements per cycle.
///
/// Uses four-way loop unrolling and software prefetch hints to maximise
/// throughput on wide out-of-order ARM server cores.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON. `a`, `b`, and `out` must be valid
/// for their respective lengths.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_add_f32_unrolled_impl(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len().min(b.len()).min(out.len());
    let mut i = 0;

    // Sixteen-element (4×128-bit register) unrolled loop with prefetch.
    while i + 16 <= len {
        // Prefetch data that will be needed in future iterations.
        let pf_idx = i + PREFETCH_DISTANCE_F32;
        if pf_idx < len {
            prefetch_read_data(a.as_ptr().add(pf_idx));
            prefetch_read_data(b.as_ptr().add(pf_idx));
        }

        let va0 = vld1q_f32(a.as_ptr().add(i));
        let va1 = vld1q_f32(a.as_ptr().add(i + 4));
        let va2 = vld1q_f32(a.as_ptr().add(i + 8));
        let va3 = vld1q_f32(a.as_ptr().add(i + 12));

        let vb0 = vld1q_f32(b.as_ptr().add(i));
        let vb1 = vld1q_f32(b.as_ptr().add(i + 4));
        let vb2 = vld1q_f32(b.as_ptr().add(i + 8));
        let vb3 = vld1q_f32(b.as_ptr().add(i + 12));

        vst1q_f32(out.as_mut_ptr().add(i), vaddq_f32(va0, vb0));
        vst1q_f32(out.as_mut_ptr().add(i + 4), vaddq_f32(va1, vb1));
        vst1q_f32(out.as_mut_ptr().add(i + 8), vaddq_f32(va2, vb2));
        vst1q_f32(out.as_mut_ptr().add(i + 12), vaddq_f32(va3, vb3));

        i += 16;
    }

    // Process any remaining 4-element chunks.
    while i + 4 <= len {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        vst1q_f32(out.as_mut_ptr().add(i), vaddq_f32(va, vb));
        i += 4;
    }

    // Scalar tail.
    while i < len {
        out[i] = a[i] + b[i];
        i += 1;
    }
}

/// Server-optimised vector addition for f32.
///
/// Uses four-way unrolled NEON with software prefetch on AArch64; falls back
/// to scalar on other architectures.
pub fn neon_add_f32_unrolled(a: &[f32], b: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_add_f32_unrolled_impl(a, b, out) }
            return;
        }
    }
    // Scalar fallback.
    let len = a.len().min(b.len()).min(out.len());
    for i in 0..len {
        out[i] = a[i] + b[i];
    }
}

// ============================================================
// Four-way unrolled dot product for f32
// ============================================================

/// NEON server-optimised dot product: four accumulator registers to fill the
/// multiply-accumulate pipeline.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dot_f32_unrolled_impl(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut i = 0;

    // Four accumulators to hide multiply-accumulate latency.
    let mut acc0 = vdupq_n_f32(0.0_f32);
    let mut acc1 = vdupq_n_f32(0.0_f32);
    let mut acc2 = vdupq_n_f32(0.0_f32);
    let mut acc3 = vdupq_n_f32(0.0_f32);

    while i + 16 <= len {
        let pf_idx = i + PREFETCH_DISTANCE_F32;
        if pf_idx < len {
            prefetch_read_data(a.as_ptr().add(pf_idx));
            prefetch_read_data(b.as_ptr().add(pf_idx));
        }

        acc0 = vfmaq_f32(
            acc0,
            vld1q_f32(a.as_ptr().add(i)),
            vld1q_f32(b.as_ptr().add(i)),
        );
        acc1 = vfmaq_f32(
            acc1,
            vld1q_f32(a.as_ptr().add(i + 4)),
            vld1q_f32(b.as_ptr().add(i + 4)),
        );
        acc2 = vfmaq_f32(
            acc2,
            vld1q_f32(a.as_ptr().add(i + 8)),
            vld1q_f32(b.as_ptr().add(i + 8)),
        );
        acc3 = vfmaq_f32(
            acc3,
            vld1q_f32(a.as_ptr().add(i + 12)),
            vld1q_f32(b.as_ptr().add(i + 12)),
        );

        i += 16;
    }

    // Merge accumulators.
    let mut acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));

    // 4-element remainder.
    while i + 4 <= len {
        acc = vfmaq_f32(
            acc,
            vld1q_f32(a.as_ptr().add(i)),
            vld1q_f32(b.as_ptr().add(i)),
        );
        i += 4;
    }

    let mut result = vaddvq_f32(acc);

    // Scalar tail.
    while i < len {
        result += a[i] * b[i];
        i += 1;
    }

    result
}

/// Server-optimised dot product for f32.
///
/// Four-way unrolled with software prefetch on AArch64.
pub fn neon_dot_f32_unrolled(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_dot_f32_unrolled_impl(a, b) };
        }
    }
    // Scalar fallback.
    let len = a.len().min(b.len());
    let mut sum = 0.0_f32;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

// ============================================================
// Four-way unrolled dot product for f64
// ============================================================

/// NEON server-optimised f64 dot product.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dot_f64_unrolled_impl(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    let mut i = 0;
    let mut acc0 = vdupq_n_f64(0.0_f64);
    let mut acc1 = vdupq_n_f64(0.0_f64);
    let mut acc2 = vdupq_n_f64(0.0_f64);
    let mut acc3 = vdupq_n_f64(0.0_f64);

    while i + 8 <= len {
        let pf_idx = i + PREFETCH_DISTANCE_F64;
        if pf_idx < len {
            prefetch_read_data(a.as_ptr().add(pf_idx));
            prefetch_read_data(b.as_ptr().add(pf_idx));
        }

        acc0 = vfmaq_f64(
            acc0,
            vld1q_f64(a.as_ptr().add(i)),
            vld1q_f64(b.as_ptr().add(i)),
        );
        acc1 = vfmaq_f64(
            acc1,
            vld1q_f64(a.as_ptr().add(i + 2)),
            vld1q_f64(b.as_ptr().add(i + 2)),
        );
        acc2 = vfmaq_f64(
            acc2,
            vld1q_f64(a.as_ptr().add(i + 4)),
            vld1q_f64(b.as_ptr().add(i + 4)),
        );
        acc3 = vfmaq_f64(
            acc3,
            vld1q_f64(a.as_ptr().add(i + 6)),
            vld1q_f64(b.as_ptr().add(i + 6)),
        );

        i += 8;
    }

    let mut acc = vaddq_f64(vaddq_f64(acc0, acc1), vaddq_f64(acc2, acc3));

    while i + 2 <= len {
        acc = vfmaq_f64(
            acc,
            vld1q_f64(a.as_ptr().add(i)),
            vld1q_f64(b.as_ptr().add(i)),
        );
        i += 2;
    }

    let mut result = vaddvq_f64(acc);

    while i < len {
        result += a[i] * b[i];
        i += 1;
    }

    result
}

/// Server-optimised dot product for f64.
pub fn neon_dot_f64_unrolled(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon_dot_f64_unrolled_impl(a, b) };
        }
    }
    let len = a.len().min(b.len());
    let mut sum = 0.0_f64;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

// ============================================================
// Simple 2D matrix multiply: C = A × B
// ============================================================

/// NEON cache-blocked 2-D matrix multiply for f32.
///
/// Computes `C = A × B` where:
/// - `A` is `a_rows × a_cols` stored in row-major order
/// - `B` is `a_cols × b_cols` stored in row-major order
/// - `C` is `a_rows × b_cols` stored in row-major order (written to `c`)
///
/// Block sizes are chosen for server-class L2 caches (≥ 1 MB):
/// - `MC = 128` rows at a time
/// - `NC = 512` columns at a time
/// - `KC = 256` reduction dimension at a time
///
/// The innermost kernel uses NEON FMA with 4-wide lanes.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_matmul_f32_impl(
    a: &[f32],
    a_rows: usize,
    a_cols: usize,
    b: &[f32],
    b_cols: usize,
    c: &mut [f32],
) {
    // Server-class block sizes (larger than mobile variant in matrix.rs).
    const MC: usize = 128;
    const NC: usize = 512;
    const KC: usize = 256;

    // Initialise C to zero.
    c.fill(0.0_f32);

    for ic in (0..a_rows).step_by(MC) {
        let mc = (ic + MC).min(a_rows) - ic;

        for pc in (0..a_cols).step_by(KC) {
            let kc = (pc + KC).min(a_cols) - pc;

            for jc in (0..b_cols).step_by(NC) {
                let nc = (jc + NC).min(b_cols) - jc;

                // Micro-kernel: i × j × p with NEON FMA.
                for i in 0..mc {
                    for j in 0..nc {
                        let c_idx = (ic + i) * b_cols + jc + j;
                        let mut acc = vdupq_n_f32(0.0_f32);
                        let mut p = 0;

                        while p + 4 <= kc {
                            let va = vld1q_f32(a.as_ptr().add((ic + i) * a_cols + pc + p));
                            let vb = vld1q_f32(b.as_ptr().add((pc + p) * b_cols + jc + j));
                            acc = vfmaq_f32(acc, va, vb);
                            p += 4;
                        }

                        let mut dot = vaddvq_f32(acc);

                        while p < kc {
                            dot += a[(ic + i) * a_cols + pc + p] * b[(pc + p) * b_cols + jc + j];
                            p += 1;
                        }

                        c[c_idx] += dot;
                    }
                }
            }
        }
    }
}

/// Simple 2-D matrix multiply for f32: `C = A × B`.
///
/// - `a`: row-major matrix of shape `a_rows × a_cols`
/// - `b`: row-major matrix of shape `a_cols × b_cols`
/// - `c`: output row-major matrix of shape `a_rows × b_cols` (zeroed on entry)
pub fn neon_matmul_f32(
    a: &[f32],
    a_rows: usize,
    a_cols: usize,
    b: &[f32],
    b_cols: usize,
    c: &mut [f32],
) {
    debug_assert_eq!(
        a.len(),
        a_rows * a_cols,
        "a slice length must be a_rows * a_cols"
    );
    debug_assert_eq!(
        b.len(),
        a_cols * b_cols,
        "b slice length must be a_cols * b_cols"
    );
    debug_assert_eq!(
        c.len(),
        a_rows * b_cols,
        "c slice length must be a_rows * b_cols"
    );

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_matmul_f32_impl(a, a_rows, a_cols, b, b_cols, c) }
            return;
        }
    }
    fallback_matmul_f32(a, a_rows, a_cols, b, b_cols, c)
}

// ============================================================
// Batch normalisation for inference (server-class)
// ============================================================

/// NEON batch normalisation for inference: `out[i] = (x[i] - mean) / sqrt(var + eps) * gamma + beta`.
///
/// Assumes per-channel scalar `mean`, `var`, `gamma`, `beta`.  This is the
/// "apply stats" path used at inference time after training has computed
/// per-batch statistics.
///
/// # Safety
///
/// Caller must ensure the CPU supports NEON.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_batch_norm_f32_server_impl(
    x: &[f32],
    mean: f32,
    var: f32,
    gamma: f32,
    beta: f32,
    eps: f32,
    out: &mut [f32],
) {
    let len = x.len().min(out.len());
    let mut i = 0;

    // Pre-compute scalar coefficients.
    let inv_std = 1.0_f32 / (var + eps).sqrt();
    let scale = gamma * inv_std;
    let bias = beta - mean * scale;

    let vscale = vdupq_n_f32(scale);
    let vbias = vdupq_n_f32(bias);

    // 16-element unrolled loop.
    while i + 16 <= len {
        let pf_idx = i + PREFETCH_DISTANCE_F32;
        if pf_idx < len {
            prefetch_read_data(x.as_ptr().add(pf_idx));
        }

        for off in [0, 4, 8, 12] {
            let vx = vld1q_f32(x.as_ptr().add(i + off));
            let vr = vfmaq_f32(vbias, vx, vscale); // x * scale + bias
            vst1q_f32(out.as_mut_ptr().add(i + off), vr);
        }
        i += 16;
    }

    // 4-element remainder.
    while i + 4 <= len {
        let vx = vld1q_f32(x.as_ptr().add(i));
        let vr = vfmaq_f32(vbias, vx, vscale);
        vst1q_f32(out.as_mut_ptr().add(i), vr);
        i += 4;
    }

    // Scalar tail.
    while i < len {
        out[i] = x[i] * scale + bias;
        i += 1;
    }
}

/// Batch normalisation for inference.
///
/// Applies pre-computed `mean`, `var`, `gamma`, `beta` to `x`:
/// `out[i] = gamma * (x[i] - mean) / sqrt(var + eps) + beta`.
pub fn neon_batch_norm_f32_server(
    x: &[f32],
    mean: f32,
    var: f32,
    gamma: f32,
    beta: f32,
    eps: f32,
    out: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { neon_batch_norm_f32_server_impl(x, mean, var, gamma, beta, eps, out) }
            return;
        }
    }
    fallback_batch_norm_f32(x, mean, var, gamma, beta, eps, out)
}

// ============================================================
// Scalar fallback implementations
// ============================================================

fn fallback_matmul_f32(
    a: &[f32],
    a_rows: usize,
    a_cols: usize,
    b: &[f32],
    b_cols: usize,
    c: &mut [f32],
) {
    c.fill(0.0_f32);
    for i in 0..a_rows {
        for k in 0..a_cols {
            let a_ik = a[i * a_cols + k];
            for j in 0..b_cols {
                c[i * b_cols + j] += a_ik * b[k * b_cols + j];
            }
        }
    }
}

fn fallback_batch_norm_f32(
    x: &[f32],
    mean: f32,
    var: f32,
    gamma: f32,
    beta: f32,
    eps: f32,
    out: &mut [f32],
) {
    let inv_std = 1.0_f32 / (var + eps).sqrt();
    let len = x.len().min(out.len());
    for i in 0..len {
        out[i] = gamma * (x[i] - mean) * inv_std + beta;
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------
    // neon_add_f32_unrolled
    // ----------------------------------------------------------

    #[test]
    fn test_add_f32_unrolled_basic() {
        let a: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![1.0_f32; 32];
        let mut out = vec![0.0_f32; 32];
        neon_add_f32_unrolled(&a, &b, &mut out);
        for i in 0..32 {
            let expected = i as f32 + 1.0;
            assert!(
                (out[i] - expected).abs() < 1e-6,
                "out[{i}]={} expected {expected}",
                out[i]
            );
        }
    }

    #[test]
    fn test_add_f32_unrolled_non_multiple() {
        // 19 elements — exercises 16-elem loop + 4-elem + scalar tail.
        let a: Vec<f32> = vec![1.0_f32; 19];
        let b: Vec<f32> = vec![2.0_f32; 19];
        let mut out = vec![0.0_f32; 19];
        neon_add_f32_unrolled(&a, &b, &mut out);
        for v in &out {
            assert!((*v - 3.0).abs() < 1e-6, "expected 3.0 got {v}");
        }
    }

    // ----------------------------------------------------------
    // neon_dot_f32_unrolled
    // ----------------------------------------------------------

    #[test]
    fn test_dot_f32_unrolled_basic() {
        // dot([1,2,...,16], [1,1,...,1]) = 136
        let a: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![1.0_f32; 16];
        let result = neon_dot_f32_unrolled(&a, &b);
        let expected: f32 = (1..=16).map(|i| i as f32).sum();
        assert!(
            (result - expected).abs() < 1e-4,
            "expected {expected} got {result}"
        );
    }

    #[test]
    fn test_dot_f32_unrolled_empty() {
        let result = neon_dot_f32_unrolled(&[], &[]);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_dot_f64_unrolled_basic() {
        let a: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        let b: Vec<f64> = vec![1.0_f64; 8];
        let result = neon_dot_f64_unrolled(&a, &b);
        let expected: f64 = (1..=8).map(|i| i as f64).sum();
        assert!((result - expected).abs() < 1e-10);
    }

    // ----------------------------------------------------------
    // neon_matmul_f32
    // ----------------------------------------------------------

    #[test]
    fn test_matmul_f32_identity() {
        // A (2×2) × I (2×2) = A
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![1.0_f32, 0.0, 0.0, 1.0]; // identity
        let mut c = vec![0.0_f32; 4];
        neon_matmul_f32(&a, 2, 2, &b, 2, &mut c);
        assert!((c[0] - 1.0).abs() < 1e-5);
        assert!((c[1] - 2.0).abs() < 1e-5);
        assert!((c[2] - 3.0).abs() < 1e-5);
        assert!((c[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_f32_known_result() {
        // [1 2] × [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
        // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![5.0_f32, 6.0, 7.0, 8.0];
        let mut c = vec![0.0_f32; 4];
        neon_matmul_f32(&a, 2, 2, &b, 2, &mut c);
        assert!((c[0] - 19.0).abs() < 1e-4, "c[0]={}", c[0]);
        assert!((c[1] - 22.0).abs() < 1e-4, "c[1]={}", c[1]);
        assert!((c[2] - 43.0).abs() < 1e-4, "c[2]={}", c[2]);
        assert!((c[3] - 50.0).abs() < 1e-4, "c[3]={}", c[3]);
    }

    #[test]
    fn test_matmul_f32_rectangular() {
        // A (2×3) × B (3×1) — non-square, exercises KC-blocking.
        // [1 2 3] × [1]   = [1+4+9]    = [14]
        // [4 5 6]   [2]     [4+10+18]    [32]
        //           [3]
        let a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0_f32, 2.0, 3.0];
        let mut c = vec![0.0_f32; 2];
        neon_matmul_f32(&a, 2, 3, &b, 1, &mut c);
        assert!((c[0] - 14.0).abs() < 1e-4, "c[0]={}", c[0]);
        assert!((c[1] - 32.0).abs() < 1e-4, "c[1]={}", c[1]);
    }

    // ----------------------------------------------------------
    // neon_batch_norm_f32_server
    // ----------------------------------------------------------

    #[test]
    fn test_batch_norm_zero_mean_unit_var() {
        // With mean=0, var=1, gamma=1, beta=0, eps=1e-5:
        // out[i] ≈ x[i] / sqrt(1 + eps) ≈ x[i]
        let x = vec![-2.0_f32, -1.0, 0.0, 1.0, 2.0];
        let mut out = vec![0.0_f32; 5];
        neon_batch_norm_f32_server(&x, 0.0, 1.0, 1.0, 0.0, 1e-5, &mut out);
        for i in 0..5 {
            assert!(
                (out[i] - x[i]).abs() < 1e-4,
                "out[{i}]={} x[{i}]={}",
                out[i],
                x[i]
            );
        }
    }

    #[test]
    fn test_batch_norm_known_values() {
        // Manually compute: mean=2, var=4, gamma=2, beta=1, eps=0
        // inv_std = 1/2, scale = 2 * 0.5 = 1, bias = 1 - 2*1 = -1
        // out[i] = x[i] * 1 + (-1)
        let x = vec![0.0_f32, 2.0, 4.0, 6.0];
        let mut out = vec![0.0_f32; 4];
        neon_batch_norm_f32_server(&x, 2.0, 4.0, 2.0, 1.0, 0.0, &mut out);
        let expected = [-1.0_f32, 1.0, 3.0, 5.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (out[i] - exp).abs() < 1e-5,
                "out[{i}]={} expected {exp}",
                out[i]
            );
        }
    }

    // ----------------------------------------------------------
    // Cross-validation: server vs fallback
    // ----------------------------------------------------------

    #[test]
    fn test_add_unrolled_matches_fallback() {
        let n = 37;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.7).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.3).collect();
        let mut ref_out = vec![0.0_f32; n];
        let mut neon_out = vec![0.0_f32; n];
        for i in 0..n {
            ref_out[i] = a[i] + b[i];
        }
        neon_add_f32_unrolled(&a, &b, &mut neon_out);
        for i in 0..n {
            assert!((ref_out[i] - neon_out[i]).abs() < 1e-5, "mismatch at i={i}");
        }
    }
}
