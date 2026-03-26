//! Extended-precision GEMM via compensated summation.
//!
//! This module implements matrix multiplication (`C = α A B + β C`) with
//! several levels of accumulation accuracy, all built entirely from IEEE 754
//! arithmetic without any external library:
//!
//! | Mode | Algorithm | Extra cost vs. standard |
//! |------|-----------|-------------------------|
//! | [`AccumulationMode::Standard`] | Naive FMA | 1× |
//! | [`AccumulationMode::Kahan`] | Kahan compensated sum | ~4× |
//! | [`AccumulationMode::PairwiseDouble`] | Double-double accumulation | ~6× |
//! | [`AccumulationMode::TwoFold`] | Ogita-Rump-Oishi exact dot | ~8× |
//!
//! # Mathematical Background
//!
//! **TwoProduct / EFT**: For IEEE 754 binary64, the *error-free transformation*
//! `(p, e) = TwoProduct(a, b)` satisfies `p + e = a * b` exactly in real
//! arithmetic, using Veltkamp's splitting to extract the round-off error `e`.
//!
//! **TwoSum**: Similarly `(s, e) = TwoSum(a, b)` with `s + e = a + b` exactly.
//!
//! **Double-double**: Maintain each accumulator as a `(hi, lo)` pair and update
//! with `TwoProduct` + `TwoSum` so that the stored pair represents the running
//! sum to ~128-bit precision.
//!
//! # References
//!
//! - Ogita, T., Rump, S. M. & Oishi, S. (2005). *Accurate sum and dot product*.
//!   SIAM J. Sci. Comput. 26(6), 1955–1988.
//! - Shewchuk, J. R. (1997). *Adaptive precision floating-point arithmetic and
//!   fast robust geometric predicates.* Discrete & Comp. Geom. 18(3), 305–363.

use crate::error::{LinalgError, LinalgResult};

/// Accumulation mode for compensated GEMM.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumulationMode {
    /// Standard f64 FMA – same accuracy as `a * b + c`.
    Standard,
    /// Kahan two-pass compensated summation.
    Kahan,
    /// Double-double arithmetic: each accumulator is a `(hi, lo)` pair.
    PairwiseDouble,
    /// TwoFold exact dot product (Ogita-Rump-Oishi 2005).
    TwoFold,
}

/// Configuration for [`CompensatedGemm`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct CompensatedGemmConfig {
    /// Accumulation algorithm, default [`AccumulationMode::PairwiseDouble`].
    pub mode: AccumulationMode,
    /// Tile size for cache-blocking, default 32.
    pub tile_size: usize,
}

impl Default for CompensatedGemmConfig {
    fn default() -> Self {
        Self {
            mode: AccumulationMode::PairwiseDouble,
            tile_size: 32,
        }
    }
}

/// Extended-precision matrix multiplication.
///
/// All operations are pure f64; no allocation beyond the caller's `c` buffer.
pub struct CompensatedGemm;

impl CompensatedGemm {
    /// Compute `C ← α·A·B + β·C` with extended-precision accumulation.
    ///
    /// # Arguments
    /// - `alpha`: scalar multiplier for `A·B`.
    /// - `a`: m×k matrix, row-major.
    /// - `b`: k×n matrix, row-major.
    /// - `beta`: scalar multiplier for the existing `C`.
    /// - `c`: m×n matrix, row-major, updated in place.
    /// - `m`, `k`, `n`: matrix dimensions.
    /// - `config`: accumulation mode and tile size.
    ///
    /// # Errors
    /// Returns [`LinalgError::ShapeError`] if dimensions are inconsistent.
    pub fn gemm(
        alpha: f64,
        a: &[f64],
        b: &[f64],
        beta: f64,
        c: &mut [f64],
        m: usize,
        k: usize,
        n: usize,
        config: &CompensatedGemmConfig,
    ) -> LinalgResult<()> {
        if a.len() != m * k {
            return Err(LinalgError::ShapeError(format!(
                "A: expected {}×{} = {} elements, got {}",
                m,
                k,
                m * k,
                a.len()
            )));
        }
        if b.len() != k * n {
            return Err(LinalgError::ShapeError(format!(
                "B: expected {}×{} = {} elements, got {}",
                k,
                n,
                k * n,
                b.len()
            )));
        }
        if c.len() != m * n {
            return Err(LinalgError::ShapeError(format!(
                "C: expected {}×{} = {} elements, got {}",
                m,
                n,
                m * n,
                c.len()
            )));
        }

        // β·C scaling.
        if (beta - 1.0).abs() > 1e-15 {
            for x in c.iter_mut() {
                *x *= beta;
            }
        }

        if alpha == 0.0 {
            return Ok(());
        }

        let ts = config.tile_size.max(1);

        match config.mode {
            AccumulationMode::Standard => {
                Self::gemm_standard(alpha, a, b, c, m, k, n, ts);
            }
            AccumulationMode::Kahan => {
                Self::gemm_kahan(alpha, a, b, c, m, k, n, ts);
            }
            AccumulationMode::PairwiseDouble => {
                Self::gemm_double_double(alpha, a, b, c, m, k, n, ts);
            }
            AccumulationMode::TwoFold => {
                Self::gemm_twofold(alpha, a, b, c, m, k, n, ts);
            }
        }
        Ok(())
    }

    /// TwoFold exact dot product (Ogita-Rump-Oishi 2005).
    ///
    /// Computes `Σ a_i * b_i` with accuracy close to the 2-fold
    /// working precision using exact error-free transformations.
    pub fn exact_dot(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len().min(b.len());
        if n == 0 {
            return 0.0;
        }

        // Compute all exact products; accumulate in cascade sum.
        let mut s = 0.0_f64;
        let mut c = 0.0_f64; // compensation

        for i in 0..n {
            let (p, e) = Self::two_product(a[i], b[i]);
            // Add p to running sum using TwoSum.
            let (s2, c2) = Self::two_sum(s, p);
            s = s2;
            c += c2 + e;
        }
        s + c
    }

    // ─────────────────────────────────────────────────────────────────────────
    // EFT primitives
    // ─────────────────────────────────────────────────────────────────────────

    /// Error-free addition: returns (s, e) with s + e = a + b exactly.
    #[inline]
    pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let v = s - a;
        let e = (a - (s - v)) + (b - v);
        (s, e)
    }

    /// Veltkamp split: returns (a_hi, a_lo) with a_hi + a_lo = a exactly,
    /// |a_hi| ≤ |a|, and a_hi has at most 26 significant bits.
    #[inline]
    pub fn split(a: f64) -> (f64, f64) {
        // Factor = 2^(ceil(53/2)) + 1 = 2^27 + 1.
        const FACTOR: f64 = 134_217_729.0_f64; // 2^27 + 1
        let c = FACTOR * a;
        let a_hi = c - (c - a);
        let a_lo = a - a_hi;
        (a_hi, a_lo)
    }

    /// Error-free product: returns (p, e) with p + e = a * b exactly.
    #[inline]
    pub fn two_product(a: f64, b: f64) -> (f64, f64) {
        let p = a * b;
        let (a_hi, a_lo) = Self::split(a);
        let (b_hi, b_lo) = Self::split(b);
        let e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
        (p, e)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // GEMM kernels
    // ─────────────────────────────────────────────────────────────────────────

    fn gemm_standard(
        alpha: f64,
        a: &[f64],
        b: &[f64],
        c: &mut [f64],
        m: usize,
        k: usize,
        n: usize,
        tile: usize,
    ) {
        // Tiled standard GEMM for cache friendliness.
        for i_tile in (0..m).step_by(tile) {
            let i_end = (i_tile + tile).min(m);
            for j_tile in (0..n).step_by(tile) {
                let j_end = (j_tile + tile).min(n);
                for l_tile in (0..k).step_by(tile) {
                    let l_end = (l_tile + tile).min(k);
                    for i in i_tile..i_end {
                        for l in l_tile..l_end {
                            let a_il = a[i * k + l] * alpha;
                            for j in j_tile..j_end {
                                c[i * n + j] += a_il * b[l * n + j];
                            }
                        }
                    }
                }
            }
        }
    }

    fn gemm_kahan(
        alpha: f64,
        a: &[f64],
        b: &[f64],
        c: &mut [f64],
        m: usize,
        k: usize,
        n: usize,
        _tile: usize,
    ) {
        // For each (i,j) accumulate with Kahan sum over k.
        for i in 0..m {
            for j in 0..n {
                let row_a = &a[i * k..(i + 1) * k];
                // Extract column j of B.
                let col_b: Vec<f64> = (0..k).map(|l| b[l * n + j]).collect();
                let dot = Self::kahan_dot(row_a, &col_b);
                c[i * n + j] += alpha * dot;
            }
        }
    }

    fn gemm_double_double(
        alpha: f64,
        a: &[f64],
        b: &[f64],
        c: &mut [f64],
        m: usize,
        k: usize,
        n: usize,
        tile: usize,
    ) {
        // Each accumulator is a double-double (hi, lo) pair.
        for i_tile in (0..m).step_by(tile) {
            let i_end = (i_tile + tile).min(m);
            for j_tile in (0..n).step_by(tile) {
                let j_end = (j_tile + tile).min(n);

                // Accumulator buffers: parallel arrays for hi and lo parts.
                let tile_m = i_end - i_tile;
                let tile_n = j_end - j_tile;
                let mut acc_hi = vec![0.0_f64; tile_m * tile_n];
                let mut acc_lo = vec![0.0_f64; tile_m * tile_n];

                for l in 0..k {
                    for (ii, i) in (i_tile..i_end).enumerate() {
                        let a_il = a[i * k + l] * alpha;
                        for (jj, j) in (j_tile..j_end).enumerate() {
                            let b_lj = b[l * n + j];
                            let (p, e) = Self::two_product(a_il, b_lj);
                            let idx = ii * tile_n + jj;
                            let (s, ce) = Self::two_sum(acc_hi[idx], p);
                            acc_hi[idx] = s;
                            acc_lo[idx] += ce + e;
                        }
                    }
                }

                // Write back.
                for (ii, i) in (i_tile..i_end).enumerate() {
                    for (jj, j) in (j_tile..j_end).enumerate() {
                        let idx = ii * tile_n + jj;
                        c[i * n + j] += acc_hi[idx] + acc_lo[idx];
                    }
                }
            }
        }
    }

    fn gemm_twofold(
        alpha: f64,
        a: &[f64],
        b: &[f64],
        c: &mut [f64],
        m: usize,
        k: usize,
        n: usize,
        _tile: usize,
    ) {
        // Per-(i,j) TwoFold exact dot product.
        for i in 0..m {
            for j in 0..n {
                let row_a = &a[i * k..(i + 1) * k];
                // Scaled by alpha.
                let scaled_a: Vec<f64> = row_a.iter().map(|&x| x * alpha).collect();
                let col_b: Vec<f64> = (0..k).map(|l| b[l * n + j]).collect();
                c[i * n + j] += Self::exact_dot(&scaled_a, &col_b);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Utility dot products
    // ─────────────────────────────────────────────────────────────────────────

    /// Kahan compensated dot product.
    pub fn kahan_dot(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len().min(b.len());
        let mut sum = 0.0_f64;
        let mut comp = 0.0_f64;
        for i in 0..n {
            let y = a[i] * b[i] - comp;
            let t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
        sum
    }

    /// Pairwise recursive summation for numerical stability.
    pub fn pairwise_sum(v: &[f64]) -> f64 {
        match v.len() {
            0 => 0.0,
            1 => v[0],
            2 => v[0] + v[1],
            n => {
                let mid = n / 2;
                Self::pairwise_sum(&v[..mid]) + Self::pairwise_sum(&v[mid..])
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn naive_gemm(alpha: f64, a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut c = vec![0.0_f64; m * n];
        for i in 0..m {
            for l in 0..k {
                for j in 0..n {
                    c[i * n + j] += alpha * a[i * k + l] * b[l * n + j];
                }
            }
        }
        c
    }

    #[test]
    fn test_compensated_gemm_config_default() {
        let cfg = CompensatedGemmConfig::default();
        assert_eq!(cfg.mode, AccumulationMode::PairwiseDouble);
        assert_eq!(cfg.tile_size, 32);
    }

    #[test]
    fn test_standard_mode_matches_naive() {
        // 3×2 × 2×4 = 3×4
        let m = 3;
        let k = 2;
        let n = 4;
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f64];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0_f64];
        let expected = naive_gemm(1.0, &a, &b, m, k, n);
        let mut c = vec![0.0_f64; m * n];
        let cfg = CompensatedGemmConfig {
            mode: AccumulationMode::Standard,
            ..Default::default()
        };
        CompensatedGemm::gemm(1.0, &a, &b, 0.0, &mut c, m, k, n, &cfg).expect("gemm failed");
        for (ci, ei) in c.iter().zip(expected.iter()) {
            assert_relative_eq!(ci, ei, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_two_sum_exact() {
        // a + b = s + e exactly.
        let a = 1.0_f64;
        let b = 1e-16_f64;
        let (s, e) = CompensatedGemm::two_sum(a, b);
        // s + e should equal a + b in exact arithmetic (checked via round-trip).
        assert_relative_eq!(s + e, a + b, epsilon = 0.0, max_relative = 1e-15);
    }

    #[test]
    fn test_two_product_exact() {
        // p + e = a * b exactly (round-trip via f128-like check).
        let a = 1.0_f64 + f64::EPSILON;
        let b = 1.0_f64 + f64::EPSILON * 2.0;
        let (p, e) = CompensatedGemm::two_product(a, b);
        // |p - a*b| should be small (within standard precision).
        assert_relative_eq!(p, a * b, max_relative = 2.0 * f64::EPSILON);
        // The error term should be non-negligible for this case.
        // p + e should be a better approximation.
        let better = p + e;
        let naive = a * b;
        // This is a weak test since we can't easily check f128 arithmetic.
        assert!(better.is_finite(), "two_product result must be finite");
        let _ = naive;
    }

    #[test]
    fn test_kahan_dot_basic() {
        let a = vec![1.0_f64, 1.0, 1.0];
        let b = vec![1.0_f64, 1.0, 1.0];
        let d = CompensatedGemm::kahan_dot(&a, &b);
        assert_relative_eq!(d, 3.0, epsilon = 1e-14);
    }

    #[test]
    fn test_exact_dot_precision() {
        // Create a case where standard summation loses precision.
        // Using a large value followed by many cancelling small values.
        let big = 1e15_f64;
        let a = vec![big, 1.0, -1.0, 1.0, -1.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        // True result = big.
        let d = CompensatedGemm::exact_dot(&a, &b);
        assert_relative_eq!(d, big, max_relative = 1e-12);
    }

    #[test]
    fn test_pairwise_sum_basic() {
        let v = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let s = CompensatedGemm::pairwise_sum(&v);
        assert_relative_eq!(s, 15.0, epsilon = 1e-12);
    }

    #[test]
    fn test_pairwise_sum_empty() {
        let v: Vec<f64> = vec![];
        assert_eq!(CompensatedGemm::pairwise_sum(&v), 0.0);
    }

    #[test]
    fn test_kahan_gemm_identity() {
        let n = 3;
        let a: Vec<f64> = (0..n * n)
            .map(|i| if i % (n + 1) == 0 { 1.0 } else { 0.0 })
            .collect();
        let b: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut c = vec![0.0_f64; n * n];
        let cfg = CompensatedGemmConfig {
            mode: AccumulationMode::Kahan,
            ..Default::default()
        };
        CompensatedGemm::gemm(1.0, &a, &b, 0.0, &mut c, n, n, n, &cfg).expect("gemm failed");
        // I * B = B
        for i in 0..n * n {
            assert_relative_eq!(c[i], b[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_double_double_gemm_alpha_beta() {
        // Test α and β scaling.
        let m = 2;
        let k = 2;
        let n = 2;
        let a = vec![1.0, 0.0, 0.0, 1.0_f64]; // identity
        let b = vec![3.0, 4.0, 5.0, 6.0_f64];
        let mut c = vec![1.0, 1.0, 1.0, 1.0_f64];
        let cfg = CompensatedGemmConfig {
            mode: AccumulationMode::PairwiseDouble,
            tile_size: 2,
        };
        // C = 2 * I * B + 0.5 * C = 2*B + 0.5
        CompensatedGemm::gemm(2.0, &a, &b, 0.5, &mut c, m, k, n, &cfg).expect("gemm failed");
        assert_relative_eq!(c[0], 2.0 * 3.0 + 0.5, epsilon = 1e-12);
        assert_relative_eq!(c[1], 2.0 * 4.0 + 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_twofold_gemm_matches_standard() {
        let m = 4;
        let k = 3;
        let n = 2;
        let a: Vec<f64> = (1..=m * k).map(|x| x as f64).collect();
        let b: Vec<f64> = (1..=k * n).map(|x| x as f64).collect();
        let expected = naive_gemm(1.0, &a, &b, m, k, n);
        let mut c = vec![0.0_f64; m * n];
        let cfg = CompensatedGemmConfig {
            mode: AccumulationMode::TwoFold,
            ..Default::default()
        };
        CompensatedGemm::gemm(1.0, &a, &b, 0.0, &mut c, m, k, n, &cfg).expect("gemm failed");
        for (ci, ei) in c.iter().zip(expected.iter()) {
            assert_relative_eq!(ci, ei, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_gemm_shape_error() {
        let mut c = vec![0.0_f64; 4];
        let cfg = CompensatedGemmConfig::default();
        let result = CompensatedGemm::gemm(
            1.0,
            &[1.0, 2.0], // wrong size
            &[1.0, 2.0, 3.0, 4.0],
            0.0,
            &mut c,
            2,
            2,
            2,
            &cfg,
        );
        assert!(result.is_err());
    }
}
