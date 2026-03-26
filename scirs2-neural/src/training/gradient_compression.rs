//! Gradient compression primitives: TopK sparsification and PowerSGD low-rank approximation.
//!
//! These algorithms reduce communication overhead in distributed training by compressing
//! gradient tensors before transmission, using error-feedback to maintain convergence.
//!
//! ## Algorithms
//!
//! - **TopK**: Keeps the top-k largest-magnitude gradient components as a sparse vector.
//! - **PowerSGD**: Projects the gradient matrix into a low-rank subspace via randomized SVD
//!   power iteration (Vogels et al., NeurIPS 2019).
//! - **RandomK**: Uniform random sparsification (baseline).
//!
//! ## Error Feedback
//!
//! Both TopK and PowerSGD maintain an *error buffer* (residual from the previous step).
//! This ensures that no gradient information is permanently discarded: the residual is
//! added back to the next gradient before compression.
//!
//! ```rust
//! use scirs2_neural::training::gradient_compression::{TopKConfig, TopKSparsifier};
//!
//! let cfg = TopKConfig { k_fraction: 0.1, use_error_feedback: true };
//! let mut sparsifier = TopKSparsifier::new(cfg, 100);
//! let grad: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
//! let sparse = sparsifier.compress(&grad);
//! assert_eq!(sparse.indices.len(), 10);
//! let dense = sparse.to_dense();
//! assert_eq!(dense.len(), 100);
//! ```

use crate::error::{NeuralError, Result as NeuralResult};
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::random::rngs::SmallRng;
use scirs2_core::random::{Rng, RngExt, SeedableRng};

// ============================================================================
// Enums
// ============================================================================

/// Method used for gradient compression.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionMethod {
    /// Top-k sparsification: retain the k highest-magnitude entries.
    TopK,
    /// PowerSGD: low-rank matrix factorisation via power iteration.
    PowerSgd,
    /// Random-k sparsification: retain k uniformly sampled entries.
    RandomK,
    /// No compression; pass gradients through unchanged.
    Uncompressed,
}

// ============================================================================
// TopK Config
// ============================================================================

/// Configuration for the TopK gradient sparsifier.
#[derive(Debug, Clone)]
pub struct TopKConfig {
    /// Fraction of gradient elements to retain. Default: `0.01` (top 1 %).
    pub k_fraction: f64,
    /// If `true`, accumulate compression residual and add it to the next step.
    pub use_error_feedback: bool,
}

impl Default for TopKConfig {
    fn default() -> Self {
        Self {
            k_fraction: 0.01,
            use_error_feedback: true,
        }
    }
}

// ============================================================================
// PowerSGD Config
// ============================================================================

/// Configuration for the PowerSGD compressor.
#[derive(Debug, Clone)]
pub struct PowerSgdConfig {
    /// Rank of the low-rank approximation. Default: `4`.
    pub rank: usize,
    /// Number of power iterations used to refine Q. Default: `1`.
    pub power_iter: usize,
    /// If `true`, accumulate compression residual. Default: `true`.
    pub use_error_feedback: bool,
    /// Skip compression when the ratio `(m*n) / (rank*(m+n))` < this threshold.
    /// Default: `2.0`.
    pub min_compression_rate: f64,
}

impl Default for PowerSgdConfig {
    fn default() -> Self {
        Self {
            rank: 4,
            power_iter: 1,
            use_error_feedback: true,
            min_compression_rate: 2.0,
        }
    }
}

// ============================================================================
// SparseGradient
// ============================================================================

/// Sparse gradient produced by TopK (or RandomK) sparsification.
#[derive(Debug, Clone)]
pub struct SparseGradient {
    /// Indices of the retained elements (sorted ascending).
    pub indices: Vec<usize>,
    /// Values of the retained elements, in the same order as `indices`.
    pub values: Vec<f64>,
    /// Length of the original (dense) gradient vector.
    pub original_len: usize,
}

impl SparseGradient {
    /// Reconstruct the dense gradient (zero-fills absent positions).
    pub fn to_dense(&self) -> Vec<f64> {
        let mut dense = vec![0.0_f64; self.original_len];
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            if idx < self.original_len {
                dense[idx] = val;
            }
        }
        dense
    }

    /// Ratio `original_len / nnz`.  Returns `1.0` when nothing is stored.
    pub fn compression_ratio(&self) -> f64 {
        if self.indices.is_empty() {
            return 1.0;
        }
        self.original_len as f64 / self.indices.len() as f64
    }
}

// ============================================================================
// TopKSparsifier
// ============================================================================

/// Sparsifies gradients by keeping only the `k_fraction` largest-magnitude entries.
///
/// With error feedback enabled, the residual from each step is accumulated and
/// injected into the gradient on the next step, ensuring no information is lost
/// in the long run.
pub struct TopKSparsifier {
    config: TopKConfig,
    /// Accumulated residual from previous compression steps.
    error_buffer: Vec<f64>,
}

impl TopKSparsifier {
    /// Create a new sparsifier.  `gradient_len` pre-allocates the error buffer.
    pub fn new(config: TopKConfig, gradient_len: usize) -> Self {
        Self {
            config,
            error_buffer: vec![0.0_f64; gradient_len],
        }
    }

    /// Compress `gradient` into a `SparseGradient`.
    ///
    /// Steps:
    /// 1. `g = gradient + error_buffer` (error-corrected gradient).
    /// 2. Determine threshold so that exactly `k` elements are kept.
    /// 3. Build sparse representation of the top-k elements.
    /// 4. `error_buffer = g - dense(sparse)` (residual for next step).
    pub fn compress(&mut self, gradient: &[f64]) -> SparseGradient {
        let n = gradient.len();

        // Resize error buffer if needed (e.g. first call or changed layer).
        if self.error_buffer.len() != n {
            self.error_buffer.resize(n, 0.0);
        }

        // Step 1: apply error feedback.
        let mut g_corrected: Vec<f64> = if self.config.use_error_feedback {
            gradient
                .iter()
                .zip(self.error_buffer.iter())
                .map(|(&gv, &ev)| gv + ev)
                .collect()
        } else {
            gradient.to_vec()
        };

        // Step 2: determine k and the threshold magnitude.
        let k = ((n as f64 * self.config.k_fraction).ceil() as usize).max(1).min(n);

        // Partial sort: collect |g| values, find the k-th largest.
        let mut magnitudes: Vec<f64> = g_corrected.iter().map(|v| v.abs()).collect();
        // nth_element equivalent: partition so element at position (n-k) is in its sorted place.
        let pivot_pos = n.saturating_sub(k);
        magnitudes.select_nth_unstable_by(pivot_pos, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let threshold = magnitudes[pivot_pos];

        // Step 3: collect indices and values where |g| >= threshold.
        // To get exactly k elements when there are ties, we use a two-pass approach.
        let mut indices: Vec<usize> = Vec::with_capacity(k);
        let mut values: Vec<f64> = Vec::with_capacity(k);

        for (i, &v) in g_corrected.iter().enumerate() {
            if v.abs() >= threshold && indices.len() < k {
                indices.push(i);
                values.push(v);
            }
        }

        // Step 4: update error buffer = g_corrected - dense(sparse).
        if self.config.use_error_feedback {
            // Zero out the error buffer first.
            for ev in self.error_buffer.iter_mut() {
                *ev = 0.0;
            }
            // error_buffer = g_corrected (start from corrected gradient).
            for (i, gv) in g_corrected.iter_mut().enumerate() {
                self.error_buffer[i] = *gv;
            }
            // Subtract the sparse values that were transmitted.
            for (&idx, &val) in indices.iter().zip(values.iter()) {
                self.error_buffer[idx] -= val;
            }
        }

        SparseGradient {
            indices,
            values,
            original_len: n,
        }
    }

    /// Decompress a sparse gradient back to a dense vector.
    pub fn decompress(sparse: &SparseGradient) -> Vec<f64> {
        sparse.to_dense()
    }

    /// Clear the error buffer (e.g., at the start of a new training phase).
    pub fn reset_error_buffer(&mut self) {
        for ev in self.error_buffer.iter_mut() {
            *ev = 0.0;
        }
    }
}

// ============================================================================
// LowRankGradient
// ============================================================================

/// Low-rank gradient approximation produced by PowerSGD.
///
/// The gradient `M` ≈ `P @ Q^T`, where
/// - `P` has shape `[m, r]` (left factor),
/// - `Q` has shape `[n, r]` (right factor),
/// - `r` is the configured rank.
#[derive(Debug, Clone)]
pub struct LowRankGradient {
    /// Left factor `P` of shape `[m, r]`.
    pub p: Array2<f64>,
    /// Right factor `Q` of shape `[n, r]`.
    pub q: Array2<f64>,
    /// Original shape `(m, n)` of the gradient matrix.
    pub shape: (usize, usize),
}

impl LowRankGradient {
    /// Reconstruct the approximated gradient `P @ Q^T` of shape `[m, n]`.
    pub fn decompress(&self) -> Array2<f64> {
        self.p.dot(&self.q.t())
    }

    /// Compression ratio: `(m * n) / (rank * (m + n))`.  Values > 1 mean compression.
    pub fn compression_ratio(&self) -> f64 {
        let (m, n) = self.shape;
        let r = self.p.shape()[1];
        if r == 0 || m + n == 0 {
            return 1.0;
        }
        (m * n) as f64 / (r * (m + n)) as f64
    }
}

// ============================================================================
// PowerSgdCompressor
// ============================================================================

/// Low-rank gradient compressor implementing PowerSGD (Vogels et al., NeurIPS 2019).
///
/// The algorithm maintains a "warm-started" right factor Q across steps, which
/// dramatically accelerates convergence of the power iteration.
pub struct PowerSgdCompressor {
    config: PowerSgdConfig,
    /// Retained Q matrix for warm-start across time steps.
    q_buffer: Option<Array2<f64>>,
    /// Accumulated compression residual.
    error_buffer: Option<Array2<f64>>,
    /// Seeded RNG for Q initialisation.
    rng: SmallRng,
}

impl PowerSgdCompressor {
    /// Create a new compressor with the given configuration and RNG seed.
    pub fn new(config: PowerSgdConfig, seed: u64) -> Self {
        Self {
            config,
            q_buffer: None,
            error_buffer: None,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// Compress a 2-D gradient matrix into a low-rank representation.
    ///
    /// # Errors
    /// Returns `NeuralError::ComputationError` if the matrix dimensions are
    /// incompatible with the configured rank.
    pub fn compress(&mut self, gradient: &Array2<f64>) -> NeuralResult<LowRankGradient> {
        let (m, n) = (gradient.shape()[0], gradient.shape()[1]);
        let rank = self.config.rank.min(m.min(n));

        if rank == 0 {
            return Err(NeuralError::ComputationError(format!(
                "PowerSGD rank is 0; gradient shape ({m}, {n}) too small"
            )));
        }

        // Check whether compression is actually beneficial.
        let ratio = (m * n) as f64 / (rank * (m + n)) as f64;
        if ratio < self.config.min_compression_rate {
            // Return a trivially rank-m matrix to indicate the full gradient is used.
            // (Caller can detect this by checking compression_ratio().)
            let p = gradient.clone();
            let q = Array2::<f64>::eye(n);
            return Ok(LowRankGradient { p, q, shape: (m, n) });
        }

        // Step 1: apply error feedback.
        let m_eff: Array2<f64> = if self.config.use_error_feedback {
            match &self.error_buffer {
                Some(buf) if buf.shape() == gradient.shape() => gradient + buf,
                _ => gradient.clone(),
            }
        } else {
            gradient.clone()
        };

        // Step 2: initialise / warm-start Q of shape [n, rank].
        let mut q = match &self.q_buffer {
            Some(qb) if qb.shape() == [n, rank] => qb.clone(),
            _ => {
                let mut qnew = Array2::<f64>::zeros((n, rank));
                for v in qnew.iter_mut() {
                    *v = self.rng.random::<f64>() * 2.0 - 1.0;
                }
                Self::orthonormalize_columns(&mut qnew);
                qnew
            }
        };

        // Step 3: power iteration.
        for _iter in 0..self.config.power_iter {
            // P = M @ Q,  then orthonormalise P.
            let mut p_tmp = m_eff.dot(&q); // [m, rank]
            Self::orthonormalize_columns(&mut p_tmp);
            // Q = M^T @ P, then orthonormalise Q.
            q = m_eff.t().dot(&p_tmp); // [n, rank]
            Self::orthonormalize_columns(&mut q);
        }

        // Step 4: final P (not normalised — carries the scale).
        let p = m_eff.dot(&q); // [m, rank]

        // Step 5: update error buffer.
        if self.config.use_error_feedback {
            let approx = p.dot(&q.t()); // [m, n]
            self.error_buffer = Some(&m_eff - &approx);
        }

        // Step 6: warm-start Q for next call.
        self.q_buffer = Some(q.clone());

        Ok(LowRankGradient { p, q, shape: (m, n) })
    }

    /// Modified Gram-Schmidt orthonormalisation of matrix columns (in-place).
    ///
    /// Transforms the columns of `matrix` into an orthonormal basis using the
    /// numerically stable modified Gram-Schmidt procedure.
    pub fn orthonormalize_columns(matrix: &mut Array2<f64>) {
        let ncols = matrix.shape()[1];
        for j in 0..ncols {
            // Orthogonalise column j against all previous columns.
            for k in 0..j {
                let col_k: Array1<f64> = matrix.slice(s![.., k]).to_owned();
                let col_j: Array1<f64> = matrix.slice(s![.., j]).to_owned();
                let proj: f64 = col_k.dot(&col_j);
                let sub = col_k.mapv(|v| v * proj);
                let mut col_j_mut = matrix.slice_mut(s![.., j]);
                col_j_mut -= &sub;
            }
            // Normalise column j.
            let col_j: Array1<f64> = matrix.slice(s![.., j]).to_owned();
            let norm = col_j.dot(&col_j).sqrt();
            if norm > 1e-12 {
                matrix.slice_mut(s![.., j]).mapv_inplace(|v| v / norm);
            }
        }
    }

    /// Clear Q buffer and error buffer (useful when the compressed layer changes shape).
    pub fn reset(&mut self) {
        self.q_buffer = None;
        self.error_buffer = None;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    // --- TopK tests ---

    #[test]
    fn test_topk_retains_exactly_k_elements() {
        let cfg = TopKConfig {
            k_fraction: 0.1,
            use_error_feedback: false,
        };
        let n = 100;
        let mut sparsifier = TopKSparsifier::new(cfg, n);
        let grad: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let sparse = sparsifier.compress(&grad);
        assert_eq!(sparse.indices.len(), 10, "should keep exactly 10 elements");
        assert_eq!(sparse.values.len(), 10);
    }

    #[test]
    fn test_topk_retains_largest_magnitude_elements() {
        let cfg = TopKConfig {
            k_fraction: 0.1,
            use_error_feedback: false,
        };
        let n = 50;
        let mut sparsifier = TopKSparsifier::new(cfg, n);
        // Large values at known positions.
        let mut grad = vec![0.001_f64; n];
        grad[3] = 10.0;
        grad[7] = 9.0;
        grad[15] = 8.0;
        grad[42] = 7.0;
        grad[49] = 6.0;
        let sparse = sparsifier.compress(&grad);
        let retained_indices: std::collections::HashSet<usize> =
            sparse.indices.iter().copied().collect();
        assert!(retained_indices.contains(&3));
        assert!(retained_indices.contains(&7));
        assert!(retained_indices.contains(&15));
        assert!(retained_indices.contains(&42));
        assert!(retained_indices.contains(&49));
    }

    #[test]
    fn test_topk_error_feedback_compensates() {
        let cfg = TopKConfig {
            k_fraction: 0.1,
            use_error_feedback: true,
        };
        let n = 20;
        let mut sparsifier = TopKSparsifier::new(cfg, n);

        // All values equal: first step compresses heavily.
        let grad1 = vec![1.0_f64; n];
        let _sparse1 = sparsifier.compress(&grad1);

        // Second step: error buffer should contain residuals from step 1.
        let grad2 = vec![0.0_f64; n];
        let sparse2 = sparsifier.compress(&grad2);
        // The accumulated residual from step 1 must be partially transmitted.
        let total_value: f64 = sparse2.values.iter().map(|v| v.abs()).sum();
        assert!(
            total_value > 0.0,
            "error feedback should transmit residual from previous step"
        );
    }

    #[test]
    fn test_topk_dense_roundtrip() {
        let cfg = TopKConfig::default();
        let n = 200;
        let mut sparsifier = TopKSparsifier::new(cfg, n);
        let grad: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let sparse = sparsifier.compress(&grad);
        let dense = sparse.to_dense();
        assert_eq!(dense.len(), n);
        // Dense values at retained indices must match original.
        for (&idx, &val) in sparse.indices.iter().zip(sparse.values.iter()) {
            assert!(
                (dense[idx] - val).abs() < 1e-12,
                "dense round-trip mismatch at index {idx}"
            );
        }
    }

    #[test]
    fn test_topk_compression_ratio() {
        let cfg = TopKConfig {
            k_fraction: 0.05,
            use_error_feedback: false,
        };
        let n = 1000;
        let mut sparsifier = TopKSparsifier::new(cfg, n);
        let grad: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let sparse = sparsifier.compress(&grad);
        let ratio = sparse.compression_ratio();
        // k = 50, ratio = 1000/50 = 20.
        assert!(
            (ratio - 20.0).abs() < 1.0,
            "expected ratio ~20, got {ratio}"
        );
    }

    #[test]
    fn test_topk_reset_error_buffer() {
        let cfg = TopKConfig {
            k_fraction: 0.1,
            use_error_feedback: true,
        };
        let n = 10;
        let mut sparsifier = TopKSparsifier::new(cfg, n);
        let grad = vec![1.0_f64; n];
        let _ = sparsifier.compress(&grad);
        sparsifier.reset_error_buffer();
        assert!(
            sparsifier.error_buffer.iter().all(|&v| v == 0.0),
            "error buffer should be zeroed after reset"
        );
    }

    // --- PowerSGD tests ---

    #[test]
    fn test_powersgd_compression_ratio() {
        let m = 64;
        let n = 64;
        let rank = 4;
        let cfg = PowerSgdConfig {
            rank,
            power_iter: 1,
            use_error_feedback: false,
            min_compression_rate: 0.0,
        };
        let mut compressor = PowerSgdCompressor::new(cfg, 42);
        let grad: Array2<f64> = Array2::ones((m, n));
        let lr = compressor.compress(&grad).expect("compress failed");
        let expected_ratio = (m * n) as f64 / (rank * (m + n)) as f64;
        let actual_ratio = lr.compression_ratio();
        assert!(
            (actual_ratio - expected_ratio).abs() < 0.01,
            "expected ratio {expected_ratio}, got {actual_ratio}"
        );
    }

    #[test]
    fn test_powersgd_decompressed_shape() {
        let m = 32;
        let n = 48;
        let cfg = PowerSgdConfig {
            rank: 4,
            power_iter: 1,
            use_error_feedback: false,
            min_compression_rate: 0.0,
        };
        let mut compressor = PowerSgdCompressor::new(cfg, 7);
        let grad: Array2<f64> = Array2::ones((m, n));
        let lr = compressor.compress(&grad).expect("compress failed");
        let decompressed = lr.decompress();
        assert_eq!(
            decompressed.shape(),
            [m, n],
            "decompressed shape should be [{m}, {n}]"
        );
    }

    #[test]
    fn test_orthonormalize_columns() {
        // Use a matrix whose 2 columns are genuinely independent.
        // Row vectors chosen to avoid rank deficiency.
        let mut mat = Array2::<f64>::from_shape_vec(
            (4, 2),
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0],
        )
        .expect("shape ok");
        PowerSgdCompressor::orthonormalize_columns(&mut mat);
        // Verify orthonormality: Q^T Q ≈ I_2.
        let qtq = mat.t().dot(&mat);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq[[i, j]] - expected).abs() < 1e-10,
                    "Q^T Q [{i},{j}] = {}, expected {expected}",
                    qtq[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_powersgd_rank1_approximates_rank1_matrix() {
        // A rank-1 matrix should be perfectly recovered with rank >= 1.
        let u: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        let v: Vec<f64> = (1..=6).map(|i| i as f64).collect();
        let u_arr = Array2::from_shape_vec((8, 1), u).expect("shape ok");
        let v_arr = Array2::from_shape_vec((1, 6), v).expect("shape ok");
        let grad = u_arr.dot(&v_arr); // [8, 6] rank-1 matrix.

        let cfg = PowerSgdConfig {
            rank: 1,
            power_iter: 5,
            use_error_feedback: false,
            min_compression_rate: 0.0,
        };
        let mut compressor = PowerSgdCompressor::new(cfg, 0);
        let lr = compressor.compress(&grad).expect("compress failed");
        let approx = lr.decompress();

        // Frobenius error should be small relative to original norm.
        let diff: f64 = (&approx - &grad).mapv(|v| v * v).sum().sqrt();
        let norm: f64 = grad.mapv(|v| v * v).sum().sqrt();
        assert!(
            diff / norm < 0.01,
            "PowerSGD should perfectly recover rank-1 matrix; relative error = {}",
            diff / norm
        );
    }

    #[test]
    fn test_powersgd_error_feedback_reduces_residual() {
        let m = 16;
        let n = 16;
        let cfg = PowerSgdConfig {
            rank: 2,
            power_iter: 2,
            use_error_feedback: true,
            min_compression_rate: 0.0,
        };
        let mut compressor = PowerSgdCompressor::new(cfg, 99);
        let grad: Array2<f64> = Array2::ones((m, n));
        // Two compression steps — error buffer should be non-trivial after step 1.
        let _lr1 = compressor.compress(&grad).expect("step 1");
        let err_norm1 = compressor
            .error_buffer
            .as_ref()
            .map(|b| b.mapv(|v| v * v).sum().sqrt())
            .unwrap_or(0.0);
        // After step 2 with gradient=0, error buffer flushes some residual.
        let zero_grad: Array2<f64> = Array2::zeros((m, n));
        let _lr2 = compressor.compress(&zero_grad).expect("step 2");
        let err_norm2 = compressor
            .error_buffer
            .as_ref()
            .map(|b| b.mapv(|v| v * v).sum().sqrt())
            .unwrap_or(0.0);
        // Error norm should decrease (residual partially sent in step 2).
        assert!(
            err_norm2 < err_norm1 + 1e-10,
            "error feedback should reduce residual; step1={err_norm1:.4} step2={err_norm2:.4}"
        );
    }

    #[test]
    fn test_compression_method_enum_non_exhaustive() {
        let method = CompressionMethod::TopK;
        // Matching with wildcard must compile (non_exhaustive).
        let _desc = match method {
            CompressionMethod::TopK => "topk",
            CompressionMethod::PowerSgd => "powersgd",
            CompressionMethod::RandomK => "randomk",
            CompressionMethod::Uncompressed => "none",
            _ => "unknown",
        };
    }

    #[test]
    fn test_sparse_gradient_empty() {
        let sparse = SparseGradient {
            indices: vec![],
            values: vec![],
            original_len: 50,
        };
        let dense = sparse.to_dense();
        assert_eq!(dense.len(), 50);
        assert!(dense.iter().all(|&v| v == 0.0));
        assert_eq!(sparse.compression_ratio(), 1.0);
    }

    #[test]
    fn test_powersgd_reset_clears_buffers() {
        let cfg = PowerSgdConfig::default();
        let mut compressor = PowerSgdCompressor::new(cfg, 0);
        let grad: Array2<f64> = Array2::ones((8, 8));
        let _ = compressor.compress(&grad).expect("compress");
        compressor.reset();
        assert!(compressor.q_buffer.is_none(), "Q buffer cleared");
        assert!(compressor.error_buffer.is_none(), "error buffer cleared");
    }

    #[test]
    fn test_topk_k_fraction_one_keeps_all() {
        let cfg = TopKConfig {
            k_fraction: 1.0,
            use_error_feedback: false,
        };
        let n = 30;
        let mut sparsifier = TopKSparsifier::new(cfg, n);
        let grad: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0)).collect();
        let sparse = sparsifier.compress(&grad);
        assert_eq!(sparse.indices.len(), n, "k_fraction=1.0 should keep all");
    }

    #[test]
    fn test_topk_small_gradient() {
        let cfg = TopKConfig {
            k_fraction: 0.5,
            use_error_feedback: false,
        };
        let mut sparsifier = TopKSparsifier::new(cfg, 4);
        let grad = vec![3.0_f64, 1.0, 4.0, 1.5];
        let sparse = sparsifier.compress(&grad);
        assert_eq!(sparse.indices.len(), 2, "50% of 4 = 2 elements");
        // Should keep the two largest: 4.0 (idx 2) and 3.0 (idx 0).
        let kept: std::collections::HashSet<usize> = sparse.indices.iter().copied().collect();
        assert!(kept.contains(&2), "idx 2 (val 4.0) should be kept");
        assert!(kept.contains(&0), "idx 0 (val 3.0) should be kept");
    }
}
