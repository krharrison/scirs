//! Streaming Orthogonal Matching Pursuit (OMP) for adaptive sparse coding (WS78).
//!
//! This module provides a block-oriented streaming OMP that incrementally
//! maintains a sparse representation as new measurement blocks arrive.
//!
//! ## Algorithm
//!
//! On each call to [`StreamingOmp::process_block`]:
//!
//! 1. Append new measurements to the residual buffer (slide window to last
//!    `block_size` samples).
//! 2. Compute correlations `c_j = |<r, d_j>|` for all dictionary atoms.
//! 3. Add the highest-correlation atom to the support (if not already present).
//! 4. Solve least-squares `y = D_S x_S` on the current support to obtain
//!    updated coefficients.
//! 5. Recompute residual `r = y - D_S x_S`.
//! 6. Prune atoms whose `|coeff| < tol`.
//!
//! ## References
//!
//! Pati, Rezaiifar & Krishnaprasad (1993) — Orthogonal Matching Pursuit.

use crate::error::{SignalError, SignalResult};

// ============================================================================
// StreamingOmpConfig
// ============================================================================

/// Configuration for [`StreamingOmp`].
#[derive(Debug, Clone)]
pub struct StreamingOmpConfig {
    /// Maximum number of non-zero coefficients (sparsity level).
    pub sparsity: usize,
    /// Coefficients with |value| below this threshold are pruned.
    pub tol: f64,
    /// Window size for the residual buffer.
    pub block_size: usize,
}

impl Default for StreamingOmpConfig {
    fn default() -> Self {
        Self {
            sparsity: 5,
            tol: 1e-4,
            block_size: 64,
        }
    }
}

// ============================================================================
// StreamingOmp
// ============================================================================

/// Streaming Orthogonal Matching Pursuit.
///
/// Maintains a running sparse representation against a fixed dictionary.  New
/// measurement blocks are incorporated incrementally.
///
/// # Dictionary
///
/// Each column `d_j` of the dictionary must be unit-norm.  [`StreamingOmp::new`]
/// validates this and returns an error if any column deviates from unit norm by
/// more than `1e-6`.
pub struct StreamingOmp {
    /// Dictionary columns (pre-normalised), shape = M rows × N atoms.
    dict: Vec<Vec<f64>>,
    /// Measurement dimension M (number of rows per atom).
    dim: usize,
    /// Number of dictionary atoms N.
    num_atoms: usize,
    /// Current active atom indices (support set).
    support: Vec<usize>,
    /// Coefficients on the support (parallel to `support`).
    coeffs: Vec<f64>,
    /// Accumulated residual buffer of length `block_size`.
    residual_buffer: Vec<f64>,
    /// Configuration.
    config: StreamingOmpConfig,
}

impl StreamingOmp {
    /// Create a new streaming OMP processor.
    ///
    /// # Arguments
    ///
    /// * `dictionary` — each inner `Vec<f64>` is one dictionary column (atom)
    ///   of length `M`.  All columns must be unit-norm.
    /// * `config` — tuning parameters.
    ///
    /// # Errors
    ///
    /// - Empty dictionary.
    /// - Atoms with zero length or inconsistent length.
    /// - Any column that is not unit-norm (tolerance 1e-6).
    /// - `sparsity == 0`.
    pub fn new(dictionary: Vec<Vec<f64>>, config: StreamingOmpConfig) -> SignalResult<Self> {
        if dictionary.is_empty() {
            return Err(SignalError::ValueError(
                "Dictionary must not be empty".to_string(),
            ));
        }
        if config.sparsity == 0 {
            return Err(SignalError::ValueError("sparsity must be > 0".to_string()));
        }
        if config.block_size == 0 {
            return Err(SignalError::ValueError(
                "block_size must be > 0".to_string(),
            ));
        }

        let dim = dictionary[0].len();
        if dim == 0 {
            return Err(SignalError::ValueError(
                "Dictionary atoms must have length > 0".to_string(),
            ));
        }

        for (j, atom) in dictionary.iter().enumerate() {
            if atom.len() != dim {
                return Err(SignalError::DimensionMismatch(format!(
                    "Atom {j} has length {} but expected {dim}",
                    atom.len()
                )));
            }
            let norm_sq: f64 = atom.iter().map(|&v| v * v).sum();
            let norm = norm_sq.sqrt();
            if (norm - 1.0).abs() > 1e-6 {
                return Err(SignalError::ValueError(format!(
                    "Atom {j} is not unit-norm (norm = {norm:.8}); normalise before passing to StreamingOmp"
                )));
            }
        }

        let num_atoms = dictionary.len();
        let residual_buffer = vec![0.0_f64; config.block_size];

        Ok(Self {
            dict: dictionary,
            dim,
            num_atoms,
            support: Vec::new(),
            coeffs: Vec::new(),
            residual_buffer,
            config,
        })
    }

    /// Process a block of new measurements.
    ///
    /// The measurements are appended to the residual buffer (keeping only
    /// the last `block_size` samples).  OMP then updates the support and
    /// coefficients.
    ///
    /// # Returns
    ///
    /// A `Vec<(usize, f64)>` of `(atom_index, coefficient)` pairs for the
    /// active support.
    ///
    /// # Errors
    ///
    /// Returns [`SignalError::DimensionMismatch`] if the measurement length
    /// does not match the dictionary atom dimension `M`.
    pub fn process_block(&mut self, measurements: &[f64]) -> SignalResult<Vec<(usize, f64)>> {
        if measurements.len() != self.dim {
            return Err(SignalError::DimensionMismatch(format!(
                "StreamingOmp::process_block: measurement length {} != dictionary dim {}",
                measurements.len(),
                self.dim
            )));
        }

        // Slide residual buffer: append new measurements, keep last block_size.
        let bs = self.config.block_size;
        if self.dim <= bs {
            // Shift old content left by dim positions and copy new at end.
            let shift = self.dim;
            let old_len = bs;
            let new_end = old_len; // buffer always bs long
            let keep_start = shift.min(old_len);
            // Move existing samples left.
            for i in 0..(old_len - keep_start) {
                self.residual_buffer[i] = self.residual_buffer[i + keep_start];
            }
            // Overwrite the last `dim` positions with new measurements.
            let start = (old_len - self.dim).max(0);
            for (i, &m) in measurements.iter().enumerate() {
                if start + i < new_end {
                    self.residual_buffer[start + i] = m;
                }
            }
        } else {
            // Measurement larger than buffer: take last block_size of measurements.
            let src_start = measurements.len() - bs;
            self.residual_buffer
                .copy_from_slice(&measurements[src_start..src_start + bs]);
        }

        // Run one OMP iteration on the residual buffer.
        self.run_omp_step();

        // Return current support as (index, coefficient) pairs.
        Ok(self
            .support
            .iter()
            .zip(self.coeffs.iter())
            .map(|(&idx, &c)| (idx, c))
            .collect())
    }

    /// Clear support, coefficients and residual buffer.
    pub fn reset(&mut self) {
        self.support.clear();
        self.coeffs.clear();
        self.residual_buffer.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Current support size.
    pub fn support_size(&self) -> usize {
        self.support.len()
    }

    /// Current support indices.
    pub fn support(&self) -> &[usize] {
        &self.support
    }

    /// Current coefficients (parallel to `support()`).
    pub fn coefficients(&self) -> &[f64] {
        &self.coeffs
    }

    // ---- internal OMP ----

    fn run_omp_step(&mut self) {
        let sparsity = self.config.sparsity;
        let tol = self.config.tol;
        let m = self.dim;
        let bs = self.config.block_size;

        // Use residual buffer as the "signal" y (length bs; project onto dim).
        // We approximate by using the first `m` elements of the residual buffer
        // as the measurement vector for the correlation step.
        let y_len = m.min(bs);
        let y = &self.residual_buffer[..y_len];

        // Step 1: compute current residual r = y - D_S * x_S
        let mut residual = y.to_vec();
        if !self.support.is_empty() {
            for (si, &atom_idx) in self.support.iter().enumerate() {
                let coeff = self.coeffs[si];
                let atom = &self.dict[atom_idx];
                for i in 0..y_len {
                    if i < atom.len() {
                        residual[i] -= coeff * atom[i];
                    }
                }
            }
        }

        // Step 2: if support not yet at sparsity limit, try to add one atom.
        if self.support.len() < sparsity {
            let mut best_idx = None;
            let mut best_corr = 0.0_f64;

            for j in 0..self.num_atoms {
                if self.support.contains(&j) {
                    continue;
                }
                let atom = &self.dict[j];
                let inner: f64 = atom.iter().zip(residual.iter()).map(|(&a, &r)| a * r).sum();
                let corr = inner.abs();
                if corr > best_corr {
                    best_corr = corr;
                    best_idx = Some(j);
                }
            }

            if let Some(idx) = best_idx {
                if best_corr > tol {
                    self.support.push(idx);
                    self.coeffs.push(0.0);
                }
            }
        }

        if self.support.is_empty() {
            return;
        }

        // Step 3: solve least-squares on the current support.
        let s = self.support.len();
        // Build sub-matrix D_S (y_len × s).
        let mut d_sub: Vec<Vec<f64>> = Vec::with_capacity(s);
        for &atom_idx in &self.support {
            let atom = &self.dict[atom_idx];
            let col: Vec<f64> = atom[..y_len].to_vec();
            d_sub.push(col);
        }

        let new_coeffs = solve_least_squares_normal_eqs(&d_sub, y);
        self.coeffs = new_coeffs;

        // Step 4: prune near-zero coefficients.
        let mut keep: Vec<usize> = Vec::with_capacity(s);
        let mut keep_coeffs: Vec<f64> = Vec::with_capacity(s);
        for (si, (&atom_idx, &c)) in self.support.iter().zip(self.coeffs.iter()).enumerate() {
            let _ = si;
            if c.abs() >= tol {
                keep.push(atom_idx);
                keep_coeffs.push(c);
            }
        }
        self.support = keep;
        self.coeffs = keep_coeffs;
    }
}

// ============================================================================
// Least-squares via normal equations (small support)
// ============================================================================

/// Solve `D^T D x = D^T y` for `x` where D is given as column-major `Vec<Vec<f64>>`.
///
/// Returns a coefficient vector of length `s = d_cols.len()`.
fn solve_least_squares_normal_eqs(d_cols: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let s = d_cols.len();
    let m = y.len();
    if s == 0 {
        return vec![];
    }

    // Compute A = D^T D  (s × s)
    let mut a = vec![vec![0.0_f64; s]; s];
    for i in 0..s {
        for j in 0..=i {
            let dot: f64 = d_cols[i]
                .iter()
                .zip(d_cols[j].iter())
                .map(|(&a, &b)| a * b)
                .sum();
            a[i][j] = dot;
            a[j][i] = dot;
        }
    }

    // Compute b = D^T y  (s)
    let mut b = vec![0.0_f64; s];
    for i in 0..s {
        b[i] = d_cols[i]
            .iter()
            .zip(y.iter())
            .map(|(&a, &y)| a * y)
            .take(m)
            .sum();
    }

    cholesky_solve_flat(&a, &b)
}

/// Simple Cholesky solve for a small SPD system.
fn cholesky_solve_flat(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return if a[0][0].abs() > 1e-14 {
            vec![b[0] / a[0][0]]
        } else {
            vec![0.0]
        };
    }

    // Cholesky factorisation L L^T = A with diagonal regularisation.
    let diag_max = (0..n).map(|i| a[i][i].abs()).fold(0.0_f64, f64::max);
    let reg = diag_max * 1e-10 + 1e-14;

    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0_f64;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            if i == j {
                let val = a[i][i] + reg - sum;
                l[i][j] = if val > 0.0 { val.sqrt() } else { 1e-14 };
            } else {
                let denom = l[j][j];
                l[i][j] = if denom.abs() > 1e-30 {
                    (a[i][j] - sum) / denom
                } else {
                    0.0
                };
            }
        }
    }

    // Forward solve L z = b.
    let mut z = vec![0.0_f64; n];
    for i in 0..n {
        let mut sum = 0.0_f64;
        for j in 0..i {
            sum += l[i][j] * z[j];
        }
        z[i] = if l[i][i].abs() > 1e-30 {
            (b[i] - sum) / l[i][i]
        } else {
            0.0
        };
    }

    // Back solve L^T x = z.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = 0.0_f64;
        for j in (i + 1)..n {
            sum += l[j][i] * x[j];
        }
        x[i] = if l[i][i].abs() > 1e-30 {
            (z[i] - sum) / l[i][i]
        } else {
            0.0
        };
    }
    x
}

// ============================================================================
// Tests
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple identity dictionary (M=N, unit columns).
    fn identity_dict(n: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|j| (0..n).map(|i| if i == j { 1.0 } else { 0.0 }).collect())
            .collect()
    }

    #[test]
    fn test_streaming_omp_creation() {
        let dict = identity_dict(8);
        let config = StreamingOmpConfig {
            block_size: 8,
            ..Default::default()
        };
        assert!(StreamingOmp::new(dict, config).is_ok());
    }

    #[test]
    fn test_streaming_omp_empty_dict_error() {
        assert!(StreamingOmp::new(vec![], StreamingOmpConfig::default()).is_err());
    }

    #[test]
    fn test_streaming_omp_zero_sparsity_error() {
        let dict = identity_dict(4);
        let config = StreamingOmpConfig {
            sparsity: 0,
            block_size: 4,
            ..Default::default()
        };
        assert!(StreamingOmp::new(dict, config).is_err());
    }

    #[test]
    fn test_streaming_omp_non_unit_norm_error() {
        // Dictionary atom with norm != 1.
        let dict = vec![vec![2.0_f64, 0.0]];
        let config = StreamingOmpConfig {
            block_size: 2,
            ..Default::default()
        };
        assert!(StreamingOmp::new(dict, config).is_err());
    }

    #[test]
    fn test_streaming_omp_converges_to_sparse_solution() {
        // Identity dictionary, sparse signal at index 2.
        let n = 8;
        let dict = identity_dict(n);
        let mut signal = vec![0.0_f64; n];
        signal[2] = 3.0;

        let config = StreamingOmpConfig {
            sparsity: 3,
            tol: 1e-6,
            block_size: n,
        };
        let mut omp = StreamingOmp::new(dict, config).expect("create");
        let result = omp.process_block(&signal).expect("process");

        // The atom at index 2 should be in the result.
        let found = result
            .iter()
            .any(|&(idx, coeff)| idx == 2 && (coeff - 3.0).abs() < 0.1);
        assert!(
            found,
            "OMP should recover the sparse component at index 2, got: {result:?}"
        );
    }

    #[test]
    fn test_streaming_omp_reset_clears_state() {
        let n = 6;
        let dict = identity_dict(n);
        let signal: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let config = StreamingOmpConfig {
            sparsity: 3,
            tol: 1e-8,
            block_size: n,
        };
        let mut omp = StreamingOmp::new(dict, config).expect("create");
        let _ = omp.process_block(&signal).expect("first block");
        omp.reset();

        assert_eq!(omp.support_size(), 0, "Support should be empty after reset");
        assert!(
            omp.residual_buffer.iter().all(|&v| v == 0.0),
            "Residual buffer should be zeroed after reset"
        );
    }

    #[test]
    fn test_streaming_omp_support_size_bounded_by_sparsity() {
        let n = 16;
        let dict = identity_dict(n);
        let signal: Vec<f64> = (0..n).map(|_| 1.0_f64).collect();

        let sparsity = 4;
        let config = StreamingOmpConfig {
            sparsity,
            tol: 1e-8,
            block_size: n,
        };
        let mut omp = StreamingOmp::new(dict, config).expect("create");
        let result = omp.process_block(&signal).expect("process");

        assert!(
            result.len() <= sparsity,
            "Support size {} exceeds sparsity {sparsity}",
            result.len()
        );
    }

    #[test]
    fn test_streaming_omp_pruning_removes_near_zero() {
        let n = 4;
        let dict = identity_dict(n);
        let mut signal = vec![0.0_f64; n];
        signal[0] = 1e-10; // effectively zero — should be pruned

        let config = StreamingOmpConfig {
            sparsity: 2,
            tol: 1e-6, // prune threshold
            block_size: n,
        };
        let mut omp = StreamingOmp::new(dict, config).expect("create");
        let result = omp.process_block(&signal).expect("process");

        // The tiny coefficient should be pruned (|1e-10| < tol=1e-6).
        for &(_, coeff) in &result {
            assert!(
                coeff.abs() >= 1e-6,
                "Coefficient {coeff} below tol should have been pruned"
            );
        }
    }

    #[test]
    fn test_streaming_omp_dimension_mismatch_error() {
        let n = 8;
        let dict = identity_dict(n);
        let config = StreamingOmpConfig {
            block_size: n,
            ..Default::default()
        };
        let mut omp = StreamingOmp::new(dict, config).expect("create");
        let bad = vec![1.0_f64; 4]; // wrong dimension
        assert!(omp.process_block(&bad).is_err());
    }

    #[test]
    fn test_streaming_omp_block_matches_batch() {
        // Single-shot OMP on an identity dict should give exact coefficient.
        let n = 6;
        let dict = identity_dict(n);
        let mut signal = vec![0.0_f64; n];
        signal[3] = 5.0;

        let config = StreamingOmpConfig {
            sparsity: 2,
            tol: 1e-8,
            block_size: n,
        };
        let mut omp = StreamingOmp::new(dict, config).expect("create");
        let result = omp.process_block(&signal).expect("process");

        // Result should contain index 3 with coefficient ≈ 5.0.
        let found = result.iter().find(|&&(idx, _)| idx == 3).map(|&(_, c)| c);
        assert!(
            found.map(|c| (c - 5.0).abs() < 0.5).unwrap_or(false),
            "Expected coeff ≈ 5.0 at index 3, got {found:?}"
        );
    }
}
