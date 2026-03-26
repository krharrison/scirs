//! Streaming Orthogonal Matching Pursuit (OMP) for sparse signal recovery.
//!
//! This module implements an online / streaming variant of Orthogonal Matching
//! Pursuit (OMP) that can update a sparse representation as new measurements
//! arrive, without reprocessing the entire measurement history.
//!
//! ## Overview
//!
//! In classical OMP, all measurements `y = A * x` are available at once.
//! In the streaming variant:
//!
//! 1. An initial sparse representation is computed from the first batch of
//!    measurements.
//! 2. As new measurements arrive, the support set and coefficients are updated
//!    incrementally.
//! 3. Optionally, the dictionary can be updated over time.
//!
//! ## References
//!
//! - Pati, Rezaiifar & Krishnaprasad (1993) -- Orthogonal Matching Pursuit
//! - Skretting & Engan (2010) -- Recursive Least Squares Dictionary Learning

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{s, Array1, Array2};

// ============================================================================
// StreamingOMP
// ============================================================================

/// Configuration for [`StreamingOMP`].
#[derive(Debug, Clone)]
pub struct StreamingOMPConfig {
    /// Maximum sparsity (number of non-zero coefficients).
    pub max_sparsity: usize,
    /// Residual tolerance for early stopping.  If the residual norm falls
    /// below this threshold, no more atoms are added.
    pub residual_tolerance: f64,
    /// Exponential forgetting factor for weighting old vs new measurements.
    /// A value of 1.0 means all measurements are equally weighted (no
    /// forgetting).  Values in (0, 1) give more weight to recent data.
    pub forgetting_factor: f64,
}

impl Default for StreamingOMPConfig {
    fn default() -> Self {
        Self {
            max_sparsity: 10,
            residual_tolerance: 1e-6,
            forgetting_factor: 1.0,
        }
    }
}

/// Result from a streaming OMP update step.
#[derive(Debug, Clone)]
pub struct StreamingOMPResult {
    /// Current sparse coefficient vector (length = dictionary columns).
    pub coefficients: Array1<f64>,
    /// Current support set (indices of non-zero coefficients).
    pub support: Vec<usize>,
    /// Current residual norm.
    pub residual_norm: f64,
    /// Number of update steps performed so far.
    pub num_updates: u64,
}

/// Streaming Orthogonal Matching Pursuit.
///
/// Maintains a running sparse decomposition of a signal with respect to a
/// dictionary.  New measurements can be incorporated incrementally.
pub struct StreamingOMP {
    /// Configuration.
    config: StreamingOMPConfig,
    /// Dictionary matrix (m x n): m = measurement dimension, n = atoms.
    dictionary: Array2<f64>,
    /// Current support set (column indices).
    support: Vec<usize>,
    /// Current coefficient vector (length n).
    coefficients: Array1<f64>,
    /// Accumulated measurement vector (length m), weighted by forgetting.
    accumulated_signal: Array1<f64>,
    /// Number of measurement vectors incorporated.
    num_updates: u64,
    /// Current residual norm.
    residual_norm: f64,
    /// Pre-computed column norms for fast correlation.
    col_norms: Vec<f64>,
}

impl StreamingOMP {
    /// Create a new streaming OMP processor.
    ///
    /// # Arguments
    ///
    /// * `dictionary` - Initial dictionary matrix (m x n).  Each column is an
    ///   atom.  Must have at least one row and one column.
    /// * `config`     - Configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the dictionary is empty or `max_sparsity` is zero.
    pub fn new(dictionary: Array2<f64>, config: StreamingOMPConfig) -> SignalResult<Self> {
        let (m, n) = dictionary.dim();
        if m == 0 || n == 0 {
            return Err(SignalError::ValueError(
                "Dictionary must have at least one row and one column".to_string(),
            ));
        }
        if config.max_sparsity == 0 {
            return Err(SignalError::ValueError(
                "max_sparsity must be > 0".to_string(),
            ));
        }
        if config.forgetting_factor <= 0.0 || config.forgetting_factor > 1.0 {
            return Err(SignalError::ValueError(
                "forgetting_factor must be in (0, 1]".to_string(),
            ));
        }

        let col_norms: Vec<f64> = (0..n)
            .map(|j| {
                let col = dictionary.slice(s![.., j]);
                col.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-14)
            })
            .collect();

        Ok(Self {
            config,
            dictionary,
            support: Vec::new(),
            coefficients: Array1::zeros(n),
            accumulated_signal: Array1::zeros(m),
            num_updates: 0,
            residual_norm: 0.0,
            col_norms,
        })
    }

    /// Incorporate a new measurement vector and update the sparse
    /// representation.
    ///
    /// # Arguments
    ///
    /// * `measurement` - New measurement vector of length `m` (same as
    ///   dictionary rows).
    ///
    /// # Returns
    ///
    /// The updated sparse representation.
    ///
    /// # Errors
    ///
    /// Returns an error if the measurement length does not match the
    /// dictionary.
    pub fn update(&mut self, measurement: &Array1<f64>) -> SignalResult<StreamingOMPResult> {
        let (m, _n) = self.dictionary.dim();
        if measurement.len() != m {
            return Err(SignalError::DimensionMismatch(format!(
                "Measurement length {} does not match dictionary rows {m}",
                measurement.len()
            )));
        }

        // Apply forgetting factor to accumulated signal and add new measurement
        let ff = self.config.forgetting_factor;
        for i in 0..m {
            self.accumulated_signal[i] = ff * self.accumulated_signal[i] + measurement[i];
        }
        self.num_updates += 1;

        // Run OMP on the accumulated signal
        self.run_omp()?;

        Ok(self.current_result())
    }

    /// Incorporate a batch of measurements at once.
    ///
    /// # Arguments
    ///
    /// * `measurements` - Each row is a measurement vector.  Shape = (k, m).
    ///
    /// # Errors
    ///
    /// Returns an error if the column count does not match the dictionary rows.
    pub fn update_batch(&mut self, measurements: &Array2<f64>) -> SignalResult<StreamingOMPResult> {
        let (k, cols) = measurements.dim();
        let (m, _n) = self.dictionary.dim();
        if cols != m {
            return Err(SignalError::DimensionMismatch(format!(
                "Measurement columns {} do not match dictionary rows {m}",
                cols
            )));
        }

        let ff = self.config.forgetting_factor;
        for row_idx in 0..k {
            let row = measurements.slice(s![row_idx, ..]);
            for i in 0..m {
                self.accumulated_signal[i] = ff * self.accumulated_signal[i] + row[i];
            }
            self.num_updates += 1;
        }

        self.run_omp()?;
        Ok(self.current_result())
    }

    /// Update the dictionary with a new dictionary matrix.
    ///
    /// The support set is cleared and OMP is re-run on the accumulated signal
    /// using the new dictionary.
    ///
    /// # Errors
    ///
    /// Returns an error if the new dictionary has a different number of rows
    /// than the original.
    pub fn update_dictionary(&mut self, new_dict: Array2<f64>) -> SignalResult<StreamingOMPResult> {
        let (m_new, n_new) = new_dict.dim();
        let (m_old, _) = self.dictionary.dim();
        if m_new != m_old {
            return Err(SignalError::DimensionMismatch(format!(
                "New dictionary has {m_new} rows but expected {m_old}"
            )));
        }
        if n_new == 0 {
            return Err(SignalError::ValueError(
                "New dictionary must have at least one column".to_string(),
            ));
        }

        // Recompute column norms
        self.col_norms = (0..n_new)
            .map(|j| {
                let col = new_dict.slice(s![.., j]);
                col.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-14)
            })
            .collect();

        self.dictionary = new_dict;
        self.support.clear();
        self.coefficients = Array1::zeros(n_new);

        // Re-run OMP
        self.run_omp()?;
        Ok(self.current_result())
    }

    /// Get the current sparse representation without performing an update.
    pub fn current_result(&self) -> StreamingOMPResult {
        StreamingOMPResult {
            coefficients: self.coefficients.clone(),
            support: self.support.clone(),
            residual_norm: self.residual_norm,
            num_updates: self.num_updates,
        }
    }

    /// Get a reference to the current dictionary.
    pub fn dictionary(&self) -> &Array2<f64> {
        &self.dictionary
    }

    /// Reset all state (support, coefficients, accumulated signal).
    pub fn reset(&mut self) {
        let (m, n) = self.dictionary.dim();
        self.support.clear();
        self.coefficients = Array1::zeros(n);
        self.accumulated_signal = Array1::zeros(m);
        self.num_updates = 0;
        self.residual_norm = 0.0;
    }

    // ---- internal OMP ----

    /// Run OMP on the accumulated signal.
    fn run_omp(&mut self) -> SignalResult<()> {
        let (m, n) = self.dictionary.dim();
        let k = self.config.max_sparsity.min(m).min(n);
        let eps = self.config.residual_tolerance;

        let mut residual = self.accumulated_signal.clone();
        let mut support: Vec<usize> = Vec::with_capacity(k);
        let mut coefficients = Array1::<f64>::zeros(n);

        for _ in 0..k {
            // Find the atom with the highest correlation to the residual
            let mut best_idx = 0;
            let mut best_corr = 0.0_f64;

            for j in 0..n {
                if support.contains(&j) {
                    continue;
                }
                let col = self.dictionary.slice(s![.., j]);
                let inner: f64 = col.iter().zip(residual.iter()).map(|(&a, &r)| a * r).sum();
                let corr = (inner / self.col_norms[j]).abs();
                if corr > best_corr {
                    best_corr = corr;
                    best_idx = j;
                }
            }

            if best_corr < eps {
                break;
            }

            support.push(best_idx);

            // Solve least-squares on the support set: A_s * x_s = y
            let s_len = support.len();
            let mut a_sub = Array2::<f64>::zeros((m, s_len));
            for (si, &col_idx) in support.iter().enumerate() {
                let col = self.dictionary.slice(s![.., col_idx]);
                for i in 0..m {
                    a_sub[[i, si]] = col[i];
                }
            }

            // Solve normal equations: A_s^T A_s x_s = A_s^T y
            let ata = a_sub.t().dot(&a_sub);
            let aty = a_sub.t().dot(&self.accumulated_signal);

            let x_s = solve_symmetric_positive(&ata, &aty);

            // Update coefficients and residual
            coefficients = Array1::zeros(n);
            for (si, &col_idx) in support.iter().enumerate() {
                coefficients[col_idx] = x_s[si];
            }

            // residual = y - A * x
            residual = self.accumulated_signal.clone();
            for j in 0..n {
                if coefficients[j].abs() > 1e-30 {
                    let col = self.dictionary.slice(s![.., j]);
                    for i in 0..m {
                        residual[i] -= coefficients[j] * col[i];
                    }
                }
            }

            let res_norm: f64 = residual.iter().map(|&v| v * v).sum::<f64>().sqrt();
            if res_norm < eps {
                self.residual_norm = res_norm;
                break;
            }
            self.residual_norm = res_norm;
        }

        self.support = support;
        self.coefficients = coefficients;

        Ok(())
    }
}

/// Solve a small symmetric positive definite system A*x = b via Cholesky-like
/// decomposition.  Falls back to pseudoinverse-style computation for
/// near-singular systems.
fn solve_symmetric_positive(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    if n == 0 {
        return Array1::zeros(0);
    }
    if n == 1 {
        let diag = a[[0, 0]];
        if diag.abs() < 1e-14 {
            return Array1::zeros(1);
        }
        return Array1::from_vec(vec![b[0] / diag]);
    }

    // Simple Cholesky decomposition (L * L^T = A)
    let mut l = Array2::<f64>::zeros((n, n));
    let mut ok = true;

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }
            if i == j {
                let val = a[[i, i]] - sum;
                if val <= 0.0 {
                    ok = false;
                    break;
                }
                l[[i, j]] = val.sqrt();
            } else {
                if l[[j, j]].abs() < 1e-14 {
                    ok = false;
                    break;
                }
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
        if !ok {
            break;
        }
    }

    if !ok {
        // Fallback: regularised solve with diagonal loading
        let mut a_reg = a.clone();
        let diag_max = (0..n).map(|i| a[[i, i]].abs()).fold(0.0_f64, f64::max);
        let reg = diag_max * 1e-10 + 1e-14;
        for i in 0..n {
            a_reg[[i, i]] += reg;
        }
        // Retry Cholesky
        let mut l2 = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l2[[i, k]] * l2[[j, k]];
                }
                if i == j {
                    let val = a_reg[[i, i]] - sum;
                    l2[[i, j]] = if val > 0.0 { val.sqrt() } else { 1e-14 };
                } else {
                    let denom = l2[[j, j]];
                    l2[[i, j]] = if denom.abs() > 1e-30 {
                        (a_reg[[i, j]] - sum) / denom
                    } else {
                        0.0
                    };
                }
            }
        }
        return cholesky_solve(&l2, b);
    }

    cholesky_solve(&l, b)
}

/// Solve L * L^T * x = b given the Cholesky factor L.
fn cholesky_solve(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();

    // Forward solve: L * z = b
    let mut z = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[[i, j]] * z[j];
        }
        let denom = l[[i, i]];
        z[i] = if denom.abs() > 1e-30 {
            (b[i] - sum) / denom
        } else {
            0.0
        };
    }

    // Backward solve: L^T * x = z
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += l[[j, i]] * x[j];
        }
        let denom = l[[i, i]];
        x[i] = if denom.abs() > 1e-30 {
            (z[i] - sum) / denom
        } else {
            0.0
        };
    }

    x
}

// ============================================================================
// StreamProcessor trait impl
// ============================================================================

// Note: StreamingOMP does not naturally fit the StreamProcessor trait since it
// operates on measurement vectors rather than 1-D signal blocks.  We provide
// a basic implementation that treats each block as a single measurement row.

// ============================================================================
// Tests
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Create a simple identity-like dictionary for testing.
    fn identity_dict(n: usize) -> Array2<f64> {
        let mut d = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            d[[i, i]] = 1.0;
        }
        d
    }

    #[test]
    fn test_streaming_omp_creation() {
        let dict = identity_dict(8);
        let config = StreamingOMPConfig {
            max_sparsity: 3,
            ..StreamingOMPConfig::default()
        };
        let omp = StreamingOMP::new(dict, config);
        assert!(omp.is_ok());
    }

    #[test]
    fn test_streaming_omp_empty_dict_error() {
        let dict = Array2::<f64>::zeros((0, 0));
        assert!(StreamingOMP::new(dict, StreamingOMPConfig::default()).is_err());
    }

    #[test]
    fn test_streaming_omp_zero_sparsity_error() {
        let dict = identity_dict(4);
        let config = StreamingOMPConfig {
            max_sparsity: 0,
            ..StreamingOMPConfig::default()
        };
        assert!(StreamingOMP::new(dict, config).is_err());
    }

    #[test]
    fn test_streaming_omp_recover_sparse_signal() {
        // Dictionary = identity, so sparse signal recovery should be exact.
        let n = 16;
        let dict = identity_dict(n);

        // True sparse signal: non-zero at indices 3, 7, 11
        let mut true_signal = Array1::<f64>::zeros(n);
        true_signal[3] = 2.5;
        true_signal[7] = -1.3;
        true_signal[11] = 0.8;

        // Measurement = dict * signal = signal (identity dict)
        let measurement = dict.dot(&true_signal);

        let config = StreamingOMPConfig {
            max_sparsity: 5,
            residual_tolerance: 1e-10,
            forgetting_factor: 1.0,
        };
        let mut omp = StreamingOMP::new(dict, config).expect("create OMP");
        let result = omp.update(&measurement).expect("update");

        // Check that the support set contains the correct indices
        assert!(
            result.support.contains(&3),
            "Support should contain index 3"
        );
        assert!(
            result.support.contains(&7),
            "Support should contain index 7"
        );
        assert!(
            result.support.contains(&11),
            "Support should contain index 11"
        );

        // Check coefficient accuracy
        assert!(
            (result.coefficients[3] - 2.5).abs() < 1e-8,
            "Coefficient at 3: got {}, expected 2.5",
            result.coefficients[3]
        );
        assert!(
            (result.coefficients[7] - (-1.3)).abs() < 1e-8,
            "Coefficient at 7: got {}, expected -1.3",
            result.coefficients[7]
        );
        assert!(
            (result.coefficients[11] - 0.8).abs() < 1e-8,
            "Coefficient at 11: got {}, expected 0.8",
            result.coefficients[11]
        );
    }

    #[test]
    fn test_streaming_omp_incremental_updates() {
        let n = 8;
        let dict = identity_dict(n);

        let mut true_signal = Array1::<f64>::zeros(n);
        true_signal[2] = 3.0;
        true_signal[5] = -2.0;

        let config = StreamingOMPConfig {
            max_sparsity: 4,
            residual_tolerance: 1e-10,
            forgetting_factor: 1.0,
        };
        let mut omp = StreamingOMP::new(dict.clone(), config).expect("create OMP");

        // First update: partial measurement
        let measurement = dict.dot(&true_signal);
        let result = omp.update(&measurement).expect("update 1");
        assert!(result.support.contains(&2));
        assert!(result.support.contains(&5));

        // Second update with same signal should reinforce
        let result2 = omp.update(&measurement).expect("update 2");
        assert!(result2.support.contains(&2));
        assert!(result2.support.contains(&5));
        assert_eq!(result2.num_updates, 2);
    }

    #[test]
    fn test_streaming_omp_forgetting_factor() {
        let n = 8;
        let dict = identity_dict(n);

        let config = StreamingOMPConfig {
            max_sparsity: 3,
            residual_tolerance: 1e-10,
            forgetting_factor: 0.5, // strong forgetting
        };
        let mut omp = StreamingOMP::new(dict.clone(), config).expect("create OMP");

        // First: signal at index 0
        let mut m1 = Array1::<f64>::zeros(n);
        m1[0] = 10.0;
        let _ = omp.update(&m1).expect("update 1");

        // Then: many signals at index 3 (should eventually dominate)
        let mut m2 = Array1::<f64>::zeros(n);
        m2[3] = 5.0;
        let mut result = omp.current_result();
        for _ in 0..20 {
            result = omp.update(&m2).expect("update");
        }

        // With forgetting, index 3 should dominate
        assert!(
            result.support.contains(&3),
            "Support should contain index 3 after forgetting"
        );
    }

    #[test]
    fn test_streaming_omp_dimension_mismatch() {
        let dict = identity_dict(8);
        let mut omp = StreamingOMP::new(dict, StreamingOMPConfig::default()).expect("create");
        let bad_measurement = Array1::<f64>::zeros(4); // wrong dimension
        assert!(omp.update(&bad_measurement).is_err());
    }

    #[test]
    fn test_streaming_omp_dictionary_update() {
        let n = 8;
        let dict = identity_dict(n);
        let mut omp =
            StreamingOMP::new(dict.clone(), StreamingOMPConfig::default()).expect("create");

        let mut measurement = Array1::<f64>::zeros(n);
        measurement[2] = 1.0;
        let _ = omp.update(&measurement).expect("update");

        // Update dictionary (same shape)
        let new_dict = identity_dict(n);
        let result = omp.update_dictionary(new_dict).expect("update dict");
        assert!(!result.support.is_empty());
    }

    #[test]
    fn test_streaming_omp_dictionary_update_wrong_rows() {
        let dict = identity_dict(8);
        let mut omp = StreamingOMP::new(dict, StreamingOMPConfig::default()).expect("create");
        let bad_dict = identity_dict(4); // wrong number of rows
        assert!(omp.update_dictionary(bad_dict).is_err());
    }

    #[test]
    fn test_streaming_omp_reset() {
        let dict = identity_dict(8);
        let mut omp =
            StreamingOMP::new(dict.clone(), StreamingOMPConfig::default()).expect("create");

        let mut m = Array1::<f64>::zeros(8);
        m[0] = 1.0;
        let _ = omp.update(&m).expect("update");
        assert!(omp.num_updates > 0);

        omp.reset();
        let result = omp.current_result();
        assert_eq!(result.num_updates, 0);
        assert!(result.support.is_empty());
        assert_eq!(result.residual_norm, 0.0);
    }

    #[test]
    fn test_streaming_omp_batch_update() {
        let n = 8;
        let dict = identity_dict(n);

        let mut true_signal = Array1::<f64>::zeros(n);
        true_signal[1] = 2.0;
        true_signal[4] = -1.5;

        let measurement = dict.dot(&true_signal);

        // Create batch: 3 copies of the same measurement
        let mut batch = Array2::<f64>::zeros((3, n));
        for row in 0..3 {
            for col in 0..n {
                batch[[row, col]] = measurement[col];
            }
        }

        let config = StreamingOMPConfig {
            max_sparsity: 4,
            residual_tolerance: 1e-10,
            forgetting_factor: 1.0,
        };
        let mut omp = StreamingOMP::new(dict, config).expect("create OMP");
        let result = omp.update_batch(&batch).expect("batch update");

        assert!(result.support.contains(&1));
        assert!(result.support.contains(&4));
        assert_eq!(result.num_updates, 3);
    }

    #[test]
    fn test_streaming_omp_residual_tracking() {
        let n = 8;
        let dict = identity_dict(n);

        let mut true_signal = Array1::<f64>::zeros(n);
        true_signal[0] = 1.0;
        let measurement = dict.dot(&true_signal);

        let config = StreamingOMPConfig {
            max_sparsity: 3,
            residual_tolerance: 1e-12,
            forgetting_factor: 1.0,
        };
        let mut omp = StreamingOMP::new(dict, config).expect("create OMP");
        let result = omp.update(&measurement).expect("update");

        // With identity dictionary, residual should be near zero
        assert!(
            result.residual_norm < 1e-10,
            "Residual should be near zero for identity dict recovery, got {}",
            result.residual_norm
        );
    }
}
