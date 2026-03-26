//! Orthogonal Matching Pursuit (OMP) for compressed sensing recovery.
//!
//! Recovers a sparse signal x from measurements y = A·x, where A is a partial
//! DFT matrix.  Each row of A corresponds to one observed frequency bin:
//!   A[i, j] = exp(-2πi · idx[i] · j / N)  (stored as cos/sin pairs).

use super::types::{CsConfig, CsResult, Measurement};
use crate::error::{FFTError, FFTResult};

/// Solver implementing Orthogonal Matching Pursuit.
pub struct OmpSolver {
    /// Algorithm configuration.
    pub config: CsConfig,
}

impl OmpSolver {
    /// Create a new OMP solver with the given configuration.
    pub fn new(config: CsConfig) -> Self {
        Self { config }
    }

    /// Build the (real-valued) partial DFT sensing matrix A.
    ///
    /// Each measurement row corresponds to one complex equation split into two real rows
    /// (real part and imaginary part), giving 2·M rows and N columns.
    fn build_matrix(indices: &[usize], n: usize) -> Vec<Vec<f64>> {
        use std::f64::consts::TAU;
        let m = indices.len();
        let mut a = vec![vec![0.0_f64; n]; 2 * m];
        for (row, &idx) in indices.iter().enumerate() {
            for col in 0..n {
                let angle = TAU * (idx as f64) * (col as f64) / (n as f64);
                a[2 * row][col] = angle.cos(); // real part
                a[2 * row + 1][col] = -angle.sin(); // imaginary part
            }
        }
        a
    }

    /// Compute y_real: interleaved [re, im, …] observations as a flat Vec<f64>.
    fn flatten_measurements(meas: &Measurement) -> Vec<f64> {
        meas.values.clone()
    }

    /// Compute r = y − A·x  (r has 2·M entries).
    fn residual(a: &[Vec<f64>], y: &[f64], x: &[f64]) -> Vec<f64> {
        let rows = a.len();
        let cols = x.len();
        let mut r = vec![0.0_f64; rows];
        for i in 0..rows {
            let mut ax_i = 0.0;
            for j in 0..cols {
                ax_i += a[i][j] * x[j];
            }
            r[i] = y[i] - ax_i;
        }
        r
    }

    /// Compute the column of A most correlated with residual r.
    /// Returns argmax_j |A[:, j]^T r|.
    fn most_correlated_column(a: &[Vec<f64>], r: &[f64], support: &[usize]) -> usize {
        let n = if a.is_empty() { 0 } else { a[0].len() };
        let rows = a.len();
        let mut best_col = 0;
        let mut best_val = f64::NEG_INFINITY;
        for j in 0..n {
            if support.contains(&j) {
                continue;
            }
            let mut dot = 0.0;
            for i in 0..rows {
                dot += a[i][j] * r[i];
            }
            let abs_dot = dot.abs();
            if abs_dot > best_val {
                best_val = abs_dot;
                best_col = j;
            }
        }
        best_col
    }

    /// Solve the least-squares problem A_S · c = y via modified Gram-Schmidt QR.
    ///
    /// A_S is the submatrix of `a` restricted to columns in `support`.
    /// Returns c of length |support|.
    fn least_squares(a: &[Vec<f64>], y: &[f64], support: &[usize]) -> FFTResult<Vec<f64>> {
        let m = a.len(); // number of measurement rows (2·M)
        let s = support.len();
        if s == 0 {
            return Ok(vec![]);
        }

        // Build A_S  (m × s)
        let mut a_s: Vec<Vec<f64>> = (0..s)
            .map(|k| (0..m).map(|i| a[i][support[k]]).collect())
            .collect();

        // Build y copy
        let mut rhs = y.to_vec();

        // Modified Gram-Schmidt QR: decompose A_S = Q · R
        let mut r_mat = vec![vec![0.0_f64; s]; s];

        for k in 0..s {
            // Compute norm of column k
            let norm_sq: f64 = (0..m).map(|i| a_s[k][i] * a_s[k][i]).sum();
            let norm = norm_sq.sqrt();
            if norm < 1e-14 {
                return Err(FFTError::ComputationError(
                    "OMP: singular sub-matrix encountered in QR".into(),
                ));
            }
            r_mat[k][k] = norm;
            for i in 0..m {
                a_s[k][i] /= norm;
            }
            // Project rhs onto q_k
            let proj_rhs: f64 = (0..m).map(|i| a_s[k][i] * rhs[i]).sum();
            // Store Q^T y for back-sub
            r_mat[k][k] = norm; // already set
                                // Orthogonalize remaining columns and rhs
            for j in (k + 1)..s {
                let dot: f64 = (0..m).map(|i| a_s[k][i] * a_s[j][i]).sum();
                r_mat[k][j] = dot;
                for i in 0..m {
                    let qk = a_s[k][i];
                    a_s[j][i] -= dot * qk;
                }
            }
            // Update rhs (project out component in direction q_k)
            for i in 0..m {
                let qk = a_s[k][i];
                rhs[i] -= proj_rhs * qk;
            }
            // Store Q^T y_k
            r_mat[k][k] = norm; // norm already set; store proj in a separate vec below
            let _ = proj_rhs; // will be computed fresh below
        }

        // Re-run with a cleaner approach: accumulate Q^T y during QR
        // Reset and redo properly
        let mut a_s2: Vec<Vec<f64>> = (0..s)
            .map(|k| (0..m).map(|i| a[i][support[k]]).collect())
            .collect();
        let mut qty = vec![0.0_f64; s]; // Q^T y
        let mut r2 = vec![vec![0.0_f64; s]; s];
        let mut y2 = y.to_vec();

        for k in 0..s {
            let norm_sq: f64 = (0..m).map(|i| a_s2[k][i] * a_s2[k][i]).sum();
            let norm = norm_sq.sqrt();
            if norm < 1e-14 {
                return Err(FFTError::ComputationError(
                    "OMP: near-zero pivot during QR".into(),
                ));
            }
            r2[k][k] = norm;
            for i in 0..m {
                a_s2[k][i] /= norm;
            }
            // Q^T y for this step
            let qk_dot_y: f64 = (0..m).map(|i| a_s2[k][i] * y2[i]).sum();
            qty[k] = qk_dot_y;
            for i in 0..m {
                let qk = a_s2[k][i];
                y2[i] -= qk_dot_y * qk;
            }
            for j in (k + 1)..s {
                let dot: f64 = (0..m).map(|i| a_s2[k][i] * a_s2[j][i]).sum();
                r2[k][j] = dot;
                for i in 0..m {
                    let qk = a_s2[k][i];
                    a_s2[j][i] -= dot * qk;
                }
            }
        }

        // Back-substitution: R · c = Q^T y
        let mut c = vec![0.0_f64; s];
        for k in (0..s).rev() {
            let mut sum = qty[k];
            for j in (k + 1)..s {
                sum -= r2[k][j] * c[j];
            }
            if r2[k][k].abs() < 1e-14 {
                c[k] = 0.0;
            } else {
                c[k] = sum / r2[k][k];
            }
        }
        Ok(c)
    }

    /// Recover the sparse signal from `measurements`.
    ///
    /// # Arguments
    /// * `measurements` – partial DFT observations.
    /// * `n` – length of the signal to recover.
    pub fn recover(&self, measurements: &Measurement, n: usize) -> FFTResult<CsResult> {
        if measurements.indices.is_empty() {
            return Err(FFTError::ValueError(
                "No measurement indices provided".into(),
            ));
        }
        if measurements.values.len() != 2 * measurements.indices.len() {
            return Err(FFTError::DimensionError(
                "values must have length 2·|indices| (re/im interleaved)".into(),
            ));
        }

        let a = Self::build_matrix(&measurements.indices, n);
        let y = Self::flatten_measurements(measurements);

        let sparsity = self.config.sparsity.min(n);
        let mut x = vec![0.0_f64; n];
        let mut support: Vec<usize> = Vec::with_capacity(sparsity);

        let mut r = Self::residual(&a, &y, &x);
        let mut iters = 0;

        for iter in 0..self.config.max_iter {
            iters = iter + 1;

            // Check convergence
            let res_norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if res_norm < self.config.tol {
                break;
            }

            // If support is full, stop
            if support.len() >= sparsity {
                break;
            }

            // Find most correlated column
            let k_star = Self::most_correlated_column(&a, &r, &support);
            support.push(k_star);
            support.sort_unstable();
            support.dedup();

            // Solve least-squares on current support
            let c = Self::least_squares(&a, &y, &support)?;

            // Update x
            x = vec![0.0_f64; n];
            for (idx, &col) in support.iter().enumerate() {
                if idx < c.len() {
                    x[col] = c[idx];
                }
            }

            r = Self::residual(&a, &y, &x);
        }

        let residual: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        Ok(CsResult {
            recovered: x,
            iterations: iters,
            residual,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn make_measurement(signal: &[f64], indices: &[usize]) -> Measurement {
        let n = signal.len();
        let m = indices.len();
        let mut values = vec![0.0_f64; 2 * m];
        for (row, &idx) in indices.iter().enumerate() {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for (j, &s) in signal.iter().enumerate() {
                let angle = TAU * (idx as f64) * (j as f64) / (n as f64);
                re += s * angle.cos();
                im += s * (-angle.sin());
            }
            values[2 * row] = re;
            values[2 * row + 1] = im;
        }
        Measurement {
            indices: indices.to_vec(),
            values,
        }
    }

    #[test]
    fn test_omp_recovers_sparse() {
        // 3-sparse signal of length 32
        let n = 32;
        let mut signal = vec![0.0_f64; n];
        signal[0] = 3.0;
        signal[5] = -2.0;
        signal[12] = 1.5;

        let indices: Vec<usize> = (0..10).collect();
        let meas = make_measurement(&signal, &indices);

        let cfg = CsConfig {
            sparsity: 3,
            max_iter: 50,
            tol: 1e-6,
        };
        let solver = OmpSolver::new(cfg);
        let result = solver.recover(&meas, n).expect("OMP should succeed");
        assert!(result.residual < 1e-4, "residual={}", result.residual);
        let err: f64 = signal
            .iter()
            .zip(result.recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(err < 0.1, "max_err={}", err);
    }

    #[test]
    fn test_omp_exact_measurements() {
        let n = 16;
        let mut signal = vec![0.0_f64; n];
        signal[2] = 5.0;
        signal[9] = -3.0;

        // Provide all measurement rows → exact recovery possible
        let indices: Vec<usize> = (0..n).collect();
        let meas = make_measurement(&signal, &indices);

        let cfg = CsConfig {
            sparsity: 2,
            max_iter: 50,
            tol: 1e-8,
        };
        let solver = OmpSolver::new(cfg);
        let result = solver.recover(&meas, n).expect("OMP should succeed");
        assert!(result.residual < 1e-6, "residual={}", result.residual);
    }

    #[test]
    fn test_omp_residual_decreases() {
        let n = 16;
        let mut signal = vec![0.0_f64; n];
        signal[3] = 4.0;
        signal[7] = -1.0;
        signal[11] = 2.5;

        let indices: Vec<usize> = (0..8).collect();
        let meas = make_measurement(&signal, &indices);

        // Run with sparsity 1 first, capture residual, then sparsity 2
        let cfg1 = CsConfig {
            sparsity: 1,
            max_iter: 10,
            tol: 1e-12,
        };
        let res1 = OmpSolver::new(cfg1).recover(&meas, n).expect("ok");

        let cfg2 = CsConfig {
            sparsity: 2,
            max_iter: 10,
            tol: 1e-12,
        };
        let res2 = OmpSolver::new(cfg2).recover(&meas, n).expect("ok");

        // More support → smaller (or equal) residual
        assert!(
            res2.residual <= res1.residual + 1e-10,
            "residual should decrease: {} vs {}",
            res2.residual,
            res1.residual
        );
    }

    #[test]
    fn test_omp_overcomplete_dict() {
        // n_measurements < n: underdetermined system
        let n = 32;
        let mut signal = vec![0.0_f64; n];
        signal[1] = 1.0;

        let indices = vec![0usize, 1, 2, 3, 4]; // only 5 measurements
        let meas = make_measurement(&signal, &indices);

        let cfg = CsConfig {
            sparsity: 1,
            max_iter: 20,
            tol: 1e-6,
        };
        let result = OmpSolver::new(cfg).recover(&meas, n).expect("ok");
        assert_eq!(result.recovered.len(), n);
    }
}
