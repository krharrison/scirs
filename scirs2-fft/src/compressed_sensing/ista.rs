//! ISTA / FISTA solver for compressed sensing via L1 minimisation.
//!
//! Solves:  min_{x}  ½ ‖y − A·x‖² + λ‖x‖₁
//!
//! A is the partial (real-valued) DFT sensing matrix built from measurement indices.
//! FISTA (Beck & Teboulle 2009) adds Nesterov momentum for O(1/k²) convergence.

use super::types::{CsConfig, CsResult, Measurement};
use crate::error::{FFTError, FFTResult};

/// Element-wise soft-threshold (proximal operator of ‖·‖₁).
///
/// `soft_threshold(v, t) = sign(v) · max(|v| − t, 0)`
pub fn soft_threshold(v: f64, t: f64) -> f64 {
    if v > t {
        v - t
    } else if v < -t {
        v + t
    } else {
        0.0
    }
}

/// Solver implementing ISTA and FISTA.
pub struct IstaSolver {
    /// Algorithm configuration (sparsity is not used directly; tol/max_iter are).
    pub config: CsConfig,
    /// L1 regularisation weight λ.
    pub lambda: f64,
    /// Use FISTA (momentum) instead of plain ISTA.
    pub use_fista: bool,
}

impl IstaSolver {
    /// Create a new solver.
    pub fn new(config: CsConfig, lambda: f64, use_fista: bool) -> Self {
        Self {
            config,
            lambda,
            use_fista,
        }
    }

    /// Build the (real-valued) partial DFT sensing matrix A  (2M × N).
    fn build_matrix(indices: &[usize], n: usize) -> Vec<Vec<f64>> {
        use std::f64::consts::TAU;
        let m = indices.len();
        let mut a = vec![vec![0.0_f64; n]; 2 * m];
        for (row, &idx) in indices.iter().enumerate() {
            for col in 0..n {
                let angle = TAU * (idx as f64) * (col as f64) / (n as f64);
                a[2 * row][col] = angle.cos();
                a[2 * row + 1][col] = -angle.sin();
            }
        }
        a
    }

    /// Compute A·x  (length 2M).
    fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        a.iter()
            .map(|row| row.iter().zip(x.iter()).map(|(aij, xj)| aij * xj).sum())
            .collect()
    }

    /// Compute Aᵀ·v  (length N).
    fn mat_t_vec(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
        if a.is_empty() {
            return vec![];
        }
        let n = a[0].len();
        let rows = a.len();
        let mut result = vec![0.0_f64; n];
        for i in 0..rows {
            for j in 0..n {
                result[j] += a[i][j] * v[i];
            }
        }
        result
    }

    /// Estimate the Lipschitz constant L = ‖AᵀA‖₂  via power iteration.
    fn estimate_lipschitz(a: &[Vec<f64>], n: usize) -> f64 {
        if n == 0 || a.is_empty() {
            return 1.0;
        }
        // Simple deterministic initialisation: all-ones normalised
        let mut v = vec![1.0_f64 / (n as f64).sqrt(); n];
        for _ in 0..30 {
            let av = Self::mat_vec(a, &v);
            let atav = Self::mat_t_vec(a, &av);
            let norm: f64 = atav.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-14 {
                break;
            }
            v = atav.iter().map(|x| x / norm).collect();
        }
        // Rayleigh quotient
        let av = Self::mat_vec(a, &v);
        let atav = Self::mat_t_vec(a, &av);
        atav.iter()
            .zip(v.iter())
            .map(|(a, vi)| a * vi)
            .sum::<f64>()
            .abs()
            .max(1e-8)
    }

    /// Compute ½‖y − A·x‖².
    fn loss(a: &[Vec<f64>], y: &[f64], x: &[f64]) -> f64 {
        let ax = Self::mat_vec(a, x);
        let diff_sq: f64 = ax
            .iter()
            .zip(y.iter())
            .map(|(ai, yi)| (ai - yi).powi(2))
            .sum();
        0.5 * diff_sq
    }

    /// Recover signal from measurements.
    pub fn recover(&self, measurements: &Measurement, n: usize) -> FFTResult<CsResult> {
        if measurements.values.len() != 2 * measurements.indices.len() {
            return Err(FFTError::DimensionError(
                "values must have length 2·|indices| (re/im interleaved)".into(),
            ));
        }

        let a = Self::build_matrix(&measurements.indices, n);
        let y = &measurements.values;

        let lipschitz = Self::estimate_lipschitz(&a, n);
        let step = 1.0 / lipschitz;
        let threshold = self.lambda * step;

        let mut x = vec![0.0_f64; n];
        let mut x_prev = x.clone();
        let mut t_k = 1.0_f64; // FISTA momentum scalar
        let mut iters = 0;

        // For FISTA the gradient is evaluated at the extrapolated point y_k
        let mut y_k = x.clone(); // FISTA auxiliary variable

        for iter in 0..self.config.max_iter {
            iters = iter + 1;

            // Point to evaluate gradient at
            let z = if self.use_fista { &y_k } else { &x };

            // Gradient of ½‖y − A·z‖²  at z:  −Aᵀ(y − Az) = Aᵀ(Az) − Aᵀy
            let az = Self::mat_vec(&a, z);
            let residual_vec: Vec<f64> = az.iter().zip(y.iter()).map(|(ai, yi)| ai - yi).collect();
            let grad = Self::mat_t_vec(&a, &residual_vec);

            // Gradient step
            let z_half: Vec<f64> = z
                .iter()
                .zip(grad.iter())
                .map(|(zi, gi)| zi - step * gi)
                .collect();

            // Soft threshold
            let x_new: Vec<f64> = z_half
                .iter()
                .map(|v| soft_threshold(*v, threshold))
                .collect();

            // Check convergence
            let diff_norm: f64 = x_new
                .iter()
                .zip(x.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let x_norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);

            if self.use_fista {
                // FISTA momentum update
                let t_next = (1.0 + (1.0 + 4.0 * t_k * t_k).sqrt()) / 2.0;
                let momentum = (t_k - 1.0) / t_next;
                y_k = x_new
                    .iter()
                    .zip(x.iter())
                    .map(|(xn, xo)| xn + momentum * (xn - xo))
                    .collect();
                t_k = t_next;
            }

            x_prev = x.clone();
            x = x_new;

            if diff_norm / x_norm < self.config.tol {
                break;
            }
        }
        let _ = x_prev; // used to suppress warning

        let ax = Self::mat_vec(&a, &x);
        let residual: f64 = ax
            .iter()
            .zip(y.iter())
            .map(|(ai, yi)| (ai - yi).powi(2))
            .sum::<f64>()
            .sqrt();

        Ok(CsResult {
            recovered: x,
            iterations: iters,
            residual,
        })
    }
}

/// FISTA-specific constructor.
impl IstaSolver {
    /// Create a FISTA solver (ISTA with Nesterov momentum).
    pub fn fista(config: CsConfig, lambda: f64) -> Self {
        Self::new(config, lambda, true)
    }

    /// Create a plain ISTA solver.
    pub fn ista(config: CsConfig, lambda: f64) -> Self {
        Self::new(config, lambda, false)
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
    fn test_ista_soft_threshold() {
        assert!((soft_threshold(3.0, 1.0) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_ista_zero_threshold() {
        assert!((soft_threshold(0.5, 1.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_ista_negative_input() {
        assert!((soft_threshold(-3.0, 1.0) - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn test_ista_converges() {
        let n = 16;
        let mut signal = vec![0.0_f64; n];
        signal[2] = 3.0;
        signal[9] = -1.5;

        let indices: Vec<usize> = (0..10).collect();
        let meas = make_measurement(&signal, &indices);

        let cfg_short = CsConfig {
            sparsity: 2,
            max_iter: 10,
            tol: 1e-12,
        };
        let cfg_long = CsConfig {
            sparsity: 2,
            max_iter: 200,
            tol: 1e-12,
        };

        let res_short = IstaSolver::ista(cfg_short, 0.01)
            .recover(&meas, n)
            .expect("ok");
        let res_long = IstaSolver::ista(cfg_long, 0.01)
            .recover(&meas, n)
            .expect("ok");

        // More iterations → smaller or equal residual
        assert!(
            res_long.residual <= res_short.residual + 1e-6,
            "ISTA should converge: {} vs {}",
            res_long.residual,
            res_short.residual
        );
    }

    #[test]
    fn test_fista_faster_than_ista() {
        let n = 32;
        let mut signal = vec![0.0_f64; n];
        signal[1] = 2.0;
        signal[15] = -1.0;
        signal[20] = 0.5;

        let indices: Vec<usize> = (0..16).collect();
        let meas = make_measurement(&signal, &indices);

        // Use a fixed budget and compare residuals
        let cfg = CsConfig {
            sparsity: 3,
            max_iter: 50,
            tol: 1e-12,
        };
        let res_ista = IstaSolver::ista(cfg.clone(), 0.05)
            .recover(&meas, n)
            .expect("ok");
        let res_fista = IstaSolver::fista(cfg, 0.05).recover(&meas, n).expect("ok");

        // FISTA should achieve a lower or equal residual in the same number of iterations
        assert!(
            res_fista.residual <= res_ista.residual + 1e-6,
            "FISTA residual {} should be <= ISTA residual {}",
            res_fista.residual,
            res_ista.residual
        );
    }

    #[test]
    fn test_cs_config_default() {
        let cfg = CsConfig::default();
        assert_eq!(cfg.sparsity, 5);
        assert_eq!(cfg.max_iter, 100);
        assert!((cfg.tol - 1e-6).abs() < 1e-15);
    }

    #[test]
    fn test_measurement_partial_fft() {
        // Measurements from a known signal should match direct DFT evaluation
        let n = 8;
        let signal = vec![1.0_f64, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
        let indices = vec![0usize, 2, 4];
        let meas = make_measurement(&signal, &indices);

        // Manually verify y[0] = sum of signal (DC component)
        let dc: f64 = signal.iter().sum();
        assert!((meas.values[0] - dc).abs() < 1e-10, "DC mismatch");
    }
}
