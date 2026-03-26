//! Nonparametric GARCH via Gaussian Process Volatility Functions
//!
//! This module implements GP-GARCH, a nonparametric extension of the classical
//! GARCH model in which the conditional variance function is replaced by a
//! Gaussian Process (GP) regression over lagged squared log-returns.
//!
//! ## Model Specification
//!
//! For a return series `r_1, …, r_T`:
//!
//! ```text
//! r_t = σ_t · z_t ,   z_t ~ N(0, 1)
//!
//! x_t = [r_{t-1}², r_{t-2}², …, r_{t-p}²]   (feature vector)
//! σ²_t = GP(x_t)                              (GP posterior mean)
//! ```
//!
//! The GP uses a **Matérn-5/2** kernel:
//!
//! ```text
//! k(x, x') = σ_f² (1 + √5 r/l + 5r²/(3l²)) exp(-√5 r/l),
//!            r = ||x - x'||₂
//! ```
//!
//! Inference is via the **FITC** (Fully Independent Training Conditional)
//! sparse approximation using `m` inducing points `Z`, which reduces the
//! O(N³) cost to O(Nm² + m³).
//!
//! ## References
//! * Rasmussen & Williams, "Gaussian Processes for Machine Learning", MIT Press 2006.
//! * Snelson & Ghahramani, "Sparse Gaussian Processes using Pseudo-inputs", NeurIPS 2005.
//! * Wilson & Ghahramani, "Copula Processes", NeurIPS 2010.

use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/// Matérn-5/2 covariance kernel.
///
/// ```text
/// k(x₁, x₂) = σ_f² (1 + √5 r/l + 5r²/(3l²)) · exp(−√5 r/l)
/// ```
///
/// where `r = ||x₁ − x₂||₂`.
pub fn matern52(x1: &[f64], x2: &[f64], length_scale: f64, signal_var: f64) -> f64 {
    debug_assert_eq!(x1.len(), x2.len(), "matern52: dimension mismatch");
    let r_sq: f64 = x1.iter().zip(x2).map(|(a, b)| (a - b) * (a - b)).sum();
    let r = r_sq.sqrt();
    let sqrt5 = 5.0_f64.sqrt();
    let l = length_scale.max(1e-12);
    let u = sqrt5 * r / l;
    signal_var * (1.0 + u + u * u / 3.0) * (-u).exp()
}

// ---------------------------------------------------------------------------
// Kernel matrix helpers
// ---------------------------------------------------------------------------

/// Build the `m × m` kernel matrix `K_ZZ` for inducing points.
fn kernel_matrix_mm(z: &[Vec<f64>], l: f64, sv: f64) -> Vec<Vec<f64>> {
    let m = z.len();
    let mut k = vec![vec![0.0f64; m]; m];
    for i in 0..m {
        for j in 0..m {
            k[i][j] = matern52(&z[i], &z[j], l, sv);
        }
    }
    k
}

/// Build the `n × m` cross-kernel matrix `K_XZ` (data × inducing).
fn kernel_matrix_nm(x: &[Vec<f64>], z: &[Vec<f64>], l: f64, sv: f64) -> Vec<Vec<f64>> {
    let n = x.len();
    let m = z.len();
    let mut k = vec![vec![0.0f64; m]; n];
    for i in 0..n {
        for j in 0..m {
            k[i][j] = matern52(&x[i], &z[j], l, sv);
        }
    }
    k
}

/// Diagonal of `K_XX`: `[k(x_i, x_i)]` for self-covariance.
fn kernel_diag(x: &[Vec<f64>], sv: f64) -> Vec<f64> {
    // Matérn-5/2 diagonal: k(x, x) = σ_f²
    x.iter().map(|_| sv).collect()
}

// ---------------------------------------------------------------------------
// Cholesky factorisation (in-place lower triangle)
// ---------------------------------------------------------------------------

/// Compute the lower Cholesky factor `L` such that `A = L Lᵀ`.
///
/// The matrix is modified in-place to contain `L` (upper triangle zeroed).
/// A small jitter `eps` is added to the diagonal for numerical stability.
fn cholesky(a: &mut Vec<Vec<f64>>, eps: f64) -> Result<()> {
    let n = a.len();
    for i in 0..n {
        a[i][i] += eps;
    }
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= a[i][k] * a[j][k];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(TimeSeriesError::NumericalInstability(format!(
                        "Cholesky: non-positive pivot {s:.3e} at ({i},{i})"
                    )));
                }
                a[i][j] = s.sqrt();
            } else {
                a[i][j] = s / a[j][j];
            }
        }
        // Zero upper triangle
        for j in (i + 1)..n {
            a[i][j] = 0.0;
        }
    }
    Ok(())
}

/// Solve `L x = b` (forward substitution, L lower triangular).
fn forward_sub(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut x = vec![0.0f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i][j] * x[j];
        }
        x[i] = s / l[i][i];
    }
    x
}

/// Solve `Lᵀ x = b` (back substitution, L lower triangular).
fn back_sub(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= l[j][i] * x[j];
        }
        x[i] = s / l[i][i];
    }
    x
}

/// Solve `A x = b` where `A = L Lᵀ` via Cholesky factors.
fn cholesky_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let y = forward_sub(l, b);
    back_sub(l, &y)
}

/// Matrix–vector product: `A x`.
fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

// ---------------------------------------------------------------------------
// K-means initialisation for inducing points
// ---------------------------------------------------------------------------

/// Initialise `m` inducing points from `X` via k-means (Lloyd, 5 iterations).
fn kmeans_init(x: &[Vec<f64>], m: usize, rng_seed: u64) -> Vec<Vec<f64>> {
    let n = x.len();
    if n == 0 || m == 0 {
        return Vec::new();
    }
    let m = m.min(n);
    let d = x[0].len();

    // Seed selection: evenly spaced indices
    let mut centres: Vec<Vec<f64>> = (0..m)
        .map(|i| x[(i * n / m) % n].clone())
        .collect();

    let mut assignments = vec![0usize; n];
    let mut state = rng_seed;

    for _iter in 0..10 {
        // Assignment step
        for (i, xi) in x.iter().enumerate() {
            let mut best = 0;
            let mut best_d = f64::MAX;
            for (c_idx, c) in centres.iter().enumerate() {
                let dist: f64 = xi.iter().zip(c).map(|(a, b)| (a - b).powi(2)).sum();
                if dist < best_d {
                    best_d = dist;
                    best = c_idx;
                }
            }
            assignments[i] = best;
        }

        // Update step
        let mut new_c = vec![vec![0.0f64; d]; m];
        let mut counts = vec![0usize; m];
        for (i, &c_idx) in assignments.iter().enumerate() {
            for k in 0..d {
                new_c[c_idx][k] += x[i][k];
            }
            counts[c_idx] += 1;
        }
        for c_idx in 0..m {
            if counts[c_idx] > 0 {
                for k in 0..d {
                    new_c[c_idx][k] /= counts[c_idx] as f64;
                }
                centres[c_idx] = new_c[c_idx].clone();
            } else {
                // Re-initialise empty cluster from random data point
                let idx = lcg(&mut state) as usize % n;
                centres[c_idx] = x[idx].clone();
            }
        }
    }
    centres
}

/// Simple LCG for index selection.
fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s >> 33
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the GP-GARCH model.
#[derive(Debug, Clone)]
pub struct GpGarchConfig {
    /// Lag order `p`: number of lagged squared returns used as features
    pub p: usize,
    /// Number of inducing points for the sparse FITC approximation
    pub n_inducing: usize,
    /// Observation noise variance `σ²_n`
    pub noise_var: f64,
    /// Matérn-5/2 length scale `l`
    pub kernel_length_scale: f64,
    /// Matérn-5/2 signal variance `σ²_f`
    pub kernel_variance: f64,
    /// Jitter for Cholesky stability
    pub jitter: f64,
}

impl Default for GpGarchConfig {
    fn default() -> Self {
        Self {
            p: 1,
            n_inducing: 20,
            noise_var: 0.01,
            kernel_length_scale: 1.0,
            kernel_variance: 1.0,
            jitter: 1e-6,
        }
    }
}

// ---------------------------------------------------------------------------
// GP-GARCH Model
// ---------------------------------------------------------------------------

/// Fitted GP-GARCH model state.
///
/// After calling [`GpGarchModel::fit`], this struct holds:
/// - The inducing point locations `Z`
/// - The Cholesky factor of `K_ZZ + K_ZX D⁻¹ K_XZ` needed for prediction
/// - The FITC posterior weight vector `α`
#[derive(Debug, Clone)]
pub struct GpGarchModel {
    /// Model configuration
    pub config: GpGarchConfig,
    /// Inducing point locations (`m × p`)
    inducing: Vec<Vec<f64>>,
    /// Posterior weight vector `α` (length `m`)
    alpha: Vec<f64>,
    /// Cholesky factor of the FITC kernel matrix (m × m, lower triangular)
    l_fitc: Vec<Vec<f64>>,
    /// Fitted flag
    fitted: bool,
    /// Approximate variance of the training labels (for uncertainty scaling)
    y_var: f64,
}

impl GpGarchModel {
    /// Create an unfitted model with the given configuration.
    pub fn new(config: GpGarchConfig) -> Self {
        Self {
            config,
            inducing: Vec::new(),
            alpha: Vec::new(),
            l_fitc: Vec::new(),
            fitted: false,
            y_var: 1.0,
        }
    }

    /// Extract the `(n-p)` feature vectors and labels from a return series.
    ///
    /// Feature `x_t = [r_{t-1}², …, r_{t-p}²]`, label `y_t = r_t²`.
    fn extract_features(returns: &[f64], p: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let n = returns.len();
        if n <= p {
            return (Vec::new(), Vec::new());
        }
        let mut x = Vec::with_capacity(n - p);
        let mut y = Vec::with_capacity(n - p);
        for t in p..n {
            let feat: Vec<f64> = (1..=p).map(|lag| returns[t - lag].powi(2)).collect();
            x.push(feat);
            y.push(returns[t].powi(2));
        }
        (x, y)
    }

    /// Fit the GP-GARCH model to a return series.
    ///
    /// # Arguments
    /// - `returns`: time series of log-returns (length ≥ p + 1)
    pub fn fit(&mut self, returns: &[f64]) -> Result<()> {
        let p = self.config.p;
        let (x, y) = Self::extract_features(returns, p);
        let n = x.len();
        if n < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: format!("need at least {} returns for p={} lag order", p + 2, p),
                required: p + 2,
                actual: returns.len(),
            });
        }

        // Empirical variance of labels
        let y_mean = y.iter().sum::<f64>() / n as f64;
        self.y_var = y.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>() / n as f64;
        let y_var = self.y_var.max(1e-10);
        let _ = y_var;

        let m = self.config.n_inducing.min(n);
        let l = self.config.kernel_length_scale;
        let sv = self.config.kernel_variance;
        let noise = self.config.noise_var;
        let jitter = self.config.jitter;

        // Initialise inducing points via k-means
        self.inducing = kmeans_init(&x, m, 0xc0ffee42);

        // Build kernel matrices
        let mut k_zz = kernel_matrix_mm(&self.inducing, l, sv);
        let k_xz = kernel_matrix_nm(&x, &self.inducing, l, sv);
        let k_xx_diag = kernel_diag(&x, sv);

        // Q_XX diagonal: q_ii = K_XZ K_ZZ^{-1} K_XZ^T diagonal
        // Compute K_ZZ^{-1} K_ZX (m × n)
        let mut k_zz_chol = k_zz.clone();
        cholesky(&mut k_zz_chol, jitter)?;

        // Solve K_ZZ^{-1} K_ZX row by row: for each data point x_i, solve K_ZZ v = K_ZX[i,:]
        let mut q_xx_diag = vec![0.0f64; n];
        for i in 0..n {
            let kxz_i: Vec<f64> = (0..m).map(|j| k_xz[i][j]).collect();
            let v = cholesky_solve(&k_zz_chol, &kxz_i);
            q_xx_diag[i] = kxz_i.iter().zip(&v).map(|(a, b)| a * b).sum();
        }

        // D = diag(K_XX - Q_XX) + σ²_n I
        let d_diag: Vec<f64> = k_xx_diag
            .iter()
            .zip(&q_xx_diag)
            .map(|(&kii, &qii)| (kii - qii).max(0.0) + noise)
            .collect();

        // Woodbury: FITC posterior weight
        // Solve (K_ZZ + K_ZX D^{-1} K_XZ) α = K_ZX D^{-1} y
        //
        // Build B = K_ZZ + K_ZX D^{-1} K_XZ
        let mut b_mat = k_zz.clone();
        for j in 0..m {
            for k in 0..m {
                let mut s = 0.0f64;
                for i in 0..n {
                    s += k_xz[i][j] * k_xz[i][k] / d_diag[i];
                }
                b_mat[j][k] += s;
            }
        }

        // rhs = K_ZX D^{-1} y
        let mut rhs = vec![0.0f64; m];
        for j in 0..m {
            let mut s = 0.0f64;
            for i in 0..n {
                s += k_xz[i][j] * y[i] / d_diag[i];
            }
            rhs[j] = s;
        }

        // Cholesky of B
        cholesky(&mut b_mat, jitter)?;
        self.l_fitc = b_mat.clone();

        // Solve B α = rhs
        self.alpha = cholesky_solve(&b_mat, &rhs);
        self.fitted = true;

        Ok(())
    }

    /// Predict conditional variance mean and uncertainty at a feature vector `x`.
    ///
    /// Returns `(mean_variance, predictive_variance)`.
    ///
    /// The mean is guaranteed to be positive (clipped at 1e-8).
    pub fn predict_variance(&self, x: &[f64]) -> Result<(f64, f64)> {
        if !self.fitted {
            return Err(TimeSeriesError::InvalidModel(
                "GpGarchModel: call fit() before predict_variance()".to_string(),
            ));
        }
        let l = self.config.kernel_length_scale;
        let sv = self.config.kernel_variance;

        // K_Zx: kernel vector between inducing points and test point
        let k_zx: Vec<f64> = self
            .inducing
            .iter()
            .map(|z| matern52(z, x, l, sv))
            .collect();

        // Posterior mean: k_Zx^T α
        let mean: f64 = k_zx.iter().zip(&self.alpha).map(|(k, a)| k * a).sum();

        // Posterior variance: k(x,x) - k_Zx^T B^{-1} k_Zx
        let v = cholesky_solve(&self.l_fitc, &k_zx);
        let post_var = (sv - k_zx.iter().zip(&v).map(|(k, vi)| k * vi).sum::<f64>()).max(0.0);

        Ok((mean.max(1e-8), post_var))
    }

    /// Multi-step ahead variance forecast.
    ///
    /// Starting from the trailing `p` values in `returns`, iteratively
    /// predicts `σ²_{T+h}` for `h = 1, …, horizon`.
    pub fn forecast(&self, returns: &[f64], horizon: usize) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(TimeSeriesError::InvalidModel(
                "GpGarchModel: call fit() before forecast()".to_string(),
            ));
        }
        let p = self.config.p;
        if returns.len() < p {
            return Err(TimeSeriesError::InsufficientData {
                message: format!("need at least {p} returns to form feature vector"),
                required: p,
                actual: returns.len(),
            });
        }

        // Build rolling feature window
        let mut window: Vec<f64> = returns[returns.len() - p..].iter().map(|v| v.powi(2)).collect();

        let mut forecasts = Vec::with_capacity(horizon);
        for _ in 0..horizon {
            let (mean_var, _) = self.predict_variance(&window)?;
            forecasts.push(mean_var);
            // Propagate: use predicted variance as next squared return proxy
            window.rotate_left(1);
            *window.last_mut().expect("window is non-empty") = mean_var;
        }
        Ok(forecasts)
    }
}

// ---------------------------------------------------------------------------
// Evaluation metric
// ---------------------------------------------------------------------------

/// QLIKE (Quasi-Likelihood) loss for volatility forecast evaluation.
///
/// Defined as:
///
/// ```text
/// QLIKE(σ², h) = mean_t [ σ²_t / h_t - log(σ²_t / h_t) - 1 ]
/// ```
///
/// where `σ²_t` is the realised variance and `h_t` is the predicted variance.
///
/// Returns `0.0` when predictions equal realised values exactly.
pub fn qlike_score(realized_var: &[f64], predicted_var: &[f64]) -> Result<f64> {
    if realized_var.len() != predicted_var.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: realized_var.len(),
            actual: predicted_var.len(),
        });
    }
    if realized_var.is_empty() {
        return Err(TimeSeriesError::InsufficientData {
            message: "qlike_score requires at least 1 observation".to_string(),
            required: 1,
            actual: 0,
        });
    }
    let n = realized_var.len();
    let mut sum = 0.0f64;
    for (&rv, &pv) in realized_var.iter().zip(predicted_var) {
        let pv = pv.max(1e-12);
        let rv = rv.max(1e-12);
        let ratio = rv / pv;
        sum += ratio - ratio.ln() - 1.0;
    }
    Ok(sum / n as f64)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate an AR(1) return series with time-varying volatility.
    fn ar1_returns(n: usize, phi: f64, sigma: f64) -> Vec<f64> {
        let mut r = Vec::with_capacity(n);
        let mut state: u64 = 0xbeef_cafe;
        let mut prev = 0.0f64;
        for _ in 0..n {
            let u1 = box_muller_u1(&mut state);
            let u2 = box_muller_u2(&mut state);
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let val = phi * prev + sigma * noise;
            r.push(val);
            prev = val;
        }
        r
    }

    fn box_muller_u1(s: &mut u64) -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (*s >> 33) as u64;
        (bits as f64 / (1u64 << 31) as f64).max(1e-12)
    }

    fn box_muller_u2(s: &mut u64) -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (*s >> 33) as u64;
        bits as f64 / (1u64 << 31) as f64
    }

    #[test]
    fn test_matern52_self_covariance() {
        let x = vec![1.0, 2.0, 3.0];
        let sv = 2.5;
        let k = matern52(&x, &x, 1.0, sv);
        assert!(
            (k - sv).abs() < 1e-10,
            "k(x,x) should equal σ_f²={sv}, got {k}"
        );
    }

    #[test]
    fn test_matern52_symmetry() {
        let x1 = vec![1.0, 0.5];
        let x2 = vec![0.3, 1.2];
        let k12 = matern52(&x1, &x2, 1.0, 1.0);
        let k21 = matern52(&x2, &x1, 1.0, 1.0);
        assert!(
            (k12 - k21).abs() < 1e-12,
            "kernel must be symmetric: k(x,y)={k12}, k(y,x)={k21}"
        );
    }

    #[test]
    fn test_matern52_positive_definite() {
        // K(x,x) > 0 for all x
        for &val in &[0.0, 1.0, -3.0, 100.0] {
            let x = vec![val];
            assert!(matern52(&x, &x, 0.5, 1.5) > 0.0);
        }
    }

    #[test]
    fn test_fitc_kzz_cholesky_succeeds() {
        let config = GpGarchConfig {
            p: 1,
            n_inducing: 5,
            noise_var: 0.01,
            kernel_length_scale: 1.0,
            kernel_variance: 1.0,
            jitter: 1e-6,
        };
        let inducing: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * 0.2]).collect();
        let mut k_zz = kernel_matrix_mm(&inducing, config.kernel_length_scale, config.kernel_variance);
        let result = cholesky(&mut k_zz, config.jitter);
        assert!(result.is_ok(), "Cholesky should succeed for SPD K_ZZ");
    }

    #[test]
    fn test_gpgarch_fit_does_not_panic() {
        let mut model = GpGarchModel::new(GpGarchConfig::default());
        let returns = ar1_returns(100, 0.3, 0.01);
        model.fit(&returns).expect("fit should not return error");
    }

    #[test]
    fn test_gpgarch_predicted_variance_positive() {
        let mut model = GpGarchModel::new(GpGarchConfig {
            p: 1,
            n_inducing: 10,
            ..Default::default()
        });
        let returns = ar1_returns(80, 0.2, 0.02);
        model.fit(&returns).expect("fit");
        let x = vec![0.0001];
        let (mean_var, _) = model.predict_variance(&x).expect("predict");
        assert!(mean_var > 0.0, "predicted variance must be positive, got {mean_var}");
    }

    #[test]
    fn test_gpgarch_forecast_length_matches_horizon() {
        let mut model = GpGarchModel::new(GpGarchConfig::default());
        let returns = ar1_returns(80, 0.1, 0.01);
        model.fit(&returns).expect("fit");
        let forecasts = model.forecast(&returns, 5).expect("forecast");
        assert_eq!(forecasts.len(), 5, "forecast length must equal horizon");
    }

    #[test]
    fn test_gpgarch_forecast_all_positive() {
        let mut model = GpGarchModel::new(GpGarchConfig::default());
        let returns = ar1_returns(80, 0.1, 0.01);
        model.fit(&returns).expect("fit");
        let forecasts = model.forecast(&returns, 3).expect("forecast");
        for &v in &forecasts {
            assert!(v > 0.0, "forecast variance must be positive, got {v}");
        }
    }

    #[test]
    fn test_gpgarch_high_vs_low_volatility() {
        let mut model = GpGarchModel::new(GpGarchConfig {
            p: 2,
            n_inducing: 15,
            noise_var: 0.001,
            kernel_length_scale: 1.0,
            kernel_variance: 1.0,
            jitter: 1e-5,
        });
        // Build training set with mixed volatility
        let low_r: Vec<f64> = ar1_returns(60, 0.0, 0.001);
        let high_r: Vec<f64> = ar1_returns(60, 0.0, 0.05);
        let mut all_r = low_r.clone();
        all_r.extend(high_r.clone());
        model.fit(&all_r).expect("fit");

        // High-volatility feature should give higher predicted variance
        let low_feat = vec![1e-6, 1e-6]; // small squared returns → low vol regime
        let high_feat = vec![0.0025, 0.0025]; // large squared returns → high vol regime

        let (low_pred, _) = model.predict_variance(&low_feat).expect("predict low");
        let (high_pred, _) = model.predict_variance(&high_feat).expect("predict high");
        assert!(
            high_pred >= low_pred,
            "high-vol features ({high_feat:?} → {high_pred:.6}) should predict ≥ low-vol ({low_feat:?} → {low_pred:.6})"
        );
    }

    #[test]
    fn test_qlike_equal_predictions_zero() {
        let v = vec![0.01, 0.02, 0.03];
        let score = qlike_score(&v, &v).expect("qlike");
        assert!(
            score.abs() < 1e-10,
            "QLIKE with equal predictions should be 0, got {score}"
        );
    }

    #[test]
    fn test_qlike_positive_for_unequal() {
        let rv = vec![0.02, 0.03, 0.01];
        let pv = vec![0.01, 0.01, 0.02];
        let score = qlike_score(&rv, &pv).expect("qlike");
        assert!(score > 0.0, "QLIKE should be positive for unequal inputs, got {score}");
    }

    #[test]
    fn test_qlike_dimension_mismatch_error() {
        let result = qlike_score(&[0.01, 0.02], &[0.01]);
        assert!(result.is_err(), "mismatched lengths should return error");
    }

    #[test]
    fn test_gpgarch_config_defaults() {
        let cfg = GpGarchConfig::default();
        assert_eq!(cfg.p, 1);
        assert_eq!(cfg.n_inducing, 20);
        assert!((cfg.noise_var - 0.01).abs() < 1e-12);
        assert!((cfg.kernel_length_scale - 1.0).abs() < 1e-12);
        assert!((cfg.kernel_variance - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_gpgarch_predict_before_fit_error() {
        let model = GpGarchModel::new(GpGarchConfig::default());
        let result = model.predict_variance(&[0.0]);
        assert!(result.is_err(), "predict before fit should return error");
    }

    #[test]
    fn test_gpgarch_forecast_before_fit_error() {
        let model = GpGarchModel::new(GpGarchConfig::default());
        let returns = ar1_returns(20, 0.1, 0.01);
        let result = model.forecast(&returns, 3);
        assert!(result.is_err(), "forecast before fit should return error");
    }
}
