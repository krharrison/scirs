//! Linear Mixed Effects Models estimated via REML.
//!
//! The model is:
//! ```text
//! y = X β + Z u + ε
//! u ~ N(0, G),  ε ~ N(0, R = σ²I)
//! ```
//! where X is the fixed-effects design matrix, Z is the random-effects
//! design matrix, β are the fixed-effects coefficients, and u are the
//! random effects.
//!
//! Estimation uses the EM algorithm for variance components (which is
//! equivalent to REML for this parameterization) combined with the
//! Henderson mixed model equations for the fixed and random effects.

use crate::error::{StatsError, StatsResult as Result};

// ---------------------------------------------------------------------------
// Structures
// ---------------------------------------------------------------------------

/// Estimated random effect for one grouping factor.
#[derive(Debug, Clone)]
pub struct RandomEffect {
    /// Level labels (one per unique group).
    pub levels: Vec<String>,
    /// BLUP estimates, one per level.
    pub values: Vec<f64>,
    /// Estimated variance component.
    pub variance: f64,
}

/// Fitted linear mixed effects model.
#[derive(Debug, Clone)]
pub struct MixedEffectsModel {
    /// Fixed effects coefficient estimates (length = number of fixed predictors).
    pub fixed_effects: Vec<f64>,
    /// Random effect components (one per grouping factor).
    pub random_effects: Vec<RandomEffect>,
    /// Residual variance σ².
    pub residual_variance: f64,
    /// REML log-likelihood at convergence.
    pub log_likelihood: f64,
    /// Akaike Information Criterion.
    pub aic: f64,
    /// Bayesian Information Criterion.
    pub bic: f64,
    /// Number of observations.
    pub n_obs: usize,
    /// Converged within tolerance.
    pub converged: bool,
    /// Number of EM iterations run.
    pub n_iter: usize,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl MixedEffectsModel {
    /// Fit a linear mixed effects model using the EM/REML algorithm.
    ///
    /// # Parameters
    /// - `x`: Fixed-effects design matrix (N × p).  Each row is one observation;
    ///   typically includes an intercept column of ones.
    /// - `z`: Random-effects design matrix (N × q).
    /// - `y`: Response vector (N).
    /// - `groups`: Group assignment for each observation (N); values in 0..n_groups.
    ///
    /// # Errors
    /// Returns an error on dimension mismatches or degenerate inputs.
    pub fn fit_reml(
        x: &[Vec<f64>],
        z: &[Vec<f64>],
        y: &[f64],
        groups: &[usize],
    ) -> Result<Self> {
        let n = y.len();
        if n == 0 {
            return Err(StatsError::InsufficientData(
                "y must be non-empty".into(),
            ));
        }
        if x.len() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "x has {} rows, y has {n}",
                x.len()
            )));
        }
        if z.len() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "z has {} rows, y has {n}",
                z.len()
            )));
        }
        if groups.len() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "groups has {} elements, y has {n}",
                groups.len()
            )));
        }

        let p = if x.is_empty() { 0 } else { x[0].len() };
        let q = if z.is_empty() { 0 } else { z[0].len() };

        // Validate row lengths
        for (i, row) in x.iter().enumerate() {
            if row.len() != p {
                return Err(StatsError::DimensionMismatch(format!(
                    "x[{i}] has {} columns, expected {p}",
                    row.len()
                )));
            }
        }
        for (i, row) in z.iter().enumerate() {
            if row.len() != q {
                return Err(StatsError::DimensionMismatch(format!(
                    "z[{i}] has {} columns, expected {q}",
                    row.len()
                )));
            }
        }

        // Find group structure
        let n_groups = groups.iter().copied().max().map(|m| m + 1).unwrap_or(0);

        // Run EM algorithm for variance components
        let (beta, u_blup, sigma2, tau2_vec, log_lik, converged, n_iter) =
            em_reml(x, z, y, groups, n_groups, p, q)?;

        // Build random effects
        let mut group_labels: Vec<Vec<String>> = (0..n_groups)
            .map(|g| vec![g.to_string()])
            .collect();
        // One RandomEffect per random-effect column in Z
        let mut random_effects = Vec::with_capacity(q);
        for k in 0..q.min(tau2_vec.len()) {
            let levels: Vec<String> = (0..n_groups).map(|g| g.to_string()).collect();
            // Extract BLUPs for this random effect column
            let values: Vec<f64> = (0..n_groups)
                .map(|g| {
                    // u_blup is indexed [group * q + col]
                    u_blup.get(g * q + k).copied().unwrap_or(0.0)
                })
                .collect();
            random_effects.push(RandomEffect {
                levels,
                values,
                variance: tau2_vec[k],
            });
        }

        // Compute AIC and BIC
        let k_params = p + q + 1; // fixed effects + variance components + sigma2
        let aic = -2.0 * log_lik + 2.0 * k_params as f64;
        let bic = -2.0 * log_lik + (k_params as f64) * (n as f64).ln();

        Ok(Self {
            fixed_effects: beta,
            random_effects,
            residual_variance: sigma2,
            log_likelihood: log_lik,
            aic,
            bic,
            n_obs: n,
            converged,
            n_iter,
        })
    }

    /// Predict for a new observation using fixed effects only.
    ///
    /// `x_new` must have the same length as the number of fixed-effects columns.
    pub fn predict(&self, x_new: &[f64]) -> Result<f64> {
        if x_new.len() != self.fixed_effects.len() {
            return Err(StatsError::DimensionMismatch(format!(
                "x_new has {} elements, expected {}",
                x_new.len(),
                self.fixed_effects.len()
            )));
        }
        Ok(x_new
            .iter()
            .zip(self.fixed_effects.iter())
            .map(|(&xi, &bi)| xi * bi)
            .sum())
    }

    /// Best Linear Unbiased Predictor for a group (marginal BLUP over random-effect columns).
    ///
    /// Returns the sum of all random effect BLUPs for the given group.
    pub fn blup(&self, group: usize) -> Result<f64> {
        let total: f64 = self
            .random_effects
            .iter()
            .map(|re| re.values.get(group).copied().unwrap_or(0.0))
            .sum();
        Ok(total)
    }

    /// Return the intraclass correlation coefficient (ICC) for the first random effect.
    pub fn icc(&self) -> f64 {
        if self.random_effects.is_empty() {
            return 0.0;
        }
        let tau2 = self.random_effects[0].variance;
        tau2 / (tau2 + self.residual_variance)
    }
}

// ---------------------------------------------------------------------------
// EM algorithm for variance components
// ---------------------------------------------------------------------------

/// Run the EM algorithm for REML estimation of the LME model.
///
/// Returns `(beta, u_blup, sigma2, tau2_vec, log_lik, converged, n_iter)`.
fn em_reml(
    x: &[Vec<f64>],
    z: &[Vec<f64>],
    y: &[f64],
    groups: &[usize],
    n_groups: usize,
    p: usize,
    q: usize,
) -> Result<(Vec<f64>, Vec<f64>, f64, Vec<f64>, f64, bool, usize)> {
    let n = y.len();
    let max_iter = 500;
    let tol = 1e-6;

    // Initialize variance components
    let y_var = sample_variance(y);
    let mut sigma2 = y_var * 0.5;
    let mut tau2_vec = vec![y_var * 0.5; q];

    let mut beta = vec![0.0_f64; p];
    let mut u_blup = vec![0.0_f64; n_groups * q];
    let mut prev_log_lik = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        // ---- E-step: compute mixed model equations ----
        // Henderson's equations: [X'X/σ²  X'Z/σ²] [β]   [X'y/σ²]
        //                        [Z'X/σ²  Z'Z/σ²+G⁻¹] [u] = [Z'y/σ²]
        // where G = diag(τ²) ⊗ I_{n_groups}

        let (beta_new, u_new) = solve_mme(x, z, y, groups, n_groups, p, q, sigma2, &tau2_vec)?;
        beta = beta_new;
        u_blup = u_new;

        // Compute residuals: e = y - Xβ - Zu
        let residuals: Vec<f64> = (0..n)
            .map(|i| {
                let xb: f64 = if p > 0 {
                    x[i].iter().zip(beta.iter()).map(|(&xi, &bi)| xi * bi).sum()
                } else {
                    0.0
                };
                let zu: f64 = if q > 0 {
                    z[i]
                        .iter()
                        .enumerate()
                        .map(|(k, &zik)| {
                            let g = groups[i];
                            zik * u_blup.get(g * q + k).copied().unwrap_or(0.0)
                        })
                        .sum()
                } else {
                    0.0
                };
                y[i] - xb - zu
            })
            .collect();

        // ---- M-step: update variance components ----
        // σ² = ||e||² / (n - p) (REML adjustment)
        let sse: f64 = residuals.iter().map(|&r| r * r).sum();
        sigma2 = (sse / (n.saturating_sub(p).max(1) as f64)).max(1e-10);

        // τ²_k = ||u_k||² / (n_groups - 1)  (REML for random effects)
        for k in 0..q {
            let ss_uk: f64 = (0..n_groups)
                .map(|g| {
                    let uk = u_blup.get(g * q + k).copied().unwrap_or(0.0);
                    uk * uk
                })
                .sum();
            tau2_vec[k] = (ss_uk / (n_groups.saturating_sub(1).max(1) as f64)).max(1e-10);
        }

        // Compute log-likelihood for convergence check
        let log_lik = reml_log_likelihood(y, x, z, groups, &beta, &u_blup, sigma2, &tau2_vec, n, p, q);

        if (log_lik - prev_log_lik).abs() < tol && iter > 5 {
            converged = true;
            break;
        }
        prev_log_lik = log_lik;
    }

    let log_lik = reml_log_likelihood(y, x, z, groups, &beta, &u_blup, sigma2, &tau2_vec, n, p, q);
    Ok((beta, u_blup, sigma2, tau2_vec, log_lik, converged, n_iter))
}

/// Solve Henderson's mixed model equations using a direct approach.
///
/// Assembles and solves the (p+n_groups*q) × (p+n_groups*q) system.
fn solve_mme(
    x: &[Vec<f64>],
    z: &[Vec<f64>],
    y: &[f64],
    groups: &[usize],
    n_groups: usize,
    p: usize,
    q: usize,
    sigma2: f64,
    tau2_vec: &[f64],
) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = y.len();
    let dim = p + n_groups * q;
    if dim == 0 {
        return Ok((vec![], vec![]));
    }

    let inv_sigma2 = 1.0 / sigma2;

    // Build the combined coefficient matrix C and right-hand side rhs
    let mut c = vec![vec![0.0_f64; dim]; dim];
    let mut rhs = vec![0.0_f64; dim];

    for i in 0..n {
        let g = groups[i];
        // Row contributions from fixed effects (indices 0..p)
        for r in 0..p {
            rhs[r] += x[i][r] * y[i] * inv_sigma2;
            for s in 0..p {
                c[r][s] += x[i][r] * x[i][s] * inv_sigma2;
            }
            for k in 0..q {
                let col = p + g * q + k;
                let val = x[i][r] * z[i][k] * inv_sigma2;
                c[r][col] += val;
                c[col][r] += val;
            }
        }

        // Row contributions from random effects
        for k in 0..q {
            let row = p + g * q + k;
            rhs[row] += z[i][k] * y[i] * inv_sigma2;
            for l in 0..q {
                let col = p + g * q + l;
                c[row][col] += z[i][k] * z[i][l] * inv_sigma2;
            }
        }
    }

    // Add G⁻¹ to the random effects diagonal block
    for g in 0..n_groups {
        for k in 0..q {
            let idx = p + g * q + k;
            let tau2 = tau2_vec.get(k).copied().unwrap_or(1.0);
            c[idx][idx] += 1.0 / tau2;
        }
    }

    // Solve C * sol = rhs using Gaussian elimination
    let sol = gaussian_elimination(&mut c, &mut rhs)?;

    let beta = sol[..p].to_vec();
    let u_blup = sol[p..].to_vec();
    Ok((beta, u_blup))
}

/// Gaussian elimination with partial pivoting.
fn gaussian_elimination(a: &mut [Vec<f64>], b: &mut [f64]) -> Result<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        // Partial pivot
        let pivot = (col..n)
            .max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap_or(std::cmp::Ordering::Equal));
        if let Some(p) = pivot {
            a.swap(col, p);
            b.swap(col, p);
        }
        if a[col][col].abs() < 1e-14 {
            // Near-singular: add a tiny regularization to the diagonal
            a[col][col] += 1e-10;
        }
        let pivot_val = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot_val;
            b[row] -= factor * b[col];
            for c in col..n {
                let val = a[col][c];
                a[row][c] -= factor * val;
            }
        }
    }
    // Back substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i][j] * x[j];
        }
        x[i] = if a[i][i].abs() > 1e-14 {
            s / a[i][i]
        } else {
            0.0
        };
    }
    Ok(x)
}

/// Approximate REML log-likelihood.
fn reml_log_likelihood(
    y: &[f64],
    x: &[Vec<f64>],
    z: &[Vec<f64>],
    groups: &[usize],
    beta: &[f64],
    u_blup: &[f64],
    sigma2: f64,
    tau2_vec: &[f64],
    n: usize,
    p: usize,
    q: usize,
) -> f64 {
    let residuals: f64 = (0..n)
        .map(|i| {
            let xb: f64 = x[i].iter().zip(beta.iter()).map(|(&xi, &bi)| xi * bi).sum();
            let zu: f64 = z[i]
                .iter()
                .enumerate()
                .map(|(k, &zik)| {
                    let g = groups[i];
                    zik * u_blup.get(g * q + k).copied().unwrap_or(0.0)
                })
                .sum();
            let r = y[i] - xb - zu;
            r * r
        })
        .sum();

    -0.5 * (n as f64) * (2.0 * std::f64::consts::PI * sigma2).ln()
        - 0.5 * residuals / sigma2
}

fn sample_variance(x: &[f64]) -> f64 {
    if x.len() <= 1 {
        return 1.0;
    }
    let m = x.iter().sum::<f64>() / x.len() as f64;
    x.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / (x.len() - 1) as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_lme() -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, Vec<usize>) {
        // 30 observations, 3 groups (10 each), 1 fixed predictor (intercept),
        // 1 random intercept per group.
        let mut x = Vec::new();
        let mut z = Vec::new();
        let mut y = Vec::new();
        let mut groups = Vec::new();
        // True: beta=[5.0], u=[0.5, -0.3, 0.1], sigma=0.4
        let u_true = [0.5_f64, -0.3, 0.1];
        let beta_true = 5.0_f64;
        let mut pseudo_rng = 12345_u64;
        for g in 0..3 {
            for k in 0..10 {
                let noise = lcg_sample(&mut pseudo_rng) * 0.8 - 0.4; // [-0.4, 0.4]
                let yi = beta_true + u_true[g] + noise;
                x.push(vec![1.0]);
                z.push(vec![1.0]);
                y.push(yi);
                groups.push(g);
            }
        }
        (x, z, y, groups)
    }

    fn lcg_sample(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (*state >> 33) as f64 / u32::MAX as f64
    }

    #[test]
    fn test_fit_reml_basic() {
        let (x, z, y, groups) = make_simple_lme();
        let model = MixedEffectsModel::fit_reml(&x, &z, &y, &groups).unwrap();

        // Fixed effect should be near 5.0
        assert!((model.fixed_effects[0] - 5.0).abs() < 1.0,
            "beta={}", model.fixed_effects[0]);
        assert!(model.residual_variance > 0.0);
        assert!(model.log_likelihood.is_finite());
        assert!(model.aic.is_finite());
        assert!(model.bic.is_finite());
    }

    #[test]
    fn test_predict() {
        let (x, z, y, groups) = make_simple_lme();
        let model = MixedEffectsModel::fit_reml(&x, &z, &y, &groups).unwrap();
        let pred = model.predict(&[1.0]).unwrap();
        assert!(pred.is_finite());
        assert!(model.predict(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_blup() {
        let (x, z, y, groups) = make_simple_lme();
        let model = MixedEffectsModel::fit_reml(&x, &z, &y, &groups).unwrap();
        let b0 = model.blup(0).unwrap();
        let b1 = model.blup(1).unwrap();
        let b2 = model.blup(2).unwrap();
        // BLUPs should be finite
        assert!(b0.is_finite());
        assert!(b1.is_finite());
        assert!(b2.is_finite());
    }

    #[test]
    fn test_icc() {
        let (x, z, y, groups) = make_simple_lme();
        let model = MixedEffectsModel::fit_reml(&x, &z, &y, &groups).unwrap();
        let icc = model.icc();
        assert!(icc >= 0.0 && icc <= 1.0, "ICC={icc}");
    }

    #[test]
    fn test_dimension_errors() {
        let x = vec![vec![1.0], vec![1.0]];
        let z = vec![vec![1.0], vec![1.0]];
        let y = vec![1.0, 2.0, 3.0]; // wrong length
        let groups = vec![0, 1];
        assert!(MixedEffectsModel::fit_reml(&x, &z, &y, &groups).is_err());

        let y2 = vec![1.0, 2.0];
        assert!(MixedEffectsModel::fit_reml(&[], &z, &y2, &groups).is_err());
    }
}
