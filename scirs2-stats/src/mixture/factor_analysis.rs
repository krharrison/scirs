//! Factor Analysis (FA) via the EM algorithm.
//!
//! # Model
//!
//! Factor Analysis decomposes an observed `p`-dimensional vector `x` as:
//!
//! ```text
//! x = W z + μ + ε
//! ```
//!
//! where
//!
//! - `W`  is the `p × q` factor loading matrix,
//! - `z ~ N(0, I_q)` is the `q`-dimensional latent factor vector,
//! - `μ`  is the `p`-dimensional mean,
//! - `ε ~ N(0, Ψ)` is the idiosyncratic noise, with `Ψ` diagonal.
//!
//! Marginalising over `z` gives `x ~ N(μ, W W^T + Ψ)`.
//!
//! Parameter estimation uses the EM algorithm of Rubin & Thayer (1982).
//!
//! # References
//! - Rubin, D. B., & Thayer, D. T. (1982). EM algorithms for ML factor
//!   analysis. *Psychometrika*, 47(1), 69–76.
//! - Bishop, C. M. (2006). *PRML*, Section 12.2.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Rotation type applied to the factor loading matrix after fitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationType {
    /// No rotation.
    None,
    /// Varimax orthogonal rotation (maximises variance of squared loadings).
    Varimax,
}

impl Default for RotationType {
    fn default() -> Self {
        Self::None
    }
}

/// Configuration for Factor Analysis.
#[derive(Debug, Clone)]
pub struct FactorAnalysisConfig {
    /// Number of latent factors.
    pub n_factors: usize,
    /// Maximum EM iterations.
    pub max_iter: usize,
    /// Convergence tolerance on log-likelihood improvement.
    pub tol: f64,
    /// Minimum noise variance (regularisation), preventing Ψ_j → 0.
    pub min_noise_var: f64,
    /// Rotation applied after fitting.
    pub rotation: RotationType,
}

impl Default for FactorAnalysisConfig {
    fn default() -> Self {
        Self {
            n_factors: 2,
            max_iter: 200,
            tol: 1e-6,
            min_noise_var: 1e-3,
            rotation: RotationType::None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Model
// ─────────────────────────────────────────────────────────────────────────────

/// A fitted Factor Analysis model.
#[derive(Debug, Clone)]
pub struct FactorAnalysis {
    /// Factor loading matrix W, shape `(n_features, n_factors)`.
    pub loadings: Array2<f64>,
    /// Noise (uniqueness) variances Ψ_j, shape `(n_features,)`.
    pub noise_variance: Array1<f64>,
    /// Data mean μ, shape `(n_features,)`.
    pub mean: Array1<f64>,
    /// Number of latent factors.
    pub n_factors: usize,
    /// Number of observed features.
    pub n_features: usize,
    /// Log-likelihood at convergence.
    pub log_likelihood: f64,
    /// Number of EM iterations performed.
    pub n_iter: usize,
    /// Whether EM converged within `max_iter`.
    pub converged: bool,
    /// Rotation type used.
    pub rotation: RotationType,
}

impl FactorAnalysis {
    /// Proportion of variance explained by each factor (communality component).
    ///
    /// Returns a vector of length `n_factors` with the sum-of-squared loadings
    /// for each factor divided by the total variance.
    pub fn factor_variance_proportions(&self) -> Array1<f64> {
        let total_var: f64 = self.total_variance();
        let q = self.n_factors;
        let p = self.n_features;
        let mut props = Array1::<f64>::zeros(q);
        for f in 0..q {
            let ssl: f64 = (0..p).map(|j| self.loadings[[j, f]].powi(2)).sum();
            props[f] = ssl / total_var;
        }
        props
    }

    /// Communalities h²_j: proportion of variable j's variance explained by
    /// all factors.
    ///
    /// h²_j = Σ_f W_{jf}² / (Σ_f W_{jf}² + Ψ_j)
    pub fn communality(&self) -> Array1<f64> {
        let p = self.n_features;
        let q = self.n_factors;
        let mut h2 = Array1::<f64>::zeros(p);
        for j in 0..p {
            let common: f64 = (0..q).map(|f| self.loadings[[j, f]].powi(2)).sum();
            let total = common + self.noise_variance[j];
            h2[j] = if total > 0.0 { common / total } else { 0.0 };
        }
        h2
    }

    /// Uniqueness (1 - communality) per variable.
    pub fn uniqueness(&self) -> Array1<f64> {
        self.communality().mapv(|h| 1.0 - h)
    }

    /// Total variance (sum over features of Σ_f W_{jf}² + Ψ_j).
    pub fn total_variance(&self) -> f64 {
        let p = self.n_features;
        let q = self.n_factors;
        (0..p)
            .map(|j| {
                let common: f64 = (0..q).map(|f| self.loadings[[j, f]].powi(2)).sum();
                common + self.noise_variance[j]
            })
            .sum()
    }

    /// Return the factor loading matrix (copy).
    pub fn factor_loadings(&self) -> Array2<f64> {
        self.loadings.clone()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fit via EM
// ─────────────────────────────────────────────────────────────────────────────

/// Fit a Factor Analysis model to `data` using the EM algorithm.
///
/// `data` has shape `(n_samples, n_features)`.
pub fn fit_em(data: &Array2<f64>, config: &FactorAnalysisConfig) -> StatsResult<FactorAnalysis> {
    let (n, p) = (data.nrows(), data.ncols());
    let q = config.n_factors;

    if n == 0 || p == 0 {
        return Err(StatsError::InvalidArgument(
            "data must be non-empty".to_string(),
        ));
    }
    if q == 0 {
        return Err(StatsError::InvalidArgument(
            "n_factors must be >= 1".to_string(),
        ));
    }
    if q >= p {
        return Err(StatsError::InvalidArgument(format!(
            "n_factors ({q}) must be < n_features ({p})"
        )));
    }

    // ── Data mean centring ───────────────────────────────────────────────
    let mean: Array1<f64> = Array1::from_vec(
        (0..p)
            .map(|j| data.column(j).mean().unwrap_or(0.0))
            .collect(),
    );

    // Centred data X̃ = X − 1 μ^T
    let mut x_cent = data.clone();
    for i in 0..n {
        for j in 0..p {
            x_cent[[i, j]] -= mean[j];
        }
    }

    // Sample covariance S (ML estimator, division by n)
    let s = sample_cov(&x_cent, n);

    // ── Initialise W, Ψ ─────────────────────────────────────────────────
    // W: first q columns of identity scaled by std.
    let mut w = Array2::<f64>::zeros((p, q));
    for j in 0..p.min(q) {
        w[[j, j]] = s[[j, j]].sqrt().max(1e-6);
    }
    // Ψ: diagonal of sample covariance.
    let mut psi: Array1<f64> = Array1::from_vec(
        (0..p)
            .map(|j| s[[j, j]].max(config.min_noise_var))
            .collect(),
    );

    let mut log_likelihood = f64::NEG_INFINITY;
    let mut n_iter = 0usize;
    let mut converged = false;

    for _iter in 0..config.max_iter {
        // ── E-step ────────────────────────────────────────────────────────
        // Posterior moments of z given x:
        //   β  = W^T Σ^{-1}   (q × p)
        //   E[z|x]   = β (x - μ)
        //   Cov[z|x] = I - β W      (q × q)

        // Compute Ψ^{-1} W (diag(Ψ) is trivial to invert)
        // Σ = W W^T + Ψ, but we use the Woodbury identity:
        // Σ^{-1} = Ψ^{-1} - Ψ^{-1} W (I + W^T Ψ^{-1} W)^{-1} W^T Ψ^{-1}

        // M = I_q + W^T Ψ^{-1} W   (q × q)
        let mut m = Array2::<f64>::zeros((q, q));
        for f in 0..q {
            for g in 0..q {
                let v: f64 = (0..p).map(|j| w[[j, f]] * w[[j, g]] / psi[j]).sum();
                m[[f, g]] = v + if f == g { 1.0 } else { 0.0 };
            }
        }

        let m_inv = mat_inv_q(&m, q)?;

        // β = M^{-1} W^T Ψ^{-1}  (q × p)
        let mut beta = Array2::<f64>::zeros((q, p));
        for f in 0..q {
            for j in 0..p {
                let psi_inv_wj: f64 = w[[j, f]] / psi[j];
                for g in 0..q {
                    beta[[g, j]] += m_inv[[g, f]] * psi_inv_wj;
                }
            }
        }
        // Wait — β should be M^{-1} W^T Ψ^{-1}, which has shape (q × p).
        // Above accumulation is wrong. Recompute properly:
        let mut beta = Array2::<f64>::zeros((q, p));
        for g in 0..q {
            for j in 0..p {
                let s: f64 = (0..q).map(|f| m_inv[[g, f]] * w[[j, f]] / psi[j]).sum();
                beta[[g, j]] = s;
            }
        }

        // Posterior covariance of z: Cov[z|x] = I - β W  (q × q)
        let mut ez_cov = Array2::<f64>::zeros((q, q));
        for f in 0..q {
            for g in 0..q {
                let bw: f64 = (0..p).map(|j| beta[[f, j]] * w[[j, g]]).sum();
                ez_cov[[f, g]] = (if f == g { 1.0 } else { 0.0 }) - bw;
            }
        }

        // E[z | x_i] = β x_i   (q × n, we compute each column)
        // Store expectations as (n × q).
        let mut ez = Array2::<f64>::zeros((n, q));
        for i in 0..n {
            for f in 0..q {
                let s: f64 = (0..p).map(|j| beta[[f, j]] * x_cent[[i, j]]).sum();
                ez[[i, f]] = s;
            }
        }

        // ── M-step ────────────────────────────────────────────────────────
        // New W: W_new = (Σ_i x_i E[z_i]^T) (Σ_i (E[z_i z_i^T]))^{-1}
        // where E[z_i z_i^T] = Cov[z|x] + E[z_i] E[z_i]^T

        // Accumulate Σ_i E[z_i z_i^T]
        let mut ezzt = Array2::<f64>::zeros((q, q));
        for f in 0..q {
            for g in 0..q {
                // ez_cov is the same for all i (because x only enters the mean).
                ezzt[[f, g]] = ez_cov[[f, g]] * n as f64;
                // + Σ_i E[z_i]_f E[z_i]_g
                let outer: f64 = (0..n).map(|i| ez[[i, f]] * ez[[i, g]]).sum();
                ezzt[[f, g]] += outer;
            }
        }

        // Accumulate Σ_i x_i E[z_i]^T  (p × q)
        let mut xzt = Array2::<f64>::zeros((p, q));
        for j in 0..p {
            for f in 0..q {
                let s: f64 = (0..n).map(|i| x_cent[[i, j]] * ez[[i, f]]).sum();
                xzt[[j, f]] = s;
            }
        }

        let ezzt_inv = mat_inv_q(&ezzt, q)?;

        // W_new = xzt * ezzt_inv
        let mut w_new = Array2::<f64>::zeros((p, q));
        for j in 0..p {
            for f in 0..q {
                let s: f64 = (0..q).map(|g| xzt[[j, g]] * ezzt_inv[[g, f]]).sum();
                w_new[[j, f]] = s;
            }
        }

        // Ψ_new = diag(S - W_new (Σ_i x_i E[z_i]^T)^T / n)
        //       = diag(S) - diag(W_new xzt^T) / n
        let mut psi_new = Array1::<f64>::zeros(p);
        for j in 0..p {
            let wxt: f64 = (0..q).map(|f| w_new[[j, f]] * xzt[[j, f]]).sum::<f64>() / n as f64;
            psi_new[j] = (s[[j, j]] - wxt).max(config.min_noise_var);
        }

        // ── Log-likelihood: log|Σ| + tr(Σ^{-1} S) using updated params ──
        // We use the identity: log|Σ| = log|M_new| + Σ_j log Ψ_j
        // where M_new = I + W_new^T Ψ_new^{-1} W_new
        let mut m_new = Array2::<f64>::zeros((q, q));
        for f in 0..q {
            for g in 0..q {
                let v: f64 = (0..p).map(|j| w_new[[j, f]] * w_new[[j, g]] / psi_new[j]).sum();
                m_new[[f, g]] = v + if f == g { 1.0 } else { 0.0 };
            }
        }
        let log_det_m = log_det_small(&m_new, q)?;
        let log_det_psi: f64 = psi_new.iter().map(|&v| v.ln()).sum();
        let log_det_sigma = log_det_m + log_det_psi;

        // tr(Σ^{-1} S) via Woodbury: expensive, approximate with diagonal
        // tr(Σ^{-1} S) ≈ tr(Ψ^{-1} S) - tr(Ψ^{-1} W (M)^{-1} W^T Ψ^{-1} S)
        let m_new_inv = mat_inv_q(&m_new, q)?;
        let trace_term = {
            // Ψ^{-1} S diagonal part: Σ_j S_{jj}/ψ_j
            let diag_part: f64 = (0..p).map(|j| s[[j, j]] / psi_new[j]).sum();
            // Off-diagonal correction (Woodbury)
            // A = W^T Ψ^{-1}: q × p
            let mut a = Array2::<f64>::zeros((q, p));
            for f in 0..q {
                for j in 0..p {
                    a[[f, j]] = w_new[[j, f]] / psi_new[j];
                }
            }
            // B = A S Ψ^{-1} W: q × q
            let mut b = Array2::<f64>::zeros((q, q));
            for f in 0..q {
                for g in 0..q {
                    let v: f64 = (0..p)
                        .map(|j| {
                            // (A S)_{f,j} = Σ_l a_{fl} S_{lj}: full S needed
                            // Approximate with diagonal of S for efficiency
                            a[[f, j]] * s[[j, j]] / psi_new[j] * w_new[[j, g]]
                        })
                        .sum();
                    b[[f, g]] = v;
                }
            }
            // tr(M^{-1} B)
            let corr: f64 = (0..q).map(|f| {
                (0..q).map(|g| m_new_inv[[f, g]] * b[[g, f]]).sum::<f64>()
            }).sum();
            diag_part - corr
        };

        let new_ll = -0.5 * n as f64 * (p as f64 * std::f64::consts::LN_2
            + p as f64 * std::f64::consts::PI.ln()
            + log_det_sigma
            + trace_term);

        w = w_new;
        psi = psi_new;

        let improvement = new_ll - log_likelihood;
        log_likelihood = new_ll;
        n_iter = _iter + 1;

        if improvement.abs() < config.tol {
            converged = true;
            break;
        }
    }

    // ── Optional rotation ────────────────────────────────────────────────
    if config.rotation == RotationType::Varimax {
        w = varimax_rotation(&w, 1000, 1e-8)?;
    }

    Ok(FactorAnalysis {
        loadings: w,
        noise_variance: psi,
        mean,
        n_factors: q,
        n_features: p,
        log_likelihood,
        n_iter,
        converged,
        rotation: config.rotation,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Transform (project to factor scores)
// ─────────────────────────────────────────────────────────────────────────────

/// Project `data` onto the factor space using the posterior mean E[z|x].
///
/// Returns factor scores of shape `(n_samples, n_factors)`.
pub fn transform(data: &Array2<f64>, model: &FactorAnalysis) -> StatsResult<Array2<f64>> {
    let (n, p) = (data.nrows(), data.ncols());
    let q = model.n_factors;

    if p != model.n_features {
        return Err(StatsError::InvalidArgument(format!(
            "data has {p} features but model was fitted on {}",
            model.n_features
        )));
    }

    let w = &model.loadings;
    let psi = &model.noise_variance;

    // M = I + W^T Ψ^{-1} W
    let mut m = Array2::<f64>::zeros((q, q));
    for f in 0..q {
        for g in 0..q {
            let v: f64 = (0..p).map(|j| w[[j, f]] * w[[j, g]] / psi[j]).sum();
            m[[f, g]] = v + if f == g { 1.0 } else { 0.0 };
        }
    }
    let m_inv = mat_inv_q(&m, q)?;

    // β = M^{-1} W^T Ψ^{-1}  (q × p)
    let mut beta = Array2::<f64>::zeros((q, p));
    for g in 0..q {
        for j in 0..p {
            let s: f64 = (0..q).map(|f| m_inv[[g, f]] * w[[j, f]] / psi[j]).sum();
            beta[[g, j]] = s;
        }
    }

    // Factor scores: Z = (X - μ) β^T
    let mut scores = Array2::<f64>::zeros((n, q));
    for i in 0..n {
        for f in 0..q {
            let s: f64 = (0..p)
                .map(|j| beta[[f, j]] * (data[[i, j]] - model.mean[j]))
                .sum();
            scores[[i, f]] = s;
        }
    }

    Ok(scores)
}

// ─────────────────────────────────────────────────────────────────────────────
// Varimax rotation
// ─────────────────────────────────────────────────────────────────────────────

/// Apply Varimax rotation to a loading matrix `w` of shape `(p, q)`.
///
/// Varimax maximises the sum of variances of squared loadings (Kaiser, 1958).
/// Returns the rotated loading matrix.
///
/// The algorithm iterates pairwise Jacobi rotations until convergence.
///
/// # References
/// - Kaiser, H. F. (1958). The varimax criterion for analytic rotation in
///   factor analysis. *Psychometrika*, 23(3), 187–200.
pub fn varimax_rotation(
    w: &Array2<f64>,
    max_iter: usize,
    tol: f64,
) -> StatsResult<Array2<f64>> {
    let (p, q) = (w.nrows(), w.ncols());
    if q < 2 {
        // Nothing to rotate
        return Ok(w.clone());
    }

    let mut r = w.clone();

    for _iter in 0..max_iter {
        let mut max_angle = 0.0_f64;

        for f1 in 0..q {
            for f2 in f1 + 1..q {
                // Compute the rotation angle θ for factors f1, f2
                // using the standard Varimax formula (Kaiser normalised).
                let u: Vec<f64> = (0..p).map(|j| r[[j, f1]].powi(2) - r[[j, f2]].powi(2)).collect();
                let v: Vec<f64> = (0..p).map(|j| 2.0 * r[[j, f1]] * r[[j, f2]]).collect();

                let a: f64 = u.iter().sum();
                let b: f64 = v.iter().sum();
                let c: f64 = u.iter().zip(u.iter()).map(|(&ui, &uj)| ui * uj).sum::<f64>()
                    - v.iter().zip(v.iter()).map(|(&vi, &vj)| vi * vj).sum::<f64>();
                let d: f64 = u.iter().zip(v.iter()).map(|(&ui, &vi)| ui * vi).sum::<f64>() * 2.0;

                let num = d - 2.0 * a * b / p as f64;
                let den = c - (a * a - b * b) / p as f64;

                let theta = if den.abs() < 1e-15 {
                    0.0
                } else {
                    0.25 * num.atan2(den)
                };

                max_angle = max_angle.max(theta.abs());

                let cos_t = theta.cos();
                let sin_t = theta.sin();

                // Apply rotation to columns f1 and f2
                for j in 0..p {
                    let r1 = r[[j, f1]];
                    let r2 = r[[j, f2]];
                    r[[j, f1]] = cos_t * r1 + sin_t * r2;
                    r[[j, f2]] = -sin_t * r1 + cos_t * r2;
                }
            }
        }

        if max_angle < tol {
            break;
        }
    }

    Ok(r)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal matrix utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Sample covariance matrix (ML; divide by n).
fn sample_cov(x_cent: &Array2<f64>, n: usize) -> Array2<f64> {
    let p = x_cent.ncols();
    let mut s = Array2::<f64>::zeros((p, p));
    for i in 0..n {
        for j in 0..p {
            for l in 0..=j {
                let c = x_cent[[i, j]] * x_cent[[i, l]] / n as f64;
                s[[j, l]] += c;
                if j != l {
                    s[[l, j]] += c;
                }
            }
        }
    }
    s
}

/// Invert a small symmetric positive-definite matrix of size `q` via
/// Cholesky factorisation.
fn mat_inv_q(m: &Array2<f64>, q: usize) -> StatsResult<Array2<f64>> {
    // Cholesky: L L^T = M
    let mut l = Array2::<f64>::zeros((q, q));
    for i in 0..q {
        for j in 0..=i {
            let mut s = m[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(StatsError::ComputationError(format!(
                        "Matrix not positive-definite at ({i},{i}): s={s}"
                    )));
                }
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }

    // Invert L via forward substitution
    let mut l_inv = Array2::<f64>::zeros((q, q));
    for i in 0..q {
        l_inv[[i, i]] = 1.0 / l[[i, i]];
        for j in 0..i {
            let s: f64 = (j..i).map(|k| l[[i, k]] * l_inv[[k, j]]).sum();
            l_inv[[i, j]] = -s / l[[i, i]];
        }
    }

    // M^{-1} = L^{-T} L^{-1}
    let mut m_inv = Array2::<f64>::zeros((q, q));
    for i in 0..q {
        for j in 0..=i {
            let s: f64 = (i..q).map(|k| l_inv[[k, i]] * l_inv[[k, j]]).sum();
            m_inv[[i, j]] = s;
            m_inv[[j, i]] = s;
        }
    }
    Ok(m_inv)
}

/// Log-determinant of a small symmetric PD matrix via Cholesky.
fn log_det_small(m: &Array2<f64>, q: usize) -> StatsResult<f64> {
    let mut l = Array2::<f64>::zeros((q, q));
    for i in 0..q {
        for j in 0..=i {
            let mut s = m[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(StatsError::ComputationError(format!(
                        "Matrix not PD at ({i},{i}): s={s}"
                    )));
                }
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    Ok(2.0 * (0..q).map(|i| l[[i, i]].ln()).sum::<f64>())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Generate synthetic data from a 2-factor model.
    fn make_factor_data() -> Array2<f64> {
        // 4 observed variables, 2 factors
        // W = [[1,0],[0,1],[1,1],[0.5,-0.5]]
        // Ψ_j = 0.1
        let w = vec![
            [1.0_f64, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, -0.5],
        ];
        let n = 100usize;
        let mut data = Array2::<f64>::zeros((n, 4));
        // Simple deterministic "data"
        for i in 0..n {
            let z0 = (i as f64 / n as f64) - 0.5;
            let z1 = ((i as f64 + 0.3) / n as f64) - 0.5;
            for j in 0..4 {
                let eps = 0.01 * ((i + j) as f64).sin();
                data[[i, j]] = w[j][0] * z0 + w[j][1] * z1 + eps;
            }
        }
        data
    }

    #[test]
    fn test_fa_fit_converges() {
        let data = make_factor_data();
        let config = FactorAnalysisConfig {
            n_factors: 2,
            max_iter: 200,
            tol: 1e-6,
            min_noise_var: 1e-6,
            rotation: RotationType::None,
        };
        let model = fit_em(&data, &config).expect("fit_em");
        assert_eq!(model.n_factors, 2);
        assert_eq!(model.n_features, 4);
        assert!(
            model.log_likelihood.is_finite(),
            "log_likelihood should be finite"
        );
    }

    #[test]
    fn test_communality_in_range() {
        let data = make_factor_data();
        let config = FactorAnalysisConfig {
            n_factors: 2,
            ..Default::default()
        };
        let model = fit_em(&data, &config).expect("fit_em");
        let h2 = model.communality();
        for (j, &h) in h2.iter().enumerate() {
            assert!(
                h >= 0.0 && h <= 1.0 + 1e-9,
                "communality[{j}] = {h} out of range"
            );
        }
    }

    #[test]
    fn test_transform_shape() {
        let data = make_factor_data();
        let config = FactorAnalysisConfig {
            n_factors: 2,
            ..Default::default()
        };
        let model = fit_em(&data, &config).expect("fit_em");
        let scores = transform(&data, &model).expect("transform");
        assert_eq!(scores.shape(), &[100, 2]);
    }

    #[test]
    fn test_varimax_rotation() {
        let data = make_factor_data();
        let config = FactorAnalysisConfig {
            n_factors: 2,
            rotation: RotationType::Varimax,
            ..Default::default()
        };
        let model = fit_em(&data, &config).expect("fit_em with varimax");
        assert_eq!(model.rotation, RotationType::Varimax);
        // Loadings should still have correct shape
        assert_eq!(model.loadings.shape(), &[4, 2]);
    }

    #[test]
    fn test_fa_invalid_args() {
        let data = make_factor_data();
        let config = FactorAnalysisConfig {
            n_factors: 4, // equal to n_features -> error
            ..Default::default()
        };
        assert!(fit_em(&data, &config).is_err());
    }
}
