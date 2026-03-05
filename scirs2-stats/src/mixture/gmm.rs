//! Gaussian Mixture Model (GMM) with full EM algorithm.
//!
//! # Overview
//!
//! A GMM models data as a weighted sum of Gaussian components:
//!
//! ```text
//! p(x) = Σ_k  w_k  N(x | μ_k, Σ_k)
//! ```
//!
//! Parameter estimation is performed via the Expectation-Maximisation (EM)
//! algorithm with numerically stable log-sum-exp arithmetic.
//!
//! # Covariance Types
//!
//! | Type        | Parameters | Description                                   |
//! |-------------|-----------|-----------------------------------------------|
//! | `Full`      | k·d²      | Each component has its own full covariance     |
//! | `Diag`      | k·d       | Each component has diagonal covariance         |
//! | `Spherical` | k         | Each component shares a single variance σ²     |
//! | `Tied`      | d²        | All components share one covariance matrix     |
//!
//! # References
//! - Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Ch. 9.
//! - McLachlan, G. J., & Peel, D. (2000). *Finite Mixture Models*.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::random::rand_distr::StandardNormal;
use scirs2_core::random::{Rng, SeedableRng};

// ─────────────────────────────────────────────────────────────────────────────
// Public enums / config
// ─────────────────────────────────────────────────────────────────────────────

/// Covariance parameterisation for each GMM component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GmmCovarianceType {
    /// Each component has an unconstrained covariance matrix (d×d).
    Full,
    /// Each component has a diagonal covariance (d variances).
    Diag,
    /// Each component has a single shared variance scalar.
    Spherical,
    /// All components share one full covariance matrix.
    Tied,
}

impl Default for GmmCovarianceType {
    fn default() -> Self {
        Self::Full
    }
}

/// Initialisation strategy for GMM component parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GmmInit {
    /// K-means++ seeding for means, identity for covariances.
    KMeansPlusPlus,
    /// Random sample rows as initial means.
    Random,
}

impl Default for GmmInit {
    fn default() -> Self {
        Self::KMeansPlusPlus
    }
}

/// Configuration for a Gaussian Mixture Model.
#[derive(Debug, Clone)]
pub struct GMMConfig {
    /// Number of Gaussian components.
    pub n_components: usize,
    /// Maximum EM iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the log-likelihood improvement.
    pub tol: f64,
    /// Covariance type.
    pub covariance_type: GmmCovarianceType,
    /// Regularisation added to diagonal of each covariance to prevent
    /// singular matrices.
    pub reg_covar: f64,
    /// Initialisation strategy.
    pub init: GmmInit,
    /// Optional RNG seed (None → thread-local entropy).
    pub seed: Option<u64>,
}

impl Default for GMMConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            max_iter: 200,
            tol: 1e-6,
            covariance_type: GmmCovarianceType::Full,
            reg_covar: 1e-6,
            init: GmmInit::KMeansPlusPlus,
            seed: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Model parameters
// ─────────────────────────────────────────────────────────────────────────────

/// Fitted parameters of a Gaussian Mixture Model.
#[derive(Debug, Clone)]
pub struct GMMModel {
    /// Component means: shape `(n_components, n_features)`.
    pub means: Array2<f64>,
    /// Component covariances.
    ///
    /// * `Full`      → Vec of `(n_features, n_features)` matrices, length `n_components`.
    /// * `Diag`      → `(n_components, n_features)` matrix of per-feature variances.
    /// * `Spherical` → `(n_components,)` vector of scalar variances.
    /// * `Tied`      → single `(n_features, n_features)` matrix (stored as the first element).
    pub covariances: Vec<Array2<f64>>,
    /// Log mixing weights (log π_k): shape `(n_components,)`.
    pub log_weights: Array1<f64>,
    /// Soft assignments (responsibilities) from the last E-step:
    /// shape `(n_samples, n_components)`.
    pub responsibilities: Array2<f64>,
    /// Log-likelihood at convergence.
    pub log_likelihood: f64,
    /// Number of EM iterations performed.
    pub n_iter: usize,
    /// Whether the algorithm converged within `max_iter`.
    pub converged: bool,
    /// Number of features.
    pub n_features: usize,
    /// Number of components.
    pub n_components: usize,
    /// Covariance type (kept for prediction & sampling).
    pub covariance_type: GmmCovarianceType,
}

impl GMMModel {
    /// Mixing weights π_k (exponentiated log-weights).
    pub fn weights(&self) -> Array1<f64> {
        self.log_weights.mapv(f64::exp)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically stable log-sum-exp over a 1-D slice.
fn log_sum_exp(logits: &[f64]) -> f64 {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = logits.iter().map(|&x| (x - max).exp()).sum();
    max + sum.ln()
}

/// Log-determinant of a positive-definite matrix via Cholesky factorisation.
/// Returns `Err` if the matrix is not positive-definite.
fn log_det_chol(mat: &Array2<f64>) -> StatsResult<f64> {
    let d = mat.nrows();
    // Simple Cholesky (L L^T) – lower-triangular factor.
    let mut l = Array2::<f64>::zeros((d, d));
    for i in 0..d {
        for j in 0..=i {
            let mut s = mat[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(StatsError::ComputationError(format!(
                        "Matrix not positive-definite at pivot ({i}, {i}): s={s}"
                    )));
                }
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    // log-det = 2 * Σ log(L_ii)
    let log_det = 2.0 * (0..d).map(|i| l[[i, i]].ln()).sum::<f64>();
    Ok(log_det)
}

/// Solve L x = b for lower-triangular L.
fn forward_solve(l: &Array2<f64>, b: &[f64]) -> Vec<f64> {
    let d = l.nrows();
    let mut x = vec![0.0_f64; d];
    for i in 0..d {
        let mut s = b[i];
        for j in 0..i {
            s -= l[[i, j]] * x[j];
        }
        x[i] = s / l[[i, i]];
    }
    x
}

/// Cholesky factorisation: returns lower-triangular L such that mat = L L^T.
fn cholesky(mat: &Array2<f64>) -> StatsResult<Array2<f64>> {
    let d = mat.nrows();
    let mut l = Array2::<f64>::zeros((d, d));
    for i in 0..d {
        for j in 0..=i {
            let mut s = mat[[i, j]];
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
    Ok(l)
}

/// Log probability of `x` under N(mean, full covariance) given Cholesky factor L.
fn log_norm_pdf_chol(x: &[f64], mean: &[f64], l: &Array2<f64>, log_det: f64) -> f64 {
    let d = x.len() as f64;
    let diff: Vec<f64> = x.iter().zip(mean.iter()).map(|(&xi, &mi)| xi - mi).collect();
    let y = forward_solve(l, &diff);
    let maha: f64 = y.iter().map(|&yi| yi * yi).sum();
    -0.5 * (d * std::f64::consts::LN_2 + d * std::f64::consts::PI.ln() + log_det + maha)
}

/// Log probability of `x` under N(mean, diag variances).
fn log_norm_pdf_diag(x: &[f64], mean: &[f64], vars: &[f64]) -> f64 {
    let d = x.len() as f64;
    let log_det: f64 = vars.iter().map(|&v| v.ln()).sum();
    let maha: f64 = x
        .iter()
        .zip(mean.iter())
        .zip(vars.iter())
        .map(|((&xi, &mi), &vi)| {
            let diff = xi - mi;
            diff * diff / vi
        })
        .sum();
    -0.5 * (d * (std::f64::consts::LN_2 + std::f64::consts::PI.ln()) + log_det + maha)
}

/// Log probability under N(mean, spherical variance σ²).
fn log_norm_pdf_spherical(x: &[f64], mean: &[f64], variance: f64) -> f64 {
    let d = x.len() as f64;
    let maha: f64 = x
        .iter()
        .zip(mean.iter())
        .map(|(&xi, &mi)| {
            let diff = xi - mi;
            diff * diff
        })
        .sum::<f64>()
        / variance;
    -0.5 * (d * (std::f64::consts::LN_2 + std::f64::consts::PI.ln() + variance.ln()) + maha)
}

// ─────────────────────────────────────────────────────────────────────────────
// Initialisation
// ─────────────────────────────────────────────────────────────────────────────

/// K-means++ initialisation: pick k centres from data rows.
fn kmeans_plusplus_init(
    data: &Array2<f64>,
    k: usize,
    rng: &mut impl Rng,
) -> StatsResult<Array2<f64>> {
    let (n, d) = (data.nrows(), data.ncols());
    if k > n {
        return Err(StatsError::InvalidArgument(format!(
            "n_components={k} > n_samples={n}"
        )));
    }
    let mut centres = Array2::<f64>::zeros((k, d));
    // Choose first centre uniformly at random.
    let first = (rng.next_u64() as usize) % n;
    centres.row_mut(0).assign(&data.row(first));

    for c in 1..k {
        // Compute squared distances to nearest centre.
        let mut dists = vec![f64::INFINITY; n];
        for i in 0..n {
            for prev in 0..c {
                let dist: f64 = data
                    .row(i)
                    .iter()
                    .zip(centres.row(prev).iter())
                    .map(|(&xi, &ci)| (xi - ci).powi(2))
                    .sum();
                if dist < dists[i] {
                    dists[i] = dist;
                }
            }
        }
        // Sample next centre proportional to squared distance.
        let total: f64 = dists.iter().sum();
        let threshold = (rng.next_u64() as f64 / u64::MAX as f64) * total;
        let mut cumsum = 0.0;
        let mut chosen = n - 1;
        for (i, &d_i) in dists.iter().enumerate() {
            cumsum += d_i;
            if cumsum >= threshold {
                chosen = i;
                break;
            }
        }
        centres.row_mut(c).assign(&data.row(chosen));
    }
    Ok(centres)
}

// ─────────────────────────────────────────────────────────────────────────────
// EM algorithm
// ─────────────────────────────────────────────────────────────────────────────

/// Fit a GMM to `data` using the EM algorithm.
///
/// `data` has shape `(n_samples, n_features)`.
pub fn fit_em(data: &Array2<f64>, config: &GMMConfig) -> StatsResult<GMMModel> {
    let (n, d) = (data.nrows(), data.ncols());
    let k = config.n_components;

    if n == 0 {
        return Err(StatsError::InvalidArgument(
            "data must have at least one row".to_string(),
        ));
    }
    if d == 0 {
        return Err(StatsError::InvalidArgument(
            "data must have at least one feature".to_string(),
        ));
    }
    if k == 0 {
        return Err(StatsError::InvalidArgument(
            "n_components must be >= 1".to_string(),
        ));
    }

    let mut rng: Box<dyn Rng> = match config.seed {
        Some(s) => Box::new(scirs2_core::random::SmallRng::seed_from_u64(s)),
        None => Box::new(scirs2_core::random::SmallRng::seed_from_u64(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(42),
        )),
    };

    // ── Initialise means ──────────────────────────────────────────────────
    let means = match config.init {
        GmmInit::KMeansPlusPlus => kmeans_plusplus_init(data, k, rng.as_mut())?,
        GmmInit::Random => {
            let mut m = Array2::<f64>::zeros((k, d));
            for i in 0..k {
                let row = (rng.next_u64() as usize) % n;
                m.row_mut(i).assign(&data.row(row));
            }
            m
        }
    };

    // ── Initialise covariances ────────────────────────────────────────────
    // Compute global variance per feature for sensible initialisation.
    let global_mean: Vec<f64> = (0..d)
        .map(|j| data.column(j).mean().unwrap_or(0.0))
        .collect();
    let global_var: Vec<f64> = (0..d)
        .map(|j| {
            let m = global_mean[j];
            let v: f64 = data.column(j).iter().map(|&x| (x - m).powi(2)).sum::<f64>()
                / (n as f64);
            v.max(config.reg_covar)
        })
        .collect();

    let covariances = init_covariances(k, d, &global_var, config);

    // ── Initialise log-weights (uniform) ─────────────────────────────────
    let log_w0 = -(k as f64).ln();
    let log_weights = Array1::from_elem(k, log_w0);

    let mut model = GMMModel {
        means,
        covariances,
        log_weights,
        responsibilities: Array2::zeros((n, k)),
        log_likelihood: f64::NEG_INFINITY,
        n_iter: 0,
        converged: false,
        n_features: d,
        n_components: k,
        covariance_type: config.covariance_type,
    };

    let mut prev_ll = f64::NEG_INFINITY;

    for iter in 0..config.max_iter {
        // E-step
        let (log_resp, ll) = e_step(data, &model)?;
        model.responsibilities = resp_from_log(&log_resp);
        model.log_likelihood = ll;

        // M-step
        m_step(data, &model.responsibilities.clone(), config, &mut model)?;

        model.n_iter = iter + 1;

        let improvement = ll - prev_ll;
        if improvement.abs() < config.tol {
            model.converged = true;
            break;
        }
        prev_ll = ll;
    }

    Ok(model)
}

/// Initialise covariance matrices to scaled identity / diagonal.
fn init_covariances(
    k: usize,
    d: usize,
    global_var: &[f64],
    config: &GMMConfig,
) -> Vec<Array2<f64>> {
    match config.covariance_type {
        GmmCovarianceType::Full | GmmCovarianceType::Tied => {
            let mut cov = Array2::<f64>::zeros((d, d));
            for j in 0..d {
                cov[[j, j]] = global_var[j];
            }
            if config.covariance_type == GmmCovarianceType::Tied {
                vec![cov]
            } else {
                vec![cov; k]
            }
        }
        GmmCovarianceType::Diag => {
            // Store as (1 × d) per component for uniformity.
            let row = Array2::from_shape_fn((1, d), |(_, j)| global_var[j]);
            vec![row; k]
        }
        GmmCovarianceType::Spherical => {
            // Store as (1 × 1): single scalar variance.
            let mean_var: f64 = global_var.iter().sum::<f64>() / d as f64;
            vec![Array2::from_elem((1, 1), mean_var); k]
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// E-step
// ─────────────────────────────────────────────────────────────────────────────

/// Compute log-responsibilities and the per-sample log-likelihood.
///
/// Returns `(log_resp, mean_log_likelihood)` where `log_resp` has shape
/// `(n_samples, n_components)`.
pub fn e_step(data: &Array2<f64>, model: &GMMModel) -> StatsResult<(Array2<f64>, f64)> {
    let (n, _d) = (data.nrows(), data.ncols());
    let k = model.n_components;

    let mut log_resp = Array2::<f64>::zeros((n, k));

    // Precompute Cholesky factors (and log-dets) for Full/Tied modes.
    let chols: Option<Vec<(Array2<f64>, f64)>> =
        match model.covariance_type {
            GmmCovarianceType::Full => {
                let mut v = Vec::with_capacity(k);
                for c in &model.covariances {
                    let mut c_reg = c.clone();
                    for j in 0..c_reg.nrows() {
                        c_reg[[j, j]] += 1e-10;
                    }
                    let l = cholesky(&c_reg)?;
                    let ld = log_det_chol(&c_reg)?;
                    v.push((l, ld));
                }
                Some(v)
            }
            GmmCovarianceType::Tied => {
                let c = &model.covariances[0];
                let mut c_reg = c.clone();
                for j in 0..c_reg.nrows() {
                    c_reg[[j, j]] += 1e-10;
                }
                let l = cholesky(&c_reg)?;
                let ld = log_det_chol(&c_reg)?;
                Some(vec![(l, ld); k])
            }
            _ => None,
        };

    for i in 0..n {
        let xi: Vec<f64> = data.row(i).iter().copied().collect();
        for c in 0..k {
            let log_w = model.log_weights[c];
            let mean_c: Vec<f64> = model.means.row(c).iter().copied().collect();

            let log_p = match model.covariance_type {
                GmmCovarianceType::Full | GmmCovarianceType::Tied => {
                    let (ref l, ld) = chols.as_ref().expect("chols always Some for Full/Tied")[c];
                    log_norm_pdf_chol(&xi, &mean_c, l, ld)
                }
                GmmCovarianceType::Diag => {
                    let vars: Vec<f64> =
                        model.covariances[c].row(0).iter().copied().collect();
                    log_norm_pdf_diag(&xi, &mean_c, &vars)
                }
                GmmCovarianceType::Spherical => {
                    let variance = model.covariances[c][[0, 0]];
                    log_norm_pdf_spherical(&xi, &mean_c, variance)
                }
            };

            log_resp[[i, c]] = log_w + log_p;
        }
        // Normalise so that Σ_c resp[i,c] = 1 in log-space.
        let row_lse = log_sum_exp(log_resp.row(i).as_slice().expect("contiguous row"));
        for c in 0..k {
            log_resp[[i, c]] -= row_lse;
        }
    }

    // Mean log-likelihood: average over samples of log p(x_i).
    let ll: f64 = (0..n)
        .map(|i| {
            let logits: Vec<f64> = (0..k)
                .map(|c| {
                    let log_w = model.log_weights[c];
                    let mean_c: Vec<f64> = model.means.row(c).iter().copied().collect();
                    let log_p = match model.covariance_type {
                        GmmCovarianceType::Full | GmmCovarianceType::Tied => {
                            let (ref l, ld) = chols.as_ref()
                                .expect("chols always Some for Full/Tied")[c];
                            let xi: Vec<f64> = data.row(i).iter().copied().collect();
                            log_norm_pdf_chol(&xi, &mean_c, l, ld)
                        }
                        GmmCovarianceType::Diag => {
                            let vars: Vec<f64> =
                                model.covariances[c].row(0).iter().copied().collect();
                            let xi: Vec<f64> = data.row(i).iter().copied().collect();
                            log_norm_pdf_diag(&xi, &mean_c, &vars)
                        }
                        GmmCovarianceType::Spherical => {
                            let variance = model.covariances[c][[0, 0]];
                            let xi: Vec<f64> = data.row(i).iter().copied().collect();
                            log_norm_pdf_spherical(&xi, &mean_c, variance)
                        }
                    };
                    log_w + log_p
                })
                .collect();
            log_sum_exp(&logits)
        })
        .sum::<f64>()
        / n as f64;

    Ok((log_resp, ll))
}

/// Convert log-responsibilities to responsibilities (softmax row-wise).
fn resp_from_log(log_resp: &Array2<f64>) -> Array2<f64> {
    let (n, k) = (log_resp.nrows(), log_resp.ncols());
    let mut resp = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        let max = log_resp
            .row(i)
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut row_sum = 0.0_f64;
        for c in 0..k {
            resp[[i, c]] = (log_resp[[i, c]] - max).exp();
            row_sum += resp[[i, c]];
        }
        if row_sum > 0.0 {
            for c in 0..k {
                resp[[i, c]] /= row_sum;
            }
        }
    }
    resp
}

// ─────────────────────────────────────────────────────────────────────────────
// M-step
// ─────────────────────────────────────────────────────────────────────────────

/// Update means, covariances, and weights from the current responsibilities.
pub fn m_step(
    data: &Array2<f64>,
    resp: &Array2<f64>,
    config: &GMMConfig,
    model: &mut GMMModel,
) -> StatsResult<()> {
    let (n, d) = (data.nrows(), data.ncols());
    let k = model.n_components;

    // Effective counts per component: N_k = Σ_i r_{ik}
    let n_k: Vec<f64> = (0..k)
        .map(|c| resp.column(c).sum().max(1e-10))
        .collect();

    // ── Log-weights ──────────────────────────────────────────────────────
    let total: f64 = n_k.iter().sum();
    for c in 0..k {
        model.log_weights[c] = (n_k[c] / total).ln();
    }

    // ── Means ────────────────────────────────────────────────────────────
    for c in 0..k {
        for j in 0..d {
            let num: f64 = (0..n).map(|i| resp[[i, c]] * data[[i, j]]).sum();
            model.means[[c, j]] = num / n_k[c];
        }
    }

    // ── Covariances ──────────────────────────────────────────────────────
    match config.covariance_type {
        GmmCovarianceType::Full => {
            for c in 0..k {
                let mut cov = Array2::<f64>::zeros((d, d));
                for i in 0..n {
                    for j in 0..d {
                        for l in 0..=j {
                            let diff_j = data[[i, j]] - model.means[[c, j]];
                            let diff_l = data[[i, l]] - model.means[[c, l]];
                            let contrib = resp[[i, c]] * diff_j * diff_l / n_k[c];
                            cov[[j, l]] += contrib;
                            if j != l {
                                cov[[l, j]] += contrib;
                            }
                        }
                    }
                }
                // Regularise diagonal
                for j in 0..d {
                    cov[[j, j]] += config.reg_covar;
                }
                model.covariances[c] = cov;
            }
        }
        GmmCovarianceType::Diag => {
            for c in 0..k {
                let mut vars = Array2::<f64>::zeros((1, d));
                for i in 0..n {
                    for j in 0..d {
                        let diff = data[[i, j]] - model.means[[c, j]];
                        vars[[0, j]] += resp[[i, c]] * diff * diff / n_k[c];
                    }
                }
                for j in 0..d {
                    vars[[0, j]] = vars[[0, j]].max(config.reg_covar);
                }
                model.covariances[c] = vars;
            }
        }
        GmmCovarianceType::Spherical => {
            for c in 0..k {
                let mut total_var = 0.0_f64;
                for i in 0..n {
                    let sq: f64 = (0..d)
                        .map(|j| {
                            let diff = data[[i, j]] - model.means[[c, j]];
                            diff * diff
                        })
                        .sum();
                    total_var += resp[[i, c]] * sq;
                }
                let variance = (total_var / (n_k[c] * d as f64)).max(config.reg_covar);
                model.covariances[c] = Array2::from_elem((1, 1), variance);
            }
        }
        GmmCovarianceType::Tied => {
            let mut cov = Array2::<f64>::zeros((d, d));
            for c in 0..k {
                for i in 0..n {
                    for j in 0..d {
                        for l in 0..=j {
                            let diff_j = data[[i, j]] - model.means[[c, j]];
                            let diff_l = data[[i, l]] - model.means[[c, l]];
                            let contrib = resp[[i, c]] * diff_j * diff_l / n as f64;
                            cov[[j, l]] += contrib;
                            if j != l {
                                cov[[l, j]] += contrib;
                            }
                        }
                    }
                }
            }
            for j in 0..d {
                cov[[j, j]] += config.reg_covar;
            }
            model.covariances[0] = cov;
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Prediction
// ─────────────────────────────────────────────────────────────────────────────

/// Compute soft assignments (posterior probabilities) for each sample.
///
/// Returns an `(n_samples, n_components)` array.
pub fn predict_proba(data: &Array2<f64>, model: &GMMModel) -> StatsResult<Array2<f64>> {
    let (log_resp, _ll) = e_step(data, model)?;
    Ok(resp_from_log(&log_resp))
}

/// Predict the most-likely component for each sample.
///
/// Returns an `(n_samples,)` array of component indices.
pub fn predict(data: &Array2<f64>, model: &GMMModel) -> StatsResult<Array1<usize>> {
    let prob = predict_proba(data, model)?;
    let n = prob.nrows();
    let mut labels = Array1::<usize>::zeros(n);
    for i in 0..n {
        let best = prob
            .row(i)
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        labels[i] = best;
    }
    Ok(labels)
}

// ─────────────────────────────────────────────────────────────────────────────
// Log-likelihood
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the total log-likelihood of `data` under the fitted model.
pub fn log_likelihood(data: &Array2<f64>, model: &GMMModel) -> StatsResult<f64> {
    let (_log_resp, mean_ll) = e_step(data, model)?;
    Ok(mean_ll * data.nrows() as f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// Model-selection criteria
// ─────────────────────────────────────────────────────────────────────────────

/// Number of free parameters in the model.
fn n_params(model: &GMMModel) -> usize {
    let k = model.n_components;
    let d = model.n_features;
    let cov_params = match model.covariance_type {
        GmmCovarianceType::Full => k * d * (d + 1) / 2,
        GmmCovarianceType::Diag => k * d,
        GmmCovarianceType::Spherical => k,
        GmmCovarianceType::Tied => d * (d + 1) / 2,
    };
    let mean_params = k * d;
    let weight_params = k - 1;
    cov_params + mean_params + weight_params
}

/// Akaike Information Criterion: AIC = 2p − 2 log L.
pub fn aic(data: &Array2<f64>, model: &GMMModel) -> StatsResult<f64> {
    let ll = log_likelihood(data, model)?;
    let p = n_params(model) as f64;
    Ok(2.0 * p - 2.0 * ll)
}

/// Bayesian Information Criterion: BIC = p log n − 2 log L.
pub fn bic(data: &Array2<f64>, model: &GMMModel) -> StatsResult<f64> {
    let ll = log_likelihood(data, model)?;
    let p = n_params(model) as f64;
    let n = data.nrows() as f64;
    Ok(p * n.ln() - 2.0 * ll)
}

// ─────────────────────────────────────────────────────────────────────────────
// Sampling
// ─────────────────────────────────────────────────────────────────────────────

/// Draw `n_samples` samples from a fitted GMM.
///
/// Uses Box-Muller / Cholesky sampling for full covariances.
pub fn sample(
    model: &GMMModel,
    n_samples: usize,
    seed: Option<u64>,
) -> StatsResult<Array2<f64>> {
    let k = model.n_components;
    let d = model.n_features;

    let mut rng: Box<dyn Rng> = match seed {
        Some(s) => Box::new(scirs2_core::random::SmallRng::seed_from_u64(s)),
        None => Box::new(scirs2_core::random::SmallRng::seed_from_u64(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|dur| dur.as_nanos() as u64)
                .unwrap_or(12345),
        )),
    };

    let weights: Vec<f64> = model.log_weights.iter().map(|&lw| lw.exp()).collect();
    // Cumulative sum for component sampling
    let cum: Vec<f64> = weights
        .iter()
        .scan(0.0_f64, |acc, &w| {
            *acc += w;
            Some(*acc)
        })
        .collect();

    // Precompute Cholesky factors for full/tied
    let chols: Option<Vec<Array2<f64>>> = match model.covariance_type {
        GmmCovarianceType::Full => {
            let mut v = Vec::with_capacity(k);
            for c in &model.covariances {
                let l = cholesky(c)?;
                v.push(l);
            }
            Some(v)
        }
        GmmCovarianceType::Tied => {
            let l = cholesky(&model.covariances[0])?;
            Some(vec![l; k])
        }
        _ => None,
    };

    let mut out = Array2::<f64>::zeros((n_samples, d));

    for i in 0..n_samples {
        // Sample component index
        let u: f64 = rng.next_u64() as f64 / u64::MAX as f64;
        let comp = cum
            .iter()
            .position(|&c| u <= c)
            .unwrap_or(k - 1);

        let mean: Vec<f64> = model.means.row(comp).iter().copied().collect();

        // Sample standard normals
        let zs: Vec<f64> = (0..d)
            .map(|_| sample_standard_normal(&mut *rng))
            .collect();

        let x: Vec<f64> = match model.covariance_type {
            GmmCovarianceType::Full | GmmCovarianceType::Tied => {
                let l = &chols.as_ref().expect("chols Some for Full/Tied")[comp];
                // x = mean + L z
                (0..d)
                    .map(|j| {
                        let lz: f64 = (0..=j).map(|jj| l[[j, jj]] * zs[jj]).sum();
                        mean[j] + lz
                    })
                    .collect()
            }
            GmmCovarianceType::Diag => {
                let vars: Vec<f64> = model.covariances[comp].row(0).iter().copied().collect();
                (0..d)
                    .map(|j| mean[j] + vars[j].sqrt() * zs[j])
                    .collect()
            }
            GmmCovarianceType::Spherical => {
                let std_dev = model.covariances[comp][[0, 0]].sqrt();
                (0..d).map(|j| mean[j] + std_dev * zs[j]).collect()
            }
        };

        for j in 0..d {
            out[[i, j]] = x[j];
        }
    }

    Ok(out)
}

/// Box-Muller transform to draw one standard normal variate.
fn sample_standard_normal(rng: &mut dyn Rng) -> f64 {
    let u1 = (rng.next_u64() as f64 + 1.0) / (u64::MAX as f64 + 1.0);
    let u2 = rng.next_u64() as f64 / u64::MAX as f64;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn make_two_cluster_data() -> Array2<f64> {
        // 20 points near (0,0) and 20 points near (5,5)
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for i in 0..20_i64 {
            rows.push(vec![i as f64 * 0.1, i as f64 * 0.1]);
        }
        for i in 0..20_i64 {
            rows.push(vec![5.0 + i as f64 * 0.1, 5.0 + i as f64 * 0.1]);
        }
        let flat: Vec<f64> = rows.iter().flatten().copied().collect();
        Array2::from_shape_vec((40, 2), flat).expect("shape ok")
    }

    #[test]
    fn test_gmm_full_convergence() {
        let data = make_two_cluster_data();
        let config = GMMConfig {
            n_components: 2,
            max_iter: 200,
            tol: 1e-6,
            covariance_type: GmmCovarianceType::Full,
            reg_covar: 1e-4,
            init: GmmInit::KMeansPlusPlus,
            seed: Some(42),
        };
        let model = fit_em(&data, &config).expect("fit should succeed");
        assert!(model.converged, "GMM should converge on clean 2-cluster data");
        assert!(
            model.log_likelihood.is_finite(),
            "log-likelihood should be finite"
        );
    }

    #[test]
    fn test_gmm_diag() {
        let data = make_two_cluster_data();
        let config = GMMConfig {
            n_components: 2,
            covariance_type: GmmCovarianceType::Diag,
            seed: Some(1),
            ..Default::default()
        };
        let model = fit_em(&data, &config).expect("fit diag");
        assert_eq!(model.n_components, 2);
        assert_eq!(model.n_features, 2);
    }

    #[test]
    fn test_gmm_spherical() {
        let data = make_two_cluster_data();
        let config = GMMConfig {
            n_components: 2,
            covariance_type: GmmCovarianceType::Spherical,
            seed: Some(7),
            ..Default::default()
        };
        let model = fit_em(&data, &config).expect("fit spherical");
        assert_eq!(model.covariances.len(), 2);
        for cov in &model.covariances {
            assert_eq!(cov.shape(), &[1, 1]);
        }
    }

    #[test]
    fn test_gmm_tied() {
        let data = make_two_cluster_data();
        let config = GMMConfig {
            n_components: 2,
            covariance_type: GmmCovarianceType::Tied,
            seed: Some(99),
            ..Default::default()
        };
        let model = fit_em(&data, &config).expect("fit tied");
        assert_eq!(model.covariances.len(), 1);
    }

    #[test]
    fn test_predict_separates_clusters() {
        let data = make_two_cluster_data();
        let config = GMMConfig {
            n_components: 2,
            seed: Some(42),
            reg_covar: 1e-4,
            ..Default::default()
        };
        let model = fit_em(&data, &config).expect("fit");
        let labels = predict(&data, &model).expect("predict");
        // First 20 and last 20 should mostly get the same label within each group.
        let first_label = labels[0];
        for i in 0..20 {
            assert_eq!(labels[i], first_label);
        }
        let second_label = labels[20];
        for i in 20..40 {
            assert_eq!(labels[i], second_label);
        }
        // The two groups must have different labels
        assert_ne!(first_label, second_label);
    }

    #[test]
    fn test_aic_bic() {
        let data = make_two_cluster_data();
        let config = GMMConfig {
            n_components: 2,
            seed: Some(42),
            reg_covar: 1e-4,
            ..Default::default()
        };
        let model = fit_em(&data, &config).expect("fit");
        let a = aic(&data, &model).expect("aic");
        let b = bic(&data, &model).expect("bic");
        assert!(a.is_finite());
        assert!(b.is_finite());
    }

    #[test]
    fn test_sample() {
        let data = make_two_cluster_data();
        let config = GMMConfig {
            n_components: 2,
            seed: Some(42),
            reg_covar: 1e-4,
            ..Default::default()
        };
        let model = fit_em(&data, &config).expect("fit");
        let samples = sample(&model, 50, Some(0)).expect("sample");
        assert_eq!(samples.shape(), &[50, 2]);
    }
}
