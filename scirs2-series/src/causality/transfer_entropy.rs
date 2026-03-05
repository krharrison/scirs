//! Transfer entropy estimation for time series
//!
//! This module provides comprehensive transfer entropy analysis:
//! - **Shannon transfer entropy**: Standard information-theoretic measure
//! - **Renyi transfer entropy**: Generalized with order parameter alpha
//! - **Conditional transfer entropy**: Controlling for confounders
//! - **Effective transfer entropy**: Surrogate-corrected for bias removal
//! - **Estimators**: Binning-based, KDE-based, and kNN-based

use crate::error::TimeSeriesError;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::validation::checkarray_finite;
use std::collections::HashMap;

use super::{fisher_yates_shuffle, CausalityResult};

/// Transfer entropy result
#[derive(Debug, Clone)]
pub struct TransferEntropyResult {
    /// Transfer entropy value (in nats)
    pub transfer_entropy: f64,
    /// P-value from significance test (if computed)
    pub p_value: Option<f64>,
    /// Number of bins used for entropy calculation
    pub bins: usize,
    /// Embedding dimension used
    pub embedding_dim: usize,
    /// Time delay used
    pub time_delay: usize,
    /// Estimator type used
    pub estimator: TransferEntropyEstimator,
    /// Standard error estimate (if available)
    pub std_error: Option<f64>,
}

/// Configuration for Shannon transfer entropy calculation
#[derive(Debug, Clone)]
pub struct TransferEntropyConfig {
    /// Number of bins for histogram-based estimation
    pub bins: usize,
    /// Embedding dimension (number of past values to consider)
    pub embedding_dim: usize,
    /// Time delay for embedding
    pub time_delay: usize,
    /// Number of bootstrap samples for significance testing (None = skip)
    pub bootstrap_samples: Option<usize>,
    /// Estimator to use
    pub estimator: TransferEntropyEstimator,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for TransferEntropyConfig {
    fn default() -> Self {
        Self {
            bins: 10,
            embedding_dim: 3,
            time_delay: 1,
            bootstrap_samples: Some(100),
            estimator: TransferEntropyEstimator::Binning,
            seed: None,
        }
    }
}

/// Transfer entropy estimator type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferEntropyEstimator {
    /// Histogram/binning based estimator
    Binning,
    /// Kernel density estimation based estimator
    KDE,
    /// k-nearest neighbor estimator (Kraskov-Stogbauer-Grassberger)
    KNN,
}

/// Configuration for Renyi transfer entropy
#[derive(Debug, Clone)]
pub struct RenyiTransferEntropyConfig {
    /// Renyi order parameter (alpha). alpha=1 reduces to Shannon
    pub alpha: f64,
    /// Number of bins for histogram estimation
    pub bins: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Time delay
    pub time_delay: usize,
    /// Bootstrap samples for significance
    pub bootstrap_samples: Option<usize>,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for RenyiTransferEntropyConfig {
    fn default() -> Self {
        Self {
            alpha: 2.0,
            bins: 10,
            embedding_dim: 3,
            time_delay: 1,
            bootstrap_samples: Some(100),
            seed: None,
        }
    }
}

/// Configuration for conditional transfer entropy
#[derive(Debug, Clone)]
pub struct ConditionalTransferEntropyConfig {
    /// Number of bins
    pub bins: usize,
    /// Embedding dimension for source and target
    pub embedding_dim: usize,
    /// Embedding dimension for conditioning variables
    pub cond_embedding_dim: usize,
    /// Time delay
    pub time_delay: usize,
    /// Bootstrap samples for significance
    pub bootstrap_samples: Option<usize>,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for ConditionalTransferEntropyConfig {
    fn default() -> Self {
        Self {
            bins: 8,
            embedding_dim: 2,
            cond_embedding_dim: 2,
            time_delay: 1,
            bootstrap_samples: Some(100),
            seed: None,
        }
    }
}

/// Configuration for effective transfer entropy (surrogate-corrected)
#[derive(Debug, Clone)]
pub struct EffectiveTransferEntropyConfig {
    /// Number of bins
    pub bins: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Time delay
    pub time_delay: usize,
    /// Number of surrogates for bias estimation
    pub n_surrogates: usize,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for EffectiveTransferEntropyConfig {
    fn default() -> Self {
        Self {
            bins: 10,
            embedding_dim: 3,
            time_delay: 1,
            n_surrogates: 100,
            seed: None,
        }
    }
}

/// Result of effective transfer entropy
#[derive(Debug, Clone)]
pub struct EffectiveTransferEntropyResult {
    /// Effective (bias-corrected) transfer entropy
    pub effective_te: f64,
    /// Raw (uncorrected) transfer entropy
    pub raw_te: f64,
    /// Mean surrogate transfer entropy (bias estimate)
    pub surrogate_mean: f64,
    /// Standard deviation of surrogate TE
    pub surrogate_std: f64,
    /// Z-score relative to surrogates
    pub z_score: f64,
    /// P-value from surrogate distribution
    pub p_value: f64,
    /// Number of surrogates used
    pub n_surrogates: usize,
}

/// Compute Shannon transfer entropy from source x to target y
///
/// Transfer entropy TE(X -> Y) measures the reduction in uncertainty about
/// the future of Y when past values of X are known, beyond what is already
/// known from past values of Y alone.
///
/// TE(X -> Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
///
/// # Arguments
///
/// * `x` - Source time series
/// * `y` - Target time series
/// * `config` - Configuration
///
/// # Returns
///
/// `TransferEntropyResult` with the transfer entropy value and statistics
pub fn shannon_transfer_entropy(
    x: &Array1<f64>,
    y: &Array1<f64>,
    config: &TransferEntropyConfig,
) -> CausalityResult<TransferEntropyResult> {
    checkarray_finite(x, "x")?;
    checkarray_finite(y, "y")?;

    if x.len() != y.len() {
        return Err(TimeSeriesError::InvalidInput(
            "Time series must have the same length".to_string(),
        ));
    }

    let required_length = config.embedding_dim * config.time_delay + 1;
    if x.len() < required_length {
        return Err(TimeSeriesError::InvalidInput(format!(
            "Time series too short (len={}) for embedding (need {})",
            x.len(),
            required_length
        )));
    }

    let te = match config.estimator {
        TransferEntropyEstimator::Binning => {
            compute_te_binning(x, y, config.bins, config.embedding_dim, config.time_delay)?
        }
        TransferEntropyEstimator::KDE => {
            compute_te_kde(x, y, config.bins, config.embedding_dim, config.time_delay)?
        }
        TransferEntropyEstimator::KNN => {
            compute_te_knn(x, y, config.embedding_dim, config.time_delay)?
        }
    };

    // Bootstrap for p-value
    let (p_value, std_error) = if let Some(n_bootstrap) = config.bootstrap_samples {
        let (p, se) = bootstrap_te_significance(x, y, config, te, n_bootstrap, config.seed)?;
        (Some(p), Some(se))
    } else {
        (None, None)
    };

    Ok(TransferEntropyResult {
        transfer_entropy: te,
        p_value,
        bins: config.bins,
        embedding_dim: config.embedding_dim,
        time_delay: config.time_delay,
        estimator: config.estimator,
        std_error,
    })
}

/// Compute Renyi transfer entropy
///
/// Generalization of Shannon transfer entropy using Renyi divergence of order alpha.
/// When alpha -> 1, this converges to Shannon transfer entropy.
///
/// # Arguments
///
/// * `x` - Source time series
/// * `y` - Target time series
/// * `config` - Configuration with Renyi order alpha
pub fn renyi_transfer_entropy(
    x: &Array1<f64>,
    y: &Array1<f64>,
    config: &RenyiTransferEntropyConfig,
) -> CausalityResult<TransferEntropyResult> {
    checkarray_finite(x, "x")?;
    checkarray_finite(y, "y")?;

    if x.len() != y.len() {
        return Err(TimeSeriesError::InvalidInput(
            "Time series must have the same length".to_string(),
        ));
    }

    if config.alpha <= 0.0 {
        return Err(TimeSeriesError::InvalidInput(
            "Renyi order alpha must be positive".to_string(),
        ));
    }

    let required_length = config.embedding_dim * config.time_delay + 1;
    if x.len() < required_length {
        return Err(TimeSeriesError::InvalidInput(
            "Time series too short for the specified embedding parameters".to_string(),
        ));
    }

    // If alpha is close to 1, use Shannon
    let te = if (config.alpha - 1.0).abs() < 1e-6 {
        compute_te_binning(x, y, config.bins, config.embedding_dim, config.time_delay)?
    } else {
        compute_renyi_te(x, y, config)?
    };

    let p_value = if let Some(n_bootstrap) = config.bootstrap_samples {
        let shannon_config = TransferEntropyConfig {
            bins: config.bins,
            embedding_dim: config.embedding_dim,
            time_delay: config.time_delay,
            bootstrap_samples: None,
            estimator: TransferEntropyEstimator::Binning,
            seed: config.seed,
        };
        let (p, _) =
            bootstrap_te_significance(x, y, &shannon_config, te, n_bootstrap, config.seed)?;
        Some(p)
    } else {
        None
    };

    Ok(TransferEntropyResult {
        transfer_entropy: te,
        p_value,
        bins: config.bins,
        embedding_dim: config.embedding_dim,
        time_delay: config.time_delay,
        estimator: TransferEntropyEstimator::Binning,
        std_error: None,
    })
}

/// Compute conditional transfer entropy: TE(X -> Y | Z)
///
/// Measures the information transfer from X to Y while controlling for
/// the influence of conditioning variables Z.
///
/// # Arguments
///
/// * `x` - Source time series
/// * `y` - Target time series
/// * `z` - Conditioning variables (2D: rows = time, cols = variables)
/// * `config` - Configuration
pub fn conditional_transfer_entropy(
    x: &Array1<f64>,
    y: &Array1<f64>,
    z: &Array2<f64>,
    config: &ConditionalTransferEntropyConfig,
) -> CausalityResult<TransferEntropyResult> {
    checkarray_finite(x, "x")?;
    checkarray_finite(y, "y")?;

    if x.len() != y.len() || x.len() != z.nrows() {
        return Err(TimeSeriesError::InvalidInput(
            "All time series must have the same length".to_string(),
        ));
    }

    let max_embed = config.embedding_dim.max(config.cond_embedding_dim);
    let required_length = max_embed * config.time_delay + 1;
    if x.len() < required_length {
        return Err(TimeSeriesError::InvalidInput(
            "Time series too short for the specified embedding parameters".to_string(),
        ));
    }

    let te = compute_conditional_te(x, y, z, config)?;

    let p_value = if let Some(n_bootstrap) = config.bootstrap_samples {
        let p = bootstrap_conditional_te_pvalue(x, y, z, config, te, n_bootstrap)?;
        Some(p)
    } else {
        None
    };

    Ok(TransferEntropyResult {
        transfer_entropy: te,
        p_value,
        bins: config.bins,
        embedding_dim: config.embedding_dim,
        time_delay: config.time_delay,
        estimator: TransferEntropyEstimator::Binning,
        std_error: None,
    })
}

/// Compute effective transfer entropy with surrogate correction
///
/// Subtracts the average transfer entropy from surrogates (shuffled source)
/// to remove estimation bias, yielding a more accurate TE estimate.
///
/// TE_effective = TE_raw - mean(TE_surrogates)
///
/// # Arguments
///
/// * `x` - Source time series
/// * `y` - Target time series
/// * `config` - Configuration with surrogate parameters
pub fn effective_transfer_entropy(
    x: &Array1<f64>,
    y: &Array1<f64>,
    config: &EffectiveTransferEntropyConfig,
) -> CausalityResult<EffectiveTransferEntropyResult> {
    checkarray_finite(x, "x")?;
    checkarray_finite(y, "y")?;

    if x.len() != y.len() {
        return Err(TimeSeriesError::InvalidInput(
            "Time series must have the same length".to_string(),
        ));
    }

    let required_length = config.embedding_dim * config.time_delay + 1;
    if x.len() < required_length {
        return Err(TimeSeriesError::InvalidInput(
            "Time series too short for the specified embedding parameters".to_string(),
        ));
    }

    // Compute raw transfer entropy
    let raw_te = compute_te_binning(x, y, config.bins, config.embedding_dim, config.time_delay)?;

    // Compute surrogate distribution
    let mut surrogate_tes = Vec::with_capacity(config.n_surrogates);
    for s in 0..config.n_surrogates {
        let mut x_shuffled = x.clone();
        let seed_val = config.seed.map(|base| base.wrapping_add(s as u64));
        fisher_yates_shuffle(&mut x_shuffled, seed_val);

        let surr_te = compute_te_binning(
            &x_shuffled,
            y,
            config.bins,
            config.embedding_dim,
            config.time_delay,
        )?;
        surrogate_tes.push(surr_te);
    }

    let n_surr = surrogate_tes.len() as f64;
    let surrogate_mean = surrogate_tes.iter().sum::<f64>() / n_surr;
    let surrogate_var = surrogate_tes
        .iter()
        .map(|&v| (v - surrogate_mean).powi(2))
        .sum::<f64>()
        / (n_surr - 1.0).max(1.0);
    let surrogate_std = surrogate_var.sqrt();

    let effective_te = raw_te - surrogate_mean;

    let z_score = if surrogate_std > 1e-15 {
        (raw_te - surrogate_mean) / surrogate_std
    } else {
        0.0
    };

    let count_above = surrogate_tes.iter().filter(|&&v| v >= raw_te).count();
    let p_value = count_above as f64 / config.n_surrogates as f64;

    Ok(EffectiveTransferEntropyResult {
        effective_te,
        raw_te,
        surrogate_mean,
        surrogate_std,
        z_score,
        p_value,
        n_surrogates: config.n_surrogates,
    })
}

// ---- Internal estimator implementations ----

/// Binning-based transfer entropy estimation
fn compute_te_binning(
    x: &Array1<f64>,
    y: &Array1<f64>,
    bins: usize,
    embedding_dim: usize,
    time_delay: usize,
) -> CausalityResult<f64> {
    let (x_embed, y_embed, y_future) = create_embeddings(x, y, embedding_dim, time_delay)?;

    let x_discrete = discretize_matrix(&x_embed, bins)?;
    let y_discrete = discretize_matrix(&y_embed, bins)?;
    let y_future_discrete = discretize_vector(&y_future, bins)?;

    compute_te_from_discrete(&x_discrete, &y_discrete, &y_future_discrete)
}

/// KDE-based transfer entropy estimation
fn compute_te_kde(
    x: &Array1<f64>,
    y: &Array1<f64>,
    bins: usize,
    embedding_dim: usize,
    time_delay: usize,
) -> CausalityResult<f64> {
    let (x_embed, y_embed, y_future) = create_embeddings(x, y, embedding_dim, time_delay)?;

    let n = x_embed.nrows();
    if n < 5 {
        return Err(TimeSeriesError::InvalidInput(
            "Too few embedded points for KDE estimation".to_string(),
        ));
    }

    // Use Silverman's rule for bandwidth
    let total_dim = (2 * embedding_dim + 1) as f64;
    let bandwidth = (4.0 / ((total_dim + 2.0) * n as f64)).powf(1.0 / (total_dim + 4.0));

    // Estimate TE via KDE ratio
    // TE = E[ ln( p(y_t | y_past, x_past) / p(y_t | y_past) ) ]
    let mut te_sum = 0.0;

    for i in 0..n {
        // Joint density p(y_t, y_past, x_past) and marginal p(y_past, x_past)
        let mut log_joint_cond = 0.0;
        let mut log_marginal_cond = 0.0;

        let mut joint_density = 0.0;
        let mut yx_density = 0.0;
        let mut y_density = 0.0;
        let mut y_only_density = 0.0;

        for j in 0..n {
            if i == j {
                continue;
            }

            // Compute kernel distances
            let y_fut_dist = (y_future[i] - y_future[j]).powi(2);

            let mut y_past_dist = 0.0;
            for d in 0..embedding_dim {
                y_past_dist += (y_embed[[i, d]] - y_embed[[j, d]]).powi(2);
            }

            let mut x_past_dist = 0.0;
            for d in 0..embedding_dim {
                x_past_dist += (x_embed[[i, d]] - x_embed[[j, d]]).powi(2);
            }

            let bw2 = bandwidth * bandwidth;
            let k_yfut = (-y_fut_dist / (2.0 * bw2)).exp();
            let k_ypast = (-y_past_dist / (2.0 * bw2)).exp();
            let k_xpast = (-x_past_dist / (2.0 * bw2)).exp();

            // p(y_future, y_past, x_past)
            joint_density += k_yfut * k_ypast * k_xpast;
            // p(y_past, x_past)
            yx_density += k_ypast * k_xpast;
            // p(y_future, y_past)
            y_density += k_yfut * k_ypast;
            // p(y_past)
            y_only_density += k_ypast;
        }

        // TE contribution: ln(p(y_t|y_past,x_past) / p(y_t|y_past))
        // = ln(p(y_t,y_past,x_past) * p(y_past)) - ln(p(y_t,y_past) * p(y_past,x_past))
        if joint_density > 1e-15
            && y_only_density > 1e-15
            && y_density > 1e-15
            && yx_density > 1e-15
        {
            te_sum += (joint_density * y_only_density / (y_density * yx_density)).ln();
        }
    }

    Ok(te_sum / n as f64)
}

/// k-nearest neighbor based transfer entropy estimation (KSG-style)
fn compute_te_knn(
    x: &Array1<f64>,
    y: &Array1<f64>,
    embedding_dim: usize,
    time_delay: usize,
) -> CausalityResult<f64> {
    let (x_embed, y_embed, y_future) = create_embeddings(x, y, embedding_dim, time_delay)?;

    let n = x_embed.nrows();
    let k = 4.min(n.saturating_sub(1)); // k nearest neighbors

    if k == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Too few embedded points for kNN estimation".to_string(),
        ));
    }

    // For each point, find the k-th nearest neighbor distance in the full joint space
    // Then count neighbors within that distance in each marginal subspace
    // TE = digamma(k) - <digamma(n_yz+1) - digamma(n_y+1) + digamma(n_yx+1)>

    let mut te_sum = 0.0;

    for i in 0..n {
        // Compute distances in full joint space (y_future, y_past, x_past)
        let mut distances = Vec::with_capacity(n);
        for j in 0..n {
            if i == j {
                continue;
            }

            let mut max_dist = (y_future[i] - y_future[j]).abs();

            for d in 0..embedding_dim {
                max_dist = max_dist.max((y_embed[[i, d]] - y_embed[[j, d]]).abs());
                max_dist = max_dist.max((x_embed[[i, d]] - x_embed[[j, d]]).abs());
            }

            distances.push((max_dist, j));
        }

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        if distances.len() < k {
            continue;
        }

        let epsilon = distances[k - 1].0;

        // Count neighbors within epsilon in each subspace
        let mut n_yz = 0; // (y_future, y_past) subspace
        let mut n_y = 0; // (y_past) subspace
        let mut n_yx = 0; // (y_past, x_past) subspace

        for j in 0..n {
            if i == j {
                continue;
            }

            // y_past distance
            let mut y_past_dist: f64 = 0.0;
            for d in 0..embedding_dim {
                y_past_dist = y_past_dist.max((y_embed[[i, d]] - y_embed[[j, d]]).abs());
            }

            // x_past distance
            let mut x_past_dist: f64 = 0.0;
            for d in 0..embedding_dim {
                x_past_dist = x_past_dist.max((x_embed[[i, d]] - x_embed[[j, d]]).abs());
            }

            let y_fut_dist = (y_future[i] - y_future[j]).abs();

            if y_fut_dist <= epsilon && y_past_dist <= epsilon {
                n_yz += 1;
            }
            if y_past_dist <= epsilon {
                n_y += 1;
            }
            if y_past_dist <= epsilon && x_past_dist <= epsilon {
                n_yx += 1;
            }
        }

        // Digamma contributions
        te_sum +=
            digamma(n_y as f64 + 1.0) - digamma(n_yz as f64 + 1.0) - digamma(n_yx as f64 + 1.0);
    }

    let te = digamma(k as f64) + te_sum / n as f64;

    Ok(te.max(0.0))
}

/// Renyi transfer entropy computation
fn compute_renyi_te(
    x: &Array1<f64>,
    y: &Array1<f64>,
    config: &RenyiTransferEntropyConfig,
) -> CausalityResult<f64> {
    let (x_embed, y_embed, y_future) =
        create_embeddings(x, y, config.embedding_dim, config.time_delay)?;

    let x_discrete = discretize_matrix(&x_embed, config.bins)?;
    let y_discrete = discretize_matrix(&y_embed, config.bins)?;
    let y_future_discrete = discretize_vector(&y_future, config.bins)?;

    let n_samples = x_discrete.nrows();
    let alpha = config.alpha;

    // Build joint and marginal probability tables
    let mut joint_counts: HashMap<(usize, Vec<usize>, Vec<usize>), usize> = HashMap::new();
    let mut marginal_yfut_ypast: HashMap<(usize, Vec<usize>), usize> = HashMap::new();
    let mut cond_ypast_xpast: HashMap<(Vec<usize>, Vec<usize>), usize> = HashMap::new();
    let mut y_only_counts: HashMap<Vec<usize>, usize> = HashMap::new();

    for i in 0..n_samples {
        let x_state = x_discrete.row(i).to_vec();
        let y_state = y_discrete.row(i).to_vec();
        let y_fut = y_future_discrete[i];

        *joint_counts
            .entry((y_fut, y_state.clone(), x_state.clone()))
            .or_insert(0) += 1;
        *marginal_yfut_ypast
            .entry((y_fut, y_state.clone()))
            .or_insert(0) += 1;
        *cond_ypast_xpast
            .entry((y_state.clone(), x_state))
            .or_insert(0) += 1;
        *y_only_counts.entry(y_state).or_insert(0) += 1;
    }

    // Renyi TE = 1/(alpha-1) * ln( sum_states p(y_fut, y_past, x_past) *
    //            [p(y_fut|y_past,x_past) / p(y_fut|y_past)]^(alpha-1) )
    let n_f = n_samples as f64;
    let mut renyi_sum = 0.0;

    for ((y_fut, y_state, x_state), &count) in &joint_counts {
        let p_joint = count as f64 / n_f;
        if p_joint <= 0.0 {
            continue;
        }

        let p_yfut_ypast = marginal_yfut_ypast
            .get(&(*y_fut, y_state.clone()))
            .copied()
            .unwrap_or(0) as f64
            / n_f;
        let p_ypast_xpast = cond_ypast_xpast
            .get(&(y_state.clone(), x_state.clone()))
            .copied()
            .unwrap_or(0) as f64
            / n_f;
        let p_ypast = y_only_counts.get(y_state).copied().unwrap_or(0) as f64 / n_f;

        if p_yfut_ypast > 0.0 && p_ypast_xpast > 0.0 && p_ypast > 0.0 {
            // p(y_fut | y_past, x_past) = p_joint / p_ypast_xpast
            // p(y_fut | y_past) = p_yfut_ypast / p_ypast
            let cond_full = p_joint / p_ypast_xpast;
            let cond_restricted = p_yfut_ypast / p_ypast;

            if cond_full > 0.0 && cond_restricted > 0.0 {
                let ratio = cond_full / cond_restricted;
                renyi_sum += p_joint * ratio.powf(alpha - 1.0);
            }
        }
    }

    let te = if (alpha - 1.0).abs() > 1e-10 && renyi_sum > 0.0 {
        renyi_sum.ln() / (alpha - 1.0)
    } else {
        0.0
    };

    Ok(te.max(0.0))
}

/// Conditional transfer entropy computation
fn compute_conditional_te(
    x: &Array1<f64>,
    y: &Array1<f64>,
    z: &Array2<f64>,
    config: &ConditionalTransferEntropyConfig,
) -> CausalityResult<f64> {
    let max_embed = config.embedding_dim.max(config.cond_embedding_dim);
    let embed_length = max_embed * config.time_delay;
    let n_points = x.len() - embed_length;

    if n_points < 5 {
        return Err(TimeSeriesError::InvalidInput(
            "Too few embedded points for conditional TE".to_string(),
        ));
    }

    let n_cond = z.ncols();

    // Build joint state representations
    let mut joint_counts: HashMap<(usize, Vec<usize>, Vec<usize>, Vec<usize>), usize> =
        HashMap::new();
    let mut marginal_yz: HashMap<(usize, Vec<usize>, Vec<usize>), usize> = HashMap::new();
    let mut cond_yxz: HashMap<(Vec<usize>, Vec<usize>, Vec<usize>), usize> = HashMap::new();
    let mut yz_only: HashMap<(Vec<usize>, Vec<usize>), usize> = HashMap::new();

    for i in 0..n_points {
        let row_idx = embed_length + i;
        let y_fut = discretize_single(y[row_idx], y, config.bins);

        let mut y_state = Vec::with_capacity(config.embedding_dim);
        let mut x_state = Vec::with_capacity(config.embedding_dim);
        let mut z_state = Vec::with_capacity(config.cond_embedding_dim * n_cond);

        for d in 0..config.embedding_dim {
            let idx = i + d * config.time_delay;
            y_state.push(discretize_single(y[idx], y, config.bins));
            x_state.push(discretize_single(x[idx], x, config.bins));
        }

        for c in 0..n_cond {
            let z_col: Array1<f64> = z.column(c).to_owned();
            for d in 0..config.cond_embedding_dim {
                let idx = i + d * config.time_delay;
                z_state.push(discretize_single(z[[idx, c]], &z_col, config.bins));
            }
        }

        *joint_counts
            .entry((y_fut, y_state.clone(), x_state.clone(), z_state.clone()))
            .or_insert(0) += 1;
        *marginal_yz
            .entry((y_fut, y_state.clone(), z_state.clone()))
            .or_insert(0) += 1;
        *cond_yxz
            .entry((y_state.clone(), x_state, z_state.clone()))
            .or_insert(0) += 1;
        *yz_only.entry((y_state, z_state)).or_insert(0) += 1;
    }

    let n_f = n_points as f64;
    let mut te = 0.0;

    for ((y_fut, y_state, x_state, z_state), &count) in &joint_counts {
        let p_joint = count as f64 / n_f;
        if p_joint <= 0.0 {
            continue;
        }

        let p_yz = marginal_yz
            .get(&(*y_fut, y_state.clone(), z_state.clone()))
            .copied()
            .unwrap_or(0) as f64
            / n_f;
        let p_yxz = cond_yxz
            .get(&(y_state.clone(), x_state.clone(), z_state.clone()))
            .copied()
            .unwrap_or(0) as f64
            / n_f;
        let p_yz_only = yz_only
            .get(&(y_state.clone(), z_state.clone()))
            .copied()
            .unwrap_or(0) as f64
            / n_f;

        if p_yz > 0.0 && p_yxz > 0.0 && p_yz_only > 0.0 {
            let ratio = (p_joint * p_yz_only) / (p_yz * p_yxz);
            if ratio > 0.0 {
                te += p_joint * ratio.ln();
            }
        }
    }

    Ok(te)
}

// ---- Embedding and discretization helpers ----

fn create_embeddings(
    x: &Array1<f64>,
    y: &Array1<f64>,
    embedding_dim: usize,
    time_delay: usize,
) -> CausalityResult<(Array2<f64>, Array2<f64>, Array1<f64>)> {
    let embed_length = embedding_dim * time_delay;
    let n_points = x.len() - embed_length;

    let mut x_embed = Array2::zeros((n_points, embedding_dim));
    let mut y_embed = Array2::zeros((n_points, embedding_dim));
    let mut y_future = Array1::zeros(n_points);

    for i in 0..n_points {
        y_future[i] = y[i + embed_length];

        for j in 0..embedding_dim {
            let idx = i + j * time_delay;
            x_embed[[i, j]] = x[idx];
            y_embed[[i, j]] = y[idx];
        }
    }

    Ok((x_embed, y_embed, y_future))
}

fn discretize_matrix(data: &Array2<f64>, bins: usize) -> CausalityResult<Array2<usize>> {
    let mut discrete = Array2::zeros((data.nrows(), data.ncols()));

    for col in 0..data.ncols() {
        let column = data.column(col);
        let min_val = column.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = column.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max_val - min_val).abs() < f64::EPSILON {
            continue;
        }

        let bin_width = (max_val - min_val) / bins as f64;

        for row in 0..data.nrows() {
            let val = data[[row, col]];
            let bin = ((val - min_val) / bin_width).floor() as usize;
            discrete[[row, col]] = bin.min(bins - 1);
        }
    }

    Ok(discrete)
}

fn discretize_vector(data: &Array1<f64>, bins: usize) -> CausalityResult<Array1<usize>> {
    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if (max_val - min_val).abs() < f64::EPSILON {
        return Ok(Array1::zeros(data.len()));
    }

    let bin_width = (max_val - min_val) / bins as f64;
    let discrete = data.mapv(|val| {
        let bin = ((val - min_val) / bin_width).floor() as usize;
        bin.min(bins - 1)
    });

    Ok(discrete)
}

fn discretize_single(value: f64, series: &Array1<f64>, bins: usize) -> usize {
    let min_val = series.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = series.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if (max_val - min_val).abs() < f64::EPSILON {
        return 0;
    }

    let bin_width = (max_val - min_val) / bins as f64;
    let bin = ((value - min_val) / bin_width).floor() as usize;
    bin.min(bins - 1)
}

fn compute_te_from_discrete(
    x_discrete: &Array2<usize>,
    y_discrete: &Array2<usize>,
    y_future_discrete: &Array1<usize>,
) -> CausalityResult<f64> {
    let mut joint_counts: HashMap<(usize, Vec<usize>, Vec<usize>), usize> = HashMap::new();
    let mut marginal_y_counts: HashMap<(usize, Vec<usize>), usize> = HashMap::new();
    let mut conditional_counts: HashMap<(Vec<usize>, Vec<usize>), usize> = HashMap::new();
    let mut y_only_counts: HashMap<Vec<usize>, usize> = HashMap::new();
    let n_samples = x_discrete.nrows();

    for i in 0..n_samples {
        let x_state = x_discrete.row(i).to_vec();
        let y_state = y_discrete.row(i).to_vec();
        let y_fut = y_future_discrete[i];

        *joint_counts
            .entry((y_fut, y_state.clone(), x_state.clone()))
            .or_insert(0) += 1;
        *marginal_y_counts
            .entry((y_fut, y_state.clone()))
            .or_insert(0) += 1;
        *conditional_counts
            .entry((y_state.clone(), x_state))
            .or_insert(0) += 1;
        *y_only_counts.entry(y_state).or_insert(0) += 1;
    }

    let mut te = 0.0;

    for (joint_key, &joint_count) in &joint_counts {
        let prob_joint = joint_count as f64 / n_samples as f64;

        if prob_joint > 0.0 {
            let (y_fut, y_state, x_state) = joint_key;

            let marginal_count = marginal_y_counts
                .get(&(*y_fut, y_state.clone()))
                .copied()
                .unwrap_or(0);
            let cond_count = conditional_counts
                .get(&(y_state.clone(), x_state.clone()))
                .copied()
                .unwrap_or(0);
            let y_only_count = y_only_counts.get(y_state).copied().unwrap_or(0);

            if marginal_count > 0 && cond_count > 0 && y_only_count > 0 {
                let prob_marginal = marginal_count as f64 / n_samples as f64;
                let prob_cond = cond_count as f64 / n_samples as f64;
                let prob_y_only = y_only_count as f64 / n_samples as f64;

                let ratio = (prob_joint * prob_y_only) / (prob_marginal * prob_cond);
                if ratio > 0.0 {
                    te += prob_joint * ratio.ln();
                }
            }
        }
    }

    Ok(te)
}

// ---- Bootstrap helpers ----

fn bootstrap_te_significance(
    x: &Array1<f64>,
    y: &Array1<f64>,
    config: &TransferEntropyConfig,
    observed_te: f64,
    n_bootstrap: usize,
    seed: Option<u64>,
) -> CausalityResult<(f64, f64)> {
    let mut te_values = Vec::with_capacity(n_bootstrap);

    for s in 0..n_bootstrap {
        let mut x_shuffled = x.clone();
        let seed_val = seed.map(|base| base.wrapping_add(s as u64));
        fisher_yates_shuffle(&mut x_shuffled, seed_val);

        let surr_te = compute_te_binning(
            &x_shuffled,
            y,
            config.bins,
            config.embedding_dim,
            config.time_delay,
        )?;
        te_values.push(surr_te);
    }

    let count = te_values.iter().filter(|&&te| te >= observed_te).count();
    let p_value = count as f64 / n_bootstrap as f64;

    let mean_te = te_values.iter().sum::<f64>() / te_values.len() as f64;
    let var_te = te_values
        .iter()
        .map(|&v| (v - mean_te).powi(2))
        .sum::<f64>()
        / (te_values.len() as f64 - 1.0).max(1.0);
    let std_error = var_te.sqrt();

    Ok((p_value, std_error))
}

fn bootstrap_conditional_te_pvalue(
    x: &Array1<f64>,
    y: &Array1<f64>,
    z: &Array2<f64>,
    config: &ConditionalTransferEntropyConfig,
    observed_te: f64,
    n_bootstrap: usize,
) -> CausalityResult<f64> {
    let mut count = 0;

    for s in 0..n_bootstrap {
        let mut x_shuffled = x.clone();
        fisher_yates_shuffle(
            &mut x_shuffled,
            config.seed.map(|b| b.wrapping_add(s as u64)),
        );

        let surr_te = compute_conditional_te(&x_shuffled, y, z, config)?;
        if surr_te >= observed_te {
            count += 1;
        }
    }

    Ok(count as f64 / n_bootstrap as f64)
}

/// Digamma function approximation
fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }

    // For small x, use recurrence: digamma(x) = digamma(x+1) - 1/x
    let mut result = 0.0;
    let mut xx = x;

    while xx < 6.0 {
        result -= 1.0 / xx;
        xx += 1.0;
    }

    // Asymptotic expansion for large x
    result += xx.ln() - 1.0 / (2.0 * xx);
    let xx2 = xx * xx;
    result -= 1.0 / (12.0 * xx2);
    result += 1.0 / (120.0 * xx2 * xx2);
    result -= 1.0 / (252.0 * xx2 * xx2 * xx2);

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_shannon_transfer_entropy() {
        let n = 50;
        let x = Array1::from_vec((0..n).map(|i| (i as f64 * 0.1).sin()).collect());
        let y = Array1::from_vec((0..n).map(|i| ((i as f64 + 1.0) * 0.1).sin()).collect());

        let config = TransferEntropyConfig {
            bins: 5,
            embedding_dim: 2,
            time_delay: 1,
            bootstrap_samples: None,
            estimator: TransferEntropyEstimator::Binning,
            seed: Some(42),
        };

        let result = shannon_transfer_entropy(&x, &y, &config).expect("Shannon TE failed");

        assert!(result.transfer_entropy >= 0.0);
        assert_eq!(result.bins, 5);
        assert_eq!(result.embedding_dim, 2);
    }

    #[test]
    fn test_shannon_te_with_bootstrap() {
        let n = 50;
        let x = Array1::from_vec((0..n).map(|i| (i as f64 * 0.1).sin()).collect());
        let y = Array1::from_vec((0..n).map(|i| ((i as f64 + 1.0) * 0.1).sin()).collect());

        let config = TransferEntropyConfig {
            bins: 5,
            embedding_dim: 2,
            time_delay: 1,
            bootstrap_samples: Some(20),
            estimator: TransferEntropyEstimator::Binning,
            seed: Some(42),
        };

        let result =
            shannon_transfer_entropy(&x, &y, &config).expect("Shannon TE with bootstrap failed");

        assert!(result.transfer_entropy >= 0.0);
        assert!(result.p_value.is_some());
        let p = result.p_value.expect("Should have p-value");
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_kde_estimator() {
        let n = 40;
        let x = Array1::from_vec((0..n).map(|i| (i as f64 * 0.2).sin()).collect());
        let y = Array1::from_vec((0..n).map(|i| ((i as f64 + 2.0) * 0.2).cos()).collect());

        let config = TransferEntropyConfig {
            bins: 5,
            embedding_dim: 2,
            time_delay: 1,
            bootstrap_samples: None,
            estimator: TransferEntropyEstimator::KDE,
            seed: None,
        };

        let result = shannon_transfer_entropy(&x, &y, &config).expect("KDE TE failed");

        assert!(result.transfer_entropy.is_finite());
    }

    #[test]
    fn test_knn_estimator() {
        let n = 40;
        let x = Array1::from_vec((0..n).map(|i| (i as f64 * 0.2).sin()).collect());
        let y = Array1::from_vec((0..n).map(|i| ((i as f64 + 2.0) * 0.2).cos()).collect());

        let config = TransferEntropyConfig {
            bins: 5,
            embedding_dim: 2,
            time_delay: 1,
            bootstrap_samples: None,
            estimator: TransferEntropyEstimator::KNN,
            seed: None,
        };

        let result = shannon_transfer_entropy(&x, &y, &config).expect("kNN TE failed");

        assert!(result.transfer_entropy >= 0.0);
    }

    #[test]
    fn test_renyi_transfer_entropy() {
        let n = 50;
        let x = Array1::from_vec((0..n).map(|i| (i as f64 * 0.1).sin()).collect());
        let y = Array1::from_vec((0..n).map(|i| ((i as f64 + 1.0) * 0.1).sin()).collect());

        let config = RenyiTransferEntropyConfig {
            alpha: 2.0,
            bins: 5,
            embedding_dim: 2,
            time_delay: 1,
            bootstrap_samples: None,
            seed: Some(42),
        };

        let result = renyi_transfer_entropy(&x, &y, &config).expect("Renyi TE failed");

        assert!(result.transfer_entropy >= 0.0);
    }

    #[test]
    fn test_renyi_alpha_one_equals_shannon() {
        let n = 50;
        let x = Array1::from_vec((0..n).map(|i| (i as f64 * 0.1).sin()).collect());
        let y = Array1::from_vec((0..n).map(|i| ((i as f64 + 1.0) * 0.1).sin()).collect());

        let shannon_config = TransferEntropyConfig {
            bins: 5,
            embedding_dim: 2,
            time_delay: 1,
            bootstrap_samples: None,
            estimator: TransferEntropyEstimator::Binning,
            seed: None,
        };

        let renyi_config = RenyiTransferEntropyConfig {
            alpha: 1.0, // Should converge to Shannon
            bins: 5,
            embedding_dim: 2,
            time_delay: 1,
            bootstrap_samples: None,
            seed: None,
        };

        let shannon_result =
            shannon_transfer_entropy(&x, &y, &shannon_config).expect("Shannon failed");
        let renyi_result = renyi_transfer_entropy(&x, &y, &renyi_config).expect("Renyi failed");

        // Should be approximately equal
        assert!(
            (shannon_result.transfer_entropy - renyi_result.transfer_entropy).abs() < 0.1,
            "Shannon ({}) and Renyi(alpha=1) ({}) should be close",
            shannon_result.transfer_entropy,
            renyi_result.transfer_entropy
        );
    }

    #[test]
    fn test_conditional_transfer_entropy() {
        let n = 50;
        let x = Array1::from_vec((0..n).map(|i| (i as f64 * 0.1).sin()).collect());
        let y = Array1::from_vec((0..n).map(|i| ((i as f64 + 1.0) * 0.1).sin()).collect());
        let z = Array2::from_shape_vec((n, 1), (0..n).map(|i| (i as f64 * 0.05).cos()).collect())
            .expect("Shape creation failed");

        let config = ConditionalTransferEntropyConfig {
            bins: 4,
            embedding_dim: 2,
            cond_embedding_dim: 2,
            time_delay: 1,
            bootstrap_samples: None,
            seed: None,
        };

        let result =
            conditional_transfer_entropy(&x, &y, &z, &config).expect("Conditional TE failed");

        assert!(result.transfer_entropy.is_finite());
    }

    #[test]
    fn test_effective_transfer_entropy() {
        let n = 50;
        let x = Array1::from_vec((0..n).map(|i| (i as f64 * 0.1).sin()).collect());
        let y = Array1::from_vec((0..n).map(|i| ((i as f64 + 1.0) * 0.1).sin()).collect());

        let config = EffectiveTransferEntropyConfig {
            bins: 5,
            embedding_dim: 2,
            time_delay: 1,
            n_surrogates: 20,
            seed: Some(42),
        };

        let result = effective_transfer_entropy(&x, &y, &config).expect("Effective TE failed");

        assert!(result.raw_te >= 0.0);
        assert!(result.surrogate_mean >= 0.0);
        assert!(result.surrogate_std >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.n_surrogates, 20);
        // Effective TE should be smaller than raw (bias removed)
        // or can be negative if there's no real signal
        assert!(result.effective_te.is_finite());
    }

    #[test]
    fn test_te_mismatched_lengths() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array1::from_vec(vec![1.0, 2.0]);

        let config = TransferEntropyConfig::default();
        let result = shannon_transfer_entropy(&x, &y, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_te_series_too_short() {
        let x = Array1::from_vec(vec![1.0, 2.0]);
        let y = Array1::from_vec(vec![1.0, 2.0]);

        let config = TransferEntropyConfig {
            bins: 5,
            embedding_dim: 3,
            time_delay: 1,
            bootstrap_samples: None,
            estimator: TransferEntropyEstimator::Binning,
            seed: None,
        };

        let result = shannon_transfer_entropy(&x, &y, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_renyi_invalid_alpha() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0]);

        let config = RenyiTransferEntropyConfig {
            alpha: -1.0,
            ..Default::default()
        };

        let result = renyi_transfer_entropy(&x, &y, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_digamma_function() {
        // digamma(1) = -gamma (Euler-Mascheroni constant)
        let d1 = digamma(1.0);
        assert!((d1 - (-0.5772156649)).abs() < 0.01);

        // digamma(2) = 1 - gamma
        let d2 = digamma(2.0);
        assert!((d2 - (1.0 - 0.5772156649)).abs() < 0.01);
    }
}
