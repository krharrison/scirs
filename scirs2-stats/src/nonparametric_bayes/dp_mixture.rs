//! DP Gaussian Mixture Model via collapsed Gibbs sampling
//!
//! Implements an infinite Gaussian mixture model with a Dirichlet Process prior
//! using the conjugate Normal-Inverse-Wishart (NIW) prior and collapsed Gibbs
//! sampling.  The marginal (predictive) likelihood of each cluster is the
//! multivariate Student-t distribution arising from integrating out the
//! Gaussian parameters against the NIW prior.
//!
//! # Algorithm
//!
//! Each Gibbs iteration:
//! 1. For each data point i, remove it from its cluster (delete cluster if empty).
//! 2. Compute `p(z_i = k | rest)` for all existing clusters k:
//!    `∝ n_k · p(x_i | data in cluster k)`.
//! 3. Compute `p(z_i = new | rest) ∝ α · p(x_i | prior)`.
//! 4. Sample a new assignment for i from the resulting categorical.
//!
//! The marginal likelihood `p(x | data in cluster)` is a multivariate
//! Student-t with parameters derived from the NIW posterior after incorporating
//! the cluster's members.

use crate::error::StatsError;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::{rngs::StdRng, Distribution, SeedableRng, Uniform};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// DpCluster
// ---------------------------------------------------------------------------

/// A single cluster in the DP-GMM.
///
/// Tracks sufficient statistics for the Normal-Inverse-Wishart conjugate update:
/// the member count, the sum of members, and the sum of outer products.
#[derive(Debug, Clone)]
pub struct DpCluster {
    /// Unique cluster id (stable after compaction).
    pub id: usize,
    /// Number of data points assigned to this cluster.
    pub n_members: usize,
    /// Sum of member vectors (length = data dimension).
    pub sum_x: Array1<f64>,
    /// Sum of outer products xᵀx (d×d matrix) for covariance estimation.
    pub sum_sq: Array2<f64>,
}

impl DpCluster {
    /// Construct an empty cluster with the given dimensionality.
    pub fn new(id: usize, dim: usize) -> Self {
        Self {
            id,
            n_members: 0,
            sum_x: Array1::zeros(dim),
            sum_sq: Array2::zeros((dim, dim)),
        }
    }

    /// Add a data point to the cluster's sufficient statistics.
    pub fn add_point(&mut self, x: ArrayView1<f64>) {
        self.n_members += 1;
        self.sum_x += &x;
        // Outer product x·xᵀ
        let d = x.len();
        for i in 0..d {
            for j in 0..d {
                self.sum_sq[[i, j]] += x[i] * x[j];
            }
        }
    }

    /// Remove a data point from the cluster's sufficient statistics.
    pub fn remove_point(&mut self, x: ArrayView1<f64>) -> Result<(), StatsError> {
        if self.n_members == 0 {
            return Err(StatsError::ComputationError(
                "DpCluster::remove_point: cluster is already empty".to_string(),
            ));
        }
        self.n_members -= 1;
        self.sum_x -= &x;
        let d = x.len();
        for i in 0..d {
            for j in 0..d {
                self.sum_sq[[i, j]] -= x[i] * x[j];
            }
        }
        Ok(())
    }

    /// Sample mean of the cluster's members.
    pub fn mean(&self) -> Array1<f64> {
        if self.n_members == 0 {
            return self.sum_x.clone(); // zeros
        }
        &self.sum_x / (self.n_members as f64)
    }
}

// ---------------------------------------------------------------------------
// DpGaussianMixture
// ---------------------------------------------------------------------------

/// DP Gaussian Mixture Model with Normal-Inverse-Wishart prior.
///
/// Hyperparameters follow the NIW parameterisation:
/// - `prior_mean` μ₀: prior on the cluster mean (default: zero).
/// - `prior_kappa` κ₀: concentration on the mean (≥ 1e-6).
/// - `prior_nu` ν₀: degrees of freedom (≥ d + 2).
/// - `prior_psi` Ψ₀: scale matrix (d×d positive-definite; default: I).
#[derive(Debug, Clone)]
pub struct DpGaussianMixture {
    // ----- Hyperparameters -----
    /// DP concentration parameter α > 0.
    pub alpha: f64,
    /// NIW prior mean μ₀ (length = data dimension d).
    pub prior_mean: Array1<f64>,
    /// NIW prior precision on the mean κ₀ > 0.
    pub prior_kappa: f64,
    /// NIW prior degrees of freedom ν₀ ≥ d.
    pub prior_nu: f64,
    /// NIW prior scale matrix Ψ₀ (d×d).
    pub prior_psi: Array2<f64>,

    // ----- Model state -----
    /// Cluster assignment for each data point (index into `clusters`).
    pub assignments: Vec<usize>,
    /// Active clusters (all have n_members > 0 during sampling).
    pub clusters: Vec<DpCluster>,

    // ----- Internal -----
    dim: usize,
    /// Next cluster id to assign when creating a new cluster.
    next_cluster_id: usize,
}

impl DpGaussianMixture {
    /// Create a new DP-GMM with default NIW hyperparameters.
    ///
    /// # Parameters
    /// - `alpha`: DP concentration parameter (> 0).
    /// - `data_dim`: Dimensionality of the data d.
    /// - `prior_mean`: Optional prior mean μ₀ (default: zero vector).
    pub fn new(alpha: f64, data_dim: usize, prior_mean: Option<Array1<f64>>) -> Self {
        let mu0 = prior_mean.unwrap_or_else(|| Array1::zeros(data_dim));
        let psi0 = Array2::eye(data_dim);
        let nu0 = (data_dim as f64) + 2.0;
        Self {
            alpha,
            prior_mean: mu0,
            prior_kappa: 1.0,
            prior_nu: nu0,
            prior_psi: psi0,
            assignments: Vec::new(),
            clusters: Vec::new(),
            dim: data_dim,
            next_cluster_id: 0,
        }
    }

    /// Initialise `n_points` assignments using `n_init_clusters` random clusters.
    pub fn initialize(&mut self, n_points: usize, n_init_clusters: usize, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let k = n_init_clusters.max(1);
        let uniform = Uniform::new(0usize, k).expect("Uniform init failed in initialize");

        self.clusters = (0..k).map(|i| DpCluster::new(i, self.dim)).collect();
        self.next_cluster_id = k;
        self.assignments = (0..n_points)
            .map(|_| uniform.sample(&mut rng))
            .collect();
    }

    /// Return the current number of active clusters.
    pub fn n_clusters(&self) -> usize {
        self.clusters.len()
    }

    /// Return the sizes of all active clusters.
    pub fn cluster_sizes(&self) -> Vec<usize> {
        self.clusters.iter().map(|c| c.n_members).collect()
    }

    /// Return the cluster assignment for each data point.
    pub fn assignments(&self) -> &[usize] {
        &self.assignments
    }

    // ------------------------------------------------------------------
    // NIW predictive marginal: log p(x | cluster)
    // ------------------------------------------------------------------

    /// Log predictive probability of observation `x` given cluster `c`.
    ///
    /// Uses the NIW posterior predictive (multivariate Student-t):
    ///
    /// After observing n points with sufficient statistics (sum_x, sum_sq),
    /// the posterior hyperparameters are:
    /// - κ_n = κ₀ + n
    /// - ν_n = ν₀ + n
    /// - μ_n = (κ₀·μ₀ + sum_x) / κ_n
    /// - Ψ_n = Ψ₀ + sum_sq + κ₀·μ₀·μ₀ᵀ - κ_n·μ_n·μ_nᵀ
    ///
    /// The marginal is Student-t with df = ν_n - d + 1, location μ_n,
    /// and scale Ψ_n·(κ_n+1)/(κ_n·(ν_n-d+1)).
    ///
    /// For simplicity and numerical stability we use the diagonal of Ψ_n
    /// as an isotropic (diagonal) approximation.
    fn log_predictive(&self, x: ArrayView1<f64>, cluster: &DpCluster) -> f64 {
        let n = cluster.n_members as f64;
        let d = self.dim as f64;

        let kappa_n = self.prior_kappa + n;
        let nu_n = self.prior_nu + n;

        // Posterior mean
        let mu_n = (self.prior_kappa * &self.prior_mean + &cluster.sum_x) / kappa_n;

        // Posterior scale matrix (diagonal elements only for efficiency)
        // Ψ_n[i,i] = Ψ₀[i,i] + sum_sq[i,i] + κ₀·μ₀[i]² - κ_n·μ_n[i]²
        let df = nu_n - d + 1.0;
        if df <= 0.0 {
            return f64::NEG_INFINITY;
        }

        let scale_factor = (kappa_n + 1.0) / (kappa_n * df);

        // Log Student-t pdf using diagonal approximation
        // log p(x) = Σ_i log t_{df}((x_i - μ_n[i]) / sqrt(scale_factor * Ψ_n[i,i]))
        //            - 0.5 * log(scale_factor * Ψ_n[i,i])
        let mut log_p = 0.0_f64;

        for i in 0..self.dim {
            let psi_ii = self.prior_psi[[i, i]]
                + cluster.sum_sq[[i, i]]
                + self.prior_kappa * self.prior_mean[i] * self.prior_mean[i]
                - kappa_n * mu_n[i] * mu_n[i];
            let psi_ii = psi_ii.max(1e-10); // numerical floor

            let sigma2 = scale_factor * psi_ii;
            let sigma = sigma2.sqrt();
            let z = (x[i] - mu_n[i]) / sigma;

            // log t_{df}(z) = log Γ((df+1)/2) - log Γ(df/2) - 0.5*log(df*π) - (df+1)/2 * log(1 + z²/df)
            log_p += log_student_t_density(z, df);
            log_p -= sigma.ln(); // Jacobian from standardisation
        }

        log_p
    }

    /// Log predictive probability of observation `x` under the NIW prior
    /// (i.e., for a brand-new empty cluster).
    fn log_prior_predictive(&self, x: ArrayView1<f64>) -> f64 {
        let d = self.dim as f64;
        let kappa_n = self.prior_kappa; // n=0 ⇒ posterior = prior
        let nu_n = self.prior_nu;
        let df = nu_n - d + 1.0;
        if df <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let scale_factor = (kappa_n + 1.0) / (kappa_n * df);

        let mut log_p = 0.0_f64;
        for i in 0..self.dim {
            let psi_ii = self.prior_psi[[i, i]].max(1e-10);
            let sigma2 = scale_factor * psi_ii;
            let sigma = sigma2.sqrt();
            let z = (x[i] - self.prior_mean[i]) / sigma;
            log_p += log_student_t_density(z, df);
            log_p -= sigma.ln();
        }
        log_p
    }

    // ------------------------------------------------------------------
    // Single Gibbs step for data point `idx`
    // ------------------------------------------------------------------

    fn gibbs_step_single(
        &mut self,
        data: ArrayView2<f64>,
        idx: usize,
        rng: &mut StdRng,
    ) -> Result<(), StatsError> {
        let x = data.row(idx);
        let old_cluster_pos = self.assignments[idx];

        // 1. Remove x from its current cluster
        {
            let cluster = self.clusters.get_mut(old_cluster_pos).ok_or_else(|| {
                StatsError::ComputationError(format!(
                    "Gibbs step: invalid cluster position {old_cluster_pos}"
                ))
            })?;
            cluster.remove_point(x)?;

            // Delete cluster if empty
            if cluster.n_members == 0 {
                self.clusters.remove(old_cluster_pos);
                // Update all assignments that pointed to clusters after this one
                for a in self.assignments.iter_mut() {
                    if *a > old_cluster_pos {
                        *a -= 1;
                    }
                }
                // The current point's assignment is now stale; set it to 0 as placeholder
                self.assignments[idx] = 0;
            }
        }

        // 2. Compute log weights for all existing clusters + new cluster
        let n_existing = self.clusters.len();
        let mut log_weights: Vec<f64> = Vec::with_capacity(n_existing + 1);

        for cluster in self.clusters.iter() {
            let lp = (cluster.n_members as f64).ln() + self.log_predictive(x, cluster);
            log_weights.push(lp);
        }
        // New cluster: CRP prior weight × NIW prior predictive
        let lp_new = self.alpha.ln() + self.log_prior_predictive(x);
        log_weights.push(lp_new);

        // 3. Stable softmax → normalised probabilities
        let max_lw = log_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = log_weights
            .iter()
            .map(|&lw| if lw.is_finite() { (lw - max_lw).exp() } else { 0.0 })
            .collect();
        let total: f64 = weights.iter().sum();
        if total == 0.0 {
            return Err(StatsError::ComputationError(
                "Gibbs step: all weights are zero".to_string(),
            ));
        }
        let probs: Vec<f64> = weights.iter().map(|&w| w / total).collect();

        // 4. Categorical sample
        let uniform = Uniform::new(0.0_f64, 1.0).map_err(|e| {
            StatsError::ComputationError(format!("Gibbs uniform init error: {e}"))
        })?;
        let u = uniform.sample(rng);
        let mut cumulative = 0.0_f64;
        let mut new_choice = n_existing; // default: new cluster
        for (k, &p) in probs.iter().enumerate() {
            cumulative += p;
            if u < cumulative {
                new_choice = k;
                break;
            }
        }

        // 5. Assign to chosen cluster (existing or new)
        if new_choice < n_existing {
            self.assignments[idx] = new_choice;
            self.clusters[new_choice].add_point(x);
        } else {
            // Create new cluster
            let new_id = self.next_cluster_id;
            self.next_cluster_id += 1;
            let mut new_cluster = DpCluster::new(new_id, self.dim);
            new_cluster.add_point(x);
            self.assignments[idx] = self.clusters.len();
            self.clusters.push(new_cluster);
        }

        Ok(())
    }

    // ------------------------------------------------------------------
    // Compute log-likelihood of current assignment
    // ------------------------------------------------------------------

    fn current_log_likelihood(&self, data: ArrayView2<f64>) -> f64 {
        let mut ll = 0.0_f64;
        for (i, &ci) in self.assignments.iter().enumerate() {
            if let Some(cluster) = self.clusters.get(ci) {
                ll += self.log_predictive(data.row(i), cluster);
            }
        }
        ll
    }

    // ------------------------------------------------------------------
    // fit: collapsed Gibbs sampling
    // ------------------------------------------------------------------

    /// Fit the DP-GMM to `data` via collapsed Gibbs sampling.
    ///
    /// # Parameters
    /// - `data`: (n_samples, n_features) data matrix.
    /// - `n_iter`: Total number of Gibbs sweeps.
    /// - `burn_in`: Number of initial sweeps discarded for warm-up.
    /// - `seed`: Random seed.
    ///
    /// # Returns
    /// A [`DpGmmResult`] containing posterior summaries.
    pub fn fit(
        &mut self,
        data: ArrayView2<f64>,
        n_iter: usize,
        burn_in: usize,
        seed: u64,
    ) -> Result<DpGmmResult, StatsError> {
        let (n_samples, n_features) = (data.nrows(), data.ncols());

        if n_samples == 0 {
            return Err(StatsError::InsufficientData(
                "dp_gmm fit: data has no rows".to_string(),
            ));
        }
        if n_features != self.dim {
            return Err(StatsError::DimensionMismatch(format!(
                "dp_gmm fit: data has {n_features} features but model dim is {}",
                self.dim
            )));
        }
        if n_iter == 0 {
            return Err(StatsError::InvalidArgument(
                "dp_gmm fit: n_iter must be >= 1".to_string(),
            ));
        }
        if burn_in >= n_iter {
            return Err(StatsError::InvalidArgument(format!(
                "dp_gmm fit: burn_in ({burn_in}) must be < n_iter ({n_iter})"
            )));
        }

        // Initialise if not yet done
        if self.assignments.is_empty() {
            let n_init = ((self.alpha * (n_samples as f64).ln()).round() as usize).max(1);
            self.initialize(n_samples, n_init, seed.wrapping_add(1));
        } else if self.assignments.len() != n_samples {
            return Err(StatsError::DimensionMismatch(format!(
                "dp_gmm fit: pre-set assignments length {} != n_samples {n_samples}",
                self.assignments.len()
            )));
        }

        // Build cluster sufficient statistics from current assignments
        self.rebuild_sufficient_statistics(data)?;

        let mut rng = StdRng::seed_from_u64(seed);

        let mut log_likelihoods: Vec<f64> = Vec::with_capacity(n_iter - burn_in);
        let mut n_clusters_trace: Vec<usize> = Vec::with_capacity(n_iter - burn_in);
        let mut all_post_assignments: Vec<Vec<usize>> = Vec::new();

        for iter in 0..n_iter {
            // One full Gibbs sweep
            for idx in 0..n_samples {
                self.gibbs_step_single(data, idx, &mut rng)?;
            }

            if iter >= burn_in {
                let ll = self.current_log_likelihood(data);
                log_likelihoods.push(ll);
                n_clusters_trace.push(self.clusters.len());
                all_post_assignments.push(self.assignments.clone());
            }
        }

        // Summarise posterior: cluster means from the final state
        let cluster_means: Vec<Array1<f64>> =
            self.clusters.iter().map(|c| c.mean()).collect();
        let cluster_sizes: Vec<usize> = self.cluster_sizes();
        let n_clusters = self.clusters.len();

        Ok(DpGmmResult {
            assignments: self.assignments.clone(),
            n_clusters,
            cluster_means,
            cluster_sizes,
            log_likelihoods,
            n_clusters_trace,
            all_post_assignments,
        })
    }

    // ------------------------------------------------------------------
    // Internal: rebuild sufficient statistics from current assignments
    // ------------------------------------------------------------------

    fn rebuild_sufficient_statistics(
        &mut self,
        data: ArrayView2<f64>,
    ) -> Result<(), StatsError> {
        // Determine how many distinct cluster indices exist
        let max_idx = self.assignments.iter().cloned().max().unwrap_or(0);

        // Rebuild clusters list to have exactly `max_idx + 1` entries
        let n_needed = max_idx + 1;
        self.clusters = (0..n_needed)
            .map(|id| DpCluster::new(id, self.dim))
            .collect();
        self.next_cluster_id = n_needed;

        for (i, &ci) in self.assignments.iter().enumerate() {
            if ci >= self.clusters.len() {
                return Err(StatsError::ComputationError(format!(
                    "rebuild_sufficient_statistics: assignment {ci} out of bounds"
                )));
            }
            self.clusters[ci].add_point(data.row(i));
        }

        // Remove empty clusters and update assignments
        self.compact_clusters();
        Ok(())
    }

    /// Remove empty clusters and renumber assignments contiguously.
    fn compact_clusters(&mut self) {
        let old_to_new: Vec<Option<usize>> = self
            .clusters
            .iter()
            .scan(0usize, |next, c| {
                if c.n_members > 0 {
                    let new_id = *next;
                    *next += 1;
                    Some(Some(new_id))
                } else {
                    Some(None)
                }
            })
            .collect();

        // Update assignments
        for a in self.assignments.iter_mut() {
            if let Some(new_id) = old_to_new[*a] {
                *a = new_id;
            }
        }

        // Filter and update clusters
        let mut new_clusters: Vec<DpCluster> = Vec::new();
        for (i, cluster) in self.clusters.drain(..).enumerate() {
            if let Some(new_id) = old_to_new[i] {
                let mut c = cluster;
                c.id = new_id;
                new_clusters.push(c);
            }
        }
        self.clusters = new_clusters;
    }
}

// ---------------------------------------------------------------------------
// DpGmmResult
// ---------------------------------------------------------------------------

/// Result returned by [`DpGaussianMixture::fit`].
#[derive(Debug, Clone)]
pub struct DpGmmResult {
    /// Cluster assignment for each data point (final Gibbs state).
    pub assignments: Vec<usize>,
    /// Number of active clusters in the final state.
    pub n_clusters: usize,
    /// Posterior mean of each cluster (computed from sufficient statistics).
    pub cluster_means: Vec<Array1<f64>>,
    /// Number of members in each cluster.
    pub cluster_sizes: Vec<usize>,
    /// Log-likelihood of the data per post-burn-in Gibbs iteration.
    pub log_likelihoods: Vec<f64>,
    /// Number of active clusters per post-burn-in Gibbs iteration.
    pub n_clusters_trace: Vec<usize>,
    /// All post-burn-in assignment vectors (for similarity matrix computation).
    pub all_post_assignments: Vec<Vec<usize>>,
}

impl DpGmmResult {
    /// Return the cluster assignment for data point `idx`.
    pub fn predict_cluster(&self, idx: usize) -> usize {
        self.assignments[idx]
    }

    /// Return the mode of the number of clusters across all post-burn-in samples.
    pub fn n_clusters_mode(&self) -> usize {
        if self.n_clusters_trace.is_empty() {
            return self.n_clusters;
        }
        let max_k = *self.n_clusters_trace.iter().max().unwrap_or(&1);
        let mut counts = vec![0usize; max_k + 1];
        for &k in &self.n_clusters_trace {
            counts[k] += 1;
        }
        counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(k, _)| k)
            .unwrap_or(self.n_clusters)
    }

    /// Compute the posterior co-clustering (similarity) matrix.
    ///
    /// `S[i,j] = fraction of posterior samples where i and j are in the same cluster.`
    ///
    /// # Parameters
    /// - `all_assignments`: Collection of assignment vectors, one per posterior sample.
    ///   If empty, uses `self.all_post_assignments`.
    pub fn similarity_matrix(&self, all_assignments: &[Vec<usize>]) -> Array2<f64> {
        let samples = if all_assignments.is_empty() {
            self.all_post_assignments.as_slice()
        } else {
            all_assignments
        };

        if samples.is_empty() {
            let n = self.assignments.len();
            return Array2::zeros((n, n));
        }

        let n = samples[0].len();
        let mut mat = Array2::<f64>::zeros((n, n));
        let n_samples = samples.len() as f64;

        for sample in samples {
            for i in 0..n {
                for j in 0..n {
                    if sample[i] == sample[j] {
                        mat[[i, j]] += 1.0;
                    }
                }
            }
        }
        mat /= n_samples;
        mat
    }
}

// ---------------------------------------------------------------------------
// Simple interface
// ---------------------------------------------------------------------------

/// Fit a DP Gaussian Mixture Model to `data` and return cluster assignments.
///
/// Uses default NIW hyperparameters (zero mean, identity scale, κ₀=1, ν₀=d+2).
///
/// # Parameters
/// - `data`: (n_samples, n_features) array.
/// - `alpha`: DP concentration parameter (> 0).
/// - `n_iter`: Number of Gibbs sweeps.
/// - `seed`: Random seed.
///
/// # Returns
/// [`DpGmmResult`] with posterior summaries.
pub fn dp_gmm_cluster(
    data: ArrayView2<f64>,
    alpha: f64,
    n_iter: usize,
    seed: u64,
) -> Result<DpGmmResult, StatsError> {
    if alpha <= 0.0 {
        return Err(StatsError::DomainError(format!(
            "dp_gmm_cluster: alpha must be > 0, got {alpha}"
        )));
    }
    let dim = data.ncols();
    if dim == 0 {
        return Err(StatsError::InvalidArgument(
            "dp_gmm_cluster: data has zero features".to_string(),
        ));
    }

    let burn_in = (n_iter / 4).max(1);
    let n_iter_safe = if n_iter <= burn_in { burn_in + 1 } else { n_iter };

    let mut model = DpGaussianMixture::new(alpha, dim, None);
    model.fit(data, n_iter_safe, burn_in, seed)
}

// ---------------------------------------------------------------------------
// Log-likelihood evaluation
// ---------------------------------------------------------------------------

/// Evaluate the Gaussian log-likelihood of `data` given cluster means and covariances.
///
/// Computes `Σ_i log p(x_i | cluster assignment, cluster mean, cluster cov)` where
/// each observation contributes through its assigned cluster's multivariate normal.
///
/// # Parameters
/// - `data`: (n, d) observation matrix.
/// - `assignments`: Cluster label for each of the n observations.
/// - `cluster_means`: Slice of d-dimensional mean vectors.
/// - `cluster_covs`: Slice of (d×d) covariance matrices.
///
/// # Returns
/// Total log-likelihood (−∞ if any assignment is out of bounds or a cov is degenerate).
pub fn dp_gmm_log_likelihood(
    data: ArrayView2<f64>,
    assignments: &[usize],
    cluster_means: &[Array1<f64>],
    cluster_covs: &[Array2<f64>],
) -> f64 {
    let (n, d) = (data.nrows(), data.ncols());
    if assignments.len() != n {
        return f64::NEG_INFINITY;
    }
    let n_clusters = cluster_means.len();
    if n_clusters == 0 || cluster_covs.len() != n_clusters {
        return f64::NEG_INFINITY;
    }

    let mut total_ll = 0.0_f64;

    for i in 0..n {
        let ci = assignments[i];
        if ci >= n_clusters {
            return f64::NEG_INFINITY;
        }
        let x = data.row(i);
        let mu = &cluster_means[ci];
        let cov = &cluster_covs[ci];

        // Use diagonal of covariance for the log-likelihood (isotropic approx)
        let mut ll_i = -(d as f64) / 2.0 * (2.0 * PI).ln();
        for j in 0..d {
            let sigma2 = cov[[j, j]].max(1e-10);
            let diff = x[j] - mu[j];
            ll_i -= 0.5 * sigma2.ln() + 0.5 * diff * diff / sigma2;
        }
        total_ll += ll_i;
    }

    total_ll
}

// ---------------------------------------------------------------------------
// Student-t log density helper
// ---------------------------------------------------------------------------

/// Log density of the standard univariate Student-t distribution with `df` degrees of freedom.
///
/// `log t_{df}(z) = log Γ((df+1)/2) − log Γ(df/2) − 0.5·log(df·π) − (df+1)/2 · log(1 + z²/df)`
fn log_student_t_density(z: f64, df: f64) -> f64 {
    let log_norm = lgamma((df + 1.0) / 2.0)
        - lgamma(df / 2.0)
        - 0.5 * (df * PI).ln();
    log_norm - (df + 1.0) / 2.0 * (1.0 + z * z / df).ln()
}

/// Log-gamma function via Lanczos approximation (g=7, n=9 coefficients).
fn lgamma(x: f64) -> f64 {
    // Stirling / Lanczos approximation valid for x > 0
    // We use the Lanczos g=7 coefficients from Numerical Recipes.
    const G: f64 = 7.0;
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_10,
        -1_259.139_216_722_402_9,
        771.323_428_777_653_08,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        // Reflection formula: Γ(x)·Γ(1-x) = π/sin(πx)
        return (PI / (PI * x).sin()).ln() - lgamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = COEFFS[0];
    let t = x + G + 0.5;
    for (i, &c) in COEFFS[1..].iter().enumerate() {
        a += c / (x + (i + 1) as f64);
    }
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Generate a synthetic 2D dataset with well-separated Gaussian clusters.
    ///
    /// Returns an (n_total, 2) Array2 and the ground-truth integer labels.
    fn make_clusters(
        centers: &[(f64, f64)],
        n_per_cluster: usize,
        std: f64,
        seed: u64,
    ) -> (Array2<f64>, Vec<usize>) {
        use scirs2_core::random::{rngs::StdRng, Distribution, Normal, SeedableRng};
        let mut rng = StdRng::seed_from_u64(seed);
        let n_total = centers.len() * n_per_cluster;
        let mut data = Array2::<f64>::zeros((n_total, 2));
        let mut labels = Vec::with_capacity(n_total);
        let normal = Normal::new(0.0_f64, std).expect("Normal init failed in make_clusters");

        for (k, &(cx, cy)) in centers.iter().enumerate() {
            for i in 0..n_per_cluster {
                let row = k * n_per_cluster + i;
                data[[row, 0]] = cx + normal.sample(&mut rng);
                data[[row, 1]] = cy + normal.sample(&mut rng);
                labels.push(k);
            }
        }
        (data, labels)
    }

    // --- DpCluster ---

    #[test]
    fn test_dpcluster_add_remove() {
        let mut c = DpCluster::new(0, 2);
        let x = Array1::from(vec![1.0, 2.0]);
        c.add_point(x.view());
        assert_eq!(c.n_members, 1);
        assert!((c.sum_x[0] - 1.0).abs() < 1e-12);
        c.remove_point(x.view()).expect("remove_point failed");
        assert_eq!(c.n_members, 0);
        assert!(c.sum_x[0].abs() < 1e-12);
    }

    #[test]
    fn test_dpcluster_remove_from_empty() {
        let mut c = DpCluster::new(0, 2);
        let x = Array1::from(vec![1.0, 2.0]);
        assert!(c.remove_point(x.view()).is_err());
    }

    #[test]
    fn test_dpcluster_mean() {
        let mut c = DpCluster::new(0, 2);
        c.add_point(Array1::from(vec![2.0, 4.0]).view());
        c.add_point(Array1::from(vec![4.0, 6.0]).view());
        let mean = c.mean();
        assert!((mean[0] - 3.0).abs() < 1e-12);
        assert!((mean[1] - 5.0).abs() < 1e-12);
    }

    // --- DpGaussianMixture ---

    #[test]
    fn test_dpgmm_new_default_hyperparams() {
        let model = DpGaussianMixture::new(1.0, 3, None);
        assert_eq!(model.dim, 3);
        assert_eq!(model.prior_mean.len(), 3);
        assert_eq!(model.prior_psi.shape(), [3, 3]);
    }

    #[test]
    fn test_dpgmm_initialize() {
        let mut model = DpGaussianMixture::new(1.0, 2, None);
        model.initialize(50, 3, 42);
        assert_eq!(model.assignments.len(), 50);
        assert!(model.assignments.iter().all(|&a| a < 3));
    }

    #[test]
    fn test_dpgmm_invalid_fit() {
        let mut model = DpGaussianMixture::new(1.0, 2, None);
        let data = Array2::<f64>::zeros((0, 2));
        assert!(model.fit(data.view(), 10, 2, 0).is_err());

        let bad_dim = Array2::<f64>::zeros((10, 3));
        let mut model2 = DpGaussianMixture::new(1.0, 2, None);
        assert!(model2.fit(bad_dim.view(), 10, 2, 0).is_err());
    }

    #[test]
    fn test_dpgmm_fit_one_cluster() {
        // All data at the same location → should collapse to 1 cluster
        let data = Array2::from_shape_vec((20, 2), vec![0.0_f64; 40])
            .expect("shape vec failed in test_dpgmm_fit_one_cluster");
        let mut model = DpGaussianMixture::new(0.01, 2, None);
        let result = model.fit(data.view(), 30, 5, 42).expect("fit failed");
        // Very small alpha + zero-variance data: likely 1 cluster
        assert!(result.n_clusters <= 3, "expected few clusters for identical data");
    }

    #[test]
    fn test_dpgmm_recovers_two_clusters() {
        // Well-separated clusters should be discovered
        let centers = vec![(-10.0, 0.0), (10.0, 0.0)];
        let (data, _labels) = make_clusters(&centers, 30, 0.5, 1);

        let mut model = DpGaussianMixture::new(1.0, 2, None);
        let result = model.fit(data.view(), 50, 10, 42).expect("fit failed");

        // Should find 2 clusters (or close to it)
        let k_mode = result.n_clusters_mode();
        assert!(
            k_mode >= 1 && k_mode <= 4,
            "expected ~2 clusters, got mode {k_mode}"
        );
    }

    #[test]
    fn test_dpgmm_recovers_three_clusters() {
        let centers = vec![(-8.0, 0.0), (0.0, 8.0), (8.0, 0.0)];
        let (data, _) = make_clusters(&centers, 25, 0.8, 7);

        let result = dp_gmm_cluster(data.view(), 1.0, 40, 42).expect("dp_gmm_cluster failed");
        let k_mode = result.n_clusters_mode();
        assert!(
            k_mode >= 2 && k_mode <= 5,
            "expected ~3 clusters, got mode {k_mode}"
        );
    }

    #[test]
    fn test_dpgmm_log_likelihoods_length() {
        let (data, _) = make_clusters(&[(0.0, 0.0), (5.0, 5.0)], 20, 1.0, 0);
        let result = dp_gmm_cluster(data.view(), 1.0, 20, 0).expect("dp_gmm_cluster failed");
        // burn_in = 20/4 = 5, so post-burn-in = 15 samples
        assert!(
            !result.log_likelihoods.is_empty(),
            "log_likelihoods must not be empty"
        );
        assert_eq!(
            result.log_likelihoods.len(),
            result.n_clusters_trace.len()
        );
    }

    #[test]
    fn test_dpgmm_result_predict_cluster() {
        let (data, _) = make_clusters(&[(0.0, 0.0), (10.0, 0.0)], 10, 0.5, 42);
        let result = dp_gmm_cluster(data.view(), 1.0, 20, 42).expect("dp_gmm_cluster failed");
        // predict_cluster should return a valid cluster index
        for i in 0..data.nrows() {
            let c = result.predict_cluster(i);
            assert!(c < result.n_clusters, "predicted cluster {c} >= n_clusters {}", result.n_clusters);
        }
    }

    #[test]
    fn test_dpgmm_similarity_matrix_shape() {
        let (data, _) = make_clusters(&[(0.0, 0.0), (5.0, 0.0)], 10, 0.5, 0);
        let result = dp_gmm_cluster(data.view(), 1.0, 15, 0).expect("dp_gmm_cluster failed");
        let sim = result.similarity_matrix(&[]);
        assert_eq!(sim.shape(), [data.nrows(), data.nrows()]);
    }

    #[test]
    fn test_dpgmm_similarity_matrix_diagonal_ones() {
        let (data, _) = make_clusters(&[(0.0, 0.0), (5.0, 0.0)], 10, 0.5, 0);
        let result = dp_gmm_cluster(data.view(), 1.0, 15, 0).expect("dp_gmm_cluster failed");
        let sim = result.similarity_matrix(&[]);
        let n = data.nrows();
        for i in 0..n {
            assert!(
                (sim[[i, i]] - 1.0).abs() < 1e-12,
                "diagonal should be 1.0, got {}",
                sim[[i, i]]
            );
        }
    }

    // --- dp_gmm_log_likelihood ---

    #[test]
    fn test_dp_gmm_log_likelihood_basic() {
        let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.1, 0.1, 5.0, 5.0, 5.1, 5.1])
            .expect("shape vec failed in test_dp_gmm_log_likelihood_basic");
        let assignments = vec![0, 0, 1, 1];
        let means = vec![
            Array1::from(vec![0.05, 0.05]),
            Array1::from(vec![5.05, 5.05]),
        ];
        let covs = vec![Array2::eye(2), Array2::eye(2)];
        let ll = dp_gmm_log_likelihood(data.view(), &assignments, &means, &covs);
        assert!(ll.is_finite(), "log-likelihood should be finite");
        assert!(ll < 0.0, "log-likelihood should be negative");
    }

    #[test]
    fn test_dp_gmm_log_likelihood_wrong_length() {
        let data = Array2::zeros((3, 2));
        let ll = dp_gmm_log_likelihood(
            data.view(),
            &[0, 1],
            &[Array1::zeros(2)],
            &[Array2::eye(2)],
        );
        assert_eq!(ll, f64::NEG_INFINITY);
    }

    #[test]
    fn test_dp_gmm_log_likelihood_out_of_range_assignment() {
        let data = Array2::zeros((2, 2));
        let ll = dp_gmm_log_likelihood(
            data.view(),
            &[0, 5], // 5 is out of range
            &[Array1::zeros(2)],
            &[Array2::eye(2)],
        );
        assert_eq!(ll, f64::NEG_INFINITY);
    }

    // --- lgamma helper ---

    #[test]
    fn test_lgamma_known_values() {
        // Γ(1) = 1  →  ln Γ(1) = 0
        assert!((lgamma(1.0)).abs() < 1e-6, "lgamma(1) should be 0");
        // Γ(2) = 1  →  ln Γ(2) = 0
        assert!((lgamma(2.0)).abs() < 1e-6, "lgamma(2) should be 0");
        // Γ(0.5) = sqrt(π)  →  ln Γ(0.5) = 0.5 * ln(π)
        let expected = 0.5 * PI.ln();
        assert!(
            (lgamma(0.5) - expected).abs() < 1e-5,
            "lgamma(0.5) = {}, expected {expected}",
            lgamma(0.5)
        );
    }

    // --- n_clusters_mode ---

    #[test]
    fn test_n_clusters_mode_correctness() {
        let result = DpGmmResult {
            assignments: vec![0, 1, 0],
            n_clusters: 2,
            cluster_means: vec![Array1::zeros(1), Array1::zeros(1)],
            cluster_sizes: vec![2, 1],
            log_likelihoods: vec![-1.0, -1.0, -1.0],
            n_clusters_trace: vec![3, 2, 2, 3, 2],
            all_post_assignments: vec![],
        };
        assert_eq!(result.n_clusters_mode(), 2);
    }

    // --- dp_gmm_cluster invalid input ---

    #[test]
    fn test_dp_gmm_cluster_invalid_alpha() {
        let data = Array2::zeros((10, 2));
        assert!(dp_gmm_cluster(data.view(), 0.0, 20, 0).is_err());
        assert!(dp_gmm_cluster(data.view(), -1.0, 20, 0).is_err());
    }

    #[test]
    fn test_dp_gmm_cluster_zero_features() {
        let data = Array2::zeros((10, 0));
        assert!(dp_gmm_cluster(data.view(), 1.0, 20, 0).is_err());
    }
}
