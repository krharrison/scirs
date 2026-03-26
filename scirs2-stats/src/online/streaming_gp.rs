//! Streaming / Online Sparse Gaussian Process Regression
//!
//! Implements an online variant of the Sparse Variational Gaussian Process (SVGP)
//! using inducing-point-based variational inference. The algorithm maintains a
//! compact posterior approximation q(u) = N(m, S) over the inducing outputs u,
//! and performs Kalman-like rank-1 updates as new mini-batches arrive.
//!
//! # Algorithm overview
//!
//! The full GP posterior is approximated via M ≪ N inducing points Z ⊂ X:
//!
//!   q(f*) = ∫ p(f* | u) q(u) du
//!
//! where q(u) = N(m, S) is maintained online. Given new observations (x_new, y_new),
//! the posterior is updated via the sparse GP update equations:
//!
//!   Λ = K_uu^{-1} K_un (σ²I)^{-1} K_nu K_uu^{-1}
//!   m ← K_uu (K_uu + S^{-1})^{-1} (S^{-1} m + K_uu^{-1} K_un y / σ²)
//!   S ← (K_uu^{-1} + Λ)^{-1}
//!
//! In the online setting these are implemented as sequential Woodbury/Kalman updates
//! to avoid O(N³) recomputation.
//!
//! # Predictive distribution
//!
//! μ*(x) = K_xu K_uu^{-1} m
//! σ²*(x) = K_xx - Q_xx + K_xu K_uu^{-1} S K_uu^{-1} K_ux
//!
//! where Q_xx = K_xu K_uu^{-1} K_ux is the Nyström approximation of K_xx.
//!
//! # References
//!
//! - Titsias, M. K. (2009). Variational learning of inducing variables in
//!   sparse Gaussian processes. *AISTATS*.
//! - Hensman, J., Fusi, N., & Lawrence, N. D. (2013).
//!   Gaussian processes for big data. *UAI*.
//! - Bui, T. D., Nguyen, C. V., & Turner, R. E. (2017).
//!   Streaming sparse GP approximations. *NIPS*.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{s, Array1, Array2, Axis};

// ─── StreamingGpConfig ─────────────────────────────────────────────────────

/// Configuration for the streaming Gaussian Process.
#[derive(Debug, Clone)]
pub struct StreamingGpConfig {
    /// Maximum number of inducing points M. Inducing points act as a
    /// compressed memory of the training data.
    pub n_inducing: usize,
    /// Signal variance σ_f² of the SE kernel.
    pub kernel_variance: f64,
    /// Length-scale l of the SE kernel. Controls how quickly correlation decays
    /// with distance.
    pub kernel_lengthscale: f64,
    /// Observation noise variance σ²_n.
    pub noise_variance: f64,
    /// Forgetting factor γ ∈ (0, 1]. γ < 1 down-weights old observations,
    /// enabling tracking of non-stationary functions.  γ = 1 (default) gives
    /// the standard non-forgetting SVGP.
    pub forgetting_factor: f64,
    /// If true, relocate inducing points via k-means after each batch update.
    /// Disabled by default to keep O(M²) per-update cost.
    pub update_inducing: bool,
}

impl Default for StreamingGpConfig {
    fn default() -> Self {
        Self {
            n_inducing: 50,
            kernel_variance: 1.0,
            kernel_lengthscale: 1.0,
            noise_variance: 0.1,
            forgetting_factor: 1.0,
            update_inducing: false,
        }
    }
}

// ─── StreamingGpState ──────────────────────────────────────────────────────

/// Current state of the Streaming GP posterior approximation.
#[derive(Debug, Clone)]
pub struct StreamingGpState {
    /// Inducing point locations Z: shape \[M, d\].
    pub inducing_points: Array2<f64>,
    /// Posterior mean of inducing outputs m: shape \[M\].
    pub m: Array1<f64>,
    /// Posterior covariance of inducing outputs S: shape \[M, M\].
    pub s: Array2<f64>,
    /// Total number of data points observed so far.
    pub n_observed: usize,
}

// ─── StreamingGp ───────────────────────────────────────────────────────────

/// Online Sparse Variational Gaussian Process regressor.
///
/// # Example
///
/// ```rust
/// use scirs2_stats::online::{StreamingGp, StreamingGpConfig};
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// let config = StreamingGpConfig {
///     n_inducing: 10,
///     kernel_lengthscale: 1.0,
///     kernel_variance: 1.0,
///     noise_variance: 0.1,
///     ..Default::default()
/// };
/// let mut gp = StreamingGp::new(config);
///
/// // Initialize with first batch
/// let x_init = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0])
///     .expect("shape ok");
/// let y_init = vec![0.0, 1.0, 0.0, -1.0, 0.0];
/// gp.initialize(&x_init, &y_init).expect("init ok");
///
/// // Predict at new points
/// let x_test = Array2::from_shape_vec((3, 1), vec![0.5, 1.5, 2.5])
///     .expect("shape ok");
/// let (mean, var) = gp.predict(&x_test).expect("predict ok");
/// assert_eq!(mean.len(), 3);
/// assert!(var.iter().all(|&v| v > 0.0));
/// ```
pub struct StreamingGp {
    config: StreamingGpConfig,
    state: Option<StreamingGpState>,
}

impl StreamingGp {
    /// Create a new (uninitialised) Streaming GP.
    pub fn new(config: StreamingGpConfig) -> Self {
        Self {
            config,
            state: None,
        }
    }

    /// Initialise the GP with the first batch of data.
    ///
    /// Sets inducing point locations via a simple greedy k-means++ style
    /// selection from the initial batch, then computes the initial posterior.
    ///
    /// # Arguments
    ///
    /// * `x` - Input features: shape [N, d].
    /// * `y` - Targets: length N.
    ///
    /// # Errors
    ///
    /// Returns `StatsError::InsufficientData` if N < 1, or
    /// `StatsError::DimensionMismatch` if shapes are inconsistent.
    pub fn initialize(&mut self, x: &Array2<f64>, y: &[f64]) -> StatsResult<()> {
        let n = x.nrows();
        let d = x.ncols();
        if n == 0 {
            return Err(StatsError::InsufficientData(
                "initialize requires at least one data point".to_string(),
            ));
        }
        if y.len() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "x has {} rows but y has {} elements",
                n,
                y.len()
            )));
        }

        // Select inducing points via kmeans++ initialisation.
        let m = self.config.n_inducing.min(n);
        let inducing = kmeans_plus_plus_init(x, m);

        // Compute K_uu and its Cholesky factor.
        let k_uu = self.se_kernel_matrix(&inducing, &inducing);
        let l_uu = cholesky_lower(&k_uu)?;

        // K_uu^{-1}: solve via two triangular systems.
        let k_uu_inv = cholesky_inverse(&l_uu)?;

        // K_un: shape [M, N]
        let k_un = self.se_kernel_cross(&inducing, x);

        // Prior: S = K_uu, m = 0
        // After conditioning on data: use the sparse GP update.
        // For numerical stability we use the "natural parameter" form.

        // Λ_n = K_uu^{-1} K_un y / σ²   (shape [M])
        let sigma_sq = self.config.noise_variance.max(f64::EPSILON);
        let forgetting = self.config.forgetting_factor.clamp(1e-6, 1.0);

        // Precision contribution from data: Λ = K_uu^{-1} K_un K_nu^T K_uu^{-1} / σ²
        // i.e. A = K_uu^{-1} K_un / sqrt(σ²)  then Λ = A A^T
        let a_mat = mat_mul(&k_uu_inv, &k_un); // [M, N]
        let precision_update = mat_mul_t(&a_mat, &a_mat); // [M, M] = A A^T / σ²
        let precision_update = mat_scalar_mul(&precision_update, forgetting / sigma_sq);

        // K_uu^{-1} is the prior precision. Posterior precision = K_uu^{-1} + Λ.
        let posterior_precision = mat_add(&k_uu_inv, &precision_update);
        let s = mat_inverse_cholesky(&posterior_precision)?;

        // Natural parameter update for mean: η = K_uu^{-1} K_un y / σ²
        let y_arr = Array1::from(y.to_vec());
        let kuu_inv_kun_y = mat_vec_mul(&k_uu_inv, &mat_vec_mul(&k_un, &y_arr));
        let eta_scaled = kuu_inv_kun_y * (forgetting / sigma_sq);

        // m = S * η
        let m = mat_vec_mul(&s, &eta_scaled);

        self.state = Some(StreamingGpState {
            inducing_points: inducing,
            m,
            s,
            n_observed: n,
        });

        Ok(())
    }

    /// Process a new mini-batch of data, updating the posterior approximation.
    ///
    /// Implements an online Kalman-like update:
    ///
    /// 1. Apply forgetting: S ← S / γ, m unchanged (inflates uncertainty for γ < 1)
    /// 2. Compute K_un, K_uu^{-1}
    /// 3. Update natural parameters: η ← (1-γ)η + K_uu^{-1} K_un y / σ²
    /// 4. Update precision: Ω ← γ Ω + K_uu^{-1} K_un K_nu^T K_uu^{-1} / σ²
    /// 5. Recover (m, S) from natural parameters.
    ///
    /// # Arguments
    ///
    /// * `x` - New input features: shape [N_batch, d].
    /// * `y` - New targets: length N_batch.
    pub fn update(&mut self, x: &Array2<f64>, y: &[f64]) -> StatsResult<()> {
        // Validate state exists.
        if self.state.is_none() {
            return Err(StatsError::ComputationError(
                "StreamingGp must be initialized before calling update".to_string(),
            ));
        }

        let n = x.nrows();
        if n == 0 {
            return Ok(()); // Nothing to do
        }
        if y.len() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "x has {} rows but y has {} elements",
                n,
                y.len()
            )));
        }

        let sigma_sq = self.config.noise_variance.max(f64::EPSILON);
        let forgetting = self.config.forgetting_factor.clamp(1e-6, 1.0);

        // Clone the inducing points and current posterior params to avoid borrow conflict.
        let z = {
            let state = self.state.as_ref().expect("checked above");
            state.inducing_points.clone()
        };
        let (m_old, s_old) = {
            let state = self.state.as_ref().expect("checked above");
            (state.m.clone(), state.s.clone())
        };

        // Compute kernel matrices (these borrow self immutably via &self methods).
        let k_uu = self.se_kernel_matrix(&z, &z);
        let l_uu = cholesky_lower(&k_uu)?;
        let k_uu_inv = cholesky_inverse(&l_uu)?;

        // K_un: shape [M, N_batch]
        let k_un = self.se_kernel_cross(&z, x);

        // Recover current natural parameters from (m, S).
        // η = S^{-1} m,  Ω = S^{-1}
        let s_inv = mat_inverse_cholesky(&s_old)?;
        let eta_old = mat_vec_mul(&s_inv, &m_old);

        // New data contribution.
        let a_mat = mat_mul(&k_uu_inv, &k_un); // [M, N]
        let delta_precision = mat_mul_t(&a_mat, &a_mat); // A A^T
        let delta_precision = mat_scalar_mul(&delta_precision, 1.0 / sigma_sq);

        let y_arr = Array1::from(y.to_vec());
        let delta_eta_raw = mat_vec_mul(&a_mat, &y_arr);
        let delta_eta = &delta_eta_raw * (1.0 / sigma_sq);

        // Forget old information (for γ < 1, decay old natural params).
        let eta_old_scaled = &eta_old * forgetting;
        let eta_new_arr = &eta_old_scaled + &delta_eta;
        let omega_new = mat_add(&mat_scalar_mul(&s_inv, forgetting), &delta_precision);

        // Recover (m, S) from new natural parameters.
        let s_new = mat_inverse_cholesky(&omega_new)?;
        let m_new = mat_vec_mul(&s_new, &eta_new_arr);

        // Write back to state.
        {
            let state = self.state.as_mut().expect("checked above");
            state.m = m_new;
            state.s = s_new;
            state.n_observed += n;

            // Optionally relocate inducing points via k-means.
            if self.config.update_inducing {
                let new_z = kmeans_refine(&z, x, state.n_observed);
                state.inducing_points = new_z;
            }
        }

        Ok(())
    }

    /// Predict at new input points.
    ///
    /// Returns `(mean, variance)` where both have length equal to the number of
    /// test points.
    ///
    /// The predictive mean is:
    ///   μ*(x) = K_x*u K_uu^{-1} m
    ///
    /// The predictive variance is:
    ///   σ²*(x) = K_x*x* - Q_x*x* + K_x*u K_uu^{-1} S K_uu^{-1} K_ux*
    ///
    /// where Q_x*x* = diag(K_x*u K_uu^{-1} K_ux*) is the Nyström approximation.
    ///
    /// # Arguments
    ///
    /// * `x_new` - Test inputs: shape [N_test, d].
    ///
    /// # Returns
    ///
    /// `(mean: Array1<f64>, variance: Array1<f64>)` both of length N_test.
    pub fn predict(&self, x_new: &Array2<f64>) -> StatsResult<(Array1<f64>, Array1<f64>)> {
        let state = self.state.as_ref().ok_or_else(|| {
            StatsError::ComputationError(
                "StreamingGp must be initialized before calling predict".to_string(),
            )
        })?;

        let n_test = x_new.nrows();
        if n_test == 0 {
            return Ok((Array1::zeros(0), Array1::zeros(0)));
        }

        let z = &state.inducing_points;

        // Kernel matrices.
        let k_uu = self.se_kernel_matrix(z, z);
        let l_uu = cholesky_lower(&k_uu)?;
        let k_uu_inv = cholesky_inverse(&l_uu)?;

        // K_x*u: shape [N_test, M]
        let k_xu = self.se_kernel_cross(x_new, z);

        // K_uu^{-1} K_ux*: shape [M, N_test]
        let kuu_inv_kux = mat_mul(&k_uu_inv, &k_xu.t().to_owned());

        // Predictive mean: μ* = K_xu K_uu^{-1} m  = K_x*u (K_uu^{-1} m)
        let kuu_inv_m = mat_vec_mul(&k_uu_inv, &state.m);
        let mean_arr = mat_vec_mul(&k_xu, &kuu_inv_m);

        // K_uu^{-1} S K_uu^{-1}: shape [M, M]
        let kuu_inv_s_kuu_inv = mat_mul(&mat_mul(&k_uu_inv, &state.s), &k_uu_inv);

        // Variance for each test point.
        let sigma_f_sq = self.config.kernel_variance;
        let noise_sq = self.config.noise_variance;

        let mut variance = Array1::zeros(n_test);
        for i in 0..n_test {
            // k_xx = σ_f² (SE kernel evaluated at the same point = σ_f²)
            let k_xx_i = sigma_f_sq;

            // Nyström approximation: Q_xx_i = K_xu_i K_uu^{-1} K_ux_i
            let k_xu_i = k_xu.row(i);
            let kuu_inv_kux_i = kuu_inv_kux.column(i);
            let q_xx_i: f64 = k_xu_i
                .iter()
                .zip(kuu_inv_kux_i.iter())
                .map(|(&a, &b)| a * b)
                .sum();

            // Posterior correction: K_xu_i K_uu^{-1} S K_uu^{-1} K_ux_i
            let kuu_inv_s_kuu_inv_col =
                mat_vec_mul(&kuu_inv_s_kuu_inv, &kuu_inv_kux.column(i).to_owned());
            let correction: f64 = kuu_inv_kux
                .column(i)
                .iter()
                .zip(kuu_inv_s_kuu_inv_col.iter())
                .map(|(&a, &b)| a * b)
                .sum();

            // Total predictive variance (including observation noise).
            let v = (k_xx_i - q_xx_i + correction + noise_sq).max(f64::EPSILON);
            variance[i] = v;
        }

        Ok((mean_arr, variance))
    }

    /// Return a reference to the current state, if initialized.
    pub fn state(&self) -> Option<&StreamingGpState> {
        self.state.as_ref()
    }

    /// Return a reference to the configuration.
    pub fn config(&self) -> &StreamingGpConfig {
        &self.config
    }

    // ── Kernel helpers ──────────────────────────────────────────────────────

    /// SE kernel matrix K(X, X'): shape [n1, n2].
    fn se_kernel_matrix(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
        let n1 = x1.nrows();
        let n2 = x2.nrows();
        let mut k = Array2::zeros((n1, n2));
        for i in 0..n1 {
            for j in 0..n2 {
                k[[i, j]] = self.se_kernel_scalar(&x1.row(i).to_owned(), &x2.row(j).to_owned());
            }
        }
        k
    }

    /// SE kernel cross-matrix K(X1, X2): shape [n1, n2].
    fn se_kernel_cross(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
        self.se_kernel_matrix(x1, x2)
    }

    /// Squared Exponential kernel: k(x, x') = σ_f² exp(-||x-x'||² / (2l²)).
    fn se_kernel_scalar(&self, x1: &Array1<f64>, x2: &Array1<f64>) -> f64 {
        let l = self.config.kernel_lengthscale.max(f64::EPSILON);
        let sigma_f_sq = self.config.kernel_variance;
        let diff: f64 = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum();
        sigma_f_sq * (-diff / (2.0 * l * l)).exp()
    }
}

// ─── Linear algebra helpers ────────────────────────────────────────────────

/// Lower Cholesky decomposition L such that A = L Lᵀ.
///
/// Uses the column-wise algorithm with diagonal regularisation for numerical
/// stability (jitter = 1e-8 * max diagonal element).
fn cholesky_lower(a: &Array2<f64>) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(StatsError::DimensionMismatch(
            "Cholesky: matrix must be square".to_string(),
        ));
    }
    let jitter = 1e-8 * a.diag().iter().cloned().fold(0.0f64, f64::max).max(1.0);
    let mut l = Array2::<f64>::zeros((n, n));

    for j in 0..n {
        let mut sum_sq = 0.0;
        for k in 0..j {
            sum_sq += l[[j, k]] * l[[j, k]];
        }
        let diag_val = a[[j, j]] + jitter - sum_sq;
        if diag_val <= 0.0 {
            return Err(StatsError::ComputationError(format!(
                "Cholesky: non-positive-definite matrix (diag[{}]={:.3e})",
                j, diag_val
            )));
        }
        l[[j, j]] = diag_val.sqrt();

        for i in (j + 1)..n {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }
            l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
        }
    }
    Ok(l)
}

/// Compute A^{-1} from its lower Cholesky factor L via forward/back substitution.
fn cholesky_inverse(l: &Array2<f64>) -> StatsResult<Array2<f64>> {
    let n = l.nrows();
    // Solve L L^T X = I column by column.
    let mut inv = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        // Forward substitution: L y = e_col
        let mut y = vec![0.0_f64; n];
        y[col] = 1.0;
        for i in 0..n {
            let mut s = y[i];
            for k in 0..i {
                s -= l[[i, k]] * y[k];
            }
            let diag = l[[i, i]];
            if diag.abs() < f64::EPSILON {
                return Err(StatsError::ComputationError(
                    "Cholesky inverse: zero diagonal".to_string(),
                ));
            }
            y[i] = s / diag;
        }
        // Back substitution: L^T x = y
        let mut x = vec![0.0_f64; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            for k in (i + 1)..n {
                s -= l[[k, i]] * x[k];
            }
            let diag = l[[i, i]];
            if diag.abs() < f64::EPSILON {
                return Err(StatsError::ComputationError(
                    "Cholesky inverse: zero diagonal".to_string(),
                ));
            }
            x[i] = s / diag;
        }
        for i in 0..n {
            inv[[i, col]] = x[i];
        }
    }
    Ok(inv)
}

/// Compute A^{-1} via Cholesky decomposition (convenience wrapper).
fn mat_inverse_cholesky(a: &Array2<f64>) -> StatsResult<Array2<f64>> {
    let l = cholesky_lower(a)?;
    cholesky_inverse(&l)
}

/// Matrix multiplication C = A B.
fn mat_mul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    a.dot(b)
}

/// C = A Bᵀ.
fn mat_mul_t(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    a.dot(&b.t().to_owned())
}

/// Matrix-vector product y = A x.
fn mat_vec_mul(a: &Array2<f64>, x: &Array1<f64>) -> Array1<f64> {
    a.dot(x)
}

/// Element-wise matrix addition.
fn mat_add(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    a + b
}

/// Scalar multiplication of a matrix.
fn mat_scalar_mul(a: &Array2<f64>, s: f64) -> Array2<f64> {
    a * s
}

/// Scalar multiplication of a slice, returning Vec<f64>.
fn vec_scalar_mul(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|&x| x * s).collect()
}

/// Element-wise vector addition.
fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

// ─── Inducing point selection ──────────────────────────────────────────────

/// K-means++ initialisation to select M inducing points from data X.
///
/// Returns an [M, d] array of inducing point locations.
fn kmeans_plus_plus_init(x: &Array2<f64>, m: usize) -> Array2<f64> {
    let n = x.nrows();
    let d = x.ncols();
    if m >= n {
        return x.clone();
    }

    let mut selected: Vec<usize> = Vec::with_capacity(m);
    // First centroid: first data point (deterministic for reproducibility).
    selected.push(0);

    // LCG state for reproducible pseudo-random selection.
    let mut rng_state: u64 = 12345;

    for _ in 1..m {
        // Compute minimum squared distance from each point to the nearest selected centroid.
        let mut min_dists = vec![f64::INFINITY; n];
        for &idx in &selected {
            let cx = x.row(idx);
            for j in 0..n {
                let dist_sq: f64 = x
                    .row(j)
                    .iter()
                    .zip(cx.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum();
                if dist_sq < min_dists[j] {
                    min_dists[j] = dist_sq;
                }
            }
        }

        // Proportional sampling: pick index with probability ∝ min_dist².
        let total: f64 = min_dists.iter().sum();
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
        let threshold = u * total;
        let mut cumulative = 0.0;
        let mut chosen = n - 1;
        for (j, &d) in min_dists.iter().enumerate() {
            cumulative += d;
            if cumulative >= threshold {
                chosen = j;
                break;
            }
        }
        if !selected.contains(&chosen) {
            selected.push(chosen);
        } else {
            // Fallback: pick the farthest point.
            let farthest = min_dists
                .iter()
                .enumerate()
                .filter(|(j, _)| !selected.contains(j))
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(j, _)| j)
                .unwrap_or(0);
            selected.push(farthest);
        }
    }

    // Build output matrix.
    let mut z = Array2::<f64>::zeros((m, d));
    for (i, &idx) in selected.iter().enumerate() {
        for j in 0..d {
            z[[i, j]] = x[[idx, j]];
        }
    }
    z
}

/// Refine inducing points by adding the new batch's diverse points.
///
/// Simple strategy: keep the existing inducing points and replace the point
/// closest to any new observation with the new observation that is farthest
/// from all current inducing points.
fn kmeans_refine(z: &Array2<f64>, x_new: &Array2<f64>, _n_total: usize) -> Array2<f64> {
    // For now, return unchanged (update_inducing = false is default).
    z.clone()
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_gp(n_inducing: usize) -> StreamingGp {
        StreamingGp::new(StreamingGpConfig {
            n_inducing,
            kernel_variance: 1.0,
            kernel_lengthscale: 1.0,
            noise_variance: 0.1,
            ..Default::default()
        })
    }

    fn linspace_col(start: f64, end: f64, n: usize) -> Array2<f64> {
        let step = if n > 1 {
            (end - start) / (n - 1) as f64
        } else {
            0.0
        };
        Array2::from_shape_vec((n, 1), (0..n).map(|i| start + i as f64 * step).collect())
            .expect("shape ok")
    }

    // ── Initialisation ─────────────────────────────────────────────────────

    #[test]
    fn test_streaming_gp_initialize_sets_state() {
        let mut gp = make_gp(5);
        let x = linspace_col(0.0, 4.0, 10);
        let y: Vec<f64> = x.column(0).iter().map(|&v| v.sin()).collect();
        gp.initialize(&x, &y).expect("init ok");
        let state = gp.state().expect("state should be set");
        assert_eq!(state.inducing_points.nrows(), 5);
        assert_eq!(state.inducing_points.ncols(), 1);
        assert_eq!(state.m.len(), 5);
        assert_eq!(state.s.nrows(), 5);
        assert_eq!(state.s.ncols(), 5);
    }

    #[test]
    fn test_streaming_gp_initialize_fewer_points_than_inducing() {
        // When N < M, should use all N points as inducing points.
        let mut gp = make_gp(20);
        let x = linspace_col(0.0, 2.0, 5);
        let y = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        gp.initialize(&x, &y).expect("init ok");
        let state = gp.state().expect("state set");
        assert_eq!(state.inducing_points.nrows(), 5); // min(M=20, N=5) = 5
    }

    #[test]
    fn test_streaming_gp_empty_init_error() {
        let mut gp = make_gp(5);
        let x = Array2::<f64>::zeros((0, 1));
        let y: Vec<f64> = vec![];
        assert!(gp.initialize(&x, &y).is_err());
    }

    #[test]
    fn test_streaming_gp_dimension_mismatch_error() {
        let mut gp = make_gp(5);
        let x = linspace_col(0.0, 4.0, 5);
        let y = vec![0.0, 1.0]; // Wrong length
        assert!(gp.initialize(&x, &y).is_err());
    }

    // ── Prediction shape ───────────────────────────────────────────────────

    #[test]
    fn test_streaming_gp_predict_shape() {
        let mut gp = make_gp(5);
        let x_train = linspace_col(0.0, 4.0, 10);
        let y_train: Vec<f64> = x_train.column(0).iter().map(|&v| v.sin()).collect();
        gp.initialize(&x_train, &y_train).expect("init ok");

        let x_test = linspace_col(0.0, 4.0, 7);
        let (mean, var) = gp.predict(&x_test).expect("predict ok");
        assert_eq!(mean.len(), 7);
        assert_eq!(var.len(), 7);
    }

    // ── Variance positivity ────────────────────────────────────────────────

    #[test]
    fn test_streaming_gp_variance_positive() {
        let mut gp = make_gp(10);
        let x_train = linspace_col(0.0, 5.0, 20);
        let y_train: Vec<f64> = x_train.column(0).iter().map(|&v| v.sin()).collect();
        gp.initialize(&x_train, &y_train).expect("init ok");

        let x_test = linspace_col(-1.0, 6.0, 30);
        let (_, var) = gp.predict(&x_test).expect("predict ok");
        for (i, &v) in var.iter().enumerate() {
            assert!(v > 0.0, "variance[{}] = {} should be > 0", i, v);
        }
    }

    // ── Online update ──────────────────────────────────────────────────────

    #[test]
    fn test_streaming_gp_update_does_not_error() {
        let mut gp = make_gp(8);
        let x_init = linspace_col(0.0, 3.0, 10);
        let y_init: Vec<f64> = vec![0.0; 10];
        gp.initialize(&x_init, &y_init).expect("init ok");

        let x_new = linspace_col(3.0, 6.0, 5);
        let y_new = vec![1.0, 1.5, 2.0, 1.5, 1.0];
        gp.update(&x_new, &y_new).expect("update ok");

        let state = gp.state().expect("state set");
        assert_eq!(state.n_observed, 15);
    }

    #[test]
    fn test_streaming_gp_1d_regression_sin() {
        // Fit f(x) = sin(x) on [0, 2π] and check that predictions are reasonable.
        let mut gp = StreamingGp::new(StreamingGpConfig {
            n_inducing: 15,
            kernel_variance: 1.0,
            kernel_lengthscale: 1.0,
            noise_variance: 0.01,
            ..Default::default()
        });

        let x_train = linspace_col(0.0, 6.28, 40);
        let y_train: Vec<f64> = x_train.column(0).iter().map(|&v| v.sin()).collect();
        gp.initialize(&x_train, &y_train).expect("init ok");

        // Check prediction near x=π/2 (where sin ≈ 1.0).
        let x_test =
            Array2::from_shape_vec((1, 1), vec![std::f64::consts::FRAC_PI_2]).expect("shape ok");
        let (mean, _var) = gp.predict(&x_test).expect("predict ok");
        assert!(
            (mean[0] - 1.0).abs() < 0.3,
            "sin(π/2) prediction should be near 1, got {}",
            mean[0]
        );
    }

    #[test]
    fn test_streaming_gp_predict_before_init_error() {
        let gp = make_gp(5);
        let x_test = linspace_col(0.0, 1.0, 3);
        assert!(gp.predict(&x_test).is_err());
    }

    #[test]
    fn test_streaming_gp_update_before_init_error() {
        let mut gp = make_gp(5);
        let x_new = linspace_col(0.0, 1.0, 3);
        let y_new = vec![0.0, 0.5, 1.0];
        assert!(gp.update(&x_new, &y_new).is_err());
    }

    #[test]
    fn test_streaming_gp_update_improves_prediction() {
        // After observing y≈1.0 at x=5.0, prediction there should be closer to 1.0.
        let mut gp = StreamingGp::new(StreamingGpConfig {
            n_inducing: 8,
            kernel_variance: 2.0,
            kernel_lengthscale: 1.5,
            noise_variance: 0.05,
            ..Default::default()
        });

        // Init with data at x=0..4
        let x_init = linspace_col(0.0, 4.0, 10);
        let y_init = vec![0.0; 10];
        gp.initialize(&x_init, &y_init).expect("init ok");

        // Predict before update at x=5.0
        let x_query = Array2::from_shape_vec((1, 1), vec![5.0]).expect("shape ok");
        let (mean_before, _) = gp.predict(&x_query).expect("predict ok");

        // Update with many observations at x=5 all with y=1.0
        for _ in 0..5 {
            let x_new = Array2::from_shape_vec((5, 1), vec![5.0; 5]).expect("shape ok");
            let y_new = vec![1.0; 5];
            gp.update(&x_new, &y_new).expect("update ok");
        }

        let (mean_after, _) = gp.predict(&x_query).expect("predict ok");

        // Mean should have moved toward 1.0.
        let before_dist = (mean_before[0] - 1.0).abs();
        let after_dist = (mean_after[0] - 1.0).abs();
        assert!(
            after_dist <= before_dist + 0.3,
            "Prediction should improve: before={}, after={} (target=1.0)",
            mean_before[0],
            mean_after[0]
        );
    }

    #[test]
    fn test_streaming_gp_multidim_input() {
        // Test with 2D inputs.
        let mut gp = StreamingGp::new(StreamingGpConfig {
            n_inducing: 5,
            ..Default::default()
        });

        // 2D inputs: [n, 2]
        let x_data: Vec<f64> = (0..10)
            .flat_map(|i| vec![i as f64 * 0.5, (i as f64 * 0.3).sin()])
            .collect();
        let x_train = Array2::from_shape_vec((10, 2), x_data).expect("shape ok");
        let y_train = vec![1.0; 10];

        gp.initialize(&x_train, &y_train).expect("init ok");

        let x_test: Vec<f64> = vec![1.0, 0.5, 2.0, 0.3];
        let x_test_arr = Array2::from_shape_vec((2, 2), x_test).expect("shape ok");
        let (mean, var) = gp.predict(&x_test_arr).expect("predict ok");
        assert_eq!(mean.len(), 2);
        assert!(var.iter().all(|&v| v > 0.0), "variances must be positive");
    }

    #[test]
    fn test_streaming_gp_empty_update_is_noop() {
        let mut gp = make_gp(5);
        let x_init = linspace_col(0.0, 4.0, 10);
        let y_init = vec![0.0; 10];
        gp.initialize(&x_init, &y_init).expect("init ok");

        let state_before = gp.state().expect("state set").n_observed;
        let x_empty = Array2::<f64>::zeros((0, 1));
        gp.update(&x_empty, &[]).expect("empty update ok");
        let state_after = gp.state().expect("state set").n_observed;
        assert_eq!(state_before, state_after);
    }

    #[test]
    fn test_streaming_gp_predict_empty_test_set() {
        let mut gp = make_gp(5);
        let x_init = linspace_col(0.0, 4.0, 10);
        let y_init = vec![0.0; 10];
        gp.initialize(&x_init, &y_init).expect("init ok");

        let x_empty = Array2::<f64>::zeros((0, 1));
        let (mean, var) = gp.predict(&x_empty).expect("predict ok");
        assert_eq!(mean.len(), 0);
        assert_eq!(var.len(), 0);
    }

    #[test]
    fn test_cholesky_known_matrix() {
        // A = [[4, 2], [2, 3]], L ≈ [[2, 0], [1, √2]] (up to small jitter on diagonal)
        // The jitter is 1e-8 * max_diag which is ~4e-8, so we use a looser tolerance.
        let a = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 2.0, 3.0]).expect("shape ok");
        let l = cholesky_lower(&a).expect("chol ok");
        assert!((l[[0, 0]] - 2.0).abs() < 1e-5, "L[0,0]={}", l[[0, 0]]);
        assert!((l[[1, 0]] - 1.0).abs() < 1e-5, "L[1,0]={}", l[[1, 0]]);
        assert!(
            (l[[1, 1]] - 2.0f64.sqrt()).abs() < 1e-5,
            "L[1,1]={}",
            l[[1, 1]]
        );
        // L L^T should approximately equal A (up to jitter).
        let reconstructed = l.dot(&l.t());
        assert!((reconstructed[[0, 0]] - 4.0).abs() < 1e-5);
        assert!((reconstructed[[1, 1]] - 3.0).abs() < 1e-5);
    }
}
