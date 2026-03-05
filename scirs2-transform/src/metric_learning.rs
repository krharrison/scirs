//! Metric Learning Algorithms
//!
//! This module provides metric learning algorithms that learn a Mahalanobis
//! distance metric M from labeled data, such that semantically similar points
//! are closer and dissimilar points are farther under the learned metric.
//!
//! ## Algorithms
//!
//! - **[`LMNN`]**: Large Margin Nearest Neighbor — pushes k-NN to be same-class
//!   targets while pulling impostor same-class neighbors away.
//! - **[`NCA`]**: Neighborhood Components Analysis — maximizes leave-one-out
//!   k-NN accuracy via stochastic softmax neighborhood.
//! - **[`MLKR`]**: Metric Learning for Kernel Regression — minimizes Nadaraya-
//!   Watson regression error with learned Mahalanobis kernel.
//! - **[`ContrastiveMetricLearner`]**: Pair-based contrastive metric learning
//!   using labeled pairs (similar/dissimilar).
//! - **[`TripletMetricLearner`]**: Triplet loss-based metric learning.
//!
//! ## Distance
//!
//! The learned Mahalanobis distance for metric M = LᵀL is:
//! d_M(x, y) = √((x-y)ᵀ M (x-y))
//!
//! ## References
//!
//! - Weinberger, K.Q., & Saul, L.K. (2009). Distance Metric Learning for
//!   Large Margin Nearest Neighbor Classification. JMLR.
//! - Goldberger, J., et al. (2005). Neighbourhood Components Analysis. NeurIPS.
//! - Weinberger, K.Q., & Tesauro, G. (2007). Metric Learning for Kernel
//!   Regression. AISTATS.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_linalg::{eigh, solve};

use crate::error::{Result, TransformError};

// ============================================================================
// Mahalanobis distance utilities
// ============================================================================

/// Compute the squared Mahalanobis distance: (x-y)ᵀ M (x-y).
pub fn mahalanobis_sq(x: &ArrayView1<f64>, y: &ArrayView1<f64>, m: &ArrayView2<f64>) -> f64 {
    let diff: Array1<f64> = x.iter().zip(y.iter()).map(|(a, b)| a - b).collect();
    let md: f64 = m.outer_iter()
        .zip(diff.iter())
        .map(|(row, &di)| {
            let mv_i: f64 = row.iter().zip(diff.iter()).map(|(mij, &dj)| mij * dj).sum();
            di * mv_i
        })
        .sum();
    md.max(0.0)
}

/// Compute the Mahalanobis distance: √((x-y)ᵀ M (x-y)).
pub fn mahalanobis(x: &ArrayView1<f64>, y: &ArrayView1<f64>, m: &ArrayView2<f64>) -> f64 {
    mahalanobis_sq(x, y, m).sqrt()
}

/// Compute the pairwise Mahalanobis distance matrix.
pub fn pairwise_mahalanobis(x: &ArrayView2<f64>, m: &ArrayView2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let mut dist = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let d = mahalanobis(&x.row(i), &x.row(j), m);
            dist[[i, j]] = d;
            dist[[j, i]] = d;
        }
    }
    dist
}

/// Transform data using the low-rank factor L where M = LᵀL.
/// Returns L @ X.T (rows are embedding dims, cols are samples).
pub fn transform_with_factor(x: &ArrayView2<f64>, l: &ArrayView2<f64>) -> Result<Array2<f64>> {
    // l is (embedding_dim, input_dim), X is (n, input_dim)
    // result is (n, embedding_dim)
    let n = x.nrows();
    let d = x.ncols();
    let e = l.ncols();

    if e != d {
        return Err(TransformError::InvalidInput(format!(
            "transform_with_factor: L has {} cols but X has {} features",
            e, d
        )));
    }

    let mut out = Array2::<f64>::zeros((n, l.nrows()));
    for i in 0..n {
        for k in 0..l.nrows() {
            let mut s = 0.0f64;
            for j in 0..d {
                s += l[[k, j]] * x[[i, j]];
            }
            out[[i, k]] = s;
        }
    }
    Ok(out)
}

// ============================================================================
// k-NN utilities
// ============================================================================

/// Find the k nearest neighbors of each point in x using Mahalanobis distance.
///
/// # Returns
/// Indices matrix of shape (n, k) — indices of k nearest neighbors.
pub fn knn_indices_mahalanobis(
    x: &ArrayView2<f64>,
    m: &ArrayView2<f64>,
    k: usize,
) -> Vec<Vec<usize>> {
    let n = x.nrows();
    (0..n).map(|i| {
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let d = mahalanobis_sq(&x.row(i), &x.row(j), m);
                (j, d)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);
        dists.into_iter().map(|(idx, _)| idx).collect()
    }).collect()
}

// ============================================================================
// LMNN — Large Margin Nearest Neighbor
// ============================================================================

/// Training result for metric learning algorithms.
#[derive(Debug, Clone)]
pub struct MetricLearningResult {
    /// Learned metric matrix M (positive semi-definite), shape (d, d)
    pub metric: Array2<f64>,
    /// Learned linear factor L where M = LᵀL, shape (embedding_dim, d)
    pub factor: Array2<f64>,
    /// Loss at each iteration
    pub loss_history: Vec<f64>,
    /// Number of iterations run
    pub n_iters: usize,
}

/// Large Margin Nearest Neighbor (LMNN) metric learning.
///
/// LMNN learns a Mahalanobis metric M = LᵀL that satisfies:
/// 1. **Pull**: Each point x_i is surrounded by its k target neighbors (same class).
/// 2. **Push**: Impostors (different-class points within margin) are penalized.
///
/// # Objective
///
/// min_{L} Σ_{i,j∈N(i)} ||L(x_i - x_j)||² + c Σ_{i,j,l} [1 + ||L(x_i-x_j)||² - ||L(x_i-x_l)||²]_+
///
/// where N(i) are the k target neighbors of x_i and x_l are impostors.
///
/// # Example
/// ```
/// use scirs2_transform::metric_learning::LMNN;
/// use scirs2_core::ndarray::{Array2, Array1};
///
/// let x = Array2::<f64>::zeros((20, 4));
/// let y = Array1::<i64>::zeros(20);
/// let mut lmnn = LMNN::new(3, 50, 1e-7).expect("LMNN::new should succeed");
/// let result = lmnn.fit(&x.view(), &y.view()).expect("LMNN fit should succeed");
/// assert_eq!(result.metric.shape(), &[4, 4]);
/// ```
#[derive(Debug, Clone)]
pub struct LMNN {
    /// Number of target neighbors k
    pub k: usize,
    /// Maximum iterations
    pub max_iter: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Regularization weight for push term
    pub margin_weight: f64,
    /// Low-rank factor L (output embedding dimension)
    pub output_dim: Option<usize>,
    /// Convergence tolerance
    pub tol: f64,
    /// Fitted linear factor L
    factor: Option<Array2<f64>>,
}

impl LMNN {
    /// Create a new LMNN instance.
    ///
    /// # Arguments
    /// * `k` - Number of target neighbors
    /// * `max_iter` - Maximum gradient descent iterations
    /// * `learning_rate` - Gradient descent step size
    pub fn new(k: usize, max_iter: usize, learning_rate: f64) -> Result<Self> {
        if k == 0 {
            return Err(TransformError::InvalidInput("LMNN: k must be > 0".to_string()));
        }
        Ok(LMNN {
            k,
            max_iter,
            learning_rate,
            margin_weight: 1.0,
            output_dim: None,
            tol: 1e-6,
            factor: None,
        })
    }

    /// Set the output embedding dimension (default: input dimension).
    pub fn with_output_dim(mut self, dim: usize) -> Self {
        self.output_dim = Some(dim);
        self
    }

    /// Set the push/margin weight (default: 1.0).
    pub fn with_margin_weight(mut self, w: f64) -> Self {
        self.margin_weight = w;
        self
    }

    /// Compute the LMNN gradient w.r.t. M = LᵀL.
    fn compute_gradient(
        x: &ArrayView2<f64>,
        labels: &[i64],
        m: &Array2<f64>,
        k: usize,
        margin_weight: f64,
    ) -> (Array2<f64>, f64) {
        let n = x.nrows();
        let d = x.ncols();

        // Find target neighbors (k nearest same-class)
        let target_neighbors: Vec<Vec<usize>> = (0..n).map(|i| {
            let mut dists: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i && labels[j] == labels[i])
                .map(|j| (j, mahalanobis_sq(&x.row(i), &x.row(j), &m.view())))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            dists.truncate(k);
            dists.into_iter().map(|(idx, _)| idx).collect()
        }).collect();

        let mut grad_m = Array2::<f64>::zeros((d, d));
        let mut loss = 0.0f64;

        // Pull gradient: penalize distance to target neighbors
        for i in 0..n {
            for &j in &target_neighbors[i] {
                let diff: Array1<f64> = (0..d).map(|k| x[[i, k]] - x[[j, k]]).collect();
                // grad += diff @ diff^T
                for a in 0..d {
                    for b in 0..d {
                        grad_m[[a, b]] += diff[a] * diff[b];
                    }
                }
                loss += mahalanobis_sq(&x.row(i), &x.row(j), &m.view());
            }
        }

        // Push gradient: penalize impostors (different-class within margin)
        let c = margin_weight;
        for i in 0..n {
            for &j in &target_neighbors[i] {
                let d_ij = mahalanobis_sq(&x.row(i), &x.row(j), &m.view());

                // Find impostors: different class with d_M(i, l) < d_ij + 1
                for l in 0..n {
                    if labels[l] == labels[i] {
                        continue;
                    }
                    let d_il = mahalanobis_sq(&x.row(i), &x.row(l), &m.view());
                    let margin_val = 1.0 + d_ij - d_il;

                    if margin_val > 0.0 {
                        loss += c * margin_val;

                        let diff_il: Array1<f64> = (0..d).map(|k| x[[i, k]] - x[[l, k]]).collect();
                        // grad += c * (diff_ij @ diff_ij^T - diff_il @ diff_il^T)
                        let diff_ij: Array1<f64> = (0..d).map(|k| x[[i, k]] - x[[j, k]]).collect();
                        for a in 0..d {
                            for b in 0..d {
                                grad_m[[a, b]] += c * (diff_ij[a] * diff_ij[b] - diff_il[a] * diff_il[b]);
                            }
                        }
                    }
                }
            }
        }

        (grad_m, loss)
    }

    /// Fit LMNN on labeled data.
    ///
    /// # Arguments
    /// * `x` - Feature matrix, shape (n, d)
    /// * `labels` - Integer class labels, shape (n,)
    pub fn fit(&mut self, x: &ArrayView2<f64>, labels: &ArrayView1<i64>) -> Result<MetricLearningResult> {
        let n = x.nrows();
        let d = x.ncols();

        if labels.len() != n {
            return Err(TransformError::InvalidInput(format!(
                "LMNN: x has {} rows but labels has {} elements",
                n, labels.len()
            )));
        }
        if n < 2 {
            return Err(TransformError::InvalidInput("LMNN requires at least 2 samples".to_string()));
        }

        let labels_vec: Vec<i64> = labels.iter().copied().collect();
        let out_dim = self.output_dim.unwrap_or(d);

        // Initialize L as identity (or identity padded/truncated)
        let mut l = Array2::<f64>::zeros((out_dim, d));
        for i in 0..out_dim.min(d) {
            l[[i, i]] = 1.0;
        }

        // M = LᵀL
        let mut m = l.t().dot(&l);
        let mut loss_history = Vec::with_capacity(self.max_iter);
        let mut prev_loss = f64::INFINITY;

        for iter in 0..self.max_iter {
            let (grad, loss) = Self::compute_gradient(x, &labels_vec, &m, self.k, self.margin_weight);
            loss_history.push(loss);

            // Gradient descent step on M
            m = m - self.learning_rate * &grad;

            // Project M onto PSD cone: M = V D_+ V^T via eigendecomposition
            // (use scirs2_linalg::eigh)
            match eigh(&m.view(), None) {
                Ok((eigenvalues, eigenvectors)) => {
                    // Clamp negative eigenvalues
                    let diag_plus: Array1<f64> = eigenvalues.mapv(|v| v.max(0.0));
                    let mut m_new = Array2::<f64>::zeros((d, d));
                    for i in 0..d {
                        if diag_plus[i] > 1e-12 {
                            let v = eigenvectors.column(i);
                            for a in 0..d {
                                for b in 0..d {
                                    m_new[[a, b]] += diag_plus[i] * v[a] * v[b];
                                }
                            }
                        }
                    }
                    m = m_new;
                }
                Err(_) => {
                    // Fallback: add small diagonal for PSD
                    for i in 0..d {
                        m[[i, i]] += 1e-8;
                    }
                }
            }

            // Update L from M: Cholesky L or eigendecomposition
            // L = sqrt(D_+) V^T
            match eigh(&m.view(), None) {
                Ok((ev, evec)) => {
                    let mut l_new = Array2::<f64>::zeros((out_dim.min(d), d));
                    let mut pairs: Vec<(f64, usize)> = ev.iter().enumerate()
                        .map(|(i, &e)| (e, i))
                        .collect();
                    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    for (k_idx, &(e_val, e_idx)) in pairs.iter().enumerate().take(out_dim.min(d)) {
                        let sqrt_e = e_val.max(0.0).sqrt();
                        for j in 0..d {
                            l_new[[k_idx, j]] = sqrt_e * evec[[j, e_idx]];
                        }
                    }
                    l = l_new;
                }
                Err(_) => {}
            }

            // Check convergence
            if (prev_loss - loss).abs() / (prev_loss.abs() + 1e-10) < self.tol {
                self.factor = Some(l.clone());
                return Ok(MetricLearningResult {
                    metric: m,
                    factor: l,
                    loss_history,
                    n_iters: iter + 1,
                });
            }
            prev_loss = loss;
        }

        self.factor = Some(l.clone());
        Ok(MetricLearningResult {
            metric: m,
            factor: l,
            loss_history,
            n_iters: self.max_iter,
        })
    }

    /// Transform data using the learned metric factor L.
    pub fn transform(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let l = self.factor.as_ref().ok_or_else(|| {
            TransformError::NotFitted("LMNN must be fitted before transform".to_string())
        })?;
        transform_with_factor(x, &l.view())
    }
}

// ============================================================================
// NCA — Neighborhood Components Analysis
// ============================================================================

/// Neighborhood Components Analysis (NCA) metric learning.
///
/// NCA learns a linear transformation A such that the leave-one-out k-NN
/// accuracy is maximized in the projected space z = Ax.
///
/// The stochastic softmax probability that x_i selects x_j as its neighbor:
/// p_{ij} = exp(-||Ax_i - Ax_j||²) / Σ_{k≠i} exp(-||Ax_i - Ax_k||²)
///
/// Objective: maximize Σ_i Σ_{j: same class} p_{ij}
///
/// # Example
/// ```
/// use scirs2_transform::metric_learning::NCA;
/// use scirs2_core::ndarray::{Array2, Array1};
///
/// let x = Array2::<f64>::zeros((15, 3));
/// let y = Array1::<i64>::zeros(15);
/// let mut nca = NCA::new(2, 30, 1e-4).expect("NCA::new should succeed");
/// let result = nca.fit(&x.view(), &y.view()).expect("NCA fit should succeed");
/// assert_eq!(result.factor.shape()[1], 3);
/// ```
#[derive(Debug, Clone)]
pub struct NCA {
    /// Output embedding dimensionality
    pub output_dim: usize,
    /// Maximum iterations
    pub max_iter: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// L2 regularization on A
    pub regularization: f64,
    /// Convergence tolerance
    pub tol: f64,
    /// Fitted transformation matrix A
    factor: Option<Array2<f64>>,
}

impl NCA {
    /// Create a new NCA instance.
    ///
    /// # Arguments
    /// * `output_dim` - Dimensionality of the embedded space
    /// * `max_iter` - Maximum gradient ascent iterations
    /// * `learning_rate` - Step size
    pub fn new(output_dim: usize, max_iter: usize, learning_rate: f64) -> Result<Self> {
        if output_dim == 0 {
            return Err(TransformError::InvalidInput("NCA: output_dim must be > 0".to_string()));
        }
        Ok(NCA {
            output_dim,
            max_iter,
            learning_rate,
            regularization: 1e-5,
            tol: 1e-6,
            factor: None,
        })
    }

    /// Set L2 regularization strength.
    pub fn with_regularization(mut self, reg: f64) -> Self {
        self.regularization = reg;
        self
    }

    /// Compute NCA objective and gradient w.r.t. A.
    fn nca_objective_gradient(
        x: &ArrayView2<f64>,
        labels: &[i64],
        a: &Array2<f64>,
        reg: f64,
    ) -> (f64, Array2<f64>) {
        let n = x.nrows();
        let d = x.ncols();
        let e = a.nrows();

        // Project: Z = X A^T  (n x e)
        // Actually A is (e, d), so z_i = A x_i
        let mut z = Array2::<f64>::zeros((n, e));
        for i in 0..n {
            for k in 0..e {
                for j in 0..d {
                    z[[i, k]] += a[[k, j]] * x[[i, j]];
                }
            }
        }

        // Compute pairwise distances in projected space
        let mut p = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            let mut sum_exp = 0.0f64;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let dist_sq: f64 = (0..e).map(|k| (z[[i, k]] - z[[j, k]]).powi(2)).sum();
                let exp_val = (-dist_sq).exp();
                p[[i, j]] = exp_val;
                sum_exp += exp_val;
            }
            if sum_exp > 1e-15 {
                for j in 0..n {
                    p[[i, j]] /= sum_exp;
                }
            }
        }

        // p_i = Σ_{j: same class} p_{ij}
        let p_i: Array1<f64> = (0..n).map(|i| {
            (0..n).filter(|&j| j != i && labels[j] == labels[i])
                .map(|j| p[[i, j]])
                .sum::<f64>()
        }).collect();

        // Objective: Σ_i p_i  (maximize = gradient ascent)
        let objective: f64 = p_i.iter().sum::<f64>();

        // Gradient: ∂f/∂A
        let mut grad_a = Array2::<f64>::zeros((e, d));

        for i in 0..n {
            // For each i: grad += 2 [p_i * Σ_k p_{ik}(z_i-z_k)x_{ik}^T
            //                        - Σ_{j: same class} p_{ij}(z_i-z_j)x_{ij}^T] A
            // Simplified: term1 = p_i * Σ_k p_{ik} f_{ik}
            //             term2 = Σ_{j: same class} p_{ij} f_{ij}
            //             f_{ij} = (z_i - z_j)(x_i - x_j)^T (outer product)

            // term1: Σ_k≠i p_{ik} outer((z_i - z_k), (x_i - x_k))
            let mut t1 = Array2::<f64>::zeros((e, d));
            for k in 0..n {
                if k == i {
                    continue;
                }
                let p_ik = p[[i, k]];
                if p_ik < 1e-15 {
                    continue;
                }
                for a in 0..e {
                    for b in 0..d {
                        t1[[a, b]] += p_ik * (z[[i, a]] - z[[k, a]]) * (x[[i, b]] - x[[k, b]]);
                    }
                }
            }

            // term2: Σ_{j: same class} p_{ij} outer((z_i - z_j), (x_i - x_j))
            let mut t2 = Array2::<f64>::zeros((e, d));
            for j in 0..n {
                if j == i || labels[j] != labels[i] {
                    continue;
                }
                let p_ij = p[[i, j]];
                if p_ij < 1e-15 {
                    continue;
                }
                for a in 0..e {
                    for b in 0..d {
                        t2[[a, b]] += p_ij * (z[[i, a]] - z[[j, a]]) * (x[[i, b]] - x[[j, b]]);
                    }
                }
            }

            // grad += 2 * (p_i * t1 - t2)
            let pi = p_i[i];
            for a in 0..e {
                for b in 0..d {
                    grad_a[[a, b]] += 2.0 * (pi * t1[[a, b]] - t2[[a, b]]);
                }
            }
        }

        // L2 regularization: objective -= reg * ||A||_F^2 / 2
        let reg_obj: f64 = reg * a.iter().map(|v| v * v).sum::<f64>() / 2.0;
        let objective_reg = objective - reg_obj;

        // grad_a -= reg * A
        let grad_reg = grad_a - a.mapv(|v| reg * v);

        (objective_reg, grad_reg)
    }

    /// Fit NCA on labeled data.
    ///
    /// # Arguments
    /// * `x` - Feature matrix, shape (n, d)
    /// * `labels` - Integer class labels, shape (n,)
    pub fn fit(&mut self, x: &ArrayView2<f64>, labels: &ArrayView1<i64>) -> Result<MetricLearningResult> {
        let n = x.nrows();
        let d = x.ncols();

        if labels.len() != n {
            return Err(TransformError::InvalidInput(format!(
                "NCA: x has {} rows but labels has {} elements",
                n, labels.len()
            )));
        }
        if n < 2 {
            return Err(TransformError::InvalidInput("NCA requires at least 2 samples".to_string()));
        }

        let labels_vec: Vec<i64> = labels.iter().copied().collect();
        let e = self.output_dim.min(d);

        // Initialize A with random small values (LCG)
        let mut a = Array2::<f64>::zeros((e, d));
        let mut state: u64 = 54321;
        for i in 0..e {
            for j in 0..d {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                let v = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                a[[i, j]] = v * 0.01;
            }
            // Initialize diagonal to 1
            if i < d {
                a[[i, i]] = 1.0;
            }
        }

        let mut loss_history = Vec::with_capacity(self.max_iter);
        let mut prev_obj = f64::NEG_INFINITY;

        for iter in 0..self.max_iter {
            let (obj, grad) = Self::nca_objective_gradient(x, &labels_vec, &a, self.regularization);
            loss_history.push(-obj); // store loss (negative objective)

            // Gradient ascent
            a = a + self.learning_rate * &grad;

            if (obj - prev_obj).abs() / (prev_obj.abs() + 1e-10) < self.tol {
                let m = a.t().dot(&a);
                self.factor = Some(a.clone());
                return Ok(MetricLearningResult {
                    metric: m,
                    factor: a,
                    loss_history,
                    n_iters: iter + 1,
                });
            }
            prev_obj = obj;
        }

        let m = a.t().dot(&a);
        self.factor = Some(a.clone());
        Ok(MetricLearningResult {
            metric: m,
            factor: a,
            loss_history,
            n_iters: self.max_iter,
        })
    }

    /// Transform data using the learned linear map A.
    pub fn transform(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let a = self.factor.as_ref().ok_or_else(|| {
            TransformError::NotFitted("NCA must be fitted before transform".to_string())
        })?;
        transform_with_factor(x, &a.view())
    }
}

// ============================================================================
// MLKR — Metric Learning for Kernel Regression
// ============================================================================

/// Metric Learning for Kernel Regression (MLKR).
///
/// MLKR learns a Mahalanobis metric M = AᵀA that minimizes the leave-one-out
/// squared error of Nadaraya-Watson kernel regression with a Gaussian kernel.
///
/// ŷ_i = Σ_{j≠i} k_M(x_i, x_j) y_j / Σ_{j≠i} k_M(x_i, x_j)
/// where k_M(x, y) = exp(-||A(x-y)||²)
///
/// # Example
/// ```
/// use scirs2_transform::metric_learning::MLKR;
/// use scirs2_core::ndarray::{Array2, Array1};
///
/// let x = Array2::<f64>::zeros((15, 3));
/// let y = Array1::<f64>::zeros(15);
/// let mut mlkr = MLKR::new(2, 50, 1e-5).expect("MLKR::new should succeed");
/// let result = mlkr.fit(&x.view(), &y.view()).expect("MLKR fit should succeed");
/// ```
#[derive(Debug, Clone)]
pub struct MLKR {
    /// Output embedding dimensionality
    pub output_dim: usize,
    /// Maximum iterations
    pub max_iter: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Convergence tolerance
    pub tol: f64,
    /// Fitted transformation matrix A
    factor: Option<Array2<f64>>,
}

impl MLKR {
    /// Create a new MLKR instance.
    ///
    /// # Arguments
    /// * `output_dim` - Embedding dimensionality
    /// * `max_iter` - Maximum gradient descent iterations
    /// * `learning_rate` - Step size
    pub fn new(output_dim: usize, max_iter: usize, learning_rate: f64) -> Result<Self> {
        if output_dim == 0 {
            return Err(TransformError::InvalidInput("MLKR: output_dim must be > 0".to_string()));
        }
        Ok(MLKR {
            output_dim,
            max_iter,
            learning_rate,
            tol: 1e-6,
            factor: None,
        })
    }

    /// Compute MLKR objective (LOO regression loss) and gradient w.r.t. A.
    fn mlkr_objective_gradient(
        x: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
        a: &Array2<f64>,
    ) -> (f64, Array2<f64>) {
        let n = x.nrows();
        let d = x.ncols();
        let e = a.nrows();

        // Project: Z = A X^T  → z_i = A x_i  (e-dimensional)
        let mut z = Array2::<f64>::zeros((n, e));
        for i in 0..n {
            for k in 0..e {
                for j in 0..d {
                    z[[i, k]] += a[[k, j]] * x[[i, j]];
                }
            }
        }

        // Pairwise kernel weights k_{ij} = exp(-||z_i - z_j||²), i ≠ j
        let mut k_mat = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dist_sq: f64 = (0..e).map(|k| (z[[i, k]] - z[[j, k]]).powi(2)).sum();
                k_mat[[i, j]] = (-dist_sq).exp();
            }
        }

        // Kernel row sums (exclude self)
        let k_sums: Array1<f64> = (0..n)
            .map(|i| (0..n).filter(|&j| j != i).map(|j| k_mat[[i, j]]).sum::<f64>())
            .collect();

        // LOO predictions ŷ_i
        let y_hat: Array1<f64> = (0..n).map(|i| {
            let denom = k_sums[i];
            if denom < 1e-15 {
                return y[i];
            }
            (0..n).filter(|&j| j != i).map(|j| k_mat[[i, j]] * y[j]).sum::<f64>() / denom
        }).collect();

        // Loss = Σ_i (ŷ_i - y_i)²
        let residuals: Array1<f64> = y_hat.iter().zip(y.iter()).map(|(yh, yi)| yh - yi).collect();
        let loss: f64 = residuals.iter().map(|r| r * r).sum();

        // Gradient w.r.t. A (simplified finite-difference approach here;
        // full analytical gradient follows from the chain rule)
        let mut grad_a = Array2::<f64>::zeros((e, d));

        for i in 0..n {
            let r_i = residuals[i];
            let s_i = k_sums[i];
            if s_i < 1e-15 {
                continue;
            }

            for j in 0..n {
                if i == j {
                    continue;
                }
                let k_ij = k_mat[[i, j]];
                if k_ij < 1e-15 {
                    continue;
                }

                // ∂ŷ_i / ∂k_{ij} = (y_j - ŷ_i) / s_i
                let dy_hat_dk_ij = (y[j] - y_hat[i]) / s_i;

                // ∂k_{ij} / ∂z_i = -2 k_{ij} (z_i - z_j)
                // ∂z_i / ∂A_{km} = x_{im} [e_k]  (only row k, column m is nonzero)
                // gradient: r_i * dy_hat_dk_ij * (-2 k_ij) * (z_i - z_j) x_i^T

                let scale = 2.0 * r_i * dy_hat_dk_ij * k_ij;
                for a_idx in 0..e {
                    for b_idx in 0..d {
                        grad_a[[a_idx, b_idx]] -= scale * (z[[i, a_idx]] - z[[j, a_idx]]) * x[[i, b_idx]];
                    }
                }
            }
        }

        (loss, grad_a)
    }

    /// Fit MLKR on regression data.
    ///
    /// # Arguments
    /// * `x` - Feature matrix, shape (n, d)
    /// * `y` - Regression targets, shape (n,)
    pub fn fit(&mut self, x: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Result<MetricLearningResult> {
        let n = x.nrows();
        let d = x.ncols();

        if y.len() != n {
            return Err(TransformError::InvalidInput(format!(
                "MLKR: x has {} rows but y has {} elements",
                n, y.len()
            )));
        }
        if n < 3 {
            return Err(TransformError::InvalidInput("MLKR requires at least 3 samples".to_string()));
        }

        let e = self.output_dim.min(d);

        // Initialize A as identity (or truncated)
        let mut a = Array2::<f64>::zeros((e, d));
        for i in 0..e.min(d) {
            a[[i, i]] = 1.0;
        }

        let mut loss_history = Vec::with_capacity(self.max_iter);
        let mut prev_loss = f64::INFINITY;

        for iter in 0..self.max_iter {
            let (loss, grad) = Self::mlkr_objective_gradient(x, y, &a);
            loss_history.push(loss);

            // Gradient descent
            a = a - self.learning_rate * &grad;

            if (prev_loss - loss).abs() / (prev_loss.abs() + 1e-10) < self.tol {
                let m = a.t().dot(&a);
                self.factor = Some(a.clone());
                return Ok(MetricLearningResult {
                    metric: m,
                    factor: a,
                    loss_history,
                    n_iters: iter + 1,
                });
            }
            prev_loss = loss;
        }

        let m = a.t().dot(&a);
        self.factor = Some(a.clone());
        Ok(MetricLearningResult {
            metric: m,
            factor: a,
            loss_history,
            n_iters: self.max_iter,
        })
    }

    /// Transform data using the learned linear map A.
    pub fn transform(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let a = self.factor.as_ref().ok_or_else(|| {
            TransformError::NotFitted("MLKR must be fitted before transform".to_string())
        })?;
        transform_with_factor(x, &a.view())
    }
}

// ============================================================================
// SiameseLoss — Contrastive Loss
// ============================================================================

/// Contrastive loss for siamese networks.
///
/// L(y, d) = (1-y) · d² + y · max(0, m - d)²
///
/// where y=0 for similar pairs, y=1 for dissimilar, d is the distance,
/// and m is the margin.
///
/// # Example
/// ```
/// use scirs2_transform::metric_learning::SiameseLoss;
///
/// let loss = SiameseLoss::new(1.0);
/// let l = loss.compute(0.3, 0); // similar pair
/// assert!(l >= 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct SiameseLoss {
    /// Margin parameter m > 0
    pub margin: f64,
}

impl SiameseLoss {
    /// Create a new contrastive loss with given margin.
    pub fn new(margin: f64) -> Self {
        SiameseLoss { margin }
    }

    /// Compute the contrastive loss for a single pair.
    ///
    /// # Arguments
    /// * `distance` - Euclidean (or Mahalanobis) distance between the pair
    /// * `label` - 0 for similar, 1 for dissimilar
    pub fn compute(&self, distance: f64, label: u8) -> f64 {
        match label {
            0 => distance * distance, // similar: pull together
            _ => {
                let margin_dist = (self.margin - distance).max(0.0);
                margin_dist * margin_dist // dissimilar: push apart
            }
        }
    }

    /// Compute the average contrastive loss over a batch of pairs.
    ///
    /// # Arguments
    /// * `distances` - Distance for each pair
    /// * `labels` - Labels (0=similar, 1=dissimilar) for each pair
    pub fn batch_loss(&self, distances: &[f64], labels: &[u8]) -> Result<f64> {
        if distances.len() != labels.len() {
            return Err(TransformError::InvalidInput(
                "SiameseLoss: distances and labels must have the same length".to_string(),
            ));
        }
        if distances.is_empty() {
            return Err(TransformError::InvalidInput(
                "SiameseLoss: batch must not be empty".to_string(),
            ));
        }
        let total: f64 = distances.iter().zip(labels.iter())
            .map(|(&d, &l)| self.compute(d, l))
            .sum();
        Ok(total / distances.len() as f64)
    }
}

// ============================================================================
// ContrastiveMetricLearner
// ============================================================================

/// Pair definition for contrastive metric learning.
#[derive(Debug, Clone)]
pub struct PairConstraint {
    /// Index of first sample
    pub i: usize,
    /// Index of second sample
    pub j: usize,
    /// 0 = must-link (similar), 1 = cannot-link (dissimilar)
    pub label: u8,
}

impl PairConstraint {
    /// Create a must-link (similar) pair constraint.
    pub fn similar(i: usize, j: usize) -> Self {
        PairConstraint { i, j, label: 0 }
    }

    /// Create a cannot-link (dissimilar) pair constraint.
    pub fn dissimilar(i: usize, j: usize) -> Self {
        PairConstraint { i, j, label: 1 }
    }
}

/// Pair-based contrastive metric learner.
///
/// Learns a linear embedding A by minimizing the sum of pairwise contrastive
/// losses over a set of labeled pairs (must-link and cannot-link).
///
/// # Example
/// ```
/// use scirs2_transform::metric_learning::{ContrastiveMetricLearner, PairConstraint};
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::<f64>::zeros((10, 4));
/// let pairs = vec![
///     PairConstraint::similar(0, 1),
///     PairConstraint::dissimilar(0, 2),
/// ];
/// let mut cml = ContrastiveMetricLearner::new(2, 1.0, 50, 1e-4).expect("ContrastiveMetricLearner::new should succeed");
/// let result = cml.fit(&x.view(), &pairs).expect("ContrastiveMetricLearner fit should succeed");
/// assert_eq!(result.factor.shape()[1], 4);
/// ```
#[derive(Debug, Clone)]
pub struct ContrastiveMetricLearner {
    /// Output embedding dimension
    pub output_dim: usize,
    /// Contrastive loss margin
    pub margin: f64,
    /// Maximum iterations
    pub max_iter: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Convergence tolerance
    pub tol: f64,
    /// Fitted transformation matrix A
    factor: Option<Array2<f64>>,
}

impl ContrastiveMetricLearner {
    /// Create a new contrastive metric learner.
    pub fn new(output_dim: usize, margin: f64, max_iter: usize, learning_rate: f64) -> Result<Self> {
        if output_dim == 0 {
            return Err(TransformError::InvalidInput(
                "ContrastiveMetricLearner: output_dim must be > 0".to_string(),
            ));
        }
        if margin <= 0.0 {
            return Err(TransformError::InvalidInput(
                "ContrastiveMetricLearner: margin must be positive".to_string(),
            ));
        }
        Ok(ContrastiveMetricLearner {
            output_dim,
            margin,
            max_iter,
            learning_rate,
            tol: 1e-6,
            factor: None,
        })
    }

    fn contrastive_loss_and_grad(
        x: &ArrayView2<f64>,
        pairs: &[PairConstraint],
        a: &Array2<f64>,
        margin: f64,
    ) -> (f64, Array2<f64>) {
        let d = x.ncols();
        let e = a.nrows();
        let n = x.nrows();

        // Project all samples
        let mut z = Array2::<f64>::zeros((n, e));
        for i in 0..n {
            for k in 0..e {
                for j in 0..d {
                    z[[i, k]] += a[[k, j]] * x[[i, j]];
                }
            }
        }

        let mut total_loss = 0.0f64;
        let mut grad_a = Array2::<f64>::zeros((e, d));

        for pair in pairs {
            let zi = z.row(pair.i);
            let zj = z.row(pair.j);

            // Euclidean distance in projected space
            let dist_sq: f64 = (0..e).map(|k| (zi[k] - zj[k]).powi(2)).sum();
            let dist = dist_sq.sqrt();

            match pair.label {
                0 => {
                    // Similar pair: loss = d²
                    total_loss += dist_sq;
                    // grad_loss/A = 2 (z_i - z_j)(x_i - x_j)^T
                    let scale = 2.0;
                    for a_idx in 0..e {
                        for b_idx in 0..d {
                            grad_a[[a_idx, b_idx]] +=
                                scale * (zi[a_idx] - zj[a_idx]) * (x[[pair.i, b_idx]] - x[[pair.j, b_idx]]);
                        }
                    }
                }
                _ => {
                    // Dissimilar pair: loss = max(0, margin - d)²
                    let slack = margin - dist;
                    if slack > 0.0 {
                        total_loss += slack * slack;
                        // grad = 2 * slack * (-1/d) * (z_i - z_j)(x_i - x_j)^T
                        if dist > 1e-10 {
                            let scale = -2.0 * slack / dist;
                            for a_idx in 0..e {
                                for b_idx in 0..d {
                                    grad_a[[a_idx, b_idx]] +=
                                        scale * (zi[a_idx] - zj[a_idx]) * (x[[pair.i, b_idx]] - x[[pair.j, b_idx]]);
                                }
                            }
                        }
                    }
                }
            }
        }

        let n_pairs = pairs.len().max(1) as f64;
        (total_loss / n_pairs, grad_a / n_pairs)
    }

    /// Fit the contrastive learner on pair constraints.
    ///
    /// # Arguments
    /// * `x` - Feature matrix, shape (n, d)
    /// * `pairs` - List of pair constraints
    pub fn fit(&mut self, x: &ArrayView2<f64>, pairs: &[PairConstraint]) -> Result<MetricLearningResult> {
        let d = x.ncols();
        let n = x.nrows();

        if pairs.is_empty() {
            return Err(TransformError::InvalidInput(
                "ContrastiveMetricLearner: pairs list is empty".to_string(),
            ));
        }

        // Validate pair indices
        for p in pairs {
            if p.i >= n || p.j >= n {
                return Err(TransformError::InvalidInput(format!(
                    "ContrastiveMetricLearner: pair index out of bounds ({}, {}) for n={}",
                    p.i, p.j, n
                )));
            }
        }

        let e = self.output_dim.min(d);
        let mut a = Array2::<f64>::zeros((e, d));
        for i in 0..e.min(d) {
            a[[i, i]] = 1.0;
        }

        let mut loss_history = Vec::with_capacity(self.max_iter);
        let mut prev_loss = f64::INFINITY;

        for iter in 0..self.max_iter {
            let (loss, grad) = Self::contrastive_loss_and_grad(x, pairs, &a, self.margin);
            loss_history.push(loss);

            a = a - self.learning_rate * &grad;

            if (prev_loss - loss).abs() / (prev_loss.abs() + 1e-10) < self.tol {
                let m = a.t().dot(&a);
                self.factor = Some(a.clone());
                return Ok(MetricLearningResult {
                    metric: m,
                    factor: a,
                    loss_history,
                    n_iters: iter + 1,
                });
            }
            prev_loss = loss;
        }

        let m = a.t().dot(&a);
        self.factor = Some(a.clone());
        Ok(MetricLearningResult {
            metric: m,
            factor: a,
            loss_history,
            n_iters: self.max_iter,
        })
    }

    /// Transform data using the learned embedding A.
    pub fn transform(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let a = self.factor.as_ref().ok_or_else(|| {
            TransformError::NotFitted("ContrastiveMetricLearner must be fitted before transform".to_string())
        })?;
        transform_with_factor(x, &a.view())
    }
}

// ============================================================================
// TripletMetricLearner
// ============================================================================

/// Triplet constraint for triplet loss metric learning.
#[derive(Debug, Clone)]
pub struct TripletConstraint {
    /// Anchor sample index
    pub anchor: usize,
    /// Positive (same class as anchor) sample index
    pub positive: usize,
    /// Negative (different class from anchor) sample index
    pub negative: usize,
}

impl TripletConstraint {
    /// Create a new triplet (anchor, positive, negative).
    pub fn new(anchor: usize, positive: usize, negative: usize) -> Self {
        TripletConstraint { anchor, positive, negative }
    }
}

/// Triplet loss-based metric learner.
///
/// Learns embedding A to minimize the triplet loss:
/// L = Σ_t max(0, ||A(a-p)||² - ||A(a-n)||² + α)
///
/// where a=anchor, p=positive, n=negative, and α is the margin.
///
/// # Example
/// ```
/// use scirs2_transform::metric_learning::{TripletMetricLearner, TripletConstraint};
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::<f64>::zeros((12, 4));
/// let triplets = vec![TripletConstraint::new(0, 1, 2)];
/// let mut tml = TripletMetricLearner::new(3, 1.0, 50, 1e-4).expect("TripletMetricLearner::new should succeed");
/// let result = tml.fit(&x.view(), &triplets).expect("TripletMetricLearner fit should succeed");
/// assert_eq!(result.factor.shape()[1], 4);
/// ```
#[derive(Debug, Clone)]
pub struct TripletMetricLearner {
    /// Output embedding dimension
    pub output_dim: usize,
    /// Triplet margin α
    pub margin: f64,
    /// Maximum iterations
    pub max_iter: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Convergence tolerance
    pub tol: f64,
    /// Fitted transformation matrix A
    factor: Option<Array2<f64>>,
}

impl TripletMetricLearner {
    /// Create a new triplet metric learner.
    ///
    /// # Arguments
    /// * `output_dim` - Embedding dimensionality
    /// * `margin` - Triplet margin α
    /// * `max_iter` - Maximum gradient descent iterations
    /// * `learning_rate` - Step size
    pub fn new(output_dim: usize, margin: f64, max_iter: usize, learning_rate: f64) -> Result<Self> {
        if output_dim == 0 {
            return Err(TransformError::InvalidInput(
                "TripletMetricLearner: output_dim must be > 0".to_string(),
            ));
        }
        if margin <= 0.0 {
            return Err(TransformError::InvalidInput(
                "TripletMetricLearner: margin must be positive".to_string(),
            ));
        }
        Ok(TripletMetricLearner {
            output_dim,
            margin,
            max_iter,
            learning_rate,
            tol: 1e-6,
            factor: None,
        })
    }

    fn triplet_loss_and_grad(
        x: &ArrayView2<f64>,
        triplets: &[TripletConstraint],
        a: &Array2<f64>,
        margin: f64,
    ) -> (f64, Array2<f64>) {
        let d = x.ncols();
        let e = a.nrows();
        let n = x.nrows();

        // Project all samples
        let mut z = Array2::<f64>::zeros((n, e));
        for i in 0..n {
            for k in 0..e {
                for j in 0..d {
                    z[[i, k]] += a[[k, j]] * x[[i, j]];
                }
            }
        }

        let mut total_loss = 0.0f64;
        let mut grad_a = Array2::<f64>::zeros((e, d));

        for t in triplets {
            let za = z.row(t.anchor);
            let zp = z.row(t.positive);
            let zn = z.row(t.negative);

            let d_ap_sq: f64 = (0..e).map(|k| (za[k] - zp[k]).powi(2)).sum();
            let d_an_sq: f64 = (0..e).map(|k| (za[k] - zn[k]).powi(2)).sum();

            let loss_t = (d_ap_sq - d_an_sq + margin).max(0.0);
            if loss_t <= 0.0 {
                continue;
            }

            total_loss += loss_t;

            // Gradient w.r.t. A
            // ∂L/∂A = 2 * [(z_a - z_p)(x_a - x_p)^T - (z_a - z_n)(x_a - x_n)^T]
            for a_idx in 0..e {
                for b_idx in 0..d {
                    let grad_ap = 2.0 * (za[a_idx] - zp[a_idx]) * (x[[t.anchor, b_idx]] - x[[t.positive, b_idx]]);
                    let grad_an = 2.0 * (za[a_idx] - zn[a_idx]) * (x[[t.anchor, b_idx]] - x[[t.negative, b_idx]]);
                    grad_a[[a_idx, b_idx]] += grad_ap - grad_an;
                }
            }
        }

        let n_triplets = triplets.len().max(1) as f64;
        (total_loss / n_triplets, grad_a / n_triplets)
    }

    /// Fit the triplet metric learner.
    ///
    /// # Arguments
    /// * `x` - Feature matrix, shape (n, d)
    /// * `triplets` - List of triplet constraints
    pub fn fit(&mut self, x: &ArrayView2<f64>, triplets: &[TripletConstraint]) -> Result<MetricLearningResult> {
        let d = x.ncols();
        let n = x.nrows();

        if triplets.is_empty() {
            return Err(TransformError::InvalidInput(
                "TripletMetricLearner: triplets list is empty".to_string(),
            ));
        }

        // Validate triplet indices
        for t in triplets {
            if t.anchor >= n || t.positive >= n || t.negative >= n {
                return Err(TransformError::InvalidInput(format!(
                    "TripletMetricLearner: triplet index out of bounds for n={}",
                    n
                )));
            }
        }

        let e = self.output_dim.min(d);
        let mut a = Array2::<f64>::zeros((e, d));
        for i in 0..e.min(d) {
            a[[i, i]] = 1.0;
        }

        let mut loss_history = Vec::with_capacity(self.max_iter);
        let mut prev_loss = f64::INFINITY;

        for iter in 0..self.max_iter {
            let (loss, grad) = Self::triplet_loss_and_grad(x, triplets, &a, self.margin);
            loss_history.push(loss);

            a = a - self.learning_rate * &grad;

            if (prev_loss - loss).abs() / (prev_loss.abs() + 1e-10) < self.tol {
                let m = a.t().dot(&a);
                self.factor = Some(a.clone());
                return Ok(MetricLearningResult {
                    metric: m,
                    factor: a,
                    loss_history,
                    n_iters: iter + 1,
                });
            }
            prev_loss = loss;
        }

        let m = a.t().dot(&a);
        self.factor = Some(a.clone());
        Ok(MetricLearningResult {
            metric: m,
            factor: a,
            loss_history,
            n_iters: self.max_iter,
        })
    }

    /// Transform data using the learned embedding A.
    pub fn transform(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let a = self.factor.as_ref().ok_or_else(|| {
            TransformError::NotFitted("TripletMetricLearner must be fitted before transform".to_string())
        })?;
        transform_with_factor(x, &a.view())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1, Array2};

    #[test]
    fn test_mahalanobis_identity() {
        let m = Array2::<f64>::eye(3);
        let x = array![1.0, 0.0, 0.0];
        let y = array![0.0, 0.0, 0.0];
        let d = mahalanobis(&x.view(), &y.view(), &m.view());
        assert!((d - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_siamese_loss_similar() {
        let loss = SiameseLoss::new(1.0);
        let l = loss.compute(0.5, 0);
        assert!((l - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_siamese_loss_dissimilar_within_margin() {
        let loss = SiameseLoss::new(2.0);
        let l = loss.compute(0.5, 1); // margin - dist = 1.5
        assert!((l - 2.25).abs() < 1e-10);
    }

    #[test]
    fn test_siamese_loss_dissimilar_outside_margin() {
        let loss = SiameseLoss::new(1.0);
        let l = loss.compute(2.0, 1); // beyond margin
        assert_eq!(l, 0.0);
    }

    #[test]
    fn test_siamese_batch_loss() {
        let loss = SiameseLoss::new(1.0);
        let dists = vec![0.3, 1.5];
        let labels = vec![0u8, 1u8];
        let bl = loss.batch_loss(&dists, &labels).expect("batch_loss should succeed");
        assert!(bl >= 0.0);
    }

    #[test]
    fn test_lmnn_fit() {
        // All same class — no impostor loss
        let x = Array2::<f64>::eye(4);
        let y = Array1::<i64>::from_vec(vec![0, 0, 0, 0]);
        let mut lmnn = LMNN::new(1, 5, 1e-6).expect("LMNN::new should succeed");
        let res = lmnn.fit(&x.view(), &y.view()).expect("LMNN fit should succeed");
        assert_eq!(res.metric.shape(), &[4, 4]);
    }

    #[test]
    fn test_nca_fit() {
        let x = Array2::<f64>::zeros((8, 3));
        let y = Array1::<i64>::from_vec(vec![0, 0, 1, 1, 0, 0, 1, 1]);
        let mut nca = NCA::new(2, 5, 1e-5).expect("NCA::new should succeed");
        let res = nca.fit(&x.view(), &y.view()).expect("NCA fit should succeed");
        assert_eq!(res.factor.shape(), &[2, 3]);
    }

    #[test]
    fn test_mlkr_fit() {
        let x = Array2::<f64>::zeros((8, 3));
        let y = Array1::<f64>::ones(8);
        let mut mlkr = MLKR::new(2, 5, 1e-6).expect("MLKR::new should succeed");
        let res = mlkr.fit(&x.view(), &y.view()).expect("MLKR fit should succeed");
        assert_eq!(res.factor.shape(), &[2, 3]);
    }

    #[test]
    fn test_contrastive_metric_learner() {
        let x = Array2::<f64>::eye(4);
        let pairs = vec![
            PairConstraint::similar(0, 1),
            PairConstraint::dissimilar(0, 2),
        ];
        let mut cml = ContrastiveMetricLearner::new(2, 1.0, 5, 1e-5).expect("ContrastiveMetricLearner::new should succeed");
        let res = cml.fit(&x.view(), &pairs).expect("ContrastiveMetricLearner fit should succeed");
        assert_eq!(res.factor.shape()[1], 4);
    }

    #[test]
    fn test_triplet_metric_learner() {
        let x = Array2::<f64>::eye(4);
        let triplets = vec![TripletConstraint::new(0, 1, 2)];
        let mut tml = TripletMetricLearner::new(3, 1.0, 5, 1e-5).expect("TripletMetricLearner::new should succeed");
        let res = tml.fit(&x.view(), &triplets).expect("TripletMetricLearner fit should succeed");
        assert_eq!(res.factor.shape()[1], 4);
    }
}
