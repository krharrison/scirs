//! Streaming PCA algorithms: Oja's rule and CCIPCA
//!
//! These algorithms process one sample at a time, making them suitable for
//! truly online settings where data arrives continuously.
//!
//! # Algorithms
//!
//! - **Oja's rule**: Online rank-1 PCA via stochastic gradient update of the
//!   weight vector. Converges to the leading eigenvector of the data covariance.
//! - **CCIPCA** (Candid Covariance-free Incremental PCA): Extends to rank-k
//!   by maintaining multiple component vectors that are orthogonalised on each
//!   update step (Weng, Zhang & Hwang, 2003).
//!
//! # Learning Rate Schedules
//!
//! - **Constant**: fixed step size `eta`
//! - **InverseT**: `eta_0 / (1 + t)` — classic 1/t decay
//! - **Amnesic**: forgetting-factor schedule `(t - l_1) / (t - l_2)` for t > l_1

use scirs2_core::ndarray::{Array1, Array2, Axis};

use crate::error::{Result, TransformError};

// ---------------------------------------------------------------------------
// Learning rate schedule
// ---------------------------------------------------------------------------

/// Learning-rate schedule used by streaming PCA algorithms.
#[derive(Debug, Clone)]
pub enum LearningRateSchedule {
    /// Fixed step size.
    Constant {
        /// The constant learning rate.
        eta: f64,
    },
    /// Classical 1/t decay: `eta_0 / (1 + t)`.
    InverseT {
        /// Initial learning rate.
        eta0: f64,
    },
    /// Amnesic schedule with forgetting factor.
    ///
    /// For step `t`:
    /// - if `t <= l1`: weight = 1.0
    /// - else: weight = `(t - l1) / (t - l2)`
    Amnesic {
        /// Cut-in step (use weight 1.0 before this).
        l1: f64,
        /// Denominator offset — must satisfy `l2 < l1`.
        l2: f64,
    },
}

impl LearningRateSchedule {
    /// Evaluate the learning rate / weight at step `t` (1-based).
    pub fn rate(&self, t: usize) -> f64 {
        let t_f = t as f64;
        match self {
            LearningRateSchedule::Constant { eta } => *eta,
            LearningRateSchedule::InverseT { eta0 } => eta0 / (1.0 + t_f),
            LearningRateSchedule::Amnesic { l1, l2 } => {
                if t_f <= *l1 {
                    1.0
                } else {
                    // (t - l1) / (t - l2)  — since l2 < l1, denominator > numerator
                    (t_f - l1) / (t_f - l2)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Online mean / variance tracker
// ---------------------------------------------------------------------------

/// Welford-style online tracker of mean and variance for a feature vector.
#[derive(Debug, Clone)]
pub struct OnlineStats {
    /// Running mean.
    mean: Array1<f64>,
    /// Welford M2 accumulator (sum of squared deviations).
    m2: Array1<f64>,
    /// Number of samples seen.
    count: usize,
}

impl OnlineStats {
    /// Create a new tracker for `dim`-dimensional data.
    pub fn new(dim: usize) -> Self {
        Self {
            mean: Array1::zeros(dim),
            m2: Array1::zeros(dim),
            count: 0,
        }
    }

    /// Update with a single observation.
    pub fn update(&mut self, x: &Array1<f64>) {
        self.count += 1;
        let n = self.count as f64;
        let delta = x - &self.mean;
        self.mean = &self.mean + &(&delta / n);
        let delta2 = x - &self.mean;
        self.m2 = &self.m2 + &(&delta * &delta2);
    }

    /// Current mean.
    pub fn mean(&self) -> &Array1<f64> {
        &self.mean
    }

    /// Current (sample) variance. Returns zeros when `count <= 1`.
    pub fn variance(&self) -> Array1<f64> {
        if self.count <= 1 {
            Array1::zeros(self.mean.len())
        } else {
            &self.m2 / (self.count - 1) as f64
        }
    }

    /// Number of observations seen.
    pub fn count(&self) -> usize {
        self.count
    }
}

// ---------------------------------------------------------------------------
// Oja's Rule (rank-1 streaming PCA)
// ---------------------------------------------------------------------------

/// Online rank-1 PCA using Oja's stochastic gradient rule.
///
/// At each step, the weight vector `w` is updated as:
///
/// ```text
/// w <- w + eta * (x^T w) * x
/// w <- w / ||w||
/// ```
///
/// This converges to the leading eigenvector of the data covariance matrix.
#[derive(Debug, Clone)]
pub struct OjaPCA {
    /// Current weight vector (the principal component).
    w: Array1<f64>,
    /// Learning-rate schedule.
    schedule: LearningRateSchedule,
    /// Online statistics tracker (for mean-centering).
    stats: OnlineStats,
    /// Number of samples processed.
    n_samples: usize,
    /// Whether to center the data (subtract running mean).
    center: bool,
}

impl OjaPCA {
    /// Create a new Oja PCA.
    ///
    /// `dim` is the dimensionality of the input space. The initial weight vector
    /// is set to a simple non-zero direction (unit vector along first axis).
    pub fn new(dim: usize, schedule: LearningRateSchedule, center: bool) -> Result<Self> {
        if dim == 0 {
            return Err(TransformError::InvalidInput(
                "Dimension must be positive".to_string(),
            ));
        }
        // Initialise w to a unit vector
        let mut w = Array1::zeros(dim);
        // Spread initial energy across all dimensions to avoid bias
        let val = 1.0 / (dim as f64).sqrt();
        for v in w.iter_mut() {
            *v = val;
        }
        Ok(Self {
            w,
            schedule,
            stats: OnlineStats::new(dim),
            n_samples: 0,
            center,
        })
    }

    /// Process a single sample and update the principal component estimate.
    pub fn update(&mut self, x: &Array1<f64>) -> Result<()> {
        if x.len() != self.w.len() {
            return Err(TransformError::InvalidInput(format!(
                "Expected dimension {}, got {}",
                self.w.len(),
                x.len()
            )));
        }
        self.stats.update(x);
        self.n_samples += 1;

        let centered = if self.center {
            x - self.stats.mean()
        } else {
            x.clone()
        };

        let eta = self.schedule.rate(self.n_samples);
        let projection = centered.dot(&self.w);
        // Oja update: w += eta * projection * x
        self.w = &self.w + &(&centered * (eta * projection));

        // Normalise
        let norm = self.w.dot(&self.w).sqrt();
        if norm > 1e-15 {
            self.w /= norm;
        }

        Ok(())
    }

    /// Return the current principal component (unit vector).
    pub fn component(&self) -> &Array1<f64> {
        &self.w
    }

    /// Project a sample onto the current component (scalar).
    pub fn project(&self, x: &Array1<f64>) -> Result<f64> {
        if x.len() != self.w.len() {
            return Err(TransformError::InvalidInput(format!(
                "Expected dimension {}, got {}",
                self.w.len(),
                x.len()
            )));
        }
        let centered = if self.center {
            x - self.stats.mean()
        } else {
            x.clone()
        };
        Ok(centered.dot(&self.w))
    }

    /// Number of samples processed.
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Return the online statistics tracker.
    pub fn stats(&self) -> &OnlineStats {
        &self.stats
    }

    /// Estimated explained variance (running estimate of eigenvalue).
    ///
    /// Computed as the mean of `(x^T w)^2` over the last seen data approximated
    /// by the running projection magnitude.
    pub fn explained_variance_estimate(&self) -> f64 {
        if self.n_samples == 0 {
            return 0.0;
        }
        // The eigenvalue is approximately the mean squared projection
        // We use the variance along the component direction
        let var = self.stats.variance();
        var.dot(&(&self.w * &self.w))
    }
}

// ---------------------------------------------------------------------------
// CCIPCA (Candid Covariance-free Incremental PCA)
// ---------------------------------------------------------------------------

/// CCIPCA: Candid Covariance-free Incremental PCA (Weng et al., 2003).
///
/// Maintains `k` component vectors that are updated one sample at a time.
/// Unlike Oja's rule this finds the top-k eigenvectors via a sequential
/// deflation strategy with amnesic weighting.
///
/// Each component vector `v_j` is updated as:
///
/// ```text
/// v_j <- (1 - w_t) * v_j + w_t * (x^T v_j / ||v_j||) * x
/// ```
///
/// where `w_t` is the amnesic weight and `x` is deflated by previously
/// extracted components before updating `v_j`.
#[derive(Debug, Clone)]
pub struct CCIPCA {
    /// Number of components to extract.
    n_components: usize,
    /// Component vectors (not necessarily unit-normalised; their norms approximate eigenvalues).
    components: Array2<f64>,
    /// Learning-rate schedule.
    schedule: LearningRateSchedule,
    /// Online statistics tracker.
    stats: OnlineStats,
    /// Number of samples processed.
    n_samples: usize,
    /// Whether to center data.
    center: bool,
    /// Dimensionality.
    dim: usize,
}

impl CCIPCA {
    /// Create a new CCIPCA estimator.
    ///
    /// # Arguments
    ///
    /// * `dim` – dimensionality of the input
    /// * `n_components` – number of principal components to extract
    /// * `schedule` – learning-rate schedule
    /// * `center` – whether to subtract running mean
    pub fn new(
        dim: usize,
        n_components: usize,
        schedule: LearningRateSchedule,
        center: bool,
    ) -> Result<Self> {
        if dim == 0 || n_components == 0 {
            return Err(TransformError::InvalidInput(
                "Dimension and n_components must be positive".to_string(),
            ));
        }
        if n_components > dim {
            return Err(TransformError::InvalidInput(format!(
                "n_components ({}) must be <= dim ({})",
                n_components, dim
            )));
        }

        // Initialise components to small distinct values for symmetry-breaking
        let mut components = Array2::zeros((n_components, dim));
        for j in 0..n_components {
            // Put main energy on dimension j, small on others
            for d in 0..dim {
                components[[j, d]] = if d == j { 1.0 } else { 1e-4 / (dim as f64) };
            }
        }

        Ok(Self {
            n_components,
            components,
            schedule,
            stats: OnlineStats::new(dim),
            n_samples: 0,
            center,
            dim,
        })
    }

    /// Process a single sample and update all component estimates.
    pub fn update(&mut self, x: &Array1<f64>) -> Result<()> {
        if x.len() != self.dim {
            return Err(TransformError::InvalidInput(format!(
                "Expected dimension {}, got {}",
                self.dim,
                x.len()
            )));
        }
        self.stats.update(x);
        self.n_samples += 1;

        let mut residual = if self.center {
            x - self.stats.mean()
        } else {
            x.clone()
        };

        let wt = self.schedule.rate(self.n_samples);

        for j in 0..self.n_components {
            let vj = self.components.row(j).to_owned();
            let vj_norm = vj.dot(&vj).sqrt();

            if vj_norm < 1e-15 {
                // Component too small; reinitialise from residual
                let r_norm = residual.dot(&residual).sqrt();
                if r_norm > 1e-15 {
                    let normed = &residual / r_norm;
                    for d in 0..self.dim {
                        self.components[[j, d]] = normed[d] * 1e-3;
                    }
                }
            } else {
                // CCIPCA update
                let vj_unit = &vj / vj_norm;
                let proj = residual.dot(&vj_unit);

                // v_j <- (1 - wt) * v_j + wt * proj * x
                let new_vj = &vj * (1.0 - wt) + &(&residual * (wt * proj));
                for d in 0..self.dim {
                    self.components[[j, d]] = new_vj[d];
                }

                // Deflate: remove projection of residual onto updated component
                let updated_vj = self.components.row(j).to_owned();
                let updated_norm = updated_vj.dot(&updated_vj).sqrt();
                if updated_norm > 1e-15 {
                    let updated_unit = &updated_vj / updated_norm;
                    let proj_residual = residual.dot(&updated_unit);
                    residual = &residual - &(&updated_unit * proj_residual);
                }
            }
        }

        Ok(())
    }

    /// Return the current principal components as unit vectors (n_components x dim).
    pub fn components(&self) -> Array2<f64> {
        let mut result = self.components.clone();
        for j in 0..self.n_components {
            let mut row = result.row_mut(j);
            let norm = row.dot(&row).sqrt();
            if norm > 1e-15 {
                row /= norm;
            }
        }
        result
    }

    /// Return approximate eigenvalues (norms of the component vectors).
    pub fn eigenvalues(&self) -> Array1<f64> {
        let mut vals = Array1::zeros(self.n_components);
        for j in 0..self.n_components {
            let row = self.components.row(j);
            vals[j] = row.dot(&row).sqrt();
        }
        vals
    }

    /// Estimated explained variance ratio for each component.
    ///
    /// Ratio of each eigenvalue to the sum of all eigenvalues.
    pub fn explained_variance_ratio(&self) -> Array1<f64> {
        let evals = self.eigenvalues();
        let total = evals.sum();
        if total < 1e-15 {
            Array1::zeros(self.n_components)
        } else {
            &evals / total
        }
    }

    /// Project a sample onto the current components.
    pub fn project(&self, x: &Array1<f64>) -> Result<Array1<f64>> {
        if x.len() != self.dim {
            return Err(TransformError::InvalidInput(format!(
                "Expected dimension {}, got {}",
                self.dim,
                x.len()
            )));
        }
        let centered = if self.center {
            x - self.stats.mean()
        } else {
            x.clone()
        };
        let comps = self.components();
        Ok(comps.dot(&centered))
    }

    /// Project a batch of samples (n_samples x dim) -> (n_samples x n_components).
    pub fn project_batch(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        if data.ncols() != self.dim {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} columns, got {}",
                self.dim,
                data.ncols()
            )));
        }
        let comps = self.components(); // (n_components x dim)
        let n = data.nrows();
        let mut result = Array2::zeros((n, self.n_components));
        for i in 0..n {
            let row = data.row(i);
            let centered = if self.center {
                row.to_owned() - self.stats.mean()
            } else {
                row.to_owned()
            };
            let proj = comps.dot(&centered);
            for j in 0..self.n_components {
                result[[i, j]] = proj[j];
            }
        }
        Ok(result)
    }

    /// Number of components.
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Number of samples processed.
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Return the online statistics tracker.
    pub fn stats(&self) -> &OnlineStats {
        &self.stats
    }

    /// Fit on a full batch of data (processes each row sequentially).
    pub fn fit_batch(&mut self, data: &Array2<f64>) -> Result<()> {
        if data.ncols() != self.dim {
            return Err(TransformError::InvalidInput(format!(
                "Expected {} columns, got {}",
                self.dim,
                data.ncols()
            )));
        }
        for i in 0..data.nrows() {
            let row = data.row(i).to_owned();
            self.update(&row)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Helper: generate a simple 2D dataset where the first component has much
    /// more variance than the second.
    fn make_simple_data(n: usize) -> Array2<f64> {
        let mut data = Array2::zeros((n, 2));
        for i in 0..n {
            let t = (i as f64) * 0.1;
            // First axis has large variance, second has small
            data[[i, 0]] = t.sin() * 10.0 + (i as f64) * 0.01;
            data[[i, 1]] = t.cos() * 0.5;
        }
        data
    }

    /// Helper: generate a higher-dimensional dataset with known structure.
    fn make_structured_data(n: usize, dim: usize) -> Array2<f64> {
        let mut data = Array2::zeros((n, dim));
        for i in 0..n {
            let t = (i as f64) * 0.05;
            // First direction has the most variance
            data[[i, 0]] = t.sin() * 20.0 + t * 0.5;
            // Second direction moderate variance
            if dim > 1 {
                data[[i, 1]] = t.cos() * 5.0;
            }
            // Rest: small noise-like
            for d in 2..dim {
                data[[i, d]] = ((i * (d + 1)) as f64 * 0.01).sin() * 0.1;
            }
        }
        data
    }

    #[test]
    fn test_oja_converges_to_first_component() {
        let data = make_simple_data(500);
        let mut oja = OjaPCA::new(2, LearningRateSchedule::InverseT { eta0: 0.5 }, true)
            .expect("should create");

        for i in 0..data.nrows() {
            let row = data.row(i).to_owned();
            oja.update(&row).expect("should update");
        }

        // The first component should align roughly with [1, 0] since that axis
        // has the most variance.
        let comp = oja.component();
        assert!(
            comp[0].abs() > comp[1].abs(),
            "First component should have larger weight on first axis: {:?}",
            comp
        );
        assert!(oja.n_samples() == 500);
        assert!(oja.explained_variance_estimate() > 0.0);
    }

    #[test]
    fn test_oja_projection() {
        let mut oja = OjaPCA::new(3, LearningRateSchedule::Constant { eta: 0.01 }, false)
            .expect("should create");

        let sample = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        oja.update(&sample).expect("update");

        let proj = oja.project(&sample).expect("project");
        // Projection should be a scalar, finite
        assert!(proj.is_finite());
    }

    #[test]
    fn test_oja_dimension_mismatch() {
        let mut oja = OjaPCA::new(3, LearningRateSchedule::Constant { eta: 0.01 }, true)
            .expect("should create");
        let bad = Array1::from_vec(vec![1.0, 2.0]);
        assert!(oja.update(&bad).is_err());
        assert!(oja.project(&bad).is_err());
    }

    #[test]
    fn test_ccipca_basic() {
        let data = make_structured_data(300, 5);
        let mut ccipca = CCIPCA::new(
            5,
            2,
            LearningRateSchedule::Amnesic { l1: 2.0, l2: 1.0 },
            true,
        )
        .expect("should create");

        ccipca.fit_batch(&data).expect("fit_batch");

        assert_eq!(ccipca.n_samples(), 300);
        assert_eq!(ccipca.n_components(), 2);

        let comps = ccipca.components();
        assert_eq!(comps.shape(), &[2, 5]);

        // Check that components are roughly unit-normalised
        for j in 0..2 {
            let row = comps.row(j);
            let norm = row.dot(&row).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-10,
                "Component {} norm = {}",
                j,
                norm
            );
        }
    }

    #[test]
    fn test_ccipca_explained_variance() {
        let data = make_structured_data(500, 4);
        let mut ccipca = CCIPCA::new(4, 3, LearningRateSchedule::InverseT { eta0: 1.0 }, true)
            .expect("should create");

        ccipca.fit_batch(&data).expect("fit_batch");

        let ratios = ccipca.explained_variance_ratio();
        assert_eq!(ratios.len(), 3);

        // Ratios should sum to approximately 1
        let total: f64 = ratios.sum();
        assert!((total - 1.0).abs() < 1e-10, "Ratios sum = {}", total);

        // First component should capture the most variance
        assert!(
            ratios[0] >= ratios[1],
            "First component should have highest ratio: {:?}",
            ratios
        );
    }

    #[test]
    fn test_ccipca_projection() {
        let data = make_structured_data(200, 3);
        let mut ccipca = CCIPCA::new(3, 2, LearningRateSchedule::Constant { eta: 0.05 }, true)
            .expect("should create");

        ccipca.fit_batch(&data).expect("fit_batch");

        // Project single sample
        let sample = data.row(0).to_owned();
        let proj = ccipca.project(&sample).expect("project");
        assert_eq!(proj.len(), 2);

        // Project batch
        let batch_proj = ccipca.project_batch(&data).expect("batch project");
        assert_eq!(batch_proj.shape(), &[200, 2]);
    }

    #[test]
    fn test_ccipca_dimension_mismatch() {
        let result = CCIPCA::new(3, 5, LearningRateSchedule::Constant { eta: 0.1 }, true);
        assert!(result.is_err()); // n_components > dim

        let mut ccipca = CCIPCA::new(3, 2, LearningRateSchedule::Constant { eta: 0.1 }, true)
            .expect("should create");
        let bad = Array1::from_vec(vec![1.0, 2.0]);
        assert!(ccipca.update(&bad).is_err());
    }

    #[test]
    fn test_learning_rate_schedules() {
        let constant = LearningRateSchedule::Constant { eta: 0.1 };
        assert!((constant.rate(1) - 0.1).abs() < 1e-15);
        assert!((constant.rate(100) - 0.1).abs() < 1e-15);

        let inverse = LearningRateSchedule::InverseT { eta0: 1.0 };
        assert!((inverse.rate(1) - 0.5).abs() < 1e-15); // 1/(1+1) = 0.5
        assert!(inverse.rate(100) < inverse.rate(10));

        let amnesic = LearningRateSchedule::Amnesic { l1: 10.0, l2: 5.0 };
        assert!((amnesic.rate(5) - 1.0).abs() < 1e-15); // t <= l1
                                                        // t=20: (20-10)/(20-5) = 10/15 = 2/3
        assert!((amnesic.rate(20) - 10.0 / 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_online_stats() {
        let mut stats = OnlineStats::new(2);
        stats.update(&Array1::from_vec(vec![2.0, 4.0]));
        stats.update(&Array1::from_vec(vec![4.0, 8.0]));
        stats.update(&Array1::from_vec(vec![6.0, 12.0]));

        let mean = stats.mean();
        assert!((mean[0] - 4.0).abs() < 1e-10);
        assert!((mean[1] - 8.0).abs() < 1e-10);

        let var = stats.variance();
        // Variance of [2,4,6] = 4.0, [4,8,12] = 16.0
        assert!((var[0] - 4.0).abs() < 1e-10);
        assert!((var[1] - 16.0).abs() < 1e-10);

        assert_eq!(stats.count(), 3);
    }
}
