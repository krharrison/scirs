//! Multi-output GP via Linear Model of Coregionalization (LMC).

use super::kernels::AdvancedKernel;
use super::linalg::{cholesky_jitter, solve_lower, solve_lower_matrix, solve_upper};
use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2};

/// Linear Model of Coregionalization multi-output GP.
///
/// f_d(x) = Σ_q a_{d,q} u_q(x) + ε_d(x)
///
/// # References
/// - Álvarez & Lawrence (2011) "Computationally Efficient Convolved Multiple Output
///   Gaussian Processes"
#[derive(Debug, Clone)]
pub struct MultiOutputGP<K: AdvancedKernel> {
    /// Shared kernel.
    pub kernel: K,
    /// Number of outputs.
    pub n_outputs: usize,
    /// Rank of the LMC coregionalization.
    pub n_latent: usize,
    /// Coregionalization matrix A: (n_outputs × n_latent).
    pub coregion_a: Array2<f64>,
    /// Per-output noise variances.
    pub noise: Vec<f64>,
    x_train: Option<Array2<f64>>,
    y_train: Option<Array2<f64>>,
    l_matrices: Vec<Array2<f64>>,
    alphas: Vec<Array1<f64>>,
}

impl<K: AdvancedKernel> MultiOutputGP<K> {
    /// Create a new LMC multi-output GP.
    pub fn new(kernel: K, n_outputs: usize, n_latent: usize) -> Self {
        let mut coregion_a = Array2::<f64>::zeros((n_outputs, n_latent));
        for d in 0..n_outputs.min(n_latent) {
            coregion_a[[d, d]] = 1.0;
        }
        Self {
            kernel,
            n_outputs,
            n_latent,
            coregion_a,
            noise: vec![0.01; n_outputs],
            x_train: None,
            y_train: None,
            l_matrices: Vec::new(),
            alphas: Vec::new(),
        }
    }

    fn coregion_scale(&self, d1: usize, d2: usize) -> f64 {
        (0..self.n_latent)
            .map(|q| self.coregion_a[[d1, q]] * self.coregion_a[[d2, q]])
            .sum()
    }

    /// Fit on multi-output observations.
    ///
    /// `x_train`: (N × D), `y_train`: (N × n_outputs).
    pub fn fit(
        &mut self,
        x_train: &Array2<f64>,
        y_train: &Array2<f64>,
    ) -> StatsResult<()> {
        let n = x_train.nrows();
        if y_train.nrows() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "x_train rows {n} ≠ y_train rows {}",
                y_train.nrows()
            )));
        }
        if y_train.ncols() != self.n_outputs {
            return Err(StatsError::DimensionMismatch(format!(
                "y_train cols {} ≠ n_outputs {}",
                y_train.ncols(),
                self.n_outputs
            )));
        }
        self.l_matrices.clear();
        self.alphas.clear();
        for d in 0..self.n_outputs {
            let scale = {
                let s = self.coregion_scale(d, d);
                if s < 1e-10 { 1.0 } else { s }
            };
            let k_base = self.kernel.matrix(x_train, x_train);
            let mut k_d = k_base.mapv(|v| scale * v);
            for i in 0..n {
                k_d[[i, i]] += self.noise[d];
            }
            let l_d = cholesky_jitter(&k_d)?;
            let y_d = y_train.column(d).to_owned();
            let alpha_half = solve_lower(&l_d, &y_d)?;
            let alpha_d = solve_upper(&l_d.t().to_owned(), &alpha_half)?;
            self.l_matrices.push(l_d);
            self.alphas.push(alpha_d);
        }
        self.x_train = Some(x_train.clone());
        self.y_train = Some(y_train.clone());
        Ok(())
    }

    /// Predict all outputs at `x_test`.
    pub fn predict(
        &self,
        x_test: &Array2<f64>,
    ) -> StatsResult<(Array2<f64>, Array2<f64>)> {
        let x_train = self.x_train.as_ref().ok_or_else(|| {
            StatsError::InvalidArgument("MultiOutputGP not fitted".into())
        })?;
        let n_star = x_test.nrows();
        let mut mean_out = Array2::<f64>::zeros((n_star, self.n_outputs));
        let mut var_out = Array2::<f64>::zeros((n_star, self.n_outputs));

        let k_star_train = self.kernel.matrix(x_test, x_train);

        for d in 0..self.n_outputs {
            let scale = {
                let s = self.coregion_scale(d, d);
                if s < 1e-10 { 1.0 } else { s }
            };
            let alpha_d = &self.alphas[d];
            let l_d = &self.l_matrices[d];

            let mean_d = k_star_train.mapv(|v| v * scale).dot(alpha_d);

            let scaled_kst = k_star_train.t().to_owned().mapv(|v| v * scale);
            let v_mat = solve_lower_matrix(l_d, &scaled_kst)?;

            for i in 0..n_star {
                let xi: Vec<f64> = x_test.row(i).iter().copied().collect();
                let k_self = self.kernel.call(&xi, &xi) * scale;
                let v_sq: f64 = v_mat.column(i).iter().map(|&v| v * v).sum();
                mean_out[[i, d]] = mean_d[i];
                var_out[[i, d]] = (k_self - v_sq + self.noise[d]).max(0.0);
            }
        }
        Ok((mean_out, var_out))
    }

    /// Return the coregionalization matrix B = A Aᵀ.
    pub fn coregionalization_matrix(&self) -> Array2<f64> {
        self.coregion_a.dot(&self.coregion_a.t())
    }
}
