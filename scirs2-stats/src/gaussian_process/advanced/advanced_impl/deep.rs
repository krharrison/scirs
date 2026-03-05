//! Deep Gaussian Process with doubly-stochastic variational inference.

use super::linalg::{cholesky_jitter, solve_lower, solve_lower_matrix, solve_upper};
use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2};

/// Configuration for one layer of a Deep GP.
#[derive(Debug, Clone)]
pub struct DeepGPLayerConfig {
    /// Number of inducing points in this layer.
    pub num_inducing: usize,
    /// Output dimensionality of this layer (input dim of next).
    pub out_dim: usize,
    /// Length scale for the RBF kernel in this layer.
    pub length_scale: f64,
    /// Output variance.
    pub variance: f64,
    /// Noise level for this layer.
    pub noise: f64,
}

impl Default for DeepGPLayerConfig {
    fn default() -> Self {
        Self {
            num_inducing: 20,
            out_dim: 1,
            length_scale: 1.0,
            variance: 1.0,
            noise: 0.01,
        }
    }
}

/// One layer of the Deep GP.
#[derive(Debug, Clone)]
struct DeepGPLayer {
    config: DeepGPLayerConfig,
    inducing_mean: Array2<f64>,
    inducing_inputs: Array2<f64>,
    chol_s: Array2<f64>,
}

impl DeepGPLayer {
    fn new(config: DeepGPLayerConfig, _in_dim: usize, inducing_inputs: Array2<f64>) -> Self {
        let m = inducing_inputs.nrows();
        let chol_s = Array2::<f64>::eye(m) * config.noise.sqrt();
        let inducing_mean = Array2::<f64>::zeros((m, config.out_dim));
        Self { config, inducing_mean, inducing_inputs, chol_s }
    }

    fn rbf(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let ls = self.config.length_scale;
        let var_k = self.config.variance;
        let sq: f64 = x1.iter().zip(x2).map(|(&a, &b)| (a - b).powi(2) / (ls * ls)).sum();
        var_k * (-0.5 * sq).exp()
    }

    fn propagate(
        &self,
        h_mean: &Array2<f64>,
        h_var: &Array2<f64>,
    ) -> StatsResult<(Array2<f64>, Array2<f64>)> {
        let n = h_mean.nrows();
        let out_d = self.config.out_dim;
        let m = self.inducing_inputs.nrows();
        let in_d = h_mean.ncols();
        let ls = self.config.length_scale;
        let var_k = self.config.variance;

        // K_uu
        let mut k_uu = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            let xi: Vec<f64> = self.inducing_inputs.row(i).iter().copied().collect();
            for j in 0..m {
                let xj: Vec<f64> = self.inducing_inputs.row(j).iter().copied().collect();
                k_uu[[i, j]] = self.rbf(&xi, &xj);
            }
            k_uu[[i, i]] += 1e-6;
        }
        let l_uu = cholesky_jitter(&k_uu)?;

        // K_fu with input uncertainty correction
        let mut k_fu = Array2::<f64>::zeros((n, m));
        for i in 0..n {
            let xi: Vec<f64> = h_mean.row(i).iter().copied().collect();
            let var_i: Vec<f64> = h_var.row(i).iter().copied().collect();
            let correction: f64 = (0..in_d.min(var_i.len()))
                .map(|d_| -0.5 * var_i[d_] / (ls * ls))
                .sum::<f64>()
                .exp();
            for j in 0..m {
                let zj: Vec<f64> = self.inducing_inputs.row(j).iter().copied().collect();
                let sq: f64 = xi.iter().zip(zj.iter())
                    .map(|(&a, &b)| (a - b).powi(2) / (ls * ls))
                    .sum();
                k_fu[[i, j]] = var_k * (-0.5 * sq).exp() * correction;
            }
        }

        // K_uu^{-1} inducing_mean
        let kinv_m = {
            let tmp = solve_lower_matrix(&l_uu, &self.inducing_mean)?;
            solve_lower_matrix(&l_uu.t().to_owned(), &tmp)?
        };
        let mean_out = k_fu.dot(&kinv_m);

        // Variance: q_fu = L_uu^{-1} K_uf
        let q_fu = solve_lower_matrix(&l_uu, &k_fu.t().to_owned())?;
        let l_inv_chol_s = solve_lower_matrix(&l_uu, &self.chol_s)?;

        let mut var_out = Array2::<f64>::zeros((n, out_d));
        for i in 0..n {
            let var_i: Vec<f64> = h_var.row(i).iter().copied().collect();
            let correction: f64 = (0..in_d.min(var_i.len()))
                .map(|d_| -0.5 * var_i[d_] / (ls * ls))
                .sum::<f64>()
                .exp();
            let k_self = var_k * correction;
            let q_sq: f64 = q_fu.column(i).iter().map(|&v| v * v).sum();
            let s_sq: f64 = l_inv_chol_s.t().dot(&q_fu.column(i).to_owned())
                .iter()
                .map(|&v| v * v)
                .sum();
            let v = (k_self - q_sq + s_sq + self.config.noise).max(1e-10);
            for od in 0..out_d {
                var_out[[i, od]] = v;
            }
        }

        Ok((mean_out, var_out))
    }
}

/// Deep Gaussian Process with doubly-stochastic variational inference.
///
/// # Reference
/// Salimbeni & Deisenroth (2017) "Doubly Stochastic Variational Inference for
/// Deep Gaussian Processes"
#[derive(Debug, Clone)]
pub struct DeepGP {
    layers: Vec<DeepGPLayer>,
    input_dim: usize,
}

impl DeepGP {
    /// Build a Deep GP.
    pub fn new(
        layer_configs: Vec<DeepGPLayerConfig>,
        x_init: &Array2<f64>,
    ) -> StatsResult<Self> {
        if layer_configs.is_empty() {
            return Err(StatsError::InvalidArgument(
                "DeepGP requires at least one layer".into(),
            ));
        }
        let input_dim = x_init.ncols();
        let mut layers = Vec::with_capacity(layer_configs.len());
        let mut cur_in_dim = input_dim;
        for config in layer_configs {
            let m = config.num_inducing.min(x_init.nrows());
            let inducing = if cur_in_dim <= x_init.ncols() {
                let cols = cur_in_dim;
                let mut ind = Array2::<f64>::zeros((m, cols));
                for i in 0..m {
                    let row_idx = i * x_init.nrows() / m;
                    for d in 0..cols {
                        ind[[i, d]] = x_init[[row_idx, d]];
                    }
                }
                ind
            } else {
                Array2::<f64>::zeros((m, cur_in_dim))
            };
            let next_dim = config.out_dim;
            let layer = DeepGPLayer::new(config, cur_in_dim, inducing);
            layers.push(layer);
            cur_in_dim = next_dim;
        }
        Ok(Self { layers, input_dim })
    }

    /// Predict by propagating `x_test` through all layers.
    pub fn predict(
        &self,
        x_test: &Array2<f64>,
    ) -> StatsResult<(Array2<f64>, Array2<f64>)> {
        if self.layers.is_empty() {
            return Err(StatsError::InvalidArgument("DeepGP has no layers".into()));
        }
        let n = x_test.nrows();
        let mut h_mean = x_test.clone();
        let mut h_var = Array2::<f64>::zeros((n, self.input_dim));
        for layer in &self.layers {
            let in_d = h_mean.ncols();
            if h_var.ncols() != in_d {
                h_var = Array2::<f64>::zeros((n, in_d));
            }
            let (new_mean, new_var) = layer.propagate(&h_mean, &h_var)?;
            h_mean = new_mean;
            h_var = new_var;
        }
        Ok((h_mean, h_var))
    }

    /// Number of layers.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }
}
