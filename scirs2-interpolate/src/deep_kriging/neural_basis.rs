//! Neural Basis Kriging (Deep Kriging)
//!
//! An MLP learns nonlinear basis functions phi(x) that map input space
//! into a latent feature space.  Ordinary kriging is then performed
//! in that transformed space, allowing the model to capture non-stationary
//! and nonlinear spatial patterns that conventional kriging cannot represent.
//!
//! ## Algorithm
//!
//! 1. **Forward**: Z = phi(X) via an MLP (input_dim -> hidden -> basis_dim).
//! 2. **Kriging**: Fit ordinary kriging on (Z, y).
//! 3. **Alternating optimisation**:
//!    - Fix kriging variogram parameters, update MLP weights via gradient descent.
//!    - Fix MLP weights, re-fit kriging variogram to the transformed residuals.
//!
//! ## References
//!
//! - Li, Z., et al. (2020). *Deep Kriging*. arXiv:2007.11972.

use crate::error::{InterpolateError, InterpolateResult};
use crate::kriging::{OrdinaryKriging, SphericalVariogram, Variogram};

use super::types::{Activation, DeepKrigingConfig};

// ---------------------------------------------------------------------------
// Simple MLP (multi-layer perceptron) for basis function learning
// ---------------------------------------------------------------------------

/// A single dense layer: y = activation(W * x + b).
#[derive(Debug, Clone)]
struct DenseLayer {
    /// Weight matrix stored row-major: weights[i * input_dim + j]
    weights: Vec<f64>,
    /// Bias vector.
    biases: Vec<f64>,
    /// Number of input features.
    input_dim: usize,
    /// Number of output features.
    output_dim: usize,
    /// Activation function.
    activation: Activation,
}

impl DenseLayer {
    /// Create a layer with He initialisation (scaled random normal).
    fn new(
        input_dim: usize,
        output_dim: usize,
        activation: Activation,
        rng: &mut SimpleRng,
    ) -> Self {
        let scale = (2.0 / input_dim as f64).sqrt();
        let weights: Vec<f64> = (0..input_dim * output_dim)
            .map(|_| rng.normal() * scale)
            .collect();
        let biases = vec![0.0; output_dim];
        Self {
            weights,
            biases,
            input_dim,
            output_dim,
            activation,
        }
    }

    /// Forward pass: returns (pre_activation, post_activation).
    fn forward(&self, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut pre = vec![0.0; self.output_dim];
        for i in 0..self.output_dim {
            let mut s = self.biases[i];
            for j in 0..self.input_dim {
                s += self.weights[i * self.input_dim + j] * input[j];
            }
            pre[i] = s;
        }
        let post: Vec<f64> = pre.iter().map(|&v| self.activation.apply(v)).collect();
        (pre, post)
    }

    /// Backward pass: given d_output (gradient w.r.t. post-activation output),
    /// compute weight/bias gradients and return d_input.
    fn backward(
        &self,
        input: &[f64],
        pre_activation: &[f64],
        d_output: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // d_pre = d_output * activation'(pre)
        let d_pre: Vec<f64> = d_output
            .iter()
            .zip(pre_activation.iter())
            .map(|(&dout, &pre)| dout * self.activation.derivative(pre))
            .collect();

        // d_weights[i][j] = d_pre[i] * input[j]
        let mut d_weights = vec![0.0; self.input_dim * self.output_dim];
        for i in 0..self.output_dim {
            for j in 0..self.input_dim {
                d_weights[i * self.input_dim + j] = d_pre[i] * input[j];
            }
        }

        // d_biases = d_pre
        let d_biases = d_pre.clone();

        // d_input[j] = sum_i W[i][j] * d_pre[i]
        let mut d_input = vec![0.0; self.input_dim];
        for j in 0..self.input_dim {
            for i in 0..self.output_dim {
                d_input[j] += self.weights[i * self.input_dim + j] * d_pre[i];
            }
        }

        (d_weights, d_biases, d_input)
    }

    /// Update weights with gradient descent.
    fn update(&mut self, d_weights: &[f64], d_biases: &[f64], lr: f64) {
        for (w, dw) in self.weights.iter_mut().zip(d_weights.iter()) {
            *w -= lr * dw;
        }
        for (b, db) in self.biases.iter_mut().zip(d_biases.iter()) {
            *b -= lr * db;
        }
    }
}

// ---------------------------------------------------------------------------
// MLP (stack of dense layers)
// ---------------------------------------------------------------------------

/// Multi-layer perceptron for learning basis functions.
#[derive(Debug, Clone)]
struct MLP {
    layers: Vec<DenseLayer>,
}

impl MLP {
    fn new(
        input_dim: usize,
        hidden_layers: &[usize],
        basis_dim: usize,
        activation: Activation,
        rng: &mut SimpleRng,
    ) -> Self {
        let mut layers = Vec::new();
        let mut prev_dim = input_dim;

        for &h in hidden_layers {
            layers.push(DenseLayer::new(prev_dim, h, activation, rng));
            prev_dim = h;
        }

        // Output layer uses Tanh for bounded basis functions
        layers.push(DenseLayer::new(prev_dim, basis_dim, Activation::Tanh, rng));

        Self { layers }
    }

    /// Forward pass through all layers.
    /// Returns (list of (input, pre_activation, post_activation) per layer, final output).
    fn forward(&self, input: &[f64]) -> (Vec<(Vec<f64>, Vec<f64>, Vec<f64>)>, Vec<f64>) {
        let mut current = input.to_vec();
        let mut cache = Vec::new();

        for layer in &self.layers {
            let (pre, post) = layer.forward(&current);
            cache.push((current.clone(), pre.clone(), post.clone()));
            current = post;
        }

        (cache, current)
    }

    /// Backward pass: given d_output, update all layer weights.
    fn backward_and_update(
        &mut self,
        cache: &[(Vec<f64>, Vec<f64>, Vec<f64>)],
        d_output: &[f64],
        lr: f64,
    ) {
        let mut d_current = d_output.to_vec();

        for (layer_idx, layer) in self.layers.iter_mut().enumerate().rev() {
            let (ref input, ref pre, _) = cache[layer_idx];
            let (d_weights, d_biases, d_input) = layer.backward(input, pre, &d_current);
            layer.update(&d_weights, &d_biases, lr);
            d_current = d_input;
        }
    }

    /// Transform a batch of points through the MLP.
    fn transform(&self, points: &[Vec<f64>]) -> Vec<Vec<f64>> {
        points
            .iter()
            .map(|p| {
                let (_, output) = self.forward(p);
                output
            })
            .collect()
    }

    /// Output dimension (basis_dim).
    fn output_dim(&self) -> usize {
        self.layers.last().map(|l| l.output_dim).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Simple PRNG (xorshift64)
// ---------------------------------------------------------------------------

/// Minimal xorshift64 PRNG for reproducible initialisation.
#[derive(Debug, Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform [0, 1)
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Approximate standard normal via Box-Muller.
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Neural Basis Kriging
// ---------------------------------------------------------------------------

/// Neural Basis Kriging interpolant.
///
/// Combines an MLP that learns nonlinear basis functions with ordinary
/// kriging in the transformed feature space.
#[derive(Debug, Clone)]
pub struct NeuralBasisKriging {
    /// The MLP that maps raw coordinates to basis features.
    mlp: MLP,
    /// Ordinary kriging fitted in the transformed space.
    kriging: OrdinaryKriging,
    /// Original training points (raw coordinates).
    train_points: Vec<Vec<f64>>,
    /// Training target values.
    train_values: Vec<f64>,
    /// Configuration used.
    config: DeepKrigingConfig,
    /// Heteroscedastic noise estimates (optional, per training point).
    noise_estimates: Option<Vec<f64>>,
}

impl NeuralBasisKriging {
    /// Fit a neural basis kriging model to scattered data.
    ///
    /// # Arguments
    ///
    /// * `points`  - Training data locations (n x d).
    /// * `values`  - Observed values at each location.
    /// * `config`  - Configuration controlling the MLP architecture and training.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is empty or dimensions are inconsistent.
    pub fn fit(
        points: Vec<Vec<f64>>,
        values: Vec<f64>,
        config: DeepKrigingConfig,
    ) -> InterpolateResult<Self> {
        if points.is_empty() {
            return Err(InterpolateError::invalid_input("no data points"));
        }
        if values.len() != points.len() {
            return Err(InterpolateError::shape_mismatch(
                format!("{}", points.len()),
                format!("{}", values.len()),
                "values",
            ));
        }

        let input_dim = points[0].len();
        let mut rng = SimpleRng::new(config.seed);

        let mut mlp = MLP::new(
            input_dim,
            &config.hidden_layers,
            config.basis_dim,
            config.activation,
            &mut rng,
        );

        // Alternating optimisation
        let mut kriging = Self::fit_kriging_on_transformed(&mlp, &points, &values)?;

        for _epoch in 0..config.epochs {
            // Phase 1: fix kriging, optimise MLP via MSE loss on kriging predictions
            Self::mlp_gradient_step(&mut mlp, &kriging, &points, &values, config.learning_rate)?;

            // Phase 2: fix MLP, re-fit kriging in transformed space
            kriging = Self::fit_kriging_on_transformed(&mlp, &points, &values)?;
        }

        Ok(Self {
            mlp,
            kriging,
            train_points: points,
            train_values: values,
            config,
            noise_estimates: None,
        })
    }

    /// Fit with heteroscedastic noise estimates.
    ///
    /// `noise_variances` is a per-point noise variance that is used to
    /// add a diagonal to the kriging covariance matrix (via nugget scaling).
    pub fn fit_heteroscedastic(
        points: Vec<Vec<f64>>,
        values: Vec<f64>,
        noise_variances: Vec<f64>,
        config: DeepKrigingConfig,
    ) -> InterpolateResult<Self> {
        if noise_variances.len() != points.len() {
            return Err(InterpolateError::shape_mismatch(
                format!("{}", points.len()),
                format!("{}", noise_variances.len()),
                "noise_variances",
            ));
        }

        // Use mean noise as the kriging nugget
        let mean_noise = if noise_variances.is_empty() {
            0.0
        } else {
            noise_variances.iter().sum::<f64>() / noise_variances.len() as f64
        };

        let input_dim = if points.is_empty() {
            return Err(InterpolateError::invalid_input("no data points"));
        } else {
            points[0].len()
        };

        let mut rng = SimpleRng::new(config.seed);
        let mut mlp = MLP::new(
            input_dim,
            &config.hidden_layers,
            config.basis_dim,
            config.activation,
            &mut rng,
        );

        let mut kriging =
            Self::fit_kriging_on_transformed_with_nugget(&mlp, &points, &values, mean_noise)?;

        for _epoch in 0..config.epochs {
            Self::mlp_gradient_step(&mut mlp, &kriging, &points, &values, config.learning_rate)?;
            kriging =
                Self::fit_kriging_on_transformed_with_nugget(&mlp, &points, &values, mean_noise)?;
        }

        Ok(Self {
            mlp,
            kriging,
            train_points: points,
            train_values: values,
            config,
            noise_estimates: Some(noise_variances),
        })
    }

    /// Predict at a new location.
    ///
    /// Returns `(estimate, variance)`.
    pub fn predict(&self, x: &[f64]) -> InterpolateResult<(f64, f64)> {
        let (_, transformed) = self.mlp.forward(x);
        self.kriging.predict(&transformed)
    }

    /// Predict at multiple locations.
    pub fn predict_batch(&self, points: &[Vec<f64>]) -> InterpolateResult<Vec<(f64, f64)>> {
        points.iter().map(|p| self.predict(p)).collect()
    }

    /// Return the transformed (basis) representation of input points.
    pub fn transform(&self, points: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.mlp.transform(points)
    }

    /// Return the training loss (MSE on training data).
    pub fn training_mse(&self) -> InterpolateResult<f64> {
        let mut total = 0.0;
        for (p, &y) in self.train_points.iter().zip(self.train_values.iter()) {
            let (pred, _) = self.predict(p)?;
            total += (pred - y) * (pred - y);
        }
        Ok(total / self.train_points.len() as f64)
    }

    /// Number of MLP parameters.
    pub fn num_parameters(&self) -> usize {
        self.mlp
            .layers
            .iter()
            .map(|l| l.weights.len() + l.biases.len())
            .sum()
    }

    /// Access to the underlying config.
    pub fn config(&self) -> &DeepKrigingConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Fit ordinary kriging in the MLP-transformed space.
    fn fit_kriging_on_transformed(
        mlp: &MLP,
        points: &[Vec<f64>],
        values: &[f64],
    ) -> InterpolateResult<OrdinaryKriging> {
        Self::fit_kriging_on_transformed_with_nugget(mlp, points, values, 0.0)
    }

    /// Fit ordinary kriging in the MLP-transformed space with a nugget.
    fn fit_kriging_on_transformed_with_nugget(
        mlp: &MLP,
        points: &[Vec<f64>],
        values: &[f64],
        nugget: f64,
    ) -> InterpolateResult<OrdinaryKriging> {
        let transformed = mlp.transform(points);

        // Estimate variogram range from max pairwise distance in transformed space
        let mut max_dist = 0.0_f64;
        for i in 0..transformed.len() {
            for j in (i + 1)..transformed.len() {
                let d: f64 = transformed[i]
                    .iter()
                    .zip(transformed[j].iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                if d > max_dist {
                    max_dist = d;
                }
            }
        }

        // Estimate sill from data variance
        let mean_val = values.iter().sum::<f64>() / values.len() as f64;
        let sill = values
            .iter()
            .map(|&v| (v - mean_val) * (v - mean_val))
            .sum::<f64>()
            / (values.len() as f64).max(1.0);
        let sill = sill.max(1e-6);

        let vgm = SphericalVariogram {
            nugget,
            sill,
            range: max_dist.max(1e-6),
        };

        OrdinaryKriging::fit(transformed, values.to_vec(), Box::new(vgm))
    }

    /// One gradient descent step on MLP weights to reduce MSE.
    fn mlp_gradient_step(
        mlp: &mut MLP,
        kriging: &OrdinaryKriging,
        points: &[Vec<f64>],
        values: &[f64],
        lr: f64,
    ) -> InterpolateResult<()> {
        let n = points.len();
        if n == 0 {
            return Ok(());
        }

        // Accumulate gradients over all training points
        for (idx, point) in points.iter().enumerate() {
            let (cache, transformed) = mlp.forward(point);

            // Predict via kriging
            let (pred, _) = kriging.predict(&transformed)?;

            // dL/d_pred = 2 * (pred - target) / n
            let d_loss = 2.0 * (pred - values[idx]) / n as f64;

            // Numerical gradient of kriging output w.r.t. its input
            let basis_dim = mlp.output_dim();
            let mut d_transformed = vec![0.0; basis_dim];
            let eps = 1e-5;
            for d in 0..basis_dim {
                let mut t_plus = transformed.clone();
                t_plus[d] += eps;
                let (pred_plus, _) = kriging.predict(&t_plus)?;

                let mut t_minus = transformed.clone();
                t_minus[d] -= eps;
                let (pred_minus, _) = kriging.predict(&t_minus)?;

                d_transformed[d] = d_loss * (pred_plus - pred_minus) / (2.0 * eps);
            }

            // Backprop through MLP
            mlp.backward_and_update(&cache, &d_transformed, lr);
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

    #[test]
    fn test_neural_basis_kriging_basic() {
        let points: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.5]).collect();
        let values: Vec<f64> = points.iter().map(|p| (p[0]).sin()).collect();

        let config = DeepKrigingConfig {
            hidden_layers: vec![16],
            learning_rate: 0.005,
            epochs: 20,
            activation: Activation::Tanh,
            basis_dim: 4,
            seed: 42,
        };

        let nk = NeuralBasisKriging::fit(points.clone(), values.clone(), config);
        assert!(nk.is_ok(), "fit should succeed");

        let nk = nk.expect("test: fit succeeded");
        // Should at least produce predictions
        let (pred, var) = nk.predict(&[1.0]).expect("test: predict succeeded");
        assert!(pred.is_finite(), "prediction should be finite");
        assert!(var >= 0.0, "variance should be non-negative");
    }

    #[test]
    fn test_neural_basis_improves_over_epochs() {
        let points: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64 * 0.5]).collect();
        let values: Vec<f64> = points.iter().map(|p| (p[0]).sin()).collect();

        // Few epochs
        let config_few = DeepKrigingConfig {
            epochs: 2,
            seed: 42,
            ..DeepKrigingConfig::default()
        };
        let nk_few = NeuralBasisKriging::fit(points.clone(), values.clone(), config_few)
            .expect("test: fit few");
        let mse_few = nk_few.training_mse().expect("test: mse few");

        // More epochs
        let config_many = DeepKrigingConfig {
            epochs: 50,
            seed: 42,
            ..DeepKrigingConfig::default()
        };
        let nk_many = NeuralBasisKriging::fit(points, values, config_many).expect("test: fit many");
        let mse_many = nk_many.training_mse().expect("test: mse many");

        // More epochs should generally not be worse (allow for some tolerance
        // since the alternating optimisation is not monotonic)
        assert!(
            mse_many < mse_few * 2.0,
            "more epochs should not drastically increase MSE: few={mse_few}, many={mse_many}"
        );
    }

    #[test]
    fn test_heteroscedastic_fit() {
        let points: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64]).collect();
        let values = vec![0.0, 1.0, 0.5, 0.8, 0.2, 0.6];
        let noise = vec![0.1, 0.1, 0.5, 0.5, 0.1, 0.1];

        let config = DeepKrigingConfig {
            epochs: 10,
            ..DeepKrigingConfig::default()
        };

        let nk = NeuralBasisKriging::fit_heteroscedastic(points, values, noise, config);
        assert!(nk.is_ok(), "heteroscedastic fit should succeed");
    }

    #[test]
    fn test_transform_shape() {
        let points: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64, i as f64 * 0.5]).collect();
        let values = vec![0.0; 5];
        let config = DeepKrigingConfig {
            basis_dim: 4,
            epochs: 5,
            ..DeepKrigingConfig::default()
        };

        let nk =
            NeuralBasisKriging::fit(points.clone(), values, config).expect("test: fit transform");
        let transformed = nk.transform(&points);
        assert_eq!(transformed.len(), 5);
        assert_eq!(transformed[0].len(), 4);
    }

    #[test]
    fn test_empty_data_error() {
        let config = DeepKrigingConfig::default();
        let result = NeuralBasisKriging::fit(vec![], vec![], config);
        assert!(result.is_err());
    }

    #[test]
    fn test_num_parameters() {
        let points: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64]).collect();
        let values = vec![0.0; 4];
        let config = DeepKrigingConfig {
            hidden_layers: vec![8],
            basis_dim: 4,
            epochs: 1,
            ..DeepKrigingConfig::default()
        };

        let nk = NeuralBasisKriging::fit(points, values, config).expect("test: fit params");
        // Layer 1: 1*8 weights + 8 biases = 16
        // Layer 2: 8*4 weights + 4 biases = 36
        // Total = 52
        assert_eq!(nk.num_parameters(), 52);
    }

    #[test]
    fn test_predict_batch() {
        let points: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64]).collect();
        let values: Vec<f64> = points.iter().map(|p| p[0] * p[0]).collect();
        let config = DeepKrigingConfig {
            epochs: 5,
            ..DeepKrigingConfig::default()
        };

        let nk = NeuralBasisKriging::fit(points, values, config).expect("test: fit batch");
        let query = vec![vec![0.5], vec![1.5], vec![2.5]];
        let results = nk.predict_batch(&query).expect("test: batch predict");
        assert_eq!(results.len(), 3);
        for (pred, var) in &results {
            assert!(pred.is_finite());
            assert!(*var >= 0.0);
        }
    }
}
