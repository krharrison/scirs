//! Feed-forward neural network for PINN with finite-difference derivatives.

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2};

/// A simple feed-forward neural network used as the PINN approximator.
///
/// Uses tanh activation functions and supports computing spatial derivatives
/// via central finite differences for physics-informed loss computation.
#[derive(Debug, Clone)]
pub struct PINNNetwork {
    /// Weight matrices for each layer transition
    weights: Vec<Array2<f64>>,
    /// Bias vectors for each layer
    biases: Vec<Array1<f64>>,
    /// Layer sizes including input and output
    layer_sizes: Vec<usize>,
}

/// Xorshift64 pseudo-random number generator for weight initialization.
fn xorshift64(state: &mut u64) -> f64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    // Map to [-1, 1] then scale
    (s as f64) / (u64::MAX as f64) * 2.0 - 1.0
}

impl PINNNetwork {
    /// Create a new feed-forward network with Xavier initialization.
    ///
    /// # Arguments
    /// * `input_dim` - Number of input features (spatial dims + optional time)
    /// * `hidden_layers` - Number of neurons per hidden layer
    /// * `output_dim` - Number of output features (typically 1 for scalar PDEs)
    ///
    /// # Errors
    /// Returns an error if `hidden_layers` is empty or dimensions are zero.
    pub fn new(
        input_dim: usize,
        hidden_layers: &[usize],
        output_dim: usize,
    ) -> IntegrateResult<Self> {
        if hidden_layers.is_empty() {
            return Err(IntegrateError::InvalidInput(
                "hidden_layers must not be empty".to_string(),
            ));
        }
        if input_dim == 0 || output_dim == 0 {
            return Err(IntegrateError::InvalidInput(
                "input_dim and output_dim must be positive".to_string(),
            ));
        }
        for (i, &size) in hidden_layers.iter().enumerate() {
            if size == 0 {
                return Err(IntegrateError::InvalidInput(format!(
                    "hidden layer {} has zero neurons",
                    i
                )));
            }
        }

        let mut layer_sizes = Vec::with_capacity(hidden_layers.len() + 2);
        layer_sizes.push(input_dim);
        layer_sizes.extend_from_slice(hidden_layers);
        layer_sizes.push(output_dim);

        let mut weights = Vec::with_capacity(layer_sizes.len() - 1);
        let mut biases = Vec::with_capacity(layer_sizes.len() - 1);
        let mut rng_state: u64 = 42_u64.wrapping_mul(7) | 1;

        for i in 0..layer_sizes.len() - 1 {
            let fan_in = layer_sizes[i];
            let fan_out = layer_sizes[i + 1];
            // Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
            let scale = (2.0 / (fan_in + fan_out) as f64).sqrt();

            let mut w = Array2::<f64>::zeros((fan_in, fan_out));
            for r in 0..fan_in {
                for c in 0..fan_out {
                    w[[r, c]] = xorshift64(&mut rng_state) * scale;
                }
            }
            weights.push(w);

            let b = Array1::<f64>::zeros(fan_out);
            biases.push(b);
        }

        Ok(Self {
            weights,
            biases,
            layer_sizes,
        })
    }

    /// Forward pass through the network with tanh activation on hidden layers.
    ///
    /// # Arguments
    /// * `x` - Input vector of size `input_dim`
    ///
    /// # Returns
    /// The scalar output of the network (first element of output layer).
    pub fn forward(&self, x: &Array1<f64>) -> IntegrateResult<f64> {
        if x.len() != self.layer_sizes[0] {
            return Err(IntegrateError::DimensionMismatch(format!(
                "expected input dim {}, got {}",
                self.layer_sizes[0],
                x.len()
            )));
        }

        let mut current = x.clone();

        for i in 0..self.weights.len() {
            // z = x * W + b
            let z = current.dot(&self.weights[i]) + &self.biases[i];

            if i < self.weights.len() - 1 {
                // Hidden layer: tanh activation
                current = z.mapv(|v| v.tanh());
            } else {
                // Output layer: linear (no activation)
                current = z;
            }
        }

        Ok(current[0])
    }

    /// Batch forward pass for multiple input points.
    ///
    /// # Arguments
    /// * `x` - Input matrix of shape (n_points, input_dim)
    ///
    /// # Returns
    /// Array of scalar outputs, one per input point.
    pub fn forward_batch(&self, x: &Array2<f64>) -> IntegrateResult<Array1<f64>> {
        let n = x.nrows();
        if x.ncols() != self.layer_sizes[0] {
            return Err(IntegrateError::DimensionMismatch(format!(
                "expected input dim {}, got {}",
                self.layer_sizes[0],
                x.ncols()
            )));
        }

        let mut result = Array1::<f64>::zeros(n);
        for i in 0..n {
            let row = x.row(i).to_owned();
            result[i] = self.forward(&row)?;
        }
        Ok(result)
    }

    /// Compute gradient du/dx via central finite differences.
    ///
    /// # Arguments
    /// * `x` - Point at which to evaluate the gradient
    /// * `h` - Step size for finite differences
    ///
    /// # Returns
    /// Gradient vector of size `input_dim`.
    pub fn gradient(&self, x: &Array1<f64>, h: f64) -> IntegrateResult<Array1<f64>> {
        let dim = x.len();
        let mut grad = Array1::<f64>::zeros(dim);

        for d in 0..dim {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[d] += h;
            x_minus[d] -= h;

            let f_plus = self.forward(&x_plus)?;
            let f_minus = self.forward(&x_minus)?;
            grad[d] = (f_plus - f_minus) / (2.0 * h);
        }

        Ok(grad)
    }

    /// Compute the Laplacian (sum of second derivatives) via finite differences.
    ///
    /// Uses the formula: d2u/dx_i^2 = (u(x+h*e_i) - 2*u(x) + u(x-h*e_i)) / h^2
    ///
    /// # Arguments
    /// * `x` - Point at which to evaluate the Laplacian
    /// * `h` - Step size for finite differences
    pub fn laplacian(&self, x: &Array1<f64>, h: f64) -> IntegrateResult<f64> {
        let dim = x.len();
        let u_center = self.forward(x)?;
        let h_sq = h * h;
        let mut lap = 0.0;

        for d in 0..dim {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[d] += h;
            x_minus[d] -= h;

            let u_plus = self.forward(&x_plus)?;
            let u_minus = self.forward(&x_minus)?;
            lap += (u_plus - 2.0 * u_center + u_minus) / h_sq;
        }

        Ok(lap)
    }

    /// Compute the second derivative d2u/dx_i^2 via finite differences.
    ///
    /// # Arguments
    /// * `x` - Point at which to evaluate
    /// * `dim` - Which spatial dimension to differentiate
    /// * `h` - Step size for finite differences
    pub fn second_derivative(&self, x: &Array1<f64>, dim: usize, h: f64) -> IntegrateResult<f64> {
        if dim >= x.len() {
            return Err(IntegrateError::InvalidInput(format!(
                "dim {} out of range for input of size {}",
                dim,
                x.len()
            )));
        }

        let u_center = self.forward(x)?;
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus[dim] += h;
        x_minus[dim] -= h;

        let u_plus = self.forward(&x_plus)?;
        let u_minus = self.forward(&x_minus)?;

        Ok((u_plus - 2.0 * u_center + u_minus) / (h * h))
    }

    /// Compute the mixed derivative d2u/(dx_i dx_j) via finite differences.
    ///
    /// Uses: (u(x+h*e_i+h*e_j) - u(x+h*e_i-h*e_j) - u(x-h*e_i+h*e_j) + u(x-h*e_i-h*e_j)) / (4*h^2)
    ///
    /// # Arguments
    /// * `x` - Point at which to evaluate
    /// * `dim_i` - First dimension index
    /// * `dim_j` - Second dimension index
    /// * `h` - Step size for finite differences
    pub fn mixed_derivative(
        &self,
        x: &Array1<f64>,
        dim_i: usize,
        dim_j: usize,
        h: f64,
    ) -> IntegrateResult<f64> {
        let n = x.len();
        if dim_i >= n || dim_j >= n {
            return Err(IntegrateError::InvalidInput(format!(
                "dim_i={} or dim_j={} out of range for input of size {}",
                dim_i, dim_j, n
            )));
        }

        // f(x+h_i+h_j)
        let mut x_pp = x.clone();
        x_pp[dim_i] += h;
        x_pp[dim_j] += h;

        // f(x+h_i-h_j)
        let mut x_pm = x.clone();
        x_pm[dim_i] += h;
        x_pm[dim_j] -= h;

        // f(x-h_i+h_j)
        let mut x_mp = x.clone();
        x_mp[dim_i] -= h;
        x_mp[dim_j] += h;

        // f(x-h_i-h_j)
        let mut x_mm = x.clone();
        x_mm[dim_i] -= h;
        x_mm[dim_j] -= h;

        let u_pp = self.forward(&x_pp)?;
        let u_pm = self.forward(&x_pm)?;
        let u_mp = self.forward(&x_mp)?;
        let u_mm = self.forward(&x_mm)?;

        Ok((u_pp - u_pm - u_mp + u_mm) / (4.0 * h * h))
    }

    /// Compute the time derivative du/dt via central finite differences.
    ///
    /// Assumes the last dimension of the input is the time variable.
    ///
    /// # Arguments
    /// * `x` - Input vector where the last element is time
    /// * `h` - Step size for finite differences
    pub fn time_derivative(&self, x: &Array1<f64>, h: f64) -> IntegrateResult<f64> {
        let t_dim = x.len() - 1;
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus[t_dim] += h;
        x_minus[t_dim] -= h;

        let u_plus = self.forward(&x_plus)?;
        let u_minus = self.forward(&x_minus)?;

        Ok((u_plus - u_minus) / (2.0 * h))
    }

    /// Get all trainable parameters (weights and biases) as a flat vector.
    pub fn parameters(&self) -> Array1<f64> {
        let n = self.n_parameters();
        let mut params = Array1::<f64>::zeros(n);
        let mut idx = 0;

        for i in 0..self.weights.len() {
            for &w in self.weights[i].iter() {
                params[idx] = w;
                idx += 1;
            }
            for &b in self.biases[i].iter() {
                params[idx] = b;
                idx += 1;
            }
        }

        params
    }

    /// Set all trainable parameters from a flat vector.
    ///
    /// # Errors
    /// Returns an error if the parameter vector length does not match `n_parameters()`.
    pub fn set_parameters(&mut self, params: &Array1<f64>) -> IntegrateResult<()> {
        let expected = self.n_parameters();
        if params.len() != expected {
            return Err(IntegrateError::DimensionMismatch(format!(
                "expected {} parameters, got {}",
                expected,
                params.len()
            )));
        }

        let mut idx = 0;
        for i in 0..self.weights.len() {
            let (rows, cols) = (self.weights[i].nrows(), self.weights[i].ncols());
            for r in 0..rows {
                for c in 0..cols {
                    self.weights[i][[r, c]] = params[idx];
                    idx += 1;
                }
            }
            let bias_len = self.biases[i].len();
            for j in 0..bias_len {
                self.biases[i][j] = params[idx];
                idx += 1;
            }
        }

        Ok(())
    }

    /// Total number of trainable parameters.
    pub fn n_parameters(&self) -> usize {
        let mut count = 0;
        for i in 0..self.weights.len() {
            count += self.weights[i].len();
            count += self.biases[i].len();
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_network_creation() {
        let net = PINNNetwork::new(2, &[8, 8], 1);
        assert!(net.is_ok());
        let net = net.expect("network should be created");
        assert_eq!(net.layer_sizes, vec![2, 8, 8, 1]);
    }

    #[test]
    fn test_network_creation_errors() {
        assert!(PINNNetwork::new(2, &[], 1).is_err());
        assert!(PINNNetwork::new(0, &[8], 1).is_err());
        assert!(PINNNetwork::new(2, &[8], 0).is_err());
        assert!(PINNNetwork::new(2, &[0, 8], 1).is_err());
    }

    #[test]
    fn test_forward_output_is_scalar() {
        let net = PINNNetwork::new(2, &[8, 8], 1).expect("network creation");
        let x = array![0.5, 0.5];
        let result = net.forward(&x);
        assert!(result.is_ok());
        // Output is a finite scalar
        let val = result.expect("forward pass");
        assert!(val.is_finite());
    }

    #[test]
    fn test_forward_dimension_mismatch() {
        let net = PINNNetwork::new(2, &[8], 1).expect("network creation");
        let x = array![0.5, 0.5, 0.5]; // wrong dim
        assert!(net.forward(&x).is_err());
    }

    #[test]
    fn test_batch_forward() {
        let net = PINNNetwork::new(2, &[8, 8], 1).expect("network creation");
        let x = Array2::from_shape_vec((3, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            .expect("array creation");
        let result = net.forward_batch(&x);
        assert!(result.is_ok());
        let vals = result.expect("batch forward");
        assert_eq!(vals.len(), 3);
        for &v in vals.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_gradient_finite_difference() {
        let net = PINNNetwork::new(2, &[16, 16], 1).expect("network creation");
        let x = array![0.5, 0.5];
        let h = 1e-5;
        let grad = net.gradient(&x, h).expect("gradient computation");
        assert_eq!(grad.len(), 2);

        // Verify gradient matches manual central differences
        for d in 0..2 {
            let mut x_p = x.clone();
            let mut x_m = x.clone();
            x_p[d] += h;
            x_m[d] -= h;
            let expected =
                (net.forward(&x_p).expect("fwd") - net.forward(&x_m).expect("fwd")) / (2.0 * h);
            assert!(
                (grad[d] - expected).abs() < 1e-10,
                "gradient mismatch in dim {}",
                d
            );
        }
    }

    #[test]
    fn test_laplacian_is_sum_of_second_derivatives() {
        let net = PINNNetwork::new(2, &[16, 16], 1).expect("network creation");
        let x = array![0.3, 0.7];
        let h = 1e-4;

        let lap = net.laplacian(&x, h).expect("laplacian");
        let d2_0 = net.second_derivative(&x, 0, h).expect("d2/dx0^2");
        let d2_1 = net.second_derivative(&x, 1, h).expect("d2/dx1^2");

        assert!(
            (lap - (d2_0 + d2_1)).abs() < 1e-6,
            "laplacian should equal sum of second derivatives"
        );
    }

    #[test]
    fn test_parameter_roundtrip() {
        let mut net = PINNNetwork::new(2, &[8, 8], 1).expect("network creation");
        let original_params = net.parameters();
        let n = net.n_parameters();
        assert!(n > 0);
        assert_eq!(original_params.len(), n);

        // Modify and restore
        let mut modified = original_params.clone();
        modified[0] = 999.0;
        net.set_parameters(&modified).expect("set params");
        let retrieved = net.parameters();
        assert!((retrieved[0] - 999.0).abs() < 1e-15);

        // Restore original
        net.set_parameters(&original_params).expect("set params");
        let restored = net.parameters();
        for i in 0..n {
            assert!(
                (restored[i] - original_params[i]).abs() < 1e-15,
                "parameter {} mismatch after roundtrip",
                i
            );
        }
    }

    #[test]
    fn test_parameter_count() {
        // Network: 2 -> 8 -> 4 -> 1
        // Layer 0: 2*8 weights + 8 biases = 24
        // Layer 1: 8*4 weights + 4 biases = 36
        // Layer 2: 4*1 weights + 1 bias = 5
        // Total: 65
        let net = PINNNetwork::new(2, &[8, 4], 1).expect("network creation");
        assert_eq!(net.n_parameters(), 65);
    }

    #[test]
    fn test_set_parameters_wrong_size() {
        let mut net = PINNNetwork::new(2, &[8], 1).expect("network creation");
        let wrong = Array1::<f64>::zeros(5);
        assert!(net.set_parameters(&wrong).is_err());
    }

    #[test]
    fn test_second_derivative_out_of_range() {
        let net = PINNNetwork::new(2, &[8], 1).expect("network creation");
        let x = array![0.5, 0.5];
        assert!(net.second_derivative(&x, 5, 1e-4).is_err());
    }

    #[test]
    fn test_mixed_derivative() {
        let net = PINNNetwork::new(2, &[16, 16], 1).expect("network creation");
        let x = array![0.3, 0.7];
        let h = 1e-4;
        let mixed = net.mixed_derivative(&x, 0, 1, h);
        assert!(mixed.is_ok());
        assert!(mixed.expect("mixed deriv").is_finite());
    }

    #[test]
    fn test_time_derivative() {
        let net = PINNNetwork::new(2, &[8, 8], 1).expect("network creation");
        let x = array![0.5, 0.3]; // [spatial, time]
        let h = 1e-5;
        let dt = net.time_derivative(&x, h);
        assert!(dt.is_ok());
        assert!(dt.expect("time deriv").is_finite());
    }
}
