// Advanced adaptive filtering algorithms
//
// This module provides comprehensive adaptive filter implementations including classical
// algorithms (LMS, RLS, NLMS) and advanced variants (Variable Step-Size LMS, Affine
// Projection Algorithm, Frequency Domain LMS, robust adaptive filters). These filters
// are used for applications such as noise cancellation, system identification, echo
// cancellation, equalization, beamforming, and channel estimation.

use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;
use std::collections::VecDeque;
use std::fmt::Debug;

/// Least Mean Squares (LMS) adaptive filter
///
/// The LMS algorithm is a simple and robust adaptive filter that minimizes
/// the mean square error between the desired signal and the filter output.
/// It uses a gradient descent approach to update the filter coefficients.
///
/// # Examples
///
/// ```ignore
/// use scirs2_signal::adaptive::LmsFilter;
///
/// let mut lms = LmsFilter::new(4, 0.01, 0.0).expect("Operation failed");
/// let (output, error, mse) = lms.adapt(1.0, 0.5).expect("Operation failed");
/// ```
#[derive(Debug, Clone)]
pub struct LmsFilter {
    /// Filter coefficients (weights)
    weights: Vec<f64>,
    /// Input buffer (delay line)
    buffer: Vec<f64>,
    /// Step size (learning rate)
    step_size: f64,
    /// Current buffer index
    buffer_index: usize,
}

impl LmsFilter {
    /// Create a new LMS filter
    ///
    /// # Arguments
    ///
    /// * `num_taps` - Number of filter taps (filter order + 1)
    /// * `step_size` - Learning rate (typically 0.001 to 0.1)
    /// * `initial_weight` - Initial value for all filter weights
    ///
    /// # Returns
    ///
    /// * A new LMS filter instance
    pub fn new(num_taps: usize, step_size: f64, initial_weight: f64) -> SignalResult<Self> {
        if num_taps == 0 {
            return Err(SignalError::ValueError(
                "Number of taps must be positive".to_string(),
            ));
        }

        if step_size <= 0.0 {
            return Err(SignalError::ValueError(
                "Step size must be positive".to_string(),
            ));
        }

        Ok(LmsFilter {
            weights: vec![initial_weight; num_taps],
            buffer: vec![0.0; num_taps],
            step_size,
            buffer_index: 0,
        })
    }

    /// Process one sample through the adaptive filter
    ///
    /// # Arguments
    ///
    /// * `input` - Input sample
    /// * `desired` - Desired output sample
    ///
    /// # Returns
    ///
    /// * Tuple of (filter_output, error, mse_estimate)
    pub fn adapt(&mut self, input: f64, desired: f64) -> SignalResult<(f64, f64, f64)> {
        // Update circular buffer with new input
        self.buffer[self.buffer_index] = input;
        self.buffer_index = (self.buffer_index + 1) % self.buffer.len();

        // Compute filter output (dot product of weights and buffered inputs)
        let mut output = 0.0;
        for i in 0..self.weights.len() {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            output += self.weights[i] * self.buffer[buffer_idx];
        }

        // Compute error
        let error = desired - output;

        // Update weights using LMS algorithm: w(n+1) = w(n) + mu * e(n) * x(n)
        for i in 0..self.weights.len() {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            self.weights[i] += self.step_size * error * self.buffer[buffer_idx];
        }

        // Estimate MSE (simple exponential average)
        let mse_estimate = error * error;

        Ok((output, error, mse_estimate))
    }

    /// Process a batch of samples
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input signal samples
    /// * `desired` - Desired output samples
    ///
    /// # Returns
    ///
    /// * Tuple of (outputs, errors, mse_estimates)
    pub fn adapt_batch(
        &mut self,
        inputs: &[f64],
        desired: &[f64],
    ) -> SignalResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        if inputs.len() != desired.len() {
            return Err(SignalError::ValueError(
                "Input and desired signals must have the same length".to_string(),
            ));
        }

        let mut outputs = Vec::with_capacity(inputs.len());
        let mut errors = Vec::with_capacity(inputs.len());
        let mut mse_estimates = Vec::with_capacity(inputs.len());

        for (&input, &des) in inputs.iter().zip(desired.iter()) {
            let (output, error, mse) = self.adapt(input, des)?;
            outputs.push(output);
            errors.push(error);
            mse_estimates.push(mse);
        }

        Ok((outputs, errors, mse_estimates))
    }

    /// Get current filter weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get current input buffer state
    pub fn buffer(&self) -> &[f64] {
        &self.buffer
    }

    /// Reset the filter to initial state
    pub fn reset(&mut self, initial_weight: f64) {
        self.weights.fill(initial_weight);
        self.buffer.fill(0.0);
        self.buffer_index = 0;
    }

    /// Set step size (learning rate)
    pub fn set_step_size(&mut self, step_size: f64) -> SignalResult<()> {
        if step_size <= 0.0 {
            return Err(SignalError::ValueError(
                "Step size must be positive".to_string(),
            ));
        }
        self.step_size = step_size;
        Ok(())
    }
}

/// Recursive Least Squares (RLS) adaptive filter
///
/// The RLS algorithm provides faster convergence than LMS but with higher
/// computational complexity. It minimizes the exponentially weighted sum
/// of squared errors and is particularly effective for non-stationary signals.
///
/// # Examples
///
/// ```ignore
/// use scirs2_signal::adaptive::RlsFilter;
///
/// let mut rls = RlsFilter::new(4, 0.99, 1000.0).expect("Operation failed");
/// let (output, error, mse) = rls.adapt(1.0, 0.5).expect("Operation failed");
/// ```
#[derive(Debug, Clone)]
pub struct RlsFilter {
    /// Filter coefficients (weights)
    weights: Vec<f64>,
    /// Input buffer (delay line)
    buffer: Vec<f64>,
    /// Inverse correlation matrix P
    p_matrix: Vec<Vec<f64>>,
    /// Forgetting factor (typically 0.95 to 0.999)
    lambda: f64,
    /// Current buffer index
    buffer_index: usize,
}

impl RlsFilter {
    /// Create a new RLS filter
    ///
    /// # Arguments
    ///
    /// * `num_taps` - Number of filter taps
    /// * `lambda` - Forgetting factor (0 < lambda <= 1.0, typically 0.99)
    /// * `delta` - Initialization parameter for P matrix (typically 100-10000)
    ///
    /// # Returns
    ///
    /// * A new RLS filter instance
    pub fn new(num_taps: usize, lambda: f64, delta: f64) -> SignalResult<Self> {
        if num_taps == 0 {
            return Err(SignalError::ValueError(
                "Number of taps must be positive".to_string(),
            ));
        }

        if lambda <= 0.0 || lambda > 1.0 {
            return Err(SignalError::ValueError(
                "Forgetting factor must be in (0, 1]".to_string(),
            ));
        }

        if delta <= 0.0 {
            return Err(SignalError::ValueError(
                "Delta must be positive".to_string(),
            ));
        }

        // Initialize P matrix as delta * I (identity matrix)
        let mut p_matrix = vec![vec![0.0; num_taps]; num_taps];
        for (i, row) in p_matrix.iter_mut().enumerate().take(num_taps) {
            row[i] = delta;
        }

        Ok(RlsFilter {
            weights: vec![0.0; num_taps],
            buffer: vec![0.0; num_taps],
            p_matrix,
            lambda,
            buffer_index: 0,
        })
    }

    /// Process one sample through the adaptive filter
    ///
    /// # Arguments
    ///
    /// * `input` - Input sample
    /// * `desired` - Desired output sample
    ///
    /// # Returns
    ///
    /// * Tuple of (filter_output, error, mse_estimate)
    pub fn adapt(&mut self, input: f64, desired: f64) -> SignalResult<(f64, f64, f64)> {
        let num_taps = self.weights.len();

        // Update circular buffer with new input
        self.buffer[self.buffer_index] = input;
        self.buffer_index = (self.buffer_index + 1) % self.buffer.len();

        // Create input vector (in proper order)
        let mut input_vec = vec![0.0; num_taps];
        for (i, input_val) in input_vec.iter_mut().enumerate().take(num_taps) {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            *input_val = self.buffer[buffer_idx];
        }

        // Compute filter output
        let output = dot_product(&self.weights, &input_vec);

        // Compute error
        let error = desired - output;

        // RLS algorithm updates
        // 1. Compute k(n) = P(n-1) * x(n) / (lambda + x(n)^T * P(n-1) * x(n))
        let mut px = matrix_vector_multiply(&self.p_matrix, &input_vec);
        let xpx = dot_product(&input_vec, &px);
        let denominator = self.lambda + xpx;

        if denominator.abs() < 1e-10 {
            return Err(SignalError::ValueError(
                "RLS denominator too small, numerical instability".to_string(),
            ));
        }

        for px_val in &mut px {
            *px_val /= denominator;
        }
        let k = px; // k(n) = P(n-1) * x(n) / denominator

        // 2. Update weights: w(n) = w(n-1) + k(n) * e(n)
        for (weight, &k_val) in self.weights.iter_mut().zip(k.iter()) {
            *weight += k_val * error;
        }

        // 3. Update P matrix: P(n) = (P(n-1) - k(n) * x(n)^T * P(n-1)) / lambda
        let mut kx_outer = vec![vec![0.0; num_taps]; num_taps];
        for (kx_row, &k_val) in kx_outer.iter_mut().zip(k.iter()) {
            for (kx_elem, &input_val) in kx_row.iter_mut().zip(input_vec.iter()) {
                *kx_elem = k_val * input_val;
            }
        }

        // P = (P - k * x^T * P) / lambda
        let p_matrix_copy = self.p_matrix.clone();
        for (p_row, kx_row) in self.p_matrix.iter_mut().zip(kx_outer.iter()) {
            for (j, p_elem) in p_row.iter_mut().enumerate() {
                let kxp = dot_product(kx_row, &get_column(&p_matrix_copy, j));
                if self.lambda.abs() < f64::EPSILON {
                    return Err(SignalError::ValueError(
                        "Forgetting factor lambda is too close to zero".to_string(),
                    ));
                }
                *p_elem = (*p_elem - kxp) / self.lambda;
            }
        }

        let mse_estimate = error * error;

        Ok((output, error, mse_estimate))
    }

    /// Process a batch of samples
    pub fn adapt_batch(
        &mut self,
        inputs: &[f64],
        desired: &[f64],
    ) -> SignalResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        if inputs.len() != desired.len() {
            return Err(SignalError::ValueError(
                "Input and desired signals must have the same length".to_string(),
            ));
        }

        let mut outputs = Vec::with_capacity(inputs.len());
        let mut errors = Vec::with_capacity(inputs.len());
        let mut mse_estimates = Vec::with_capacity(inputs.len());

        for (&input, &des) in inputs.iter().zip(desired.iter()) {
            let (output, error, mse) = self.adapt(input, des)?;
            outputs.push(output);
            errors.push(error);
            mse_estimates.push(mse);
        }

        Ok((outputs, errors, mse_estimates))
    }

    /// Get current filter weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Reset the filter to initial state
    pub fn reset(&mut self, delta: f64) -> SignalResult<()> {
        if delta <= 0.0 {
            return Err(SignalError::ValueError(
                "Delta must be positive".to_string(),
            ));
        }

        let num_taps = self.weights.len();
        self.weights.fill(0.0);
        self.buffer.fill(0.0);
        self.buffer_index = 0;

        // Reinitialize P matrix
        for i in 0..num_taps {
            for j in 0..num_taps {
                self.p_matrix[i][j] = if i == j { delta } else { 0.0 };
            }
        }

        Ok(())
    }
}

/// Normalized LMS (NLMS) adaptive filter
///
/// The NLMS algorithm normalizes the step size by the input signal power,
/// providing better performance for signals with varying power levels.
#[derive(Debug, Clone)]
pub struct NlmsFilter {
    /// Filter coefficients (weights)
    weights: Vec<f64>,
    /// Input buffer (delay line)
    buffer: Vec<f64>,
    /// Step size (learning rate)
    step_size: f64,
    /// Regularization parameter to avoid division by zero
    epsilon: f64,
    /// Current buffer index
    buffer_index: usize,
}

impl NlmsFilter {
    /// Create a new NLMS filter
    ///
    /// # Arguments
    ///
    /// * `num_taps` - Number of filter taps
    /// * `step_size` - Learning rate (typically 0.1 to 2.0)
    /// * `epsilon` - Regularization parameter (typically 1e-6)
    pub fn new(num_taps: usize, step_size: f64, epsilon: f64) -> SignalResult<Self> {
        if num_taps == 0 {
            return Err(SignalError::ValueError(
                "Number of taps must be positive".to_string(),
            ));
        }

        if step_size <= 0.0 {
            return Err(SignalError::ValueError(
                "Step size must be positive".to_string(),
            ));
        }

        if epsilon <= 0.0 {
            return Err(SignalError::ValueError(
                "Epsilon must be positive".to_string(),
            ));
        }

        Ok(NlmsFilter {
            weights: vec![0.0; num_taps],
            buffer: vec![0.0; num_taps],
            step_size,
            epsilon,
            buffer_index: 0,
        })
    }

    /// Process one sample through the adaptive filter
    pub fn adapt(&mut self, input: f64, desired: f64) -> SignalResult<(f64, f64, f64)> {
        // Update circular buffer
        self.buffer[self.buffer_index] = input;
        self.buffer_index = (self.buffer_index + 1) % self.buffer.len();

        // Compute filter output
        let mut output = 0.0;
        for i in 0..self.weights.len() {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            output += self.weights[i] * self.buffer[buffer_idx];
        }

        // Compute error
        let error = desired - output;

        // Compute input power (norm squared)
        let input_power: f64 = self.buffer.iter().map(|&x| x * x).sum();
        let normalized_step = self.step_size / (input_power + self.epsilon);

        // Update weights using NLMS algorithm
        for i in 0..self.weights.len() {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            self.weights[i] += normalized_step * error * self.buffer[buffer_idx];
        }

        let mse_estimate = error * error;

        Ok((output, error, mse_estimate))
    }

    /// Process a batch of samples
    pub fn adapt_batch(
        &mut self,
        inputs: &[f64],
        desired: &[f64],
    ) -> SignalResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        if inputs.len() != desired.len() {
            return Err(SignalError::ValueError(
                "Input and desired signals must have the same length".to_string(),
            ));
        }

        let mut outputs = Vec::with_capacity(inputs.len());
        let mut errors = Vec::with_capacity(inputs.len());
        let mut mse_estimates = Vec::with_capacity(inputs.len());

        for (&input, &des) in inputs.iter().zip(desired.iter()) {
            let (output, error, mse) = self.adapt(input, des)?;
            outputs.push(output);
            errors.push(error);
            mse_estimates.push(mse);
        }

        Ok((outputs, errors, mse_estimates))
    }

    /// Get current filter weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Reset the filter
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
        self.buffer.fill(0.0);
        self.buffer_index = 0;
    }
}

/// Variable Step-Size LMS (VS-LMS) adaptive filter
///
/// The VS-LMS algorithm automatically adjusts the step size based on the gradient
/// estimation to achieve faster convergence and better steady-state performance.
#[derive(Debug, Clone)]
pub struct VsLmsFilter {
    /// Filter coefficients (weights)
    weights: Vec<f64>,
    /// Input buffer (delay line)
    buffer: Vec<f64>,
    /// Current step size
    step_size: f64,
    /// Initial step size
    initial_step_size: f64,
    /// Step size adaptation parameter
    alpha: f64,
    /// Gradient power estimate
    gradient_power: f64,
    /// Current buffer index
    buffer_index: usize,
    /// Previous error for gradient estimation
    prev_error: f64,
}

impl VsLmsFilter {
    /// Create a new VS-LMS filter
    ///
    /// # Arguments
    ///
    /// * `num_taps` - Number of filter taps
    /// * `initial_step_size` - Initial learning rate
    /// * `alpha` - Step size adaptation parameter (typically 0.01-0.1)
    pub fn new(num_taps: usize, initial_step_size: f64, alpha: f64) -> SignalResult<Self> {
        if num_taps == 0 {
            return Err(SignalError::ValueError(
                "Number of taps must be positive".to_string(),
            ));
        }

        if initial_step_size <= 0.0 {
            return Err(SignalError::ValueError(
                "Initial step size must be positive".to_string(),
            ));
        }

        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(SignalError::ValueError(
                "Alpha must be in (0, 1)".to_string(),
            ));
        }

        Ok(VsLmsFilter {
            weights: vec![0.0; num_taps],
            buffer: vec![0.0; num_taps],
            step_size: initial_step_size,
            initial_step_size,
            alpha,
            gradient_power: 1.0,
            buffer_index: 0,
            prev_error: 0.0,
        })
    }

    /// Process one sample through the adaptive filter
    pub fn adapt(&mut self, input: f64, desired: f64) -> SignalResult<(f64, f64, f64)> {
        // Update circular buffer
        self.buffer[self.buffer_index] = input;
        self.buffer_index = (self.buffer_index + 1) % self.buffer.len();

        // Compute filter output
        let mut output = 0.0;
        for i in 0..self.weights.len() {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            output += self.weights[i] * self.buffer[buffer_idx];
        }

        // Compute error
        let error = desired - output;

        // Estimate gradient correlation
        let gradient_correlation = error * self.prev_error;

        // Update gradient power estimate
        self.gradient_power =
            (1.0 - self.alpha) * self.gradient_power + self.alpha * gradient_correlation.abs();

        // Adapt step size
        if gradient_correlation > 0.0 {
            self.step_size *= 1.05;
        } else {
            self.step_size *= 0.95;
        }

        // Bound step size
        self.step_size = self
            .step_size
            .clamp(self.initial_step_size * 0.01, self.initial_step_size * 10.0);

        // Update weights using current step size
        for i in 0..self.weights.len() {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            self.weights[i] += self.step_size * error * self.buffer[buffer_idx];
        }

        self.prev_error = error;
        let mse_estimate = error * error;

        Ok((output, error, mse_estimate))
    }

    /// Get current filter weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get current step size
    pub fn current_step_size(&self) -> f64 {
        self.step_size
    }

    /// Reset the filter
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
        self.buffer.fill(0.0);
        self.buffer_index = 0;
        self.step_size = self.initial_step_size;
        self.gradient_power = 1.0;
        self.prev_error = 0.0;
    }
}

/// Affine Projection Algorithm (APA) adaptive filter
///
/// APA uses multiple previous input vectors to accelerate convergence,
/// especially effective for highly correlated input signals.
#[derive(Debug, Clone)]
pub struct ApaFilter {
    /// Filter coefficients (weights)
    weights: Vec<f64>,
    /// Input matrix (K x num_taps)
    input_matrix: Vec<Vec<f64>>,
    /// Projection order K
    projection_order: usize,
    /// Step size
    step_size: f64,
    /// Regularization parameter
    delta: f64,
    /// Current matrix row index
    current_row: usize,
}

impl ApaFilter {
    /// Create a new APA filter
    ///
    /// # Arguments
    ///
    /// * `num_taps` - Number of filter taps
    /// * `projection_order` - Projection order K (typically 2-10)
    /// * `step_size` - Learning rate
    /// * `delta` - Regularization parameter
    pub fn new(
        num_taps: usize,
        projection_order: usize,
        step_size: f64,
        delta: f64,
    ) -> SignalResult<Self> {
        if num_taps == 0 {
            return Err(SignalError::ValueError(
                "Number of taps must be positive".to_string(),
            ));
        }

        if projection_order == 0 {
            return Err(SignalError::ValueError(
                "Projection order must be positive".to_string(),
            ));
        }

        if step_size <= 0.0 {
            return Err(SignalError::ValueError(
                "Step size must be positive".to_string(),
            ));
        }

        Ok(ApaFilter {
            weights: vec![0.0; num_taps],
            input_matrix: vec![vec![0.0; num_taps]; projection_order],
            projection_order,
            step_size,
            delta,
            current_row: 0,
        })
    }

    /// Process one sample through the adaptive filter
    pub fn adapt(&mut self, input: &[f64], desired: f64) -> SignalResult<(f64, f64, f64)> {
        if input.len() != self.weights.len() {
            return Err(SignalError::ValueError(
                "Input length must match number of taps".to_string(),
            ));
        }

        // Update input matrix
        self.input_matrix[self.current_row] = input.to_vec();
        self.current_row = (self.current_row + 1) % self.projection_order;

        // Compute filter output
        let output = dot_product(&self.weights, input);
        let error = desired - output;

        // Compute all errors for the projection order
        let mut errors = vec![0.0; self.projection_order];
        let mut outputs = vec![0.0; self.projection_order];

        for k in 0..self.projection_order {
            outputs[k] = dot_product(&self.weights, &self.input_matrix[k]);
            errors[k] =
                if k == (self.current_row + self.projection_order - 1) % self.projection_order {
                    error
                } else {
                    -outputs[k]
                };
        }

        // Compute input correlation matrix X^T * X
        let mut correlation_matrix = vec![vec![0.0; self.projection_order]; self.projection_order];
        for (i, row) in correlation_matrix
            .iter_mut()
            .enumerate()
            .take(self.projection_order)
        {
            for (j, cell) in row.iter_mut().enumerate().take(self.projection_order) {
                *cell = dot_product(&self.input_matrix[i], &self.input_matrix[j]);
                if i == j {
                    *cell += self.delta; // Regularization
                }
            }
        }

        // Solve for step size vector: alpha = (X^T * X + delta * I)^(-1) * e
        let step_vector = solve_linear_system_small(&correlation_matrix, &errors)?;

        // Update weights: w = w + mu * X^T * alpha
        for (k, input_row) in self
            .input_matrix
            .iter()
            .enumerate()
            .take(self.projection_order)
        {
            for (i, weight) in self.weights.iter_mut().enumerate() {
                *weight += self.step_size * step_vector[k] * input_row[i];
            }
        }

        let mse_estimate = error * error;
        Ok((output, error, mse_estimate))
    }

    /// Get current filter weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Reset the filter
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
        for row in &mut self.input_matrix {
            row.fill(0.0);
        }
        self.current_row = 0;
    }
}

/// Frequency Domain LMS (FDLMS) adaptive filter
///
/// FDLMS operates in the frequency domain for computational efficiency
/// with long filters, using overlap-save processing.
pub struct FdlmsFilter {
    /// Filter length
    filter_length: usize,
    /// Block size (typically 2 * filter_length)
    block_size: usize,
    /// Filter coefficients in frequency domain
    freq_weights: Vec<Complex64>,
    /// Input buffer for overlap-save
    input_buffer: VecDeque<f64>,
    /// Step size
    step_size: f64,
    /// Leakage factor for weight constraint
    leakage: f64,
}

impl FdlmsFilter {
    /// Create a new FDLMS filter
    ///
    /// # Arguments
    ///
    /// * `filter_length` - Number of filter taps
    /// * `step_size` - Learning rate
    /// * `leakage` - Leakage factor (0.999-1.0)
    pub fn new(filter_length: usize, step_size: f64, leakage: f64) -> SignalResult<Self> {
        if filter_length == 0 {
            return Err(SignalError::ValueError(
                "Filter length must be positive".to_string(),
            ));
        }

        if step_size <= 0.0 {
            return Err(SignalError::ValueError(
                "Step size must be positive".to_string(),
            ));
        }

        let block_size = 2 * filter_length;

        Ok(FdlmsFilter {
            filter_length,
            block_size,
            freq_weights: vec![Complex64::new(0.0, 0.0); block_size],
            input_buffer: VecDeque::with_capacity(block_size),
            step_size,
            leakage,
        })
    }

    /// Process a block of samples
    pub fn adapt_block(
        &mut self,
        inputs: &[f64],
        desired: &[f64],
    ) -> SignalResult<(Vec<f64>, Vec<f64>)> {
        if inputs.len() != desired.len() {
            return Err(SignalError::ValueError(
                "Input and desired signals must have same length".to_string(),
            ));
        }

        let mut outputs = Vec::with_capacity(inputs.len());
        let mut errors = Vec::with_capacity(inputs.len());

        // Process in blocks
        for (input_chunk, desired_chunk) in inputs
            .chunks(self.filter_length)
            .zip(desired.chunks(self.filter_length))
        {
            // Fill input buffer
            for &sample in input_chunk {
                if self.input_buffer.len() >= self.block_size {
                    self.input_buffer.pop_front();
                }
                self.input_buffer.push_back(sample);
            }

            if self.input_buffer.len() == self.block_size {
                let (block_outputs, block_errors) =
                    self.process_block(input_chunk, desired_chunk)?;
                outputs.extend(block_outputs);
                errors.extend(block_errors);
            }
        }

        Ok((outputs, errors))
    }

    fn process_block(
        &mut self,
        inputs: &[f64],
        desired: &[f64],
    ) -> SignalResult<(Vec<f64>, Vec<f64>)> {
        // Convert input buffer to f64 vec for FFT
        let input_f64: Vec<f64> = self.input_buffer.iter().copied().collect();

        // FFT of input using scirs2_fft
        let input_freq = scirs2_fft::fft(&input_f64, Some(self.block_size))
            .map_err(|e| SignalError::ComputationError(format!("FFT failed: {}", e)))?;

        // Frequency domain filtering
        let freq_output: Vec<Complex64> = input_freq
            .iter()
            .zip(self.freq_weights.iter())
            .map(|(&x, &w)| x * w)
            .collect();

        // IFFT to get time domain output using scirs2_fft
        let time_output = scirs2_fft::ifft(&freq_output, Some(self.block_size))
            .map_err(|e| SignalError::ComputationError(format!("IFFT failed: {}", e)))?;

        // Extract outputs (last half due to overlap-save)
        let outputs: Vec<f64> = time_output[self.filter_length..]
            .iter()
            .take(inputs.len())
            .map(|c| c.re)
            .collect();

        // Compute errors
        let errors: Vec<f64> = outputs
            .iter()
            .zip(desired.iter())
            .map(|(&out, &des)| des - out)
            .collect();

        // Update weights in frequency domain
        self.update_weights(&input_freq, &errors)?;

        Ok((outputs, errors))
    }

    fn update_weights(&mut self, freq_input: &[Complex64], errors: &[f64]) -> SignalResult<()> {
        // Create error signal in frequency domain
        let mut error_padded_f64 = vec![0.0f64; self.block_size];
        for (i, &err) in errors.iter().enumerate() {
            if self.filter_length + i < self.block_size {
                error_padded_f64[self.filter_length + i] = err;
            }
        }

        // FFT of error using scirs2_fft
        let error_freq = scirs2_fft::fft(&error_padded_f64, Some(self.block_size))
            .map_err(|e| SignalError::ComputationError(format!("FFT failed: {}", e)))?;

        // Update frequency domain weights
        for i in 0..self.block_size {
            let gradient = freq_input[i].conj() * error_freq[i];
            self.freq_weights[i] = self.leakage * self.freq_weights[i]
                + Complex64::new(self.step_size, 0.0) * gradient;
        }

        Ok(())
    }

    /// Get current filter weights (time domain)
    pub fn weights(&self) -> Vec<f64> {
        // Convert frequency domain weights to time domain using scirs2_fft
        match scirs2_fft::ifft(&self.freq_weights, Some(self.block_size)) {
            Ok(time_weights) => time_weights[..self.filter_length]
                .iter()
                .map(|c| c.re)
                .collect(),
            Err(_) => vec![0.0; self.filter_length],
        }
    }

    /// Reset the filter
    pub fn reset(&mut self) {
        self.freq_weights.fill(Complex64::new(0.0, 0.0));
        self.input_buffer.clear();
    }
}

/// Least Mean Fourth (LMF) robust adaptive filter
///
/// LMF uses fourth-order moments instead of second-order, providing
/// better performance in the presence of impulsive noise.
#[derive(Debug, Clone)]
pub struct LmfFilter {
    /// Filter coefficients (weights)
    weights: Vec<f64>,
    /// Input buffer (delay line)
    buffer: Vec<f64>,
    /// Step size
    step_size: f64,
    /// Current buffer index
    buffer_index: usize,
}

impl LmfFilter {
    /// Create a new LMF filter
    pub fn new(num_taps: usize, step_size: f64) -> SignalResult<Self> {
        if num_taps == 0 {
            return Err(SignalError::ValueError(
                "Number of taps must be positive".to_string(),
            ));
        }

        if step_size <= 0.0 {
            return Err(SignalError::ValueError(
                "Step size must be positive".to_string(),
            ));
        }

        Ok(LmfFilter {
            weights: vec![0.0; num_taps],
            buffer: vec![0.0; num_taps],
            step_size,
            buffer_index: 0,
        })
    }

    /// Process one sample through the adaptive filter
    pub fn adapt(&mut self, input: f64, desired: f64) -> SignalResult<(f64, f64, f64)> {
        // Update circular buffer
        self.buffer[self.buffer_index] = input;
        self.buffer_index = (self.buffer_index + 1) % self.buffer.len();

        // Compute filter output
        let mut output = 0.0;
        for i in 0..self.weights.len() {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            output += self.weights[i] * self.buffer[buffer_idx];
        }

        // Compute error
        let error = desired - output;

        // LMF weight update: w(n+1) = w(n) + mu * e^3(n) * x(n)
        let error_cubed = error * error * error;
        for i in 0..self.weights.len() {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            self.weights[i] += self.step_size * error_cubed * self.buffer[buffer_idx];
        }

        let mse_estimate = error * error;
        Ok((output, error, mse_estimate))
    }

    /// Get current filter weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Reset the filter
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
        self.buffer.fill(0.0);
        self.buffer_index = 0;
    }
}

/// Set-Membership LMS (SM-LMS) adaptive filter
///
/// SM-LMS updates weights only when the output error exceeds a threshold,
/// reducing computational complexity and providing robustness.
#[derive(Debug, Clone)]
pub struct SmLmsFilter {
    /// Filter coefficients (weights)
    weights: Vec<f64>,
    /// Input buffer (delay line)
    buffer: Vec<f64>,
    /// Step size
    step_size: f64,
    /// Error bound threshold
    error_bound: f64,
    /// Current buffer index
    buffer_index: usize,
    /// Update counter
    update_count: u64,
    /// Total sample counter
    sample_count: u64,
}

impl SmLmsFilter {
    /// Create a new SM-LMS filter
    ///
    /// # Arguments
    ///
    /// * `num_taps` - Number of filter taps
    /// * `step_size` - Learning rate
    /// * `error_bound` - Error threshold for updates
    pub fn new(num_taps: usize, step_size: f64, error_bound: f64) -> SignalResult<Self> {
        if num_taps == 0 {
            return Err(SignalError::ValueError(
                "Number of taps must be positive".to_string(),
            ));
        }

        if step_size <= 0.0 {
            return Err(SignalError::ValueError(
                "Step size must be positive".to_string(),
            ));
        }

        if error_bound <= 0.0 {
            return Err(SignalError::ValueError(
                "Error bound must be positive".to_string(),
            ));
        }

        Ok(SmLmsFilter {
            weights: vec![0.0; num_taps],
            buffer: vec![0.0; num_taps],
            step_size,
            error_bound,
            buffer_index: 0,
            update_count: 0,
            sample_count: 0,
        })
    }

    /// Process one sample through the adaptive filter
    pub fn adapt(&mut self, input: f64, desired: f64) -> SignalResult<(f64, f64, f64)> {
        // Update circular buffer
        self.buffer[self.buffer_index] = input;
        self.buffer_index = (self.buffer_index + 1) % self.buffer.len();

        // Compute filter output
        let mut output = 0.0;
        for i in 0..self.weights.len() {
            let buffer_idx = (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
            output += self.weights[i] * self.buffer[buffer_idx];
        }

        // Compute error
        let error = desired - output;
        self.sample_count += 1;

        // Update weights only if error exceeds bound
        if error.abs() > self.error_bound {
            // Compute input power for normalization
            let input_power: f64 = self.buffer.iter().map(|&x| x * x).sum();
            let normalization = if input_power > 1e-12 {
                input_power + 1e-12
            } else {
                1e-12
            };

            let normalized_step = self.step_size / normalization;

            for i in 0..self.weights.len() {
                let buffer_idx =
                    (self.buffer_index + self.buffer.len() - 1 - i) % self.buffer.len();
                self.weights[i] += normalized_step * error * self.buffer[buffer_idx];
            }

            self.update_count += 1;
        }

        let mse_estimate = error * error;
        Ok((output, error, mse_estimate))
    }

    /// Get current filter weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get update statistics
    pub fn update_statistics(&self) -> (u64, u64, f64) {
        let update_ratio = if self.sample_count > 0 {
            self.update_count as f64 / self.sample_count as f64
        } else {
            0.0
        };
        (self.update_count, self.sample_count, update_ratio)
    }

    /// Reset the filter
    pub fn reset(&mut self) {
        self.weights.fill(0.0);
        self.buffer.fill(0.0);
        self.buffer_index = 0;
        self.update_count = 0;
        self.sample_count = 0;
    }
}

/// Unified adaptive filter interface wrapping all algorithms
///
/// Provides a consistent API for all adaptive filter types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdaptiveMethod {
    /// Least Mean Squares
    Lms,
    /// Normalized LMS
    Nlms,
    /// Recursive Least Squares
    Rls,
    /// Variable Step-Size LMS
    VsLms,
    /// Least Mean Fourth
    Lmf,
    /// Set-Membership LMS
    SmLms,
}

/// Configuration for the unified adaptive filter
#[derive(Debug, Clone)]
pub struct AdaptiveFilterConfig {
    /// Filter order (number of taps)
    pub order: usize,
    /// Adaptation method
    pub method: AdaptiveMethod,
    /// Step size / learning rate
    pub step_size: f64,
    /// Forgetting factor (for RLS only, default 0.99)
    pub forgetting_factor: f64,
    /// Regularization (for NLMS epsilon, APA delta, RLS delta)
    pub regularization: f64,
    /// Error bound (for SM-LMS)
    pub error_bound: f64,
    /// Alpha adaptation parameter (for VS-LMS)
    pub alpha: f64,
}

impl Default for AdaptiveFilterConfig {
    fn default() -> Self {
        Self {
            order: 8,
            method: AdaptiveMethod::Lms,
            step_size: 0.01,
            forgetting_factor: 0.99,
            regularization: 1e-6,
            error_bound: 0.1,
            alpha: 0.05,
        }
    }
}

/// Unified adaptive filter that wraps all methods
pub struct AdaptiveFilter {
    /// Internal filter state
    inner: AdaptiveFilterInner,
}

enum AdaptiveFilterInner {
    Lms(LmsFilter),
    Nlms(NlmsFilter),
    Rls(RlsFilter),
    VsLms(VsLmsFilter),
    Lmf(LmfFilter),
    SmLms(SmLmsFilter),
}

impl AdaptiveFilter {
    /// Create a new adaptive filter with the given order and method
    pub fn new(order: usize, method: AdaptiveMethod) -> SignalResult<Self> {
        let config = AdaptiveFilterConfig {
            order,
            method,
            ..Default::default()
        };
        Self::from_config(&config)
    }

    /// Create a new adaptive filter from configuration
    pub fn from_config(config: &AdaptiveFilterConfig) -> SignalResult<Self> {
        let inner = match config.method {
            AdaptiveMethod::Lms => {
                AdaptiveFilterInner::Lms(LmsFilter::new(config.order, config.step_size, 0.0)?)
            }
            AdaptiveMethod::Nlms => AdaptiveFilterInner::Nlms(NlmsFilter::new(
                config.order,
                config.step_size,
                config.regularization,
            )?),
            AdaptiveMethod::Rls => AdaptiveFilterInner::Rls(RlsFilter::new(
                config.order,
                config.forgetting_factor,
                1.0 / config.regularization,
            )?),
            AdaptiveMethod::VsLms => AdaptiveFilterInner::VsLms(VsLmsFilter::new(
                config.order,
                config.step_size,
                config.alpha,
            )?),
            AdaptiveMethod::Lmf => {
                AdaptiveFilterInner::Lmf(LmfFilter::new(config.order, config.step_size)?)
            }
            AdaptiveMethod::SmLms => AdaptiveFilterInner::SmLms(SmLmsFilter::new(
                config.order,
                config.step_size,
                config.error_bound,
            )?),
        };
        Ok(Self { inner })
    }

    /// Process one sample through the filter
    ///
    /// Returns (output, error, mse_estimate)
    pub fn filter(&mut self, input: f64, desired: f64) -> SignalResult<(f64, f64, f64)> {
        match &mut self.inner {
            AdaptiveFilterInner::Lms(f) => f.adapt(input, desired),
            AdaptiveFilterInner::Nlms(f) => f.adapt(input, desired),
            AdaptiveFilterInner::Rls(f) => f.adapt(input, desired),
            AdaptiveFilterInner::VsLms(f) => f.adapt(input, desired),
            AdaptiveFilterInner::Lmf(f) => f.adapt(input, desired),
            AdaptiveFilterInner::SmLms(f) => f.adapt(input, desired),
        }
    }

    /// Process a batch of samples through the filter
    ///
    /// Returns (outputs, errors, mse_estimates)
    pub fn filter_batch(
        &mut self,
        inputs: &[f64],
        desired: &[f64],
    ) -> SignalResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        if inputs.len() != desired.len() {
            return Err(SignalError::ValueError(
                "Input and desired signals must have the same length".to_string(),
            ));
        }

        let mut outputs = Vec::with_capacity(inputs.len());
        let mut errors = Vec::with_capacity(inputs.len());
        let mut mse_estimates = Vec::with_capacity(inputs.len());

        for (&inp, &des) in inputs.iter().zip(desired.iter()) {
            let (output, error, mse) = self.filter(inp, des)?;
            outputs.push(output);
            errors.push(error);
            mse_estimates.push(mse);
        }

        Ok((outputs, errors, mse_estimates))
    }

    /// Get current filter weights
    pub fn weights(&self) -> Vec<f64> {
        match &self.inner {
            AdaptiveFilterInner::Lms(f) => f.weights().to_vec(),
            AdaptiveFilterInner::Nlms(f) => f.weights().to_vec(),
            AdaptiveFilterInner::Rls(f) => f.weights().to_vec(),
            AdaptiveFilterInner::VsLms(f) => f.weights().to_vec(),
            AdaptiveFilterInner::Lmf(f) => f.weights().to_vec(),
            AdaptiveFilterInner::SmLms(f) => f.weights().to_vec(),
        }
    }
}

// Helper functions for matrix operations

/// Compute dot product of two vectors
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Multiply matrix by vector
fn matrix_vector_multiply(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; matrix.len()];
    for i in 0..matrix.len() {
        result[i] = dot_product(&matrix[i], vector);
    }
    result
}

/// Get column from matrix
fn get_column(matrix: &[Vec<f64>], col: usize) -> Vec<f64> {
    matrix.iter().map(|row| row[col]).collect()
}

/// Solve small linear system using Gaussian elimination (for APA)
fn solve_linear_system_small(matrix: &[Vec<f64>], rhs: &[f64]) -> SignalResult<Vec<f64>> {
    let n = matrix.len();
    if n != rhs.len() {
        return Err(SignalError::ValueError(
            "Matrix and RHS dimensions must match".to_string(),
        ));
    }

    // Create augmented matrix
    let mut aug_matrix: Vec<Vec<f64>> = matrix
        .iter()
        .zip(rhs.iter())
        .map(|(row, &b)| {
            let mut aug_row = row.clone();
            aug_row.push(b);
            aug_row
        })
        .collect();

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug_matrix[k][i].abs() > aug_matrix[max_row][i].abs() {
                max_row = k;
            }
        }

        // Swap rows
        aug_matrix.swap(i, max_row);

        // Check for singular matrix
        if aug_matrix[i][i].abs() < 1e-12 {
            return Err(SignalError::ComputationError(
                "Matrix is singular or near-singular".to_string(),
            ));
        }

        // Eliminate column
        for k in (i + 1)..n {
            if aug_matrix[i][i].abs() < f64::EPSILON {
                return Err(SignalError::ValueError(format!(
                    "Singular matrix detected at row {}",
                    i
                )));
            }
            let factor = aug_matrix[k][i] / aug_matrix[i][i];
            for j in i..=n {
                aug_matrix[k][j] -= factor * aug_matrix[i][j];
            }
        }
    }

    // Back substitution
    let mut solution = vec![0.0; n];
    for i in (0..n).rev() {
        solution[i] = aug_matrix[i][n];
        for j in (i + 1)..n {
            solution[i] -= aug_matrix[i][j] * solution[j];
        }
        solution[i] /= aug_matrix[i][i];
    }

    Ok(solution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lms_creation() {
        let lms = LmsFilter::new(4, 0.01, 0.0)
            .expect("LMS filter creation should succeed with valid parameters");
        assert_eq!(lms.weights().len(), 4);
        assert_eq!(lms.buffer().len(), 4);

        // Test error conditions
        assert!(LmsFilter::new(0, 0.01, 0.0).is_err());
        assert!(LmsFilter::new(4, 0.0, 0.0).is_err());
    }

    #[test]
    fn test_lms_adapt() {
        let mut lms = LmsFilter::new(2, 0.1, 0.0)
            .expect("LMS filter creation should succeed with valid parameters");

        // Test single adaptation
        let (output, error, _mse) = lms.adapt(1.0, 0.5).expect("Operation failed");

        // Initially weights are zero, so output should be zero
        assert_relative_eq!(output, 0.0, epsilon = 1e-10);
        assert_relative_eq!(error, 0.5, epsilon = 1e-10);

        // Weights should be updated
        assert!(!lms.weights().iter().all(|&w| w == 0.0));
    }

    #[test]
    fn test_lms_batch() {
        let mut lms = LmsFilter::new(2, 0.05, 0.0).expect("Operation failed");

        let inputs = vec![1.0, 0.5, -0.3, 0.8];
        let desired = vec![0.1, 0.2, 0.3, 0.4];

        let (_outputs, errors, _mse) = lms
            .adapt_batch(&inputs, &desired)
            .expect("Operation failed");

        assert_eq!(_outputs.len(), 4);
        assert_eq!(errors.len(), 4);

        // Error should be bounded
        assert!(errors.iter().all(|&e| e.abs() < 10.0));
    }

    #[test]
    fn test_lms_system_identification() {
        let mut lms = LmsFilter::new(3, 0.01, 0.0).expect("Operation failed");

        // Target system: h = [0.5, -0.3, 0.2]
        let target_system = [0.5, -0.3, 0.2];

        // Generate training data
        let mut inputs = Vec::new();
        let mut desired = Vec::new();

        for i in 0..100 {
            let input = (i as f64 * 0.1).sin();
            inputs.push(input);

            let output = if i >= 2 {
                target_system[0] * inputs[i]
                    + target_system[1] * inputs[i - 1]
                    + target_system[2] * inputs[i - 2]
            } else {
                0.0
            };
            desired.push(output);
        }

        let (_outputs, _errors, _mse) = lms
            .adapt_batch(&inputs, &desired)
            .expect("Operation failed");

        // Check if weights converged towards target (approximately)
        for (i, &target_weight) in target_system.iter().enumerate() {
            let weight_diff = (lms.weights()[i] - target_weight).abs();
            assert!(
                weight_diff < 1.0,
                "Weight {} difference {} too large",
                i,
                weight_diff
            );
        }
    }

    #[test]
    fn test_rls_creation() {
        let rls = RlsFilter::new(3, 0.99, 100.0).expect("Operation failed");
        assert_eq!(rls.weights().len(), 3);

        // Test error conditions
        assert!(RlsFilter::new(0, 0.99, 100.0).is_err());
        assert!(RlsFilter::new(3, 0.0, 100.0).is_err());
        assert!(RlsFilter::new(3, 1.1, 100.0).is_err());
        assert!(RlsFilter::new(3, 0.99, 0.0).is_err());
    }

    #[test]
    fn test_rls_adapt() {
        let mut rls = RlsFilter::new(2, 0.99, 100.0).expect("Operation failed");

        let (output, error, _mse) = rls.adapt(1.0, 0.5).expect("Operation failed");

        assert_relative_eq!(output, 0.0, epsilon = 1e-10);
        assert_relative_eq!(error, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_nlms_creation() {
        let nlms = NlmsFilter::new(4, 0.5, 1e-6).expect("Operation failed");
        assert_eq!(nlms.weights().len(), 4);

        assert!(NlmsFilter::new(0, 0.5, 1e-6).is_err());
        assert!(NlmsFilter::new(4, 0.0, 1e-6).is_err());
        assert!(NlmsFilter::new(4, 0.5, 0.0).is_err());
    }

    #[test]
    fn test_nlms_adapt() {
        let mut nlms = NlmsFilter::new(2, 0.5, 1e-6).expect("Operation failed");

        let (output, error, _mse) = nlms.adapt(1.0, 0.3).expect("Operation failed");

        assert_relative_eq!(output, 0.0, epsilon = 1e-10);
        assert_relative_eq!(error, 0.3, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_operations() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert_relative_eq!(result, 32.0, epsilon = 1e-10);

        let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let vector = vec![5.0, 6.0];
        let result = matrix_vector_multiply(&matrix, &vector);
        assert_eq!(result.len(), 2);
        assert_relative_eq!(result[0], 17.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 39.0, epsilon = 1e-10);

        let column = get_column(&matrix, 0);
        assert_eq!(column, vec![1.0, 3.0]);
    }

    #[test]
    fn test_convergence_comparison() {
        let target_system = [0.8, -0.4];
        let num_samples = 50;

        let mut lms = LmsFilter::new(2, 0.05, 0.0).expect("Operation failed");
        let mut rls = RlsFilter::new(2, 0.99, 100.0).expect("Operation failed");

        let mut lms_errors = Vec::new();
        let mut rls_errors = Vec::new();

        for i in 0..num_samples {
            let input = (i as f64 * 0.2).sin();
            let desired = if i >= 1 {
                target_system[0] * input + target_system[1] * (((i - 1) as f64) * 0.2).sin()
            } else {
                target_system[0] * input
            };

            let (_out_lms, err_lms, _) = lms.adapt(input, desired).expect("Operation failed");
            let (_out_rls, err_rls, _) = rls.adapt(input, desired).expect("Operation failed");

            lms_errors.push(err_lms.abs());
            rls_errors.push(err_rls.abs());
        }

        let lms_final_error = lms_errors.iter().rev().take(10).sum::<f64>() / 10.0;
        let rls_final_error = rls_errors.iter().rev().take(10).sum::<f64>() / 10.0;

        assert!(
            lms_final_error < 2.0,
            "LMS final error too large: {}",
            lms_final_error
        );
        assert!(
            rls_final_error < 2.0,
            "RLS final error too large: {}",
            rls_final_error
        );
    }

    #[test]
    fn test_vs_lms_creation() {
        let vs_lms = VsLmsFilter::new(4, 0.01, 0.05).expect("Operation failed");
        assert_eq!(vs_lms.weights().len(), 4);
        assert_relative_eq!(vs_lms.current_step_size(), 0.01, epsilon = 1e-10);

        assert!(VsLmsFilter::new(0, 0.01, 0.05).is_err());
        assert!(VsLmsFilter::new(4, 0.0, 0.05).is_err());
        assert!(VsLmsFilter::new(4, 0.01, 0.0).is_err());
        assert!(VsLmsFilter::new(4, 0.01, 1.0).is_err());
    }

    #[test]
    fn test_vs_lms_adapt() {
        let mut vs_lms = VsLmsFilter::new(2, 0.1, 0.01).expect("Operation failed");

        let (output, error, _mse) = vs_lms.adapt(1.0, 0.5).expect("Operation failed");

        assert_relative_eq!(output, 0.0, epsilon = 1e-10);
        assert_relative_eq!(error, 0.5, epsilon = 1e-10);

        let initial_step = vs_lms.current_step_size();

        for _ in 0..10 {
            vs_lms.adapt(1.0, 0.5).expect("Operation failed");
        }

        assert_ne!(vs_lms.current_step_size(), initial_step);
    }

    #[test]
    fn test_apa_creation() {
        let apa = ApaFilter::new(4, 3, 0.1, 0.01).expect("Operation failed");
        assert_eq!(apa.weights().len(), 4);

        assert!(ApaFilter::new(0, 3, 0.1, 0.01).is_err());
        assert!(ApaFilter::new(4, 0, 0.1, 0.01).is_err());
        assert!(ApaFilter::new(4, 3, 0.0, 0.01).is_err());
    }

    #[test]
    fn test_apa_adapt() {
        let mut apa = ApaFilter::new(2, 2, 0.1, 0.01).expect("Operation failed");
        let input = vec![1.0, 0.5];

        let (output, error, _mse) = apa.adapt(&input, 0.3).expect("Operation failed");

        assert_relative_eq!(output, 0.0, epsilon = 1e-10);
        assert_relative_eq!(error, 0.3, epsilon = 1e-10);

        // Test wrong input size
        let wrong_input = vec![1.0];
        assert!(apa.adapt(&wrong_input, 0.3).is_err());
    }

    #[test]
    fn test_fdlms_creation() {
        let fdlms = FdlmsFilter::new(8, 0.01, 0.999).expect("Operation failed");
        assert_eq!(fdlms.weights().len(), 8);

        assert!(FdlmsFilter::new(0, 0.01, 0.999).is_err());
        assert!(FdlmsFilter::new(8, 0.0, 0.999).is_err());
    }

    #[test]
    fn test_fdlms_adapt_block() {
        let mut fdlms = FdlmsFilter::new(4, 0.01, 0.999).expect("Operation failed");
        let inputs = vec![1.0, 0.5, -0.3, 0.8, 0.2, -0.1, 0.6, -0.4];
        let desired = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        let (outputs, errors) = fdlms
            .adapt_block(&inputs, &desired)
            .expect("Operation failed");

        assert!(!outputs.is_empty());
        assert!(!errors.is_empty());
        assert_eq!(outputs.len(), errors.len());

        // Test wrong input size
        let wrong_desired = vec![0.1, 0.2];
        assert!(fdlms.adapt_block(&inputs, &wrong_desired).is_err());
    }

    #[test]
    fn test_lmf_creation() {
        let lmf = LmfFilter::new(4, 0.01).expect("Operation failed");
        assert_eq!(lmf.weights().len(), 4);

        assert!(LmfFilter::new(0, 0.01).is_err());
        assert!(LmfFilter::new(4, 0.0).is_err());
    }

    #[test]
    fn test_lmf_adapt() {
        let mut lmf = LmfFilter::new(2, 0.01).expect("Operation failed");

        let (output, error, _mse) = lmf.adapt(1.0, 0.5).expect("Operation failed");

        assert_relative_eq!(output, 0.0, epsilon = 1e-10);
        assert_relative_eq!(error, 0.5, epsilon = 1e-10);

        for _ in 0..50 {
            lmf.adapt(1.0, 0.5).expect("Operation failed");
        }

        let (final_output, _, _) = lmf.adapt(1.0, 0.5).expect("Operation failed");
        assert!(final_output.abs() > 1e-6);
    }

    #[test]
    fn test_sm_lms_creation() {
        let sm_lms = SmLmsFilter::new(4, 0.1, 0.1).expect("Operation failed");
        assert_eq!(sm_lms.weights().len(), 4);

        assert!(SmLmsFilter::new(0, 0.1, 0.1).is_err());
        assert!(SmLmsFilter::new(4, 0.0, 0.1).is_err());
        assert!(SmLmsFilter::new(4, 0.1, 0.0).is_err());
    }

    #[test]
    fn test_sm_lms_adapt() {
        let mut sm_lms = SmLmsFilter::new(2, 0.1, 0.05).expect("Operation failed");

        // Small error - should not trigger update
        let (_output, error, _mse) = sm_lms.adapt(1.0, 0.01).expect("Operation failed");
        assert_relative_eq!(_output, 0.0, epsilon = 1e-10);
        assert_relative_eq!(error, 0.01, epsilon = 1e-10);

        let (update_count, sample_count, _) = sm_lms.update_statistics();
        assert_eq!(update_count, 0);
        assert_eq!(sample_count, 1);

        // Large error - should trigger update
        sm_lms.adapt(1.0, 0.5).expect("Operation failed");
        let (update_count, sample_count, update_ratio) = sm_lms.update_statistics();
        assert_eq!(update_count, 1);
        assert_eq!(sample_count, 2);
        assert_relative_eq!(update_ratio, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_sm_lms_selective_updates() {
        let mut sm_lms = SmLmsFilter::new(3, 0.1, 0.1).expect("Operation failed");

        let inputs = [1.0, 0.5, -0.3, 0.8, 0.2];
        let target_errors = [0.05, 0.15, 0.08, 0.2, 0.03];

        for (&input, &target_error) in inputs.iter().zip(target_errors.iter()) {
            sm_lms.adapt(input, target_error).expect("Operation failed");
        }

        let (update_count, sample_count, _) = sm_lms.update_statistics();
        assert_eq!(sample_count, 5);
        assert!(update_count < sample_count);
        assert!(update_count > 0);
    }

    #[test]
    fn test_advanced_algorithm_convergence_comparison() {
        let target_system = [0.6, -0.4, 0.2];
        let num_samples = 100;

        let mut lms = LmsFilter::new(3, 0.01, 0.0).expect("Operation failed");
        let mut vs_lms = VsLmsFilter::new(3, 0.01, 0.05).expect("Operation failed");
        let mut nlms = NlmsFilter::new(3, 0.5, 1e-6).expect("Operation failed");
        let mut lmf = LmfFilter::new(3, 0.001).expect("Operation failed");

        let mut lms_errors = Vec::new();
        let mut vs_lms_errors = Vec::new();
        let mut nlms_errors = Vec::new();
        let mut lmf_errors = Vec::new();

        for i in 0..num_samples {
            let input = (i as f64 * 0.1).sin();
            let desired = if i >= 2 {
                target_system[0] * input
                    + target_system[1] * ((i - 1) as f64 * 0.1).sin()
                    + target_system[2] * ((i - 2) as f64 * 0.1).sin()
            } else if i >= 1 {
                target_system[0] * input + target_system[1] * ((i - 1) as f64 * 0.1).sin()
            } else {
                target_system[0] * input
            };

            let (_, err_lms, _) = lms.adapt(input, desired).expect("Operation failed");
            let (_, err_vs_lms, _) = vs_lms.adapt(input, desired).expect("Operation failed");
            let (_, err_nlms, _) = nlms.adapt(input, desired).expect("Operation failed");
            let (_, err_lmf, _) = lmf.adapt(input, desired).expect("Operation failed");

            lms_errors.push(err_lms.abs());
            vs_lms_errors.push(err_vs_lms.abs());
            nlms_errors.push(err_nlms.abs());
            lmf_errors.push(err_lmf.abs());
        }

        let final_window = 20;
        let lms_final_error =
            lms_errors.iter().rev().take(final_window).sum::<f64>() / final_window as f64;
        let vs_lms_final_error =
            vs_lms_errors.iter().rev().take(final_window).sum::<f64>() / final_window as f64;
        let nlms_final_error =
            nlms_errors.iter().rev().take(final_window).sum::<f64>() / final_window as f64;
        let lmf_final_error =
            lmf_errors.iter().rev().take(final_window).sum::<f64>() / final_window as f64;

        assert!(
            lms_final_error < 1.0,
            "LMS final error too large: {}",
            lms_final_error
        );
        assert!(
            vs_lms_final_error < 1.0,
            "VS-LMS final error too large: {}",
            vs_lms_final_error
        );
        assert!(
            nlms_final_error < 1.0,
            "NLMS final error too large: {}",
            nlms_final_error
        );
        assert!(
            lmf_final_error < 1.5,
            "LMF final error too large: {}",
            lmf_final_error
        );
    }

    #[test]
    fn test_solve_linear_system_small() {
        let matrix = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let rhs = vec![5.0, 6.0];

        let solution = solve_linear_system_small(&matrix, &rhs).expect("Operation failed");

        assert_relative_eq!(solution[0], 1.8, epsilon = 1e-10);
        assert_relative_eq!(solution[1], 1.4, epsilon = 1e-10);

        let singular_matrix = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert!(solve_linear_system_small(&singular_matrix, &rhs).is_err());

        let wrong_rhs = vec![5.0];
        assert!(solve_linear_system_small(&matrix, &wrong_rhs).is_err());
    }

    #[test]
    fn test_algorithm_reset_functionality() {
        let mut vs_lms = VsLmsFilter::new(3, 0.01, 0.05).expect("Operation failed");
        let mut apa = ApaFilter::new(3, 2, 0.1, 0.01).expect("Operation failed");
        let mut lmf = LmfFilter::new(3, 0.01).expect("Operation failed");
        let mut sm_lms = SmLmsFilter::new(3, 0.1, 0.1).expect("Operation failed");

        for i in 0..10 {
            let input = i as f64;
            vs_lms.adapt(input, 0.5).expect("Operation failed");
            apa.adapt(&[input, input * 0.5, input * 0.2], 0.5)
                .expect("Operation failed");
            lmf.adapt(input, 0.5).expect("Operation failed");
            sm_lms.adapt(input, 0.5).expect("Operation failed");
        }

        assert!(vs_lms.weights().iter().any(|&w| w != 0.0));
        assert!(apa.weights().iter().any(|&w| w != 0.0));
        assert!(lmf.weights().iter().any(|&w| w != 0.0));
        assert!(sm_lms.weights().iter().any(|&w| w != 0.0));

        vs_lms.reset();
        apa.reset();
        lmf.reset();
        sm_lms.reset();

        assert!(vs_lms.weights().iter().all(|&w| w == 0.0));
        assert!(apa.weights().iter().all(|&w| w == 0.0));
        assert!(lmf.weights().iter().all(|&w| w == 0.0));
        assert!(sm_lms.weights().iter().all(|&w| w == 0.0));

        let (update_count, sample_count, _) = sm_lms.update_statistics();
        assert_eq!(update_count, 0);
        assert_eq!(sample_count, 0);
    }

    #[test]
    fn test_unified_adaptive_filter_lms() {
        let mut af = AdaptiveFilter::new(4, AdaptiveMethod::Lms).expect("Operation failed");
        let (output, error, _mse) = af.filter(1.0, 0.5).expect("Operation failed");
        assert_relative_eq!(output, 0.0, epsilon = 1e-10);
        assert_relative_eq!(error, 0.5, epsilon = 1e-10);
        assert_eq!(af.weights().len(), 4);
    }

    #[test]
    fn test_unified_adaptive_filter_nlms() {
        let mut af = AdaptiveFilter::new(4, AdaptiveMethod::Nlms).expect("Operation failed");
        let (output, error, _mse) = af.filter(1.0, 0.3).expect("Operation failed");
        assert_relative_eq!(output, 0.0, epsilon = 1e-10);
        assert_relative_eq!(error, 0.3, epsilon = 1e-10);
    }

    #[test]
    fn test_unified_adaptive_filter_rls() {
        let mut af = AdaptiveFilter::new(4, AdaptiveMethod::Rls).expect("Operation failed");
        let (output, error, _mse) = af.filter(1.0, 0.5).expect("Operation failed");
        assert_relative_eq!(output, 0.0, epsilon = 1e-10);
        assert_relative_eq!(error, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_unified_adaptive_filter_batch() {
        let mut af = AdaptiveFilter::new(3, AdaptiveMethod::Lms).expect("Operation failed");
        let inputs = vec![1.0, 0.5, -0.3, 0.8, 0.2];
        let desired = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let (outputs, errors, _mse) = af
            .filter_batch(&inputs, &desired)
            .expect("Operation failed");
        assert_eq!(outputs.len(), 5);
        assert_eq!(errors.len(), 5);
    }

    #[test]
    fn test_unified_adaptive_filter_config() {
        let config = AdaptiveFilterConfig {
            order: 8,
            method: AdaptiveMethod::VsLms,
            step_size: 0.05,
            alpha: 0.03,
            ..Default::default()
        };
        let mut af = AdaptiveFilter::from_config(&config).expect("Operation failed");
        let (_output, _error, _mse) = af.filter(1.0, 0.5).expect("Operation failed");
        assert_eq!(af.weights().len(), 8);
    }
}
