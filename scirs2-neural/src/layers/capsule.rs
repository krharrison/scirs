//! Capsule network layers
//!
//! Implements capsule networks as described in:
//! - "Dynamic Routing Between Capsules", Sabour, Frosst & Hinton (2017)
//!   <https://arxiv.org/abs/1710.09829>
//! - "Matrix Capsules with EM Routing", Hinton, Sabour & Frosst (2018)
//!   <https://openreview.net/forum?id=HJWLfGWRb>
//!
//! ## Key Concepts
//!
//! Capsules are groups of neurons whose activity vector represents the
//! instantiation parameters of a specific type of entity.  The length of
//! the activity vector encodes the probability that the entity exists; its
//! orientation encodes the entity's properties.

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array, Array1, Array2, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Squash activation
// ---------------------------------------------------------------------------

/// Squash activation function for capsule networks.
///
/// Maps a vector v to a new vector whose magnitude lies in [0,1) while
/// preserving orientation:
///
/// ```text
/// squash(v) = (||v||² / (1 + ||v||²)) * (v / ||v||)
/// ```
///
/// # Arguments
/// * `v` - Input vector slice (f64 components)
///
/// # Returns
/// Squashed vector with same length as `v`.
pub fn squash(v: &[f64]) -> Vec<f64> {
    let sq_norm: f64 = v.iter().map(|x| x * x).sum();
    if sq_norm < 1e-12 {
        return vec![0.0; v.len()];
    }
    let scale = sq_norm / (1.0 + sq_norm);
    let norm = sq_norm.sqrt();
    v.iter().map(|x| scale * x / norm).collect()
}

/// Generic squash for ndarray Array1<F>.
fn squash_array<F: Float>(v: &Array1<F>) -> Array1<F> {
    let sq_norm = v.iter().fold(F::zero(), |acc, &x| acc + x * x);
    let eps = F::from(1e-12).unwrap_or(F::zero());
    if sq_norm < eps {
        return Array1::zeros(v.len());
    }
    let one = F::one();
    let scale = sq_norm / (one + sq_norm);
    let norm = sq_norm.sqrt();
    v.mapv(|x| scale * x / norm)
}

// ---------------------------------------------------------------------------
// PrimaryCapsules
// ---------------------------------------------------------------------------

/// Primary capsule layer.
///
/// Converts a standard convolutional feature map into a set of capsule vectors
/// by applying a linear projection and the squash non-linearity.
///
/// **Input layout** (flattened spatial features):
/// `[batch, num_channels * spatial]`  where `num_channels` must be divisible
/// by `caps_dim` to yield `num_caps = (num_channels * spatial) / caps_dim`
/// capsules.
///
/// **Output layout**: `[batch, num_caps, caps_dim]` — but returned as a flat
/// `IxDyn` array with shape `[batch, num_caps * caps_dim]` to stay compatible
/// with the `Layer` trait.
pub struct PrimaryCapsules {
    /// Number of capsule types (output capsule groups)
    num_caps: usize,
    /// Dimensionality of each capsule vector
    caps_dim: usize,
    /// Linear projection weights: shape [input_features, num_caps * caps_dim]
    weights: Array2<f64>,
    /// Bias: shape [num_caps * caps_dim]
    bias: Array1<f64>,
    /// Gradient buffers
    dweights: Array2<f64>,
    dbias: Array1<f64>,
    /// Cached input for backward pass
    cached_input: Option<Array2<f64>>,
}

impl std::fmt::Debug for PrimaryCapsules {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrimaryCapsules")
            .field("num_caps", &self.num_caps)
            .field("caps_dim", &self.caps_dim)
            .finish()
    }
}

impl PrimaryCapsules {
    /// Create a new `PrimaryCapsules` layer.
    ///
    /// # Arguments
    /// * `input_features` – total number of input features per sample
    /// * `num_caps` – number of output capsules
    /// * `caps_dim` – dimensionality of each capsule vector
    pub fn new(input_features: usize, num_caps: usize, caps_dim: usize) -> Result<Self> {
        if input_features == 0 || num_caps == 0 || caps_dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "PrimaryCapsules: dimensions must be > 0".to_string(),
            ));
        }
        let out_dim = num_caps * caps_dim;
        // He initialisation: std = sqrt(2 / fan_in)
        let std = (2.0 / input_features as f64).sqrt();
        let mut weights = Array2::<f64>::zeros((input_features, out_dim));
        // Simple deterministic initialisation (avoids RNG dependency in layer)
        for i in 0..input_features {
            for j in 0..out_dim {
                let idx = (i * out_dim + j) as f64;
                // pseudo-random via sin
                weights[[i, j]] = std * (idx * 0.6180339887).sin();
            }
        }
        let bias = Array1::<f64>::zeros(out_dim);
        Ok(Self {
            num_caps,
            caps_dim,
            dweights: Array2::zeros((input_features, out_dim)),
            dbias: Array1::zeros(out_dim),
            weights,
            bias,
            cached_input: None,
        })
    }

    /// Forward pass: linear projection then per-capsule squash.
    ///
    /// # Arguments
    /// * `x` – 2D array of shape `[batch, input_features]`
    ///
    /// # Returns
    /// Array of shape `[batch, num_caps * caps_dim]`.
    pub fn forward_2d(&mut self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let batch = x.nrows();
        // linear: [batch, out_dim]
        let pre = x.dot(&self.weights);
        // add bias
        let pre = pre + &self.bias.view().insert_axis(scirs2_core::ndarray::Axis(0));
        self.cached_input = Some(x.clone());
        // squash each capsule vector
        let out_dim = self.num_caps * self.caps_dim;
        let mut out = Array2::<f64>::zeros((batch, out_dim));
        for b in 0..batch {
            for c in 0..self.num_caps {
                let start = c * self.caps_dim;
                let end = start + self.caps_dim;
                let slice: Vec<f64> = (start..end).map(|i| pre[[b, i]]).collect();
                let sq = squash(&slice);
                for (d, val) in sq.into_iter().enumerate() {
                    out[[b, start + d]] = val;
                }
            }
        }
        Ok(out)
    }

    /// Number of capsules
    pub fn num_caps(&self) -> usize {
        self.num_caps
    }

    /// Capsule dimensionality
    pub fn caps_dim(&self) -> usize {
        self.caps_dim
    }
}

impl<F: Float + Debug + ScalarOperand + NumAssign + 'static> crate::layers::Layer<F>
    for PrimaryCapsules
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() < 2 {
            return Err(NeuralError::ShapeMismatch(
                "PrimaryCapsules expects at least 2D input".to_string(),
            ));
        }
        let batch = shape[0];
        let feat: usize = shape[1..].iter().product();
        let out_dim = self.num_caps * self.caps_dim;
        // Convert F → f64 for computation
        let flat: Vec<f64> = input.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
        let x2d = Array2::from_shape_vec((batch, feat), flat)
            .map_err(|e| NeuralError::ShapeMismatch(e.to_string()))?;
        let pre = x2d.dot(&self.weights);
        let pre = pre + &self.bias.view().insert_axis(scirs2_core::ndarray::Axis(0));
        let mut out = Array2::<f64>::zeros((batch, out_dim));
        for b in 0..batch {
            for c in 0..self.num_caps {
                let start = c * self.caps_dim;
                let end = start + self.caps_dim;
                let slice: Vec<f64> = (start..end).map(|i| pre[[b, i]]).collect();
                let sq = squash(&slice);
                for (d, val) in sq.into_iter().enumerate() {
                    out[[b, start + d]] = val;
                }
            }
        }
        // Convert back to F
        let out_f: Vec<F> = out
            .iter()
            .map(|&v| F::from(v).unwrap_or(F::zero()))
            .collect();
        Array::from_shape_vec(IxDyn(&[batch, out_dim]), out_f)
            .map_err(|e| NeuralError::ShapeMismatch(e.to_string()))
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        let lr = learning_rate.to_f64().unwrap_or(0.001);
        self.weights.zip_mut_with(&self.dweights, |w, &dw| {
            *w -= lr * dw;
        });
        self.bias.zip_mut_with(&self.dbias, |b, &db| {
            *b -= lr * db;
        });
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn layer_type(&self) -> &str {
        "PrimaryCapsules"
    }

    fn parameter_count(&self) -> usize {
        self.weights.len() + self.bias.len()
    }
}

// ---------------------------------------------------------------------------
// DynamicRouting
// ---------------------------------------------------------------------------

/// Dynamic routing-by-agreement algorithm (Sabour et al. 2017).
///
/// Iteratively refines routing coefficients `c_{ij}` so that capsule `j` in
/// the higher layer receives votes primarily from lower-level capsules that
/// agree with its current pose estimate.
///
/// # Algorithm
///
/// ```text
/// for iteration in 0..routing_iters:
///     c = softmax(b, axis=caps_j)
///     s_j = sum_i c_{ij} * u_hat_{j|i}
///     v_j = squash(s_j)
///     b_{ij} += dot(u_hat_{j|i}, v_j)
/// ```
///
/// where `u_hat_{j|i} = W_{ij} * u_i` are the vote vectors.
pub struct DynamicRouting {
    /// Number of lower-level capsules (in_caps)
    in_caps: usize,
    /// Number of higher-level capsules (out_caps)
    out_caps: usize,
    /// Dimensionality of lower-level capsule vectors
    in_dim: usize,
    /// Dimensionality of higher-level capsule vectors
    out_dim: usize,
    /// Number of routing iterations
    routing_iters: usize,
    /// Transformation matrices: shape [in_caps, out_caps, out_dim, in_dim]
    transform_weights: Vec<f64>,
}

impl std::fmt::Debug for DynamicRouting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicRouting")
            .field("in_caps", &self.in_caps)
            .field("out_caps", &self.out_caps)
            .field("in_dim", &self.in_dim)
            .field("out_dim", &self.out_dim)
            .field("routing_iters", &self.routing_iters)
            .finish()
    }
}

impl DynamicRouting {
    /// Create a new `DynamicRouting` layer.
    ///
    /// # Arguments
    /// * `in_caps` – number of input capsules
    /// * `out_caps` – number of output capsules
    /// * `in_dim` – input capsule dimension
    /// * `out_dim` – output capsule dimension
    /// * `routing_iters` – number of routing iterations (typically 3)
    pub fn new(
        in_caps: usize,
        out_caps: usize,
        in_dim: usize,
        out_dim: usize,
        routing_iters: usize,
    ) -> Result<Self> {
        if in_caps == 0 || out_caps == 0 || in_dim == 0 || out_dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "DynamicRouting: all dimensions must be > 0".to_string(),
            ));
        }
        if routing_iters == 0 {
            return Err(NeuralError::InvalidArgument(
                "DynamicRouting: routing_iters must be >= 1".to_string(),
            ));
        }
        let total = in_caps * out_caps * out_dim * in_dim;
        let std = (2.0 / (in_dim as f64)).sqrt();
        let transform_weights: Vec<f64> = (0..total)
            .map(|k| std * ((k as f64) * 0.6180339887).sin())
            .collect();
        Ok(Self {
            in_caps,
            out_caps,
            in_dim,
            out_dim,
            routing_iters,
            transform_weights,
        })
    }

    /// Perform dynamic routing.
    ///
    /// # Arguments
    /// * `u` – input capsule vectors, shape `[batch, in_caps, in_dim]`
    ///
    /// # Returns
    /// Output capsule vectors, shape `[batch, out_caps, out_dim]`.
    pub fn route(&self, u: &Array2<f64>) -> Result<Vec<f64>> {
        // u has shape [in_caps, in_dim] (single sample)
        if u.nrows() != self.in_caps || u.ncols() != self.in_dim {
            return Err(NeuralError::ShapeMismatch(format!(
                "DynamicRouting route: expected [{}, {}], got [{}, {}]",
                self.in_caps,
                self.in_dim,
                u.nrows(),
                u.ncols()
            )));
        }
        // Compute vote vectors u_hat[i,j] = W[i,j] * u[i]
        // W[i,j] has shape [out_dim, in_dim]
        let mut u_hat = vec![0.0f64; self.in_caps * self.out_caps * self.out_dim];
        for i in 0..self.in_caps {
            for j in 0..self.out_caps {
                let w_offset = (i * self.out_caps + j) * self.out_dim * self.in_dim;
                for d in 0..self.out_dim {
                    let mut sum = 0.0f64;
                    for k in 0..self.in_dim {
                        sum += self.transform_weights[w_offset + d * self.in_dim + k] * u[[i, k]];
                    }
                    u_hat[(i * self.out_caps + j) * self.out_dim + d] = sum;
                }
            }
        }
        // Routing logits b[i,j] initialised to 0
        let mut b = vec![0.0f64; self.in_caps * self.out_caps];
        let mut v = vec![0.0f64; self.out_caps * self.out_dim];
        for _ in 0..self.routing_iters {
            // c = softmax(b) over out_caps dim for each in_caps i
            let mut c = vec![0.0f64; self.in_caps * self.out_caps];
            for i in 0..self.in_caps {
                let max_b = (0..self.out_caps)
                    .map(|j| b[i * self.out_caps + j])
                    .fold(f64::NEG_INFINITY, f64::max);
                let mut sum_exp = 0.0f64;
                for j in 0..self.out_caps {
                    let e = (b[i * self.out_caps + j] - max_b).exp();
                    c[i * self.out_caps + j] = e;
                    sum_exp += e;
                }
                for j in 0..self.out_caps {
                    c[i * self.out_caps + j] /= sum_exp.max(1e-12);
                }
            }
            // s[j] = sum_i c[i,j] * u_hat[i,j]
            let mut s = vec![0.0f64; self.out_caps * self.out_dim];
            for i in 0..self.in_caps {
                for j in 0..self.out_caps {
                    let coeff = c[i * self.out_caps + j];
                    let uhat_offset = (i * self.out_caps + j) * self.out_dim;
                    let s_offset = j * self.out_dim;
                    for d in 0..self.out_dim {
                        s[s_offset + d] += coeff * u_hat[uhat_offset + d];
                    }
                }
            }
            // v[j] = squash(s[j])
            for j in 0..self.out_caps {
                let offset = j * self.out_dim;
                let slice = &s[offset..offset + self.out_dim];
                let sq = squash(slice);
                for (d, val) in sq.into_iter().enumerate() {
                    v[j * self.out_dim + d] = val;
                }
            }
            // Update b: b[i,j] += dot(u_hat[i,j], v[j])
            for i in 0..self.in_caps {
                for j in 0..self.out_caps {
                    let uhat_offset = (i * self.out_caps + j) * self.out_dim;
                    let v_offset = j * self.out_dim;
                    let dot: f64 = (0..self.out_dim)
                        .map(|d| u_hat[uhat_offset + d] * v[v_offset + d])
                        .sum();
                    b[i * self.out_caps + j] += dot;
                }
            }
        }
        Ok(v)
    }

    /// Number of output capsules
    pub fn out_caps(&self) -> usize {
        self.out_caps
    }

    /// Output capsule dimension
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }
}

impl<F: Float + Debug + ScalarOperand + NumAssign + 'static> crate::layers::Layer<F>
    for DynamicRouting
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() < 2 {
            return Err(NeuralError::ShapeMismatch(
                "DynamicRouting expects [batch, in_caps * in_dim] or [batch, in_caps, in_dim]"
                    .to_string(),
            ));
        }
        let batch = shape[0];
        let expected_flat = self.in_caps * self.in_dim;
        let flat_input: Vec<f64> = input.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
        let mut out_data = Vec::with_capacity(batch * self.out_caps * self.out_dim);
        for b in 0..batch {
            let slice = &flat_input[b * expected_flat..(b + 1) * expected_flat];
            let u = Array2::from_shape_vec((self.in_caps, self.in_dim), slice.to_vec())
                .map_err(|e| NeuralError::ShapeMismatch(e.to_string()))?;
            let v = self.route(&u)?;
            out_data.extend(v);
        }
        let out_f: Vec<F> = out_data
            .iter()
            .map(|&v| F::from(v).unwrap_or(F::zero()))
            .collect();
        Array::from_shape_vec(IxDyn(&[batch, self.out_caps * self.out_dim]), out_f)
            .map_err(|e| NeuralError::ShapeMismatch(e.to_string()))
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, _lr: F) -> Result<()> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn layer_type(&self) -> &str {
        "DynamicRouting"
    }

    fn parameter_count(&self) -> usize {
        self.transform_weights.len()
    }
}

// ---------------------------------------------------------------------------
// CapsuleLayer
// ---------------------------------------------------------------------------

/// General capsule layer combining transformation, routing and squash.
///
/// Acts as a complete capsule layer: takes `in_caps` lower-level capsule
/// vectors of size `in_dim`, computes votes via learnable transformation
/// matrices, then applies dynamic routing to produce `out_caps` capsule
/// vectors of size `out_dim`.
pub struct CapsuleLayer {
    routing: DynamicRouting,
}

impl std::fmt::Debug for CapsuleLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CapsuleLayer")
            .field("routing", &self.routing)
            .finish()
    }
}

impl CapsuleLayer {
    /// Create a new `CapsuleLayer`.
    ///
    /// # Arguments
    /// * `in_caps` – number of input capsules
    /// * `out_caps` – number of output capsules
    /// * `in_dim` – input capsule dimensionality
    /// * `out_dim` – output capsule dimensionality
    /// * `routing_iters` – dynamic routing iterations (default 3)
    pub fn new(
        in_caps: usize,
        out_caps: usize,
        in_dim: usize,
        out_dim: usize,
        routing_iters: usize,
    ) -> Result<Self> {
        let routing = DynamicRouting::new(in_caps, out_caps, in_dim, out_dim, routing_iters)?;
        Ok(Self { routing })
    }

    /// Number of output capsules
    pub fn out_caps(&self) -> usize {
        self.routing.out_caps()
    }

    /// Output capsule dimension
    pub fn out_dim(&self) -> usize {
        self.routing.out_dim()
    }
}

impl<F: Float + Debug + ScalarOperand + NumAssign + 'static> crate::layers::Layer<F>
    for CapsuleLayer
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        self.routing.forward(input)
    }

    fn backward(
        &self,
        input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        self.routing.backward(input, grad_output)
    }

    fn update(&mut self, lr: F) -> Result<()> {
        self.routing.update(lr)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn layer_type(&self) -> &str {
        "CapsuleLayer"
    }

    fn parameter_count(&self) -> usize {
        <DynamicRouting as crate::layers::Layer<F>>::parameter_count(&self.routing)
    }
}

// ---------------------------------------------------------------------------
// EM Routing (Matrix Capsules – Hinton et al. 2018)
// ---------------------------------------------------------------------------

/// Configuration for EM routing.
#[derive(Debug, Clone)]
pub struct EMRoutingConfig {
    /// Number of input capsules
    pub in_caps: usize,
    /// Number of output capsules
    pub out_caps: usize,
    /// Pose matrix dimension (d×d, so pose_dim = d²)
    pub pose_dim: usize,
    /// Number of EM iterations
    pub em_iters: usize,
    /// Initial inverse temperature β_v (votes)
    pub beta_v: f64,
    /// Initial inverse temperature β_a (activations)
    pub beta_a: f64,
}

impl EMRoutingConfig {
    /// Standard EM routing config
    pub fn standard(in_caps: usize, out_caps: usize) -> Self {
        Self {
            in_caps,
            out_caps,
            pose_dim: 16, // 4×4 pose matrices
            em_iters: 3,
            beta_v: 0.01,
            beta_a: 0.01,
        }
    }
}

/// Expectation-Maximization routing for Matrix Capsules (Hinton et al. 2018).
///
/// Each capsule is represented by a pose matrix (flattened to `pose_dim`) and
/// an activation probability.  Routing iterates E-step (assignment
/// probabilities) and M-step (Gaussian parameter estimation) to cluster lower
/// capsule votes around output capsule means.
pub struct EMRouting {
    config: EMRoutingConfig,
    /// Transformation matrices: [in_caps, out_caps, pose_dim, pose_dim]
    transform: Vec<f64>,
}

impl std::fmt::Debug for EMRouting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EMRouting")
            .field("config", &self.config)
            .finish()
    }
}

impl EMRouting {
    /// Create a new `EMRouting` layer.
    pub fn new(config: EMRoutingConfig) -> Result<Self> {
        let EMRoutingConfig {
            in_caps,
            out_caps,
            pose_dim,
            ..
        } = config;
        if in_caps == 0 || out_caps == 0 || pose_dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "EMRouting: dimensions must be > 0".to_string(),
            ));
        }
        let total = in_caps * out_caps * pose_dim * pose_dim;
        let std = (2.0 / pose_dim as f64).sqrt();
        let transform: Vec<f64> = (0..total)
            .map(|k| std * ((k as f64) * 0.6180339887).sin())
            .collect();
        Ok(Self { config, transform })
    }

    /// Run EM routing for a single sample.
    ///
    /// # Arguments
    /// * `poses_in` – input poses, shape `[in_caps, pose_dim]`
    /// * `activations_in` – input activations, shape `[in_caps]`
    ///
    /// # Returns
    /// `(poses_out, activations_out)` with shapes `[out_caps, pose_dim]` and `[out_caps]`.
    pub fn route_sample(
        &self,
        poses_in: &[f64],
        activations_in: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let EMRoutingConfig {
            in_caps,
            out_caps,
            pose_dim,
            em_iters,
            beta_v,
            beta_a,
        } = self.config;
        if poses_in.len() != in_caps * pose_dim {
            return Err(NeuralError::ShapeMismatch(format!(
                "EMRouting: poses_in length {} != {}",
                poses_in.len(),
                in_caps * pose_dim
            )));
        }
        if activations_in.len() != in_caps {
            return Err(NeuralError::ShapeMismatch(format!(
                "EMRouting: activations_in length {} != {}",
                activations_in.len(),
                in_caps
            )));
        }
        // Compute vote poses V[i,j] = W[i,j] * M[i]
        // V shape: [in_caps, out_caps, pose_dim]
        let mut votes = vec![0.0f64; in_caps * out_caps * pose_dim];
        for i in 0..in_caps {
            for j in 0..out_caps {
                let w_offset = (i * out_caps + j) * pose_dim * pose_dim;
                let m_offset = i * pose_dim;
                let v_offset = (i * out_caps + j) * pose_dim;
                for d in 0..pose_dim {
                    let mut s = 0.0f64;
                    for k in 0..pose_dim {
                        s += self.transform[w_offset + d * pose_dim + k] * poses_in[m_offset + k];
                    }
                    votes[v_offset + d] = s;
                }
            }
        }
        // Initialise routing assignment R[i,j] = 1/out_caps
        let init_r = 1.0 / out_caps as f64;
        let mut r = vec![init_r; in_caps * out_caps];
        let mut mu = vec![0.0f64; out_caps * pose_dim];
        let mut sigma_sq = vec![1.0f64; out_caps * pose_dim];
        let mut a_out = vec![0.5f64; out_caps];
        for _iter in 0..em_iters {
            // M-step: compute mean and variance for each output capsule
            for j in 0..out_caps {
                let mut sum_r = 0.0f64;
                for i in 0..in_caps {
                    sum_r += r[i * out_caps + j] * activations_in[i];
                }
                sum_r = sum_r.max(1e-12);
                // mean
                for d in 0..pose_dim {
                    let mut s = 0.0f64;
                    for i in 0..in_caps {
                        s += r[i * out_caps + j] * activations_in[i]
                            * votes[(i * out_caps + j) * pose_dim + d];
                    }
                    mu[j * pose_dim + d] = s / sum_r;
                }
                // variance
                for d in 0..pose_dim {
                    let mut s = 0.0f64;
                    for i in 0..in_caps {
                        let diff = votes[(i * out_caps + j) * pose_dim + d]
                            - mu[j * pose_dim + d];
                        s += r[i * out_caps + j] * activations_in[i] * diff * diff;
                    }
                    sigma_sq[j * pose_dim + d] = (s / sum_r).max(1e-6);
                }
                // activation: logistic sigmoid of (beta_a - beta_v * sum_d 0.5*log(2π e σ²_d))
                let cost: f64 = (0..pose_dim)
                    .map(|d| {
                        let s = sigma_sq[j * pose_dim + d];
                        0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * s).ln()
                    })
                    .sum();
                let logit = beta_a - beta_v * cost;
                a_out[j] = 1.0 / (1.0 + (-logit).exp());
            }
            // E-step: update assignment probabilities
            for i in 0..in_caps {
                let mut log_p = vec![0.0f64; out_caps];
                for j in 0..out_caps {
                    let mut lp = 0.0f64;
                    for d in 0..pose_dim {
                        let diff = votes[(i * out_caps + j) * pose_dim + d]
                            - mu[j * pose_dim + d];
                        let sv = sigma_sq[j * pose_dim + d];
                        lp += -0.5 * diff * diff / sv
                            - 0.5 * (2.0 * std::f64::consts::PI * sv).ln();
                    }
                    log_p[j] = lp + a_out[j].ln().max(-50.0);
                }
                // softmax
                let max_lp = log_p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = log_p.iter().map(|&lp| (lp - max_lp).exp()).collect();
                let sum_exp: f64 = exps.iter().sum::<f64>().max(1e-12);
                for j in 0..out_caps {
                    r[i * out_caps + j] = exps[j] / sum_exp;
                }
            }
        }
        Ok((mu, a_out))
    }

    /// Number of output capsules
    pub fn out_caps(&self) -> usize {
        self.config.out_caps
    }

    /// Pose dimension
    pub fn pose_dim(&self) -> usize {
        self.config.pose_dim
    }
}

impl<F: Float + Debug + ScalarOperand + NumAssign + 'static> crate::layers::Layer<F> for EMRouting {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() < 2 {
            return Err(NeuralError::ShapeMismatch(
                "EMRouting expects [batch, in_caps*(pose_dim+1)]".to_string(),
            ));
        }
        let batch = shape[0];
        let in_caps = self.config.in_caps;
        let pose_dim = self.config.pose_dim;
        let expected = in_caps * (pose_dim + 1);
        let flat: Vec<f64> = input.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
        let out_caps = self.config.out_caps;
        let mut out_data = Vec::with_capacity(batch * out_caps * (pose_dim + 1));
        for b in 0..batch {
            let slice = &flat[b * expected..(b + 1) * expected];
            let poses_in = &slice[..in_caps * pose_dim];
            let acts_in = &slice[in_caps * pose_dim..];
            let (poses_out, acts_out) = self.route_sample(poses_in, acts_in)?;
            // interleave: [poses, acts]
            out_data.extend_from_slice(&poses_out);
            out_data.extend_from_slice(&acts_out);
        }
        let out_f: Vec<F> = out_data
            .iter()
            .map(|&v| F::from(v).unwrap_or(F::zero()))
            .collect();
        Array::from_shape_vec(IxDyn(&[batch, out_caps * (pose_dim + 1)]), out_f)
            .map_err(|e| NeuralError::ShapeMismatch(e.to_string()))
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, _lr: F) -> Result<()> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn layer_type(&self) -> &str {
        "EMRouting"
    }

    fn parameter_count(&self) -> usize {
        self.transform.len()
    }
}

// ---------------------------------------------------------------------------
// CapsuleNetwork
// ---------------------------------------------------------------------------

/// Configuration for a full capsule network with reconstruction decoder.
#[derive(Debug, Clone)]
pub struct CapsuleNetworkConfig {
    /// Input feature dimension
    pub input_dim: usize,
    /// Number of primary capsules
    pub num_primary_caps: usize,
    /// Dimensionality of primary capsule vectors
    pub primary_caps_dim: usize,
    /// Number of output (class) capsules
    pub num_class_caps: usize,
    /// Dimensionality of output capsule vectors (length encodes class probability)
    pub class_caps_dim: usize,
    /// Number of dynamic routing iterations
    pub routing_iters: usize,
    /// Whether to include a reconstruction decoder
    pub use_decoder: bool,
    /// Hidden dimension for the reconstruction decoder
    pub decoder_hidden_dim: usize,
}

impl CapsuleNetworkConfig {
    /// Standard capsule network for MNIST-like tasks.
    pub fn standard(input_dim: usize, num_classes: usize) -> Self {
        Self {
            input_dim,
            num_primary_caps: 32,
            primary_caps_dim: 8,
            num_class_caps: num_classes,
            class_caps_dim: 16,
            routing_iters: 3,
            use_decoder: true,
            decoder_hidden_dim: 512,
        }
    }

    /// Tiny config for testing.
    pub fn tiny(input_dim: usize, num_classes: usize) -> Self {
        Self {
            input_dim,
            num_primary_caps: 4,
            primary_caps_dim: 4,
            num_class_caps: num_classes,
            class_caps_dim: 8,
            routing_iters: 2,
            use_decoder: false,
            decoder_hidden_dim: 64,
        }
    }
}

/// Full capsule network with optional reconstruction decoder.
///
/// Architecture:
/// 1. `PrimaryCapsules` – converts input features to initial capsule vectors
/// 2. `CapsuleLayer` (dynamic routing) – routes to class capsules
/// 3. Length of class capsule vector → class probability
/// 4. *(Optional)* reconstruction decoder via 3 fully-connected layers
pub struct CapsuleNetwork {
    config: CapsuleNetworkConfig,
    primary: PrimaryCapsules,
    routing: DynamicRouting,
    /// Decoder weights (3 layers): each element is (W: [in, out], b: [out])
    decoder_layers: Vec<(Vec<f64>, Vec<f64>, usize, usize)>,
}

impl std::fmt::Debug for CapsuleNetwork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CapsuleNetwork")
            .field("config", &self.config)
            .finish()
    }
}

impl CapsuleNetwork {
    /// Create a new `CapsuleNetwork`.
    pub fn new(config: CapsuleNetworkConfig) -> Result<Self> {
        let primary = PrimaryCapsules::new(
            config.input_dim,
            config.num_primary_caps,
            config.primary_caps_dim,
        )?;
        let routing = DynamicRouting::new(
            config.num_primary_caps,
            config.num_class_caps,
            config.primary_caps_dim,
            config.class_caps_dim,
            config.routing_iters,
        )?;
        let mut decoder_layers = Vec::new();
        if config.use_decoder {
            // Layer 1: num_class_caps * class_caps_dim → decoder_hidden_dim
            let in1 = config.num_class_caps * config.class_caps_dim;
            let out1 = config.decoder_hidden_dim;
            decoder_layers.push(Self::make_fc_layer(in1, out1));
            // Layer 2: decoder_hidden_dim → decoder_hidden_dim
            decoder_layers.push(Self::make_fc_layer(out1, out1));
            // Layer 3: decoder_hidden_dim → input_dim
            decoder_layers.push(Self::make_fc_layer(out1, config.input_dim));
        }
        Ok(Self {
            config,
            primary,
            routing,
            decoder_layers,
        })
    }

    fn make_fc_layer(in_dim: usize, out_dim: usize) -> (Vec<f64>, Vec<f64>, usize, usize) {
        let std = (2.0 / in_dim as f64).sqrt();
        let w: Vec<f64> = (0..in_dim * out_dim)
            .map(|k| std * ((k as f64) * 0.6180339887).sin())
            .collect();
        let b: Vec<f64> = vec![0.0; out_dim];
        (w, b, in_dim, out_dim)
    }

    fn fc_forward(layer: &(Vec<f64>, Vec<f64>, usize, usize), x: &[f64]) -> Vec<f64> {
        let (w, b, in_dim, out_dim) = layer;
        let mut out = b.clone();
        for j in 0..*out_dim {
            let mut s = 0.0f64;
            for i in 0..*in_dim {
                s += w[j * in_dim + i] * x[i];
            }
            out[j] += s;
        }
        out
    }

    fn relu(x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| v.max(0.0)).collect()
    }

    fn sigmoid(x: &[f64]) -> Vec<f64> {
        x.iter()
            .map(|&v| 1.0 / (1.0 + (-v).exp()))
            .collect()
    }

    /// Run the forward pass for a single sample.
    ///
    /// # Arguments
    /// * `x` – flat input vector of length `input_dim`
    ///
    /// # Returns
    /// `CapsuleNetworkOutput` containing class probabilities and optional reconstruction.
    pub fn forward_sample(&self, x: &[f64]) -> Result<CapsuleNetworkOutput> {
        if x.len() != self.config.input_dim {
            return Err(NeuralError::ShapeMismatch(format!(
                "CapsuleNetwork: expected input length {}, got {}",
                self.config.input_dim,
                x.len()
            )));
        }
        // Primary capsules
        let x2d = Array2::from_shape_vec((1, self.config.input_dim), x.to_vec())
            .map_err(|e| NeuralError::ShapeMismatch(e.to_string()))?;
        let primary_out_raw = x2d.dot(&self.primary.weights);
        let primary_out_raw =
            primary_out_raw + &self.primary.bias.view().insert_axis(scirs2_core::ndarray::Axis(0));
        let out_dim = self.config.num_primary_caps * self.config.primary_caps_dim;
        let mut primary_out = vec![0.0f64; out_dim];
        for c in 0..self.config.num_primary_caps {
            let start = c * self.config.primary_caps_dim;
            let end = start + self.config.primary_caps_dim;
            let slice: Vec<f64> = (start..end).map(|i| primary_out_raw[[0, i]]).collect();
            let sq = squash(&slice);
            for (d, val) in sq.into_iter().enumerate() {
                primary_out[start + d] = val;
            }
        }
        // Dynamic routing
        let u = Array2::from_shape_vec(
            (self.config.num_primary_caps, self.config.primary_caps_dim),
            primary_out.clone(),
        )
        .map_err(|e| NeuralError::ShapeMismatch(e.to_string()))?;
        let v = self.routing.route(&u)?;
        // Class probabilities = L2 norm of each class capsule
        let class_probs: Vec<f64> = (0..self.config.num_class_caps)
            .map(|c| {
                let offset = c * self.config.class_caps_dim;
                let sq_norm: f64 = (0..self.config.class_caps_dim)
                    .map(|d| v[offset + d] * v[offset + d])
                    .sum();
                sq_norm.sqrt()
            })
            .collect();
        // Reconstruction (optional)
        let reconstruction = if self.config.use_decoder && !self.decoder_layers.is_empty() {
            let mut h = v.clone();
            for (layer_idx, layer) in self.decoder_layers.iter().enumerate() {
                let pre = Self::fc_forward(layer, &h);
                h = if layer_idx < self.decoder_layers.len() - 1 {
                    Self::relu(&pre)
                } else {
                    Self::sigmoid(&pre)
                };
            }
            Some(h)
        } else {
            None
        };
        Ok(CapsuleNetworkOutput {
            class_probs,
            class_capsules: v,
            reconstruction,
        })
    }

    /// Compute margin loss for capsule networks.
    ///
    /// `L_c = T_c * max(0, m+ - ||v_c||)² + λ * (1-T_c) * max(0, ||v_c|| - m-)²`
    ///
    /// where `T_c = 1` if class `c` is present.
    pub fn margin_loss(
        &self,
        class_probs: &[f64],
        target_class: usize,
        m_plus: f64,
        m_minus: f64,
        lambda: f64,
    ) -> Result<f64> {
        if target_class >= self.config.num_class_caps {
            return Err(NeuralError::InvalidArgument(format!(
                "target_class {} >= num_class_caps {}",
                target_class, self.config.num_class_caps
            )));
        }
        let loss: f64 = class_probs
            .iter()
            .enumerate()
            .map(|(c, &p)| {
                let t = if c == target_class { 1.0 } else { 0.0 };
                let present_loss = t * (m_plus - p).max(0.0).powi(2);
                let absent_loss = lambda * (1.0 - t) * (p - m_minus).max(0.0).powi(2);
                present_loss + absent_loss
            })
            .sum();
        Ok(loss)
    }
}

/// Output from a forward pass through `CapsuleNetwork`.
#[derive(Debug, Clone)]
pub struct CapsuleNetworkOutput {
    /// L2 norms of class capsule vectors (class probabilities), len = num_class_caps
    pub class_probs: Vec<f64>,
    /// Raw class capsule vectors (flattened), len = num_class_caps * class_caps_dim
    pub class_capsules: Vec<f64>,
    /// Optional reconstruction of the input
    pub reconstruction: Option<Vec<f64>>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_squash_zero() {
        let v = vec![0.0f64; 4];
        let out = squash(&v);
        assert_eq!(out.len(), 4);
        for &x in &out {
            assert!(x.abs() < 1e-10);
        }
    }

    #[test]
    fn test_squash_magnitude() {
        let v = vec![1.0, 0.0, 0.0, 0.0];
        let out = squash(&v);
        let norm: f64 = out.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm < 1.0, "squash magnitude should be < 1, got {norm}");
        assert!(norm > 0.0, "squash magnitude should be > 0");
    }

    #[test]
    fn test_primary_capsules_new() {
        let caps = PrimaryCapsules::new(64, 8, 4).expect("construction failed");
        assert_eq!(caps.num_caps(), 8);
        assert_eq!(caps.caps_dim(), 4);
    }

    #[test]
    fn test_dynamic_routing_route() {
        let dr = DynamicRouting::new(4, 2, 4, 4, 2).expect("construction failed");
        let u = Array2::<f64>::from_shape_vec(
            (4, 4),
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        )
        .expect("array creation");
        let v = dr.route(&u).expect("routing failed");
        assert_eq!(v.len(), 2 * 4);
        // magnitudes should be in (0, 1)
        for c in 0..2 {
            let norm: f64 = (0..4)
                .map(|d| v[c * 4 + d] * v[c * 4 + d])
                .sum::<f64>()
                .sqrt();
            assert!(norm < 1.0, "capsule norm should be < 1");
        }
    }

    #[test]
    fn test_capsule_network_forward() {
        let config = CapsuleNetworkConfig::tiny(16, 3);
        let net = CapsuleNetwork::new(config).expect("construction failed");
        let x = vec![0.1f64; 16];
        let out = net.forward_sample(&x).expect("forward failed");
        assert_eq!(out.class_probs.len(), 3);
        for &p in &out.class_probs {
            assert!(p >= 0.0 && p <= 1.0, "class prob out of range: {p}");
        }
    }

    #[test]
    fn test_em_routing() {
        let config = EMRoutingConfig::standard(4, 3);
        let em = EMRouting::new(config).expect("construction failed");
        let poses_in = vec![0.1f64; 4 * 16];
        let acts_in = vec![0.5f64; 4];
        let (poses_out, acts_out) = em.route_sample(&poses_in, &acts_in).expect("em route failed");
        assert_eq!(poses_out.len(), 3 * 16);
        assert_eq!(acts_out.len(), 3);
        for &a in &acts_out {
            assert!(a >= 0.0 && a <= 1.0);
        }
    }
}
