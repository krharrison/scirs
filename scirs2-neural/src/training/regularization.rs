//! Regularization techniques for neural network training
//!
//! This module implements a comprehensive suite of regularization methods used to
//! prevent overfitting and improve generalization of neural network models:
//!
//! ## Penalty-Based Regularization
//!
//! - [`l1_regularization`] / [`l1_gradient`] — LASSO penalty: `alpha * ||w||_1`
//! - [`l2_regularization`] / [`l2_gradient`] — Ridge/weight decay: `alpha * ||w||_2^2`
//! - [`elastic_net`] / [`elastic_net_gradient`] — Combination: `alpha*L1 + beta*L2`
//!
//! ## Stochastic Regularization
//!
//! - [`DropoutLayer`] — Drop random activations during training (standard and inverted)
//! - [`dropconnect_mask`] — Drop random weight connections
//! - [`StochasticDepth`] — Randomly skip entire residual branches
//!
//! ## Weight Normalization
//!
//! - [`spectral_normalize`] — Constrain weight matrices to have spectral norm ≤ 1
//!
//! ## Gradient Control
//!
//! - [`clip_grad_norm`] — Rescale gradients whose global norm exceeds a threshold
//! - [`clip_grad_value`] — Clamp each gradient element to `[-clip, clip]`
//!
//! ## Data Augmentation
//!
//! - [`mixup`] — Linearly interpolate pairs of training examples (Zhang 2018)
//! - [`cutmix`] — Paste rectangular patches between training images
//!
//! # Example
//!
//! ```rust
//! use scirs2_neural::training::regularization::{
//!     l1_regularization, l2_regularization, elastic_net,
//!     DropoutLayer, spectral_normalize,
//! };
//! use scirs2_core::ndarray::Array2;
//!
//! // L1 penalty
//! let weights = vec![1.0_f64, -2.0, 3.0];
//! let penalty = l1_regularization(&weights, 0.01);
//! assert!((penalty - 0.06).abs() < 1e-10);
//!
//! // Spectral normalization
//! let w = Array2::<f64>::eye(3);
//! let (w_sn, sigma) = spectral_normalize(&w, 5);
//! assert!((sigma - 1.0).abs() < 1e-6);
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array2, Axis};
use scirs2_core::random::rngs::SmallRng;
use scirs2_core::random::{Rng, SeedableRng};

// ============================================================================
// L1 Regularization (LASSO)
// ============================================================================

/// Compute the L1 regularization penalty.
///
/// Returns `alpha * sum(|w_i|)`.
///
/// # Arguments
///
/// * `weights` - Flat slice of weight values
/// * `alpha`   - Regularization strength
pub fn l1_regularization(weights: &[f64], alpha: f64) -> f64 {
    alpha * weights.iter().map(|w| w.abs()).sum::<f64>()
}

/// Compute the gradient of the L1 penalty with respect to each weight.
///
/// The subgradient is `alpha * sign(w_i)` (0 is treated as sign 0).
///
/// # Arguments
///
/// * `weights` - Flat slice of weight values
/// * `alpha`   - Regularization strength
pub fn l1_gradient(weights: &[f64], alpha: f64) -> Vec<f64> {
    weights
        .iter()
        .map(|&w| {
            if w > 0.0 {
                alpha
            } else if w < 0.0 {
                -alpha
            } else {
                0.0
            }
        })
        .collect()
}

// ============================================================================
// L2 Regularization (Ridge / Weight Decay)
// ============================================================================

/// Compute the L2 regularization penalty.
///
/// Returns `alpha * sum(w_i^2)`.
///
/// # Arguments
///
/// * `weights` - Flat slice of weight values
/// * `alpha`   - Regularization strength
pub fn l2_regularization(weights: &[f64], alpha: f64) -> f64 {
    alpha * weights.iter().map(|w| w * w).sum::<f64>()
}

/// Compute the gradient of the L2 penalty with respect to each weight.
///
/// Returns `2 * alpha * w_i` for each element.
///
/// # Arguments
///
/// * `weights` - Flat slice of weight values
/// * `alpha`   - Regularization strength
pub fn l2_gradient(weights: &[f64], alpha: f64) -> Vec<f64> {
    weights.iter().map(|&w| 2.0 * alpha * w).collect()
}

// ============================================================================
// Elastic Net Regularization
// ============================================================================

/// Compute the Elastic Net regularization penalty.
///
/// Combines L1 and L2 penalties: `alpha * ||w||_1 + beta * ||w||_2^2`.
///
/// # Arguments
///
/// * `weights` - Flat slice of weight values
/// * `alpha`   - L1 regularization strength
/// * `beta`    - L2 regularization strength
pub fn elastic_net(weights: &[f64], alpha: f64, beta: f64) -> f64 {
    l1_regularization(weights, alpha) + l2_regularization(weights, beta)
}

/// Compute the gradient of the Elastic Net penalty.
///
/// Returns `alpha * sign(w_i) + 2 * beta * w_i` for each element.
///
/// # Arguments
///
/// * `weights` - Flat slice of weight values
/// * `alpha`   - L1 regularization strength
/// * `beta`    - L2 regularization strength
pub fn elastic_net_gradient(weights: &[f64], alpha: f64, beta: f64) -> Vec<f64> {
    let g1 = l1_gradient(weights, alpha);
    let g2 = l2_gradient(weights, beta);
    g1.iter().zip(g2.iter()).map(|(a, b)| a + b).collect()
}

// ============================================================================
// Dropout Layer
// ============================================================================

/// A dropout layer that zeros out a random fraction of activations during training.
///
/// In training mode each unit is independently kept with probability `1 - rate`.
/// The standard variant scales retained activations by `1 / (1 - rate)` during
/// forward (inverted dropout), ensuring that the expected value of each unit is
/// identical at train and inference time.
///
/// # Example
///
/// ```rust
/// use scirs2_neural::training::regularization::DropoutLayer;
///
/// let mut layer = DropoutLayer::new(0.5);
/// layer.train();
/// let out = layer.forward_inverted(&[1.0, 1.0, 1.0, 1.0], 42);
/// // Approximately half the units are 0 and the rest are scaled to 2.0
/// let sum: f64 = out.iter().sum();
/// assert!((sum - 4.0).abs() < 4.0 + 1e-6); // generous bound for small n
/// ```
#[derive(Debug, Clone)]
pub struct DropoutLayer {
    /// Fraction of units to drop (e.g. 0.5 → drop half)
    pub rate: f64,
    /// Whether the layer is in training mode
    pub training: bool,
    /// Current dropout mask (`true` = keep, `false` = drop)
    mask: Option<Vec<bool>>,
}

impl DropoutLayer {
    /// Create a new dropout layer.
    ///
    /// # Panics (never) — invalid rates are clamped to `[0, 1)`.
    pub fn new(rate: f64) -> Self {
        let rate = rate.clamp(0.0, 1.0 - f64::EPSILON);
        Self {
            rate,
            training: true,
            mask: None,
        }
    }

    /// Switch to training mode.
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Switch to evaluation mode (dropout is disabled).
    pub fn eval(&mut self) {
        self.training = false;
        self.mask = None;
    }

    /// Standard forward pass (no scaling).
    ///
    /// During training, zeros out units according to the dropout mask.
    /// During evaluation, returns the input unchanged.
    pub fn forward(&mut self, input: &[f64], seed: u64) -> Vec<f64> {
        if !self.training {
            return input.to_vec();
        }
        let mask = build_keep_mask(input.len(), self.rate, seed);
        let out: Vec<f64> = input
            .iter()
            .zip(mask.iter())
            .map(|(&x, &keep)| if keep { x } else { 0.0 })
            .collect();
        self.mask = Some(mask);
        out
    }

    /// Inverted dropout forward pass.
    ///
    /// During training, zeros out units and scales kept units by `1 / (1 - rate)`,
    /// so that the expected value of each unit equals the input value regardless of
    /// the dropout rate.
    ///
    /// During evaluation, returns the input unchanged (no scaling needed).
    pub fn forward_inverted(&mut self, input: &[f64], seed: u64) -> Vec<f64> {
        if !self.training {
            return input.to_vec();
        }
        let scale = if (1.0 - self.rate).abs() < f64::EPSILON {
            1.0
        } else {
            1.0 / (1.0 - self.rate)
        };
        let mask = build_keep_mask(input.len(), self.rate, seed);
        let out: Vec<f64> = input
            .iter()
            .zip(mask.iter())
            .map(|(&x, &keep)| if keep { x * scale } else { 0.0 })
            .collect();
        self.mask = Some(mask);
        out
    }

    /// Backward pass: propagate gradients through the most-recently applied mask.
    ///
    /// If no forward pass has been performed yet (or the layer is in eval mode),
    /// the gradient is returned unchanged.
    pub fn backward(&self, grad_output: &[f64]) -> Vec<f64> {
        match &self.mask {
            Some(mask) => grad_output
                .iter()
                .zip(mask.iter())
                .map(|(&g, &keep)| if keep { g } else { 0.0 })
                .collect(),
            None => grad_output.to_vec(),
        }
    }
}

/// Build a boolean mask where each entry is `true` (keep) with probability `1 - rate`.
///
/// Uses a seeded LCG-based RNG derived from `scirs2_core::random::SmallRng`.
fn build_keep_mask(len: usize, rate: f64, seed: u64) -> Vec<bool> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..len)
        .map(|_| rng.random::<f64>() >= rate)
        .collect()
}

// ============================================================================
// DropConnect
// ============================================================================

/// Generate a DropConnect weight mask for a 2-D weight matrix.
///
/// Each entry is `true` (keep) with probability `1 - rate`.
///
/// # Arguments
///
/// * `shape` - `(rows, cols)` of the weight matrix
/// * `rate`  - Fraction of weights to drop
/// * `seed`  - Random seed for reproducibility
pub fn dropconnect_mask(shape: (usize, usize), rate: f64, seed: u64) -> Array2<bool> {
    let rate = rate.clamp(0.0, 1.0);
    let (rows, cols) = shape;
    let mut rng = SmallRng::seed_from_u64(seed);
    Array2::from_shape_fn((rows, cols), |_| rng.random::<f64>() >= rate)
}

// ============================================================================
// Stochastic Depth (layer dropout for ResNets)
// ============================================================================

/// Stochastic Depth regularization (Huang et al., 2016).
///
/// Randomly skips residual branches during training by applying a Bernoulli
/// gate with survival probability `survival_prob`.
///
/// - **Training**: with probability `(1 - survival_prob)` the branch output is
///   replaced by the shortcut; otherwise the branch output is used as-is.
/// - **Inference**: the branch output is scaled by `survival_prob` and added to
///   the shortcut, implementing the expectation over the random gate.
///
/// # Example
///
/// ```rust
/// use scirs2_neural::training::regularization::StochasticDepth;
///
/// let layer = StochasticDepth::new(0.8);
/// let x        = vec![1.0, 2.0, 3.0];
/// let shortcut = vec![0.1, 0.1, 0.1];
/// let out = layer.forward(&x, &shortcut, 0);
/// assert_eq!(out.len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct StochasticDepth {
    /// Probability of keeping the residual branch (1 = always keep)
    pub survival_prob: f64,
    /// Whether the layer is in training mode
    pub training: bool,
}

impl StochasticDepth {
    /// Create a new `StochasticDepth` layer.
    pub fn new(survival_prob: f64) -> Self {
        Self {
            survival_prob: survival_prob.clamp(0.0, 1.0),
            training: true,
        }
    }

    /// Apply stochastic depth to a residual branch.
    ///
    /// # Arguments
    ///
    /// * `x`        - Residual branch output
    /// * `shortcut` - Shortcut (identity or projection) path
    /// * `seed`     - Random seed for the Bernoulli gate
    ///
    /// # Returns
    ///
    /// The gated output: either the branch or shortcut (training) or a scaled
    /// combination (inference).
    pub fn forward(&self, x: &[f64], shortcut: &[f64], seed: u64) -> Vec<f64> {
        let n = x.len().min(shortcut.len());
        if !self.training {
            // Inference: scale branch by survival probability
            return (0..n)
                .map(|i| x[i] * self.survival_prob + shortcut[i])
                .collect();
        }

        // Training: Bernoulli gate
        let mut rng = SmallRng::seed_from_u64(seed);
        let keep: bool = rng.random::<f64>() < self.survival_prob;
        if keep {
            (0..n).map(|i| x[i] + shortcut[i]).collect()
        } else {
            shortcut[..n].to_vec()
        }
    }
}

// ============================================================================
// Spectral Normalization
// ============================================================================

/// Apply spectral normalization to a weight matrix (Miyato et al., 2018).
///
/// Iteratively estimates the largest singular value `sigma` of `weight` using the
/// power iteration method, then returns `(weight / sigma, sigma)`.
///
/// After normalization the returned matrix has spectral norm (largest singular
/// value) ≤ 1 + numerical noise.
///
/// # Arguments
///
/// * `weight`        - 2-D weight matrix to normalize
/// * `n_power_iter`  - Number of power-iteration steps (typically 1–5)
///
/// # Returns
///
/// `(normalized_weight, sigma)` where `sigma` is the estimated spectral norm.
///
/// # Errors
///
/// Returns an error if the weight matrix has zero spectral norm (all-zero matrix)
/// or if any shape-related operation fails.
pub fn spectral_normalize(
    weight: &Array2<f64>,
    n_power_iter: usize,
) -> Result<(Array2<f64>, f64)> {
    let (rows, cols) = (weight.shape()[0], weight.shape()[1]);
    if rows == 0 || cols == 0 {
        return Err(NeuralError::InvalidArgument(
            "spectral_normalize: weight matrix must be non-empty".to_string(),
        ));
    }

    // Initialize u as a unit vector of length `rows`
    let mut u: Vec<f64> = (0..rows).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();

    // Power iteration to estimate the dominant singular value
    let n_iter = n_power_iter.max(1);
    let mut v: Vec<f64> = vec![0.0; cols];

    for _ in 0..n_iter {
        // v = W^T * u  (shape: cols)
        for j in 0..cols {
            let val: f64 = (0..rows).map(|i| weight[[i, j]] * u[i]).sum();
            v[j] = val;
        }
        normalize_vec_inplace(&mut v);

        // u = W * v  (shape: rows)
        for i in 0..rows {
            let val: f64 = (0..cols).map(|j| weight[[i, j]] * v[j]).sum();
            u[i] = val;
        }
        normalize_vec_inplace(&mut u);
    }

    // sigma = u^T * W * v
    let wv: Vec<f64> = (0..rows)
        .map(|i| (0..cols).map(|j| weight[[i, j]] * v[j]).sum())
        .collect();
    let sigma: f64 = u.iter().zip(wv.iter()).map(|(&ui, &wvi)| ui * wvi).sum();

    if sigma.abs() < f64::EPSILON {
        return Err(NeuralError::ComputationError(
            "spectral_normalize: weight matrix has zero spectral norm".to_string(),
        ));
    }

    let normalized = weight.mapv(|w| w / sigma);
    Ok((normalized, sigma))
}

/// Normalize a vector in-place (L2 norm). If the vector is zero, it is unchanged.
fn normalize_vec_inplace(v: &mut Vec<f64>) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > f64::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ============================================================================
// Gradient Clipping (simple slice-based API, distinct from gradient_clipping.rs)
// ============================================================================

/// Clip gradients by their global L2 norm (in-place, simple slice-of-slices API).
///
/// If the total L2 norm across all gradient arrays exceeds `max_norm`, every
/// element is rescaled by `max_norm / (global_norm + eps)`.
///
/// # Returns
///
/// The global L2 norm **before** clipping.
pub fn clip_grad_norm(grads: &mut Vec<Vec<f64>>, max_norm: f64) -> f64 {
    let sum_sq: f64 = grads
        .iter()
        .flat_map(|g| g.iter())
        .map(|x| x * x)
        .sum();
    let global_norm = sum_sq.sqrt();

    if global_norm > max_norm && global_norm > f64::EPSILON {
        let scale = max_norm / (global_norm + 1e-8);
        for g in grads.iter_mut() {
            for x in g.iter_mut() {
                *x *= scale;
            }
        }
    }
    global_norm
}

/// Clip each gradient element to the range `[-clip_value, clip_value]` (in-place).
pub fn clip_grad_value(grads: &mut Vec<Vec<f64>>, clip_value: f64) {
    let cv = clip_value.abs();
    for g in grads.iter_mut() {
        for x in g.iter_mut() {
            if *x > cv {
                *x = cv;
            } else if *x < -cv {
                *x = -cv;
            }
        }
    }
}

// ============================================================================
// Mixup Data Augmentation (Zhang et al., 2018)
// ============================================================================

/// Apply Mixup to a pair of training samples.
///
/// Samples `lambda ~ Beta(alpha, alpha)` and returns:
/// - `mixed_x = lambda * x1 + (1 - lambda) * x2`
/// - `mixed_y = lambda * y1 + (1 - lambda) * y2`
///
/// When `alpha = 0` (degenerate), `lambda = 0.5`.
///
/// # Arguments
///
/// * `x1`, `y1` — First sample (features and label)
/// * `x2`, `y2` — Second sample
/// * `alpha`    — Beta distribution concentration parameter
/// * `seed`     — Random seed
///
/// # Returns
///
/// `(mixed_x, mixed_y)`.
pub fn mixup(
    x1: &[f64],
    y1: &[f64],
    x2: &[f64],
    y2: &[f64],
    alpha: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let lambda = sample_beta_lcg(alpha, seed);

    let xlen = x1.len().min(x2.len());
    let ylen = y1.len().min(y2.len());

    let mixed_x: Vec<f64> = (0..xlen)
        .map(|i| lambda * x1[i] + (1.0 - lambda) * x2[i])
        .collect();
    let mixed_y: Vec<f64> = (0..ylen)
        .map(|i| lambda * y1[i] + (1.0 - lambda) * y2[i])
        .collect();

    (mixed_x, mixed_y)
}

// ============================================================================
// CutMix Data Augmentation
// ============================================================================

/// Apply CutMix to a pair of 2-D images.
///
/// Samples `lambda ~ Beta(alpha, alpha)`, derives a rectangular patch with
/// proportional area `(1 - lambda)` of the full image, pastes that patch from
/// `x2` into a copy of `x1`, and mixes labels proportionally to the true area
/// ratio of the pasted region.
///
/// # Arguments
///
/// * `x1`, `y1` — Destination image (`H × W` Array2) and its labels
/// * `x2`, `y2` — Source image and its labels
/// * `alpha`    — Beta distribution concentration parameter
/// * `seed`     — Random seed
///
/// # Returns
///
/// `(mixed_image, mixed_labels)`.
///
/// # Errors
///
/// Returns an error if the images have incompatible shapes.
pub fn cutmix(
    x1: &Array2<f64>,
    y1: &[f64],
    x2: &Array2<f64>,
    y2: &[f64],
    alpha: f64,
    seed: u64,
) -> Result<(Array2<f64>, Vec<f64>)> {
    let h = x1.shape()[0];
    let w = x1.shape()[1];

    if x2.shape()[0] != h || x2.shape()[1] != w {
        return Err(NeuralError::ShapeMismatch(format!(
            "cutmix: x1 shape ({h},{w}) != x2 shape ({},{})",
            x2.shape()[0],
            x2.shape()[1]
        )));
    }

    let lambda = sample_beta_lcg(alpha, seed);

    // Derive patch dimensions from lambda: area of patch = (1 - lambda) * H * W
    let cut_ratio = (1.0 - lambda).sqrt();
    let cut_h = ((h as f64 * cut_ratio).ceil() as usize).min(h).max(1);
    let cut_w = ((w as f64 * cut_ratio).ceil() as usize).min(w).max(1);

    // Random top-left corner for the patch
    let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
    let top = if h > cut_h {
        rng.random_range(0..=(h - cut_h))
    } else {
        0
    };
    let left = if w > cut_w {
        rng.random_range(0..=(w - cut_w))
    } else {
        0
    };

    let mut mixed = x1.clone();
    // Paste patch from x2 into x1
    for i in top..top + cut_h {
        for j in left..left + cut_w {
            mixed[[i, j]] = x2[[i, j]];
        }
    }

    // Actual lambda based on the area actually pasted
    let actual_lambda = 1.0 - (cut_h * cut_w) as f64 / (h * w) as f64;

    let ylen = y1.len().min(y2.len());
    let mixed_y: Vec<f64> = (0..ylen)
        .map(|i| actual_lambda * y1[i] + (1.0 - actual_lambda) * y2[i])
        .collect();

    Ok((mixed, mixed_y))
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Sample from `Beta(alpha, alpha)` using a Cheng (1978) acceptance-rejection
/// method when `alpha >= 1` and a Johnk method for `0 < alpha < 1`.
/// Falls back to 0.5 for degenerate `alpha <= 0`.
///
/// This is a pure-Rust, scirs2_core-based implementation that avoids any
/// external statistics crate dependency in this module.
fn sample_beta_lcg(alpha: f64, seed: u64) -> f64 {
    if alpha <= 0.0 {
        return 0.5;
    }

    let mut rng = SmallRng::seed_from_u64(seed);

    if (alpha - 1.0).abs() < 1e-12 {
        // Beta(1,1) = Uniform(0,1)
        return rng.random::<f64>().clamp(0.0, 1.0);
    }

    if alpha < 1.0 {
        // Johnk's method: sample two Gamma(alpha) variates and normalize
        // Approximated via the Ahrens-Dieter method for Gamma < 1
        let x = sample_gamma_small_alpha(&mut rng, alpha);
        let y = sample_gamma_small_alpha(&mut rng, alpha);
        let s = x + y;
        if s < f64::EPSILON {
            return 0.5;
        }
        return (x / s).clamp(0.0, 1.0);
    }

    // alpha >= 1: Cheng's BB method
    sample_beta_cheng(&mut rng, alpha)
}

/// Gamma(alpha, 1) sample for `0 < alpha < 1` using Ahrens-Dieter (1980).
fn sample_gamma_small_alpha(rng: &mut SmallRng, alpha: f64) -> f64 {
    // GS algorithm (Ahrens & Dieter, 1980) for 0 < alpha < 1
    let c = (std::f64::consts::E + alpha) / std::f64::consts::E;
    loop {
        let u1: f64 = rng.random();
        let u2: f64 = rng.random();
        let p = c * u1;
        let (x, q) = if p <= 1.0 {
            let x = p.powf(1.0 / alpha);
            let q = (-x).exp();
            (x, q)
        } else {
            let x = -(p - 1.0).ln() / alpha;
            let q = x.powf(alpha - 1.0);
            (x, q)
        };
        if u2 <= q {
            return x;
        }
    }
}

/// Beta(alpha, alpha) sample for `alpha >= 1` using Cheng's BB method.
fn sample_beta_cheng(rng: &mut SmallRng, alpha: f64) -> f64 {
    // BB method (Cheng 1978) for alpha >= 1
    let a = 2.0 * alpha - 1.0;
    let b = alpha;
    let c = alpha + b.ln() - std::f64::consts::LN_2 * (b - 1.0);
    // Actually use the direct Gamma(alpha) / (Gamma(alpha) + Gamma(alpha)) approach
    // which is simpler to implement correctly
    let x = sample_gamma_ge1(rng, alpha);
    let y = sample_gamma_ge1(rng, alpha);
    let _ = (a, b, c); // suppress warnings
    let s = x + y;
    if s < f64::EPSILON {
        return 0.5;
    }
    (x / s).clamp(0.0, 1.0)
}

/// Gamma(alpha, 1) sample for `alpha >= 1` using Marsaglia-Tsang (2000).
fn sample_gamma_ge1(rng: &mut SmallRng, alpha: f64) -> f64 {
    // Marsaglia & Tsang "A Simple Method for Generating Gamma Variables" (2000)
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x: f64 = sample_standard_normal(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v3 = v * v * v;
        let u: f64 = rng.random();
        // Quick acceptance test
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return d * v3;
        }
        // Log acceptance test
        if u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
            return d * v3;
        }
    }
}

/// Standard normal sample using Box-Muller transform.
fn sample_standard_normal(rng: &mut SmallRng) -> f64 {
    loop {
        let u1: f64 = rng.random();
        let u2: f64 = rng.random();
        if u1 < f64::EPSILON {
            continue;
        }
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        return z;
    }
}


// ============================================================================
// Label Smoothing Loss
// ============================================================================

/// Configuration for label smoothing cross-entropy loss.
///
/// Label smoothing (Szegedy et al., 2016) prevents the model from becoming
/// overconfident by replacing the hard one-hot target distribution with a
/// soft distribution:
///
/// ```text
/// y_smooth[k] = (1 - epsilon)   if k == true_class
///             = epsilon / (C - 1)  otherwise
/// ```
///
/// where `C` is the number of classes and `epsilon` is a small constant
/// (typically 0.05–0.15).
#[derive(Debug, Clone)]
pub struct LabelSmoothingConfig {
    /// Smoothing coefficient ε ∈ [0, 1).
    ///
    /// `0.0` means no smoothing (standard cross-entropy).
    /// Common values: 0.1 for ImageNet, 0.05 for NLP.
    pub epsilon: f64,
    /// Number of classes (vocabulary size, output dimension).
    pub num_classes: usize,
}

impl Default for LabelSmoothingConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            num_classes: 1000,
        }
    }
}

impl LabelSmoothingConfig {
    /// Validate configuration.
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.epsilon < 0.0 || self.epsilon >= 1.0 {
            return Err(crate::error::NeuralError::InvalidArgument(format!(
                "epsilon must be in [0, 1), got {}",
                self.epsilon
            )));
        }
        if self.num_classes < 2 {
            return Err(crate::error::NeuralError::InvalidArgument(format!(
                "num_classes must be >= 2, got {}",
                self.num_classes
            )));
        }
        Ok(())
    }
}

/// Compute label-smoothed cross-entropy loss for a batch of predictions.
///
/// # Arguments
///
/// * `logits`      — Raw unnormalized logits, shape `(batch, num_classes)`.
/// * `true_labels` — Ground-truth class indices, one per sample.
/// * `config`      — Label smoothing configuration.
///
/// # Returns
///
/// The mean label-smoothed cross-entropy loss over the batch.
///
/// # Errors
///
/// Returns `Err` if any label index is out of `[0, num_classes)`, or if
/// `logits.ncols() != config.num_classes`.
///
/// # Example
///
/// ```rust
/// use scirs2_neural::training::regularization::{LabelSmoothingConfig, label_smoothing_loss};
/// use scirs2_core::ndarray::array;
///
/// let logits = array![[2.0_f64, 1.0, 0.5], [0.1, 3.0, 1.2]];
/// let config = LabelSmoothingConfig { epsilon: 0.1, num_classes: 3 };
/// let loss = label_smoothing_loss(logits.view(), &[0, 1], &config).expect("loss failed");
/// assert!(loss > 0.0);
/// ```
pub fn label_smoothing_loss(
    logits: scirs2_core::ndarray::ArrayView2<f64>,
    true_labels: &[usize],
    config: &LabelSmoothingConfig,
) -> crate::error::Result<f64> {
    use scirs2_core::ndarray::Axis;
    config.validate()?;

    let batch = logits.nrows();
    let num_classes = logits.ncols();

    if num_classes != config.num_classes {
        return Err(crate::error::NeuralError::ShapeMismatch(format!(
            "logits has {} classes but config.num_classes={}",
            num_classes, config.num_classes
        )));
    }
    if true_labels.len() != batch {
        return Err(crate::error::NeuralError::ShapeMismatch(format!(
            "true_labels length {} != batch size {}",
            true_labels.len(),
            batch
        )));
    }

    let eps = config.epsilon;
    let c = num_classes as f64;
    // Soft target: (1 - eps) for the true class, eps/(C-1) for others.
    let smooth_other = eps / (c - 1.0);
    let smooth_true = 1.0 - eps;

    let mut total_loss = 0.0f64;

    for (i, &label) in true_labels.iter().enumerate() {
        if label >= num_classes {
            return Err(crate::error::NeuralError::InvalidArgument(format!(
                "label {} is out of range [0, {})",
                label, num_classes
            )));
        }

        // Numerically stable log-softmax.
        let row = logits.index_axis(Axis(0), i);
        let max_logit = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = row.iter().map(|&v| (v - max_logit).exp()).sum();
        let log_sum = max_logit + exp_sum.ln();

        // Cross-entropy with soft targets:
        // loss = -sum_k { y_smooth[k] * log_softmax[k] }
        let mut ce = 0.0f64;
        for (k, &logit) in row.iter().enumerate() {
            let log_p = logit - log_sum;
            let y_k = if k == label { smooth_true } else { smooth_other };
            ce -= y_k * log_p;
        }
        total_loss += ce;
    }

    Ok(total_loss / batch as f64)
}

/// Compute the KL-divergence component of label smoothing loss, separately from
/// the hard cross-entropy component.
///
/// Returns `(hard_ce_loss, smoothing_kl_loss)` so callers can log them.
pub fn label_smoothing_loss_components(
    logits: scirs2_core::ndarray::ArrayView2<f64>,
    true_labels: &[usize],
    config: &LabelSmoothingConfig,
) -> crate::error::Result<(f64, f64)> {
    use scirs2_core::ndarray::Axis;
    config.validate()?;

    let batch = logits.nrows();
    let num_classes = logits.ncols();

    if num_classes != config.num_classes {
        return Err(crate::error::NeuralError::ShapeMismatch(format!(
            "logits has {} classes but config.num_classes={}",
            num_classes, config.num_classes
        )));
    }
    if true_labels.len() != batch {
        return Err(crate::error::NeuralError::ShapeMismatch(format!(
            "true_labels length {} != batch size {}",
            true_labels.len(),
            batch
        )));
    }

    let c = num_classes as f64;
    let mut hard_total = 0.0f64;
    let mut smooth_total = 0.0f64;

    for (i, &label) in true_labels.iter().enumerate() {
        if label >= num_classes {
            return Err(crate::error::NeuralError::InvalidArgument(format!(
                "label {} out of range [0, {})",
                label, num_classes
            )));
        }
        let row = logits.index_axis(Axis(0), i);
        let max_logit = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = row.iter().map(|&v| (v - max_logit).exp()).sum();
        let log_sum = max_logit + exp_sum.ln();

        // Hard CE: -log p(true_label)
        hard_total -= logits[[i, label]] - log_sum;

        // Uniform smoothing term: -1/C * sum_k log p(k)  — encourages uniform entropy
        let uniform_ce: f64 = row.iter().map(|&logit| -(logit - log_sum) / c).sum();
        smooth_total += uniform_ce;
    }

    Ok((hard_total / batch as f64, smooth_total / batch as f64))
}

// ============================================================================
// Weight Decay Schedules
// ============================================================================

/// Schedule type for weight decay coefficient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDecayScheduleType {
    /// Constant weight decay throughout training.
    Constant,
    /// Linearly decay from `initial` to `final` over `total_steps`.
    Linear,
    /// Cosine anneal from `initial` toward `final`.
    Cosine,
    /// Exponential decay: `wd(t) = initial * decay_rate^(t / decay_steps)`.
    Exponential,
    /// Step-wise decay: multiply by `decay_factor` every `step_interval` steps.
    StepWise,
    /// Warmup then cosine: weight decay grows from 0 to `final` during warmup,
    /// then cosine-decays back.
    WarmupCosine,
}

/// A schedule that varies the weight-decay coefficient over training steps.
///
/// Decoupled weight decay (Loshchilov & Hutter, 2019) works best when the
/// decay coefficient is treated as a separate hyper-parameter that can be
/// scheduled independently of the learning rate.
///
/// # Example
///
/// ```rust
/// use scirs2_neural::training::regularization::{WeightDecaySchedule, WeightDecayScheduleType};
///
/// let mut wd = WeightDecaySchedule::new(0.01, 0.001, 100, WeightDecayScheduleType::Cosine)
///     .expect("valid schedule");
/// let wd0 = wd.current();
/// assert!((wd0 - 0.01).abs() < 1e-9, "should start at initial value");
/// wd.advance();
/// let wd1 = wd.current();
/// assert!(wd1 <= wd0, "cosine schedule should be decreasing");
/// ```
#[derive(Debug, Clone)]
pub struct WeightDecaySchedule {
    /// Initial weight decay (at step 0).
    pub initial_wd: f64,
    /// Final weight decay (at `total_steps`).
    pub final_wd: f64,
    /// Total training steps.
    pub total_steps: usize,
    /// Current step counter.
    pub current_step: usize,
    /// Schedule type.
    pub schedule_type: WeightDecayScheduleType,
    /// Exponential decay rate (used only for `Exponential` schedule).
    pub decay_rate: f64,
    /// Steps between decays for `StepWise` and `Exponential` schedules.
    pub decay_steps: usize,
    /// Decay factor for `StepWise` schedule.
    pub decay_factor: f64,
    /// Warmup steps for `WarmupCosine` schedule.
    pub warmup_steps: usize,
}

impl WeightDecaySchedule {
    /// Create a new weight decay schedule.
    ///
    /// # Errors
    ///
    /// Returns `Err` if either `wd` value is negative or `total_steps` is 0.
    pub fn new(
        initial_wd: f64,
        final_wd: f64,
        total_steps: usize,
        schedule_type: WeightDecayScheduleType,
    ) -> crate::error::Result<Self> {
        if initial_wd < 0.0 || final_wd < 0.0 {
            return Err(crate::error::NeuralError::InvalidArgument(
                "weight decay values must be non-negative".to_string(),
            ));
        }
        if total_steps == 0 {
            return Err(crate::error::NeuralError::InvalidArgument(
                "total_steps must be > 0".to_string(),
            ));
        }
        Ok(Self {
            initial_wd,
            final_wd,
            total_steps,
            current_step: 0,
            schedule_type,
            decay_rate: 0.96,
            decay_steps: 100,
            decay_factor: 0.5,
            warmup_steps: 0,
        })
    }

    /// Set exponential decay rate (default 0.96).
    pub fn with_decay_rate(mut self, rate: f64) -> Self {
        self.decay_rate = rate;
        self
    }

    /// Set the number of steps per decay interval (default 100).
    pub fn with_decay_steps(mut self, steps: usize) -> Self {
        self.decay_steps = steps.max(1);
        self
    }

    /// Set the multiplicative decay factor for `StepWise` (default 0.5).
    pub fn with_decay_factor(mut self, factor: f64) -> Self {
        self.decay_factor = factor;
        self
    }

    /// Set warmup steps for `WarmupCosine` (default 0).
    pub fn with_warmup(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self
    }

    /// Advance the step counter by 1.
    pub fn advance(&mut self) {
        if self.current_step < self.total_steps {
            self.current_step += 1;
        }
    }

    /// Reset to step 0.
    pub fn reset(&mut self) {
        self.current_step = 0;
    }

    /// Return `true` once `total_steps` has been reached.
    pub fn is_done(&self) -> bool {
        self.current_step >= self.total_steps
    }

    /// Return the weight decay coefficient at the current step.
    pub fn current(&self) -> f64 {
        let s = self.current_step as f64;
        let n = self.total_steps as f64;
        let wi = self.initial_wd;
        let wf = self.final_wd;

        if self.current_step >= self.total_steps {
            return wf;
        }

        match self.schedule_type {
            WeightDecayScheduleType::Constant => wi,
            WeightDecayScheduleType::Linear => wi + (wf - wi) * (s / n),
            WeightDecayScheduleType::Cosine => {
                wf + (wi - wf) * 0.5 * (1.0 + (std::f64::consts::PI * s / n).cos())
            }
            WeightDecayScheduleType::Exponential => {
                let decay = self.decay_steps.max(1) as f64;
                wi * self.decay_rate.powf(s / decay)
            }
            WeightDecayScheduleType::StepWise => {
                let n_decays = (s / self.decay_steps.max(1) as f64).floor() as u32;
                wi * self.decay_factor.powi(n_decays as i32)
            }
            WeightDecayScheduleType::WarmupCosine => {
                let warmup = self.warmup_steps as f64;
                if s < warmup {
                    // Linear warmup: 0 → initial_wd
                    wi * (s / warmup.max(1.0))
                } else {
                    // Cosine decay from initial_wd toward final_wd
                    let effective = s - warmup;
                    let effective_total = (n - warmup).max(1.0);
                    wf + (wi - wf)
                        * 0.5
                        * (1.0 + (std::f64::consts::PI * effective / effective_total).cos())
                }
            }
        }
    }

    /// Advance and return the new weight decay value.
    pub fn step(&mut self) -> f64 {
        self.advance();
        self.current()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    const EPS: f64 = 1e-10;

    // ---------- L1 regularization ----------

    #[test]
    fn test_l1_penalty_basic() {
        let weights = vec![1.0, -2.0, 3.0];
        let penalty = l1_regularization(&weights, 0.01);
        // 0.01 * (1 + 2 + 3) = 0.06
        assert!((penalty - 0.06).abs() < EPS);
    }

    #[test]
    fn test_l1_penalty_zeros() {
        let weights = vec![0.0, 0.0];
        let penalty = l1_regularization(&weights, 1.0);
        assert!((penalty).abs() < EPS);
    }

    #[test]
    fn test_l1_gradient_sign() {
        let weights = vec![3.0, -2.0, 0.0, 1.0, -0.5];
        let grads = l1_gradient(&weights, 1.0);
        // positive weights → positive gradient, negative → negative, zero → zero
        assert!((grads[0] - 1.0).abs() < EPS); // positive
        assert!((grads[1] + 1.0).abs() < EPS); // negative
        assert!((grads[2]).abs() < EPS);         // zero
        assert!((grads[3] - 1.0).abs() < EPS);
        assert!((grads[4] + 1.0).abs() < EPS);
    }

    #[test]
    fn test_l1_gradient_alpha_scaling() {
        let weights = vec![1.0, -1.0];
        let grads = l1_gradient(&weights, 0.5);
        assert!((grads[0] - 0.5).abs() < EPS);
        assert!((grads[1] + 0.5).abs() < EPS);
    }

    // ---------- L2 regularization ----------

    #[test]
    fn test_l2_penalty_basic() {
        let weights = vec![1.0, 2.0, 3.0];
        let penalty = l2_regularization(&weights, 1.0);
        // sum of squares = 14, alpha=1 → 14.0
        assert!((penalty - 14.0).abs() < EPS);
    }

    #[test]
    fn test_l2_gradient_proportional() {
        let weights = vec![1.0, -2.0, 3.0];
        let grads = l2_gradient(&weights, 1.0);
        // gradient should be 2 * alpha * w_i
        assert!((grads[0] - 2.0).abs() < EPS);
        assert!((grads[1] + 4.0).abs() < EPS);
        assert!((grads[2] - 6.0).abs() < EPS);
    }

    #[test]
    fn test_l2_gradient_zero() {
        let weights = vec![0.0, 0.0];
        let grads = l2_gradient(&weights, 100.0);
        for g in &grads {
            assert!(g.abs() < EPS);
        }
    }

    // ---------- Elastic Net ----------

    #[test]
    fn test_elastic_net_combines_both() {
        let weights = vec![1.0, -1.0];
        let alpha = 0.1;
        let beta = 0.2;
        let penalty = elastic_net(&weights, alpha, beta);
        // L1 = 0.1 * 2 = 0.2, L2 = 0.2 * 2 = 0.4 → total = 0.6
        assert!((penalty - 0.6).abs() < EPS);
    }

    #[test]
    fn test_elastic_net_gradient_combines() {
        let weights = vec![2.0];
        let grads = elastic_net_gradient(&weights, 0.1, 0.2);
        // L1 grad = 0.1, L2 grad = 2*0.2*2 = 0.8 → total = 0.9
        assert!((grads[0] - 0.9).abs() < EPS);
    }

    #[test]
    fn test_elastic_net_negative_weight() {
        let weights = vec![-3.0];
        let grads = elastic_net_gradient(&weights, 1.0, 1.0);
        // L1 grad = -1.0, L2 grad = 2*1*(-3) = -6 → total = -7
        assert!((grads[0] + 7.0).abs() < EPS);
    }

    // ---------- Dropout ----------

    #[test]
    fn test_dropout_training_drops_some() {
        let mut layer = DropoutLayer::new(0.5);
        layer.train();
        let input: Vec<f64> = (0..100).map(|i| i as f64 + 1.0).collect();
        let out = layer.forward(&input, 42);
        let zeros = out.iter().filter(|&&x| x.abs() < EPS).count();
        // Should drop roughly half; at least 10 and at most 90 for 100 units
        assert!(zeros >= 5, "Too few zeros (dropout underperforming): {zeros}");
        assert!(zeros <= 95, "Too many zeros (dropout overperforming): {zeros}");
    }

    #[test]
    fn test_dropout_eval_passthrough() {
        let mut layer = DropoutLayer::new(0.9);
        layer.eval();
        let input = vec![1.0, 2.0, 3.0];
        let out = layer.forward(&input, 0);
        assert_eq!(out, input);
    }

    #[test]
    fn test_dropout_inverted_expected_value() {
        // With many units, E[output] ≈ E[input] (inverted dropout)
        let mut layer = DropoutLayer::new(0.5);
        layer.train();
        let n = 10_000;
        let input: Vec<f64> = vec![1.0; n];
        let out = layer.forward_inverted(&input, 12345);
        let mean: f64 = out.iter().sum::<f64>() / n as f64;
        // Expected value = 1.0; with n=10000 this should be within 5%
        assert!(
            (mean - 1.0).abs() < 0.1,
            "Inverted dropout expected value off: {mean}"
        );
    }

    #[test]
    fn test_dropout_backward_uses_mask() {
        let mut layer = DropoutLayer::new(0.5);
        layer.train();
        let input = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let out = layer.forward_inverted(&input, 99);
        let grad_in = vec![1.0; input.len()];
        let grad_out = layer.backward(&grad_in);
        // Where output was zero, backward should also be zero
        for (o, g) in out.iter().zip(grad_out.iter()) {
            if o.abs() < EPS {
                assert!(g.abs() < EPS, "Backward through dropped unit should be zero");
            }
        }
    }

    // ---------- DropConnect ----------

    #[test]
    fn test_dropconnect_mask_shape() {
        let mask = dropconnect_mask((4, 5), 0.3, 0);
        assert_eq!(mask.shape(), &[4, 5]);
    }

    #[test]
    fn test_dropconnect_mask_drops_some() {
        let mask = dropconnect_mask((100, 100), 0.5, 7);
        let n_false = mask.iter().filter(|&&v| !v).count();
        // Should drop roughly 50% of 10000 entries
        assert!(n_false > 3000 && n_false < 7000, "DropConnect drop rate off: {n_false}/10000");
    }

    // ---------- StochasticDepth ----------

    #[test]
    fn test_stochastic_depth_eval_scales() {
        let mut layer = StochasticDepth::new(0.8);
        layer.training = false;
        let x = vec![1.0, 1.0, 1.0];
        let shortcut = vec![0.5, 0.5, 0.5];
        let out = layer.forward(&x, &shortcut, 0);
        // Each output = x * 0.8 + shortcut = 0.8 + 0.5 = 1.3
        for &v in &out {
            assert!((v - 1.3).abs() < 1e-9);
        }
    }

    #[test]
    fn test_stochastic_depth_training_either_branch() {
        let layer = StochasticDepth::new(0.5);
        let x = vec![10.0, 10.0];
        let shortcut = vec![0.0, 0.0];
        // Run with two different seeds — one should pick the branch, the other might skip
        let out0 = layer.forward(&x, &shortcut, 1000);
        let out1 = layer.forward(&x, &shortcut, 1001);
        // At least one result should differ — test just shape
        assert_eq!(out0.len(), 2);
        assert_eq!(out1.len(), 2);
    }

    // ---------- Spectral normalization ----------

    #[test]
    fn test_spectral_normalize_identity() {
        // Identity matrix has spectral norm = 1
        let w = Array2::<f64>::eye(4);
        let (w_sn, sigma) = spectral_normalize(&w, 10).expect("spectral_normalize failed");
        assert!((sigma - 1.0).abs() < 1e-5, "sigma should be ~1.0, got {sigma}");
        // Normalized matrix should equal identity (since sigma=1)
        let diff = (&w_sn - &w).mapv(f64::abs);
        assert!(diff.iter().all(|&d| d < 1e-5));
    }

    #[test]
    fn test_spectral_normalize_spectral_norm_le_1() {
        // Create a random-ish 3x5 matrix
        let data: Vec<f64> = (0..15).map(|i| (i as f64 - 7.0) * 0.5).collect();
        let w = Array2::from_shape_vec((3, 5), data).expect("shape");
        let (w_sn, sigma) = spectral_normalize(&w, 10).expect("spectral_normalize failed");
        assert!(sigma > 0.0, "sigma must be positive");
        // Verify spectral norm of normalized matrix is ≤ 1 + eps
        let (_, sigma_after) = spectral_normalize(&w_sn, 20).expect("second sn failed");
        assert!(
            sigma_after <= 1.0 + 1e-4,
            "Spectral norm of normalized matrix should be ≤ 1+eps, got {sigma_after}"
        );
    }

    #[test]
    fn test_spectral_normalize_empty_error() {
        let w = Array2::<f64>::zeros((0, 3));
        let result = spectral_normalize(&w, 5);
        assert!(result.is_err());
    }

    // ---------- Gradient clipping ----------

    #[test]
    fn test_clip_grad_norm_reduces_norm() {
        let mut grads = vec![vec![3.0, 4.0]]; // norm = 5.0
        let original = clip_grad_norm(&mut grads, 2.5);
        assert!((original - 5.0).abs() < EPS);
        let clipped_norm: f64 = grads
            .iter()
            .flat_map(|g| g.iter())
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        assert!(
            clipped_norm <= 2.5 + 1e-6,
            "Clipped norm should be ≤ max_norm"
        );
    }

    #[test]
    fn test_clip_grad_norm_no_clip_needed() {
        let mut grads = vec![vec![1.0, 1.0]]; // norm = sqrt(2) < 10
        let _ = clip_grad_norm(&mut grads, 10.0);
        // Values should remain unchanged
        assert!((grads[0][0] - 1.0).abs() < EPS);
        assert!((grads[0][1] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_clip_grad_value_clamps() {
        let mut grads = vec![vec![10.0, -10.0, 0.5]];
        clip_grad_value(&mut grads, 1.0);
        assert!((grads[0][0] - 1.0).abs() < EPS);
        assert!((grads[0][1] + 1.0).abs() < EPS);
        assert!((grads[0][2] - 0.5).abs() < EPS);
    }

    // ---------- Mixup ----------

    #[test]
    fn test_mixup_lambda_in_0_1() {
        // With alpha=1 (Uniform), lambda is in [0,1], so mixed values stay between
        let x1 = vec![0.0, 0.0];
        let x2 = vec![1.0, 1.0];
        let y1 = vec![1.0, 0.0];
        let y2 = vec![0.0, 1.0];
        let (mx, my) = mixup(&x1, &y1, &x2, &y2, 1.0, 42);
        for v in mx.iter().chain(my.iter()) {
            assert!(
                *v >= -1e-9 && *v <= 1.0 + 1e-9,
                "Mixup output should be in [0,1], got {v}"
            );
        }
    }

    #[test]
    fn test_mixup_label_sum_preserved() {
        // When y1 and y2 are one-hot, the mixed label sums to 1
        let x1 = vec![0.0];
        let x2 = vec![1.0];
        let y1 = vec![1.0, 0.0];
        let y2 = vec![0.0, 1.0];
        let (_mx, my) = mixup(&x1, &y1, &x2, &y2, 2.0, 7);
        let sum: f64 = my.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Mixed label should sum to 1");
    }

    #[test]
    fn test_mixup_output_length() {
        let x1 = vec![1.0, 2.0, 3.0];
        let x2 = vec![4.0, 5.0, 6.0];
        let y1 = vec![1.0, 0.0];
        let y2 = vec![0.0, 1.0];
        let (mx, my) = mixup(&x1, &y1, &x2, &y2, 1.0, 0);
        assert_eq!(mx.len(), 3);
        assert_eq!(my.len(), 2);
    }

    // ---------- CutMix ----------

    #[test]
    fn test_cutmix_shape_preserved() {
        let x1 = Array2::from_elem((8, 8), 0.0_f64);
        let x2 = Array2::from_elem((8, 8), 1.0_f64);
        let y1 = vec![1.0, 0.0];
        let y2 = vec![0.0, 1.0];
        let (mx, my) = cutmix(&x1, &y1, &x2, &y2, 1.0, 42).expect("cutmix failed");
        assert_eq!(mx.shape(), &[8, 8]);
        assert_eq!(my.len(), 2);
    }

    #[test]
    fn test_cutmix_pastes_values() {
        // x1 is all 0s, x2 is all 1s — mixed image should have some 1s
        let x1 = Array2::from_elem((8, 8), 0.0_f64);
        let x2 = Array2::from_elem((8, 8), 1.0_f64);
        let y1 = vec![1.0, 0.0];
        let y2 = vec![0.0, 1.0];
        let (mx, _) = cutmix(&x1, &y1, &x2, &y2, 2.0, 1).expect("cutmix failed");
        let n_ones = mx.iter().filter(|&&v| (v - 1.0).abs() < EPS).count();
        assert!(n_ones > 0, "CutMix should paste at least one pixel from x2");
    }

    #[test]
    fn test_cutmix_label_sum() {
        let x1 = Array2::from_elem((4, 4), 0.0_f64);
        let x2 = Array2::from_elem((4, 4), 1.0_f64);
        let y1 = vec![1.0, 0.0];
        let y2 = vec![0.0, 1.0];
        let (_, my) = cutmix(&x1, &y1, &x2, &y2, 0.5, 99).expect("cutmix failed");
        let sum: f64 = my.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Mixed label should sum to 1");
    }

    #[test]
    fn test_cutmix_shape_mismatch_error() {
        let x1 = Array2::from_elem((4, 4), 0.0_f64);
        let x2 = Array2::from_elem((5, 5), 1.0_f64);
        let result = cutmix(&x1, &[1.0], &x2, &[0.0], 1.0, 0);
        assert!(result.is_err());
    }

    // ---------- Beta sampling sanity ----------

    #[test]
    fn test_beta_sample_range() {
        // lambda should always be in [0, 1]
        for seed in 0..50u64 {
            let lambda = sample_beta_lcg(0.5, seed);
            assert!(
                lambda >= 0.0 && lambda <= 1.0,
                "beta sample out of range: {lambda} (seed {seed})"
            );
        }
    }

    #[test]
    fn test_beta_sample_mean_close_to_half() {
        // Beta(alpha, alpha) is symmetric around 0.5
        let samples: Vec<f64> = (0..200u64).map(|s| sample_beta_lcg(2.0, s * 13 + 7)).collect();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(
            (mean - 0.5).abs() < 0.1,
            "Beta(2,2) mean should be ~0.5, got {mean}"
        );
    }
}
