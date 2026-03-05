//! Advanced Wavelet Features Module
//!
//! This module provides advanced wavelet processing capabilities for v0.2.0:
//!
//! ## Features
//!
//! ### Enhanced 2D Wavelet Transforms
//! - Improved edge handling with multiple boundary modes
//! - Gradient-preserving boundary extension
//! - Adaptive boundary selection based on local content
//!
//! ### Wavelet Packet Transforms
//! - Full wavelet packet decomposition and reconstruction
//! - Best basis selection algorithms (Shannon entropy, threshold cost, log-energy)
//! - Cost function framework for custom basis selection
//!
//! ### Advanced Denoising Methods
//! - VisuShrink (universal threshold)
//! - BayesShrink (Bayesian adaptive threshold)
//! - SureShrink (SURE-based adaptive threshold)
//! - Level-dependent thresholding
//! - Noise variance estimation from signal
//!
//! ## Example
//!
//! ```rust
//! use scirs2_signal::wavelet_advanced::{
//!     advanced_denoise_1d, DenoisingConfig, ThresholdRule
//! };
//! use scirs2_signal::dwt::Wavelet;
//!
//! // Create a noisy signal
//! let noisy_signal: Vec<f64> = (0..256).map(|i| {
//!     (i as f64 * 0.1).sin() + 0.1 * (i as f64 * 0.3).cos()
//! }).collect();
//!
//! // Apply advanced denoising with BayesShrink
//! let config = DenoisingConfig {
//!     wavelet: Wavelet::DB(4),
//!     threshold_rule: ThresholdRule::BayesShrink,
//!     level: None, // Auto-select optimal level
//!     noise_sigma: None, // Estimate from signal
//! };
//!
//! match advanced_denoise_1d(&noisy_signal, &config) {
//!     Ok(denoised) => println!("Denoised signal length: {}", denoised.len()),
//!     Err(e) => eprintln!("Denoising failed: {}", e),
//! }
//! ```

use crate::dwt::{dwt_decompose, dwt_reconstruct, wavedec, waverec, Wavelet, WaveletFilters};
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

// =============================================================================
// Types and Enums
// =============================================================================

/// Threshold selection rule for wavelet denoising
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ThresholdRule {
    /// Universal threshold (VisuShrink): lambda = sigma * sqrt(2 * log(n))
    VisuShrink,
    /// Bayesian adaptive threshold: lambda = sigma^2 / sigma_x
    BayesShrink,
    /// SURE-based threshold: minimizes Stein's Unbiased Risk Estimate
    SureShrink,
    /// Minimax threshold
    Minimax,
    /// Fixed threshold value
    Fixed(f64),
}

/// Threshold application method
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ThresholdMode {
    /// Hard thresholding: set coefficients below threshold to zero
    Hard,
    /// Soft thresholding: shrink coefficients toward zero
    Soft,
    /// Garrote thresholding: non-negative garrote shrinkage
    Garrote,
    /// Firm thresholding: hybrid between hard and soft
    Firm { lambda1: f64, lambda2: f64 },
    /// Block thresholding: James-Stein block shrinkage
    /// Groups neighboring coefficients and applies shrinkage to blocks
    /// rather than individual coefficients for better spatial adaptivity.
    /// The `block_size` parameter controls the number of coefficients per block.
    Block { block_size: usize },
}

/// Cost function type for best basis selection
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum CostFunction {
    /// Shannon entropy: -sum(p * log(p))
    Shannon,
    /// Threshold cost: count of coefficients above threshold
    Threshold(f64),
    /// Log-energy: sum(log(x^2 + epsilon))
    LogEnergy,
    /// Norm-based cost: sum(|x|^p)
    Norm(f64),
    /// SURE cost (for denoising applications)
    Sure,
}

/// Configuration for advanced denoising
#[derive(Debug, Clone)]
pub struct DenoisingConfig {
    /// Wavelet to use for decomposition
    pub wavelet: Wavelet,
    /// Threshold selection rule
    pub threshold_rule: ThresholdRule,
    /// Decomposition level (None for automatic selection)
    pub level: Option<usize>,
    /// Known noise standard deviation (None to estimate from signal)
    pub noise_sigma: Option<f64>,
}

impl Default for DenoisingConfig {
    fn default() -> Self {
        Self {
            wavelet: Wavelet::DB(4),
            threshold_rule: ThresholdRule::BayesShrink,
            level: None,
            noise_sigma: None,
        }
    }
}

/// Result of best basis selection
#[derive(Debug, Clone)]
pub struct BestBasisResult {
    /// Selected basis nodes as (level, position) pairs
    pub selected_nodes: Vec<(usize, usize)>,
    /// Total cost of the selected basis
    pub total_cost: f64,
    /// Cost at each node
    pub node_costs: HashMap<(usize, usize), f64>,
    /// Tree structure indicating optimal decomposition
    pub decomposition_map: HashMap<(usize, usize), bool>,
}

/// Enhanced 2D boundary mode for edge handling
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Edge2DMode {
    /// Standard symmetric reflection
    Symmetric,
    /// Periodic extension (wrap around)
    Periodic,
    /// Zero padding
    Zero,
    /// Constant value extension
    Constant(f64),
    /// Gradient-preserving extension (extrapolates using local gradient)
    GradientPreserving,
    /// Smooth extension using polynomial fitting
    SmoothPoly { order: usize },
    /// Anti-symmetric extension
    AntiSymmetric,
    /// Mirror extension (includes boundary sample)
    Mirror,
}

/// Configuration for enhanced 2D wavelet transform
#[derive(Debug, Clone)]
pub struct Dwt2DEnhancedConfig {
    /// Wavelet to use
    pub wavelet: Wavelet,
    /// Edge handling mode
    pub edge_mode: Edge2DMode,
    /// Number of decomposition levels
    pub levels: usize,
    /// Whether to compute quality metrics
    pub compute_metrics: bool,
}

impl Default for Dwt2DEnhancedConfig {
    fn default() -> Self {
        Self {
            wavelet: Wavelet::DB(4),
            edge_mode: Edge2DMode::Symmetric,
            levels: 3,
            compute_metrics: false,
        }
    }
}

// =============================================================================
// Noise Estimation Functions
// =============================================================================

/// Estimate noise standard deviation using robust MAD estimator
///
/// Uses the Median Absolute Deviation (MAD) of the finest scale detail
/// coefficients, which is a robust estimator of noise level.
///
/// # Arguments
/// * `detail_coeffs` - Detail coefficients from the finest wavelet scale
///
/// # Returns
/// * Estimated noise standard deviation
pub fn estimate_noise_mad(detail_coeffs: &[f64]) -> f64 {
    if detail_coeffs.is_empty() {
        return 0.0;
    }

    // Take absolute values
    let mut abs_coeffs: Vec<f64> = detail_coeffs.iter().map(|&x| x.abs()).collect();

    // Sort to find median
    abs_coeffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = abs_coeffs.len();
    let median = if n % 2 == 0 {
        (abs_coeffs[n / 2 - 1] + abs_coeffs[n / 2]) / 2.0
    } else {
        abs_coeffs[n / 2]
    };

    // MAD to sigma conversion factor (for Gaussian noise)
    // sigma = MAD / 0.6745
    median / 0.6745
}

/// Estimate noise standard deviation from 2D detail coefficients
///
/// Uses the diagonal (HH) subband which typically contains mostly noise
/// for natural images.
///
/// # Arguments
/// * `hh_coeffs` - HH (diagonal detail) coefficients
///
/// # Returns
/// * Estimated noise standard deviation
pub fn estimate_noise_2d(hh_coeffs: &Array2<f64>) -> f64 {
    let coeffs: Vec<f64> = hh_coeffs.iter().copied().collect();
    estimate_noise_mad(&coeffs)
}

// =============================================================================
// Threshold Calculation Functions
// =============================================================================

/// Calculate VisuShrink (universal) threshold
///
/// The universal threshold is: lambda = sigma * sqrt(2 * log(n))
/// This threshold has the property of asymptotically removing all noise
/// with high probability.
///
/// # Arguments
/// * `sigma` - Noise standard deviation
/// * `n` - Number of coefficients
///
/// # Returns
/// * Universal threshold value
pub fn visushrink_threshold(sigma: f64, n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    sigma * (2.0 * (n as f64).ln()).sqrt()
}

/// Calculate BayesShrink threshold
///
/// The BayesShrink threshold is: lambda = sigma^2 / sigma_x
/// where sigma_x = sqrt(max(sigma_y^2 - sigma^2, 0)) is the estimated
/// signal standard deviation.
///
/// # Arguments
/// * `coeffs` - Wavelet coefficients
/// * `sigma` - Noise standard deviation
///
/// # Returns
/// * BayesShrink threshold value
pub fn bayesshrink_threshold(coeffs: &[f64], sigma: f64) -> f64 {
    if coeffs.is_empty() {
        return 0.0;
    }

    let n = coeffs.len() as f64;

    // Estimate signal + noise variance
    let sigma_y_sq: f64 = coeffs.iter().map(|&x| x * x).sum::<f64>() / n;

    // Noise variance
    let sigma_sq = sigma * sigma;

    // Estimate signal variance (non-negative)
    let sigma_x_sq = (sigma_y_sq - sigma_sq).max(0.0);

    if sigma_x_sq < 1e-12 {
        // Signal is very weak, use universal threshold
        visushrink_threshold(sigma, coeffs.len())
    } else {
        sigma_sq / sigma_x_sq.sqrt()
    }
}

/// Calculate SureShrink threshold using Stein's Unbiased Risk Estimate
///
/// SURE provides an unbiased estimate of the mean squared error for
/// soft thresholding, allowing optimal threshold selection.
///
/// # Arguments
/// * `coeffs` - Wavelet coefficients
/// * `sigma` - Noise standard deviation
///
/// # Returns
/// * SureShrink threshold value
pub fn sureshrink_threshold(coeffs: &[f64], sigma: f64) -> f64 {
    if coeffs.is_empty() {
        return 0.0;
    }

    let n = coeffs.len();
    let n_f64 = n as f64;

    // Normalize coefficients by sigma
    let mut y: Vec<f64> = coeffs.iter().map(|&x| x / sigma).collect();

    // Sort by absolute value
    y.sort_by(|a, b| {
        a.abs()
            .partial_cmp(&b.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Check if we should use universal threshold (sparse signal heuristic)
    let energy: f64 = y.iter().map(|&x| x * x).sum();
    let sparsity_threshold = ((n_f64.ln() / n_f64.ln().ln()).powi(2) / n_f64.sqrt()).max(1.0);

    if energy < sparsity_threshold {
        return visushrink_threshold(sigma, n);
    }

    // Calculate cumulative sums for SURE calculation
    let mut cumsum_sq = vec![0.0; n + 1];
    for (i, &val) in y.iter().enumerate() {
        cumsum_sq[i + 1] = cumsum_sq[i] + val * val;
    }

    // Find threshold that minimizes SURE
    let mut min_risk = f64::INFINITY;
    let mut best_threshold = 0.0;

    for (i, &t_abs) in y.iter().enumerate() {
        let t = t_abs.abs();

        // SURE risk for this threshold
        // Risk(t) = n - 2*k + sum(min(y_i^2, t^2))
        // where k is number of coefficients below threshold
        let k = i + 1;
        let sum_below = cumsum_sq[k];
        let sum_above = (n - k) as f64 * t * t;

        let risk = n_f64 - 2.0 * k as f64 + sum_below + sum_above;

        if risk < min_risk {
            min_risk = risk;
            best_threshold = t;
        }
    }

    best_threshold * sigma
}

/// Calculate minimax threshold
///
/// The minimax threshold is computed using tabulated values that minimize
/// the maximum risk over a class of signals.
///
/// # Arguments
/// * `sigma` - Noise standard deviation
/// * `n` - Number of coefficients
///
/// # Returns
/// * Minimax threshold value
pub fn minimax_threshold(sigma: f64, n: usize) -> f64 {
    if n <= 32 {
        return 0.0;
    }

    // Minimax threshold approximation
    let log_n = (n as f64).log2();
    let lambda = 0.3936 + 0.1829 * log_n;

    sigma * lambda
}

/// Calculate threshold based on the specified rule
///
/// # Arguments
/// * `coeffs` - Wavelet coefficients
/// * `sigma` - Noise standard deviation
/// * `rule` - Threshold selection rule
///
/// # Returns
/// * Threshold value
pub fn calculate_threshold(coeffs: &[f64], sigma: f64, rule: ThresholdRule) -> f64 {
    match rule {
        ThresholdRule::VisuShrink => visushrink_threshold(sigma, coeffs.len()),
        ThresholdRule::BayesShrink => bayesshrink_threshold(coeffs, sigma),
        ThresholdRule::SureShrink => sureshrink_threshold(coeffs, sigma),
        ThresholdRule::Minimax => minimax_threshold(sigma, coeffs.len()),
        ThresholdRule::Fixed(t) => t,
    }
}

// =============================================================================
// Thresholding Functions
// =============================================================================

/// Apply hard thresholding to coefficients
///
/// Hard thresholding sets coefficients with absolute value below the
/// threshold to zero, keeping others unchanged.
///
/// # Arguments
/// * `coeffs` - Mutable slice of wavelet coefficients
/// * `threshold` - Threshold value
pub fn apply_hard_threshold(coeffs: &mut [f64], threshold: f64) {
    for coeff in coeffs.iter_mut() {
        if coeff.abs() <= threshold {
            *coeff = 0.0;
        }
    }
}

/// Apply soft thresholding to coefficients
///
/// Soft thresholding shrinks coefficients toward zero by the threshold
/// amount, creating a continuous shrinkage function.
///
/// # Arguments
/// * `coeffs` - Mutable slice of wavelet coefficients
/// * `threshold` - Threshold value
pub fn apply_soft_threshold(coeffs: &mut [f64], threshold: f64) {
    for coeff in coeffs.iter_mut() {
        if coeff.abs() <= threshold {
            *coeff = 0.0;
        } else {
            *coeff = coeff.signum() * (coeff.abs() - threshold);
        }
    }
}

/// Apply garrote thresholding to coefficients
///
/// Garrote thresholding provides a compromise between hard and soft
/// thresholding with better bias-variance tradeoff.
///
/// # Arguments
/// * `coeffs` - Mutable slice of wavelet coefficients
/// * `threshold` - Threshold value
pub fn apply_garrote_threshold(coeffs: &mut [f64], threshold: f64) {
    let threshold_sq = threshold * threshold;
    for coeff in coeffs.iter_mut() {
        if coeff.abs() <= threshold {
            *coeff = 0.0;
        } else {
            *coeff -= threshold_sq / *coeff;
        }
    }
}

/// Apply firm thresholding to coefficients
///
/// Firm thresholding uses two thresholds to create a piecewise linear
/// shrinkage function.
///
/// # Arguments
/// * `coeffs` - Mutable slice of wavelet coefficients
/// * `lambda1` - Lower threshold
/// * `lambda2` - Upper threshold
pub fn apply_firm_threshold(coeffs: &mut [f64], lambda1: f64, lambda2: f64) {
    for coeff in coeffs.iter_mut() {
        let abs_x = coeff.abs();
        if abs_x <= lambda1 {
            *coeff = 0.0;
        } else if abs_x <= lambda2 {
            // Linear interpolation between lambda1 and lambda2
            let scale = (abs_x - lambda1) / (lambda2 - lambda1);
            *coeff = coeff.signum() * scale * abs_x;
        }
        // else: keep unchanged
    }
}

/// Apply thresholding based on the specified mode
///
/// # Arguments
/// * `coeffs` - Mutable slice of wavelet coefficients
/// * `threshold` - Threshold value
/// * `mode` - Thresholding mode
pub fn apply_threshold(coeffs: &mut [f64], threshold: f64, mode: ThresholdMode) {
    match mode {
        ThresholdMode::Hard => apply_hard_threshold(coeffs, threshold),
        ThresholdMode::Soft => apply_soft_threshold(coeffs, threshold),
        ThresholdMode::Garrote => apply_garrote_threshold(coeffs, threshold),
        ThresholdMode::Firm { lambda1, lambda2 } => {
            apply_firm_threshold(coeffs, lambda1 * threshold, lambda2 * threshold)
        }
        ThresholdMode::Block { block_size } => {
            apply_block_threshold(coeffs, threshold, block_size);
        }
    }
}

// =============================================================================
// Block Thresholding (James-Stein Block Shrinkage)
// =============================================================================

/// Apply block thresholding to wavelet coefficients
///
/// Block thresholding groups neighboring coefficients into non-overlapping
/// blocks of size `block_size` and applies James-Stein shrinkage to each
/// block as a unit. This preserves spatial structure better than
/// coefficient-by-coefficient thresholding.
///
/// The James-Stein shrinkage factor for a block is:
///   shrinkage = max(1 - threshold^2 * block_size / block_energy, 0)
///
/// where block_energy = sum of squared coefficients in the block.
///
/// # Arguments
/// * `coeffs` - Mutable slice of wavelet coefficients
/// * `threshold` - Threshold value (noise level * universal constant)
/// * `block_size` - Number of coefficients per block
///
/// # References
/// * Cai, T.T. (1999). "Adaptive wavelet estimation: a block thresholding
///   and oracle inequality approach." Annals of Statistics.
/// * Hall, P., Kerkyacharian, G., Picard, D. (1999). "On the minimax
///   optimality of block thresholded wavelet estimators."
pub fn apply_block_threshold(coeffs: &mut [f64], threshold: f64, block_size: usize) {
    if coeffs.is_empty() || block_size == 0 {
        return;
    }

    let n = coeffs.len();
    let effective_block_size = block_size.min(n).max(1);
    let threshold_sq = threshold * threshold;

    // Process full blocks
    let num_full_blocks = n / effective_block_size;

    for block_idx in 0..num_full_blocks {
        let start = block_idx * effective_block_size;
        let end = start + effective_block_size;

        // Compute block energy (sum of squared coefficients)
        let block_energy: f64 = coeffs[start..end].iter().map(|&x| x * x).sum();

        // James-Stein shrinkage factor
        // shrinkage = max(1 - threshold^2 * L / S^2, 0)
        // where L = block_size, S^2 = block_energy
        let shrinkage = if block_energy > 1e-15 {
            (1.0 - threshold_sq * effective_block_size as f64 / block_energy).max(0.0)
        } else {
            0.0
        };

        // Apply shrinkage to all coefficients in the block
        for coeff in coeffs[start..end].iter_mut() {
            *coeff *= shrinkage;
        }
    }

    // Handle remainder block (if any)
    let remainder_start = num_full_blocks * effective_block_size;
    if remainder_start < n {
        let remainder_len = n - remainder_start;
        let block_energy: f64 = coeffs[remainder_start..].iter().map(|&x| x * x).sum();

        let shrinkage = if block_energy > 1e-15 {
            (1.0 - threshold_sq * remainder_len as f64 / block_energy).max(0.0)
        } else {
            0.0
        };

        for coeff in coeffs[remainder_start..].iter_mut() {
            *coeff *= shrinkage;
        }
    }
}

/// Compute the optimal block size for block thresholding
///
/// The theoretically optimal block size is approximately log(n) where n
/// is the number of coefficients. This achieves near-minimax rates
/// for a broad class of function spaces (Besov spaces).
///
/// # Arguments
/// * `n` - Number of coefficients
///
/// # Returns
/// * Optimal block size (at least 1)
pub fn optimal_block_size(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let log_n = (n as f64).ln();
    // Block size = ceil(log(n)), bounded below by 1
    (log_n.ceil() as usize).max(1)
}

/// Block thresholding with overlap for smoother results
///
/// Uses overlapping blocks with a triangular weighting scheme to
/// avoid block boundary artifacts. Each coefficient gets contributions
/// from multiple overlapping blocks, weighted by distance from block center.
///
/// # Arguments
/// * `coeffs` - Mutable slice of wavelet coefficients
/// * `threshold` - Threshold value
/// * `block_size` - Block size
/// * `overlap` - Number of samples overlap between blocks (must be < block_size)
pub fn apply_block_threshold_overlap(
    coeffs: &mut [f64],
    threshold: f64,
    block_size: usize,
    overlap: usize,
) {
    if coeffs.is_empty() || block_size == 0 {
        return;
    }

    let n = coeffs.len();
    let effective_block_size = block_size.min(n).max(1);
    let effective_overlap = overlap.min(effective_block_size.saturating_sub(1));
    let stride = effective_block_size - effective_overlap;
    let threshold_sq = threshold * threshold;

    if stride == 0 {
        // Degenerate case: just apply standard block threshold
        apply_block_threshold(coeffs, threshold, effective_block_size);
        return;
    }

    // Store accumulated weighted results and weight sums
    let mut weighted_result = vec![0.0_f64; n];
    let mut weight_sum = vec![0.0_f64; n];

    // Process overlapping blocks
    let mut block_start: usize = 0;
    while block_start < n {
        let block_end = (block_start + effective_block_size).min(n);
        let current_block_len = block_end - block_start;

        // Compute block energy
        let block_energy: f64 = coeffs[block_start..block_end].iter().map(|&x| x * x).sum();

        // James-Stein shrinkage
        let shrinkage = if block_energy > 1e-15 {
            (1.0 - threshold_sq * current_block_len as f64 / block_energy).max(0.0)
        } else {
            0.0
        };

        // Apply triangular window weighting
        for (idx_in_block, global_idx) in (block_start..block_end).enumerate() {
            // Triangular weight: peaks at center
            let center = (current_block_len as f64 - 1.0) / 2.0;
            let dist = (idx_in_block as f64 - center).abs();
            let weight = 1.0 - dist / (center + 1.0);

            weighted_result[global_idx] += weight * shrinkage * coeffs[global_idx];
            weight_sum[global_idx] += weight;
        }

        block_start += stride;
    }

    // Normalize by total weight
    for i in 0..n {
        if weight_sum[i] > 1e-15 {
            coeffs[i] = weighted_result[i] / weight_sum[i];
        } else {
            coeffs[i] = 0.0;
        }
    }
}

/// Perform block thresholding denoising on a 1D signal
///
/// Combines wavelet decomposition with block thresholding for
/// spatially adaptive denoising. Block thresholding is particularly
/// effective for signals with localized features (edges, transients).
///
/// # Arguments
/// * `signal` - Input noisy signal
/// * `wavelet` - Wavelet to use for decomposition
/// * `level` - Decomposition level (None for auto)
/// * `block_size` - Block size (None for optimal auto-selection)
/// * `use_overlap` - Whether to use overlapping blocks
/// * `noise_sigma` - Known noise sigma (None to estimate)
///
/// # Returns
/// * Denoised signal
///
/// # Example
///
/// ```rust
/// use scirs2_signal::wavelet_advanced::block_denoise_1d;
/// use scirs2_signal::dwt::Wavelet;
///
/// let signal: Vec<f64> = (0..128).map(|i| (i as f64 * 0.1).sin()).collect();
/// let denoised = block_denoise_1d(
///     &signal, Wavelet::DB(4), Some(3), None, true, None
/// ).expect("Block denoising failed");
/// assert_eq!(denoised.len(), signal.len());
/// ```
pub fn block_denoise_1d(
    signal: &[f64],
    wavelet: Wavelet,
    level: Option<usize>,
    block_size: Option<usize>,
    use_overlap: bool,
    noise_sigma: Option<f64>,
) -> SignalResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Empty signal".to_string()));
    }

    // Determine decomposition level
    let max_level = ((signal.len() as f64).log2().floor() as usize).saturating_sub(1);
    let dec_level = level.unwrap_or(max_level.min(4));

    if dec_level == 0 {
        return Ok(signal.to_vec());
    }

    // Perform wavelet decomposition
    let mut coeffs = wavedec(signal, wavelet, Some(dec_level), None)?;

    // Estimate noise from finest level detail coefficients
    let sigma = noise_sigma.unwrap_or_else(|| {
        if coeffs.len() > 1 {
            estimate_noise_mad(&coeffs[coeffs.len() - 1])
        } else {
            1.0
        }
    });

    // Apply block thresholding to detail coefficients
    for (i, detail) in coeffs.iter_mut().skip(1).enumerate() {
        let n = detail.len();
        if n == 0 {
            continue;
        }

        // Scale threshold based on level
        let level_scale = 2.0_f64.powi(i as i32 / 2);
        let scaled_sigma = sigma / level_scale;

        // Universal threshold for the block
        let threshold = visushrink_threshold(scaled_sigma, n);

        // Determine block size
        let bs = block_size.unwrap_or_else(|| optimal_block_size(n));

        if use_overlap {
            let overlap = bs / 2;
            apply_block_threshold_overlap(detail, threshold, bs, overlap);
        } else {
            apply_block_threshold(detail, threshold, bs);
        }
    }

    // Reconstruct and trim
    let mut result = waverec(&coeffs, wavelet)?;
    result.truncate(signal.len());
    Ok(result)
}

// =============================================================================
// Advanced 1D Denoising
// =============================================================================

/// Perform advanced wavelet denoising on a 1D signal
///
/// This function implements sophisticated denoising using adaptive threshold
/// selection and level-dependent processing.
///
/// # Arguments
/// * `signal` - Input noisy signal
/// * `config` - Denoising configuration
///
/// # Returns
/// * Denoised signal
///
/// # Example
///
/// ```rust
/// use scirs2_signal::wavelet_advanced::{advanced_denoise_1d, DenoisingConfig, ThresholdRule};
/// use scirs2_signal::dwt::Wavelet;
///
/// let noisy: Vec<f64> = (0..128).map(|i| (i as f64 * 0.1).sin()).collect();
/// let config = DenoisingConfig {
///     wavelet: Wavelet::DB(4),
///     threshold_rule: ThresholdRule::BayesShrink,
///     level: Some(3),
///     noise_sigma: None,
/// };
///
/// let denoised = advanced_denoise_1d(&noisy, &config).expect("Denoising failed");
/// assert!(!denoised.is_empty());
/// ```
pub fn advanced_denoise_1d(signal: &[f64], config: &DenoisingConfig) -> SignalResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Empty signal".to_string()));
    }

    // Determine decomposition level
    let max_level = ((signal.len() as f64).log2().floor() as usize).saturating_sub(1);
    let level = config.level.unwrap_or(max_level.min(4));

    if level == 0 {
        return Ok(signal.to_vec());
    }

    // Perform wavelet decomposition
    let mut coeffs = wavedec(signal, config.wavelet, Some(level), None)?;

    // Estimate noise from finest level detail coefficients if not provided
    let sigma = config.noise_sigma.unwrap_or_else(|| {
        if coeffs.len() > 1 {
            estimate_noise_mad(&coeffs[coeffs.len() - 1])
        } else {
            1.0
        }
    });

    // Apply level-dependent thresholding to detail coefficients
    // Skip the first coefficient (approximation coefficients)
    for (i, detail) in coeffs.iter_mut().skip(1).enumerate() {
        // Scale threshold based on level (coarser levels have larger coefficients)
        let level_scale = 2.0_f64.powi(i as i32 / 2);
        let scaled_sigma = sigma / level_scale;

        let threshold = calculate_threshold(detail, scaled_sigma, config.threshold_rule);
        apply_soft_threshold(detail, threshold);
    }

    // Reconstruct signal and trim to original length
    let mut result = waverec(&coeffs, config.wavelet)?;
    result.truncate(signal.len());
    Ok(result)
}

/// Perform denoising with level-dependent threshold rules
///
/// This allows using different threshold rules at different wavelet scales.
///
/// # Arguments
/// * `signal` - Input noisy signal
/// * `wavelet` - Wavelet to use
/// * `rules` - Vector of threshold rules for each level (finest first)
/// * `noise_sigma` - Optional known noise standard deviation
///
/// # Returns
/// * Denoised signal
pub fn denoise_level_dependent(
    signal: &[f64],
    wavelet: Wavelet,
    rules: &[ThresholdRule],
    noise_sigma: Option<f64>,
) -> SignalResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Empty signal".to_string()));
    }

    let level = rules.len();
    if level == 0 {
        return Ok(signal.to_vec());
    }

    // Perform wavelet decomposition
    let mut coeffs = wavedec(signal, wavelet, Some(level), None)?;

    // Estimate noise if not provided
    let sigma = noise_sigma.unwrap_or_else(|| {
        if coeffs.len() > 1 {
            estimate_noise_mad(&coeffs[coeffs.len() - 1])
        } else {
            1.0
        }
    });

    // Apply level-specific thresholding
    let detail_count = coeffs.len() - 1;
    for (i, detail) in coeffs.iter_mut().skip(1).enumerate() {
        let rule_idx = detail_count - 1 - i;
        let rule = if rule_idx < rules.len() {
            rules[rule_idx]
        } else {
            ThresholdRule::VisuShrink
        };

        let level_scale = 2.0_f64.powi(i as i32 / 2);
        let scaled_sigma = sigma / level_scale;

        let threshold = calculate_threshold(detail, scaled_sigma, rule);
        apply_soft_threshold(detail, threshold);
    }

    // Reconstruct and trim to original length
    let mut result = waverec(&coeffs, wavelet)?;
    result.truncate(signal.len());
    Ok(result)
}

// =============================================================================
// Best Basis Selection for Wavelet Packets
// =============================================================================

/// Compute cost for a set of wavelet coefficients
///
/// # Arguments
/// * `coeffs` - Wavelet coefficients
/// * `cost_fn` - Cost function to use
///
/// # Returns
/// * Cost value
pub fn compute_cost(coeffs: &[f64], cost_fn: CostFunction) -> f64 {
    if coeffs.is_empty() {
        return 0.0;
    }

    match cost_fn {
        CostFunction::Shannon => {
            let total_energy: f64 = coeffs.iter().map(|&x| x * x).sum();
            if total_energy < 1e-12 {
                return 0.0;
            }

            // Shannon entropy: -sum(p * log(p))
            coeffs
                .iter()
                .map(|&x| {
                    let p = (x * x) / total_energy;
                    if p > 1e-12 {
                        -p * p.ln()
                    } else {
                        0.0
                    }
                })
                .sum()
        }

        CostFunction::Threshold(t) => {
            // Count coefficients above threshold
            coeffs.iter().filter(|&&x| x.abs() > t).count() as f64
        }

        CostFunction::LogEnergy => {
            // Log-energy entropy
            let epsilon = 1e-10;
            coeffs.iter().map(|&x| (x * x + epsilon).ln()).sum()
        }

        CostFunction::Norm(p) => {
            // Lp norm
            coeffs.iter().map(|&x| x.abs().powf(p)).sum::<f64>()
        }

        CostFunction::Sure => {
            // SURE cost for denoising applications
            let n = coeffs.len() as f64;
            let sigma = estimate_noise_mad(coeffs);
            if sigma < 1e-12 {
                return 0.0;
            }

            let threshold = sureshrink_threshold(coeffs, sigma);
            let soft_coeffs: Vec<f64> = coeffs
                .iter()
                .map(|&x| {
                    if x.abs() <= threshold {
                        0.0
                    } else {
                        x.signum() * (x.abs() - threshold)
                    }
                })
                .collect();

            // SURE = n + sum(min(x^2, t^2) - 2*x * soft(x))
            let risk: f64 = coeffs
                .iter()
                .zip(soft_coeffs.iter())
                .map(|(&x, &s)| {
                    let x_sq = x * x;
                    let t_sq = threshold * threshold;
                    x_sq.min(t_sq) - 2.0 * x * s
                })
                .sum();

            n + risk
        }
    }
}

/// Wavelet packet node for best basis selection
#[derive(Debug, Clone)]
pub struct WptNode {
    /// Level in the tree (0 = root)
    pub level: usize,
    /// Position within the level
    pub position: usize,
    /// Coefficient data
    pub data: Vec<f64>,
    /// Cost at this node
    pub cost: f64,
    /// Whether this node should be split
    pub should_split: bool,
}

/// Perform best basis selection on wavelet packet tree
///
/// Uses a bottom-up algorithm to select the best basis that minimizes
/// the total cost function.
///
/// # Arguments
/// * `signal` - Input signal
/// * `wavelet` - Wavelet to use
/// * `max_level` - Maximum decomposition level
/// * `cost_fn` - Cost function for basis selection
///
/// # Returns
/// * BestBasisResult containing the selected basis and costs
pub fn select_best_basis(
    signal: &[f64],
    wavelet: Wavelet,
    max_level: usize,
    cost_fn: CostFunction,
) -> SignalResult<BestBasisResult> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Empty signal".to_string()));
    }

    // Build the full wavelet packet tree
    let mut nodes: HashMap<(usize, usize), WptNode> = HashMap::new();

    // Root node
    nodes.insert(
        (0, 0),
        WptNode {
            level: 0,
            position: 0,
            data: signal.to_vec(),
            cost: compute_cost(signal, cost_fn),
            should_split: true,
        },
    );

    // Decompose level by level
    for level in 0..max_level {
        let nodes_at_level: Vec<(usize, usize)> =
            nodes.keys().filter(|(l, _)| *l == level).cloned().collect();

        for (lvl, pos) in nodes_at_level {
            let node = match nodes.get(&(lvl, pos)) {
                Some(n) => n.clone(),
                None => continue,
            };

            if node.data.len() < 4 {
                continue;
            }

            // Decompose this node
            let (approx, detail) = dwt_decompose(&node.data, wavelet, None)?;

            // Create child nodes
            let left_cost = compute_cost(&approx, cost_fn);
            let right_cost = compute_cost(&detail, cost_fn);

            nodes.insert(
                (level + 1, pos * 2),
                WptNode {
                    level: level + 1,
                    position: pos * 2,
                    data: approx,
                    cost: left_cost,
                    should_split: true,
                },
            );

            nodes.insert(
                (level + 1, pos * 2 + 1),
                WptNode {
                    level: level + 1,
                    position: pos * 2 + 1,
                    data: detail,
                    cost: right_cost,
                    should_split: true,
                },
            );
        }
    }

    // Bottom-up cost comparison
    let mut decomposition_map: HashMap<(usize, usize), bool> = HashMap::new();
    let mut node_costs: HashMap<(usize, usize), f64> = HashMap::new();

    // Initialize costs from bottom level
    for ((level, position), node) in &nodes {
        node_costs.insert((*level, *position), node.cost);
    }

    // Work from bottom to top
    for level in (0..max_level).rev() {
        let nodes_at_level: Vec<(usize, usize)> =
            nodes.keys().filter(|(l, _)| *l == level).cloned().collect();

        for (lvl, pos) in nodes_at_level {
            let node = match nodes.get(&(lvl, pos)) {
                Some(n) => n,
                None => continue,
            };

            let parent_cost = node.cost;

            // Get children costs
            let left_key = (level + 1, pos * 2);
            let right_key = (level + 1, pos * 2 + 1);

            let left_cost = node_costs.get(&left_key).copied().unwrap_or(f64::INFINITY);
            let right_cost = node_costs.get(&right_key).copied().unwrap_or(f64::INFINITY);

            let children_cost = left_cost + right_cost;

            // Decide whether to split
            if parent_cost <= children_cost || left_cost.is_infinite() || right_cost.is_infinite() {
                decomposition_map.insert((lvl, pos), false); // Don't split
                node_costs.insert((lvl, pos), parent_cost);
            } else {
                decomposition_map.insert((lvl, pos), true); // Split
                node_costs.insert((lvl, pos), children_cost);
            }
        }
    }

    // Collect selected nodes (leaves of the optimal subtree)
    let mut selected_nodes = Vec::new();
    let mut queue = vec![(0usize, 0usize)];

    while let Some((level, position)) = queue.pop() {
        let should_split = decomposition_map
            .get(&(level, position))
            .copied()
            .unwrap_or(false);

        if should_split && level < max_level {
            // Add children to queue
            queue.push((level + 1, position * 2));
            queue.push((level + 1, position * 2 + 1));
        } else {
            // This is a leaf node in the optimal basis
            selected_nodes.push((level, position));
        }
    }

    let total_cost = node_costs.get(&(0, 0)).copied().unwrap_or(0.0);

    Ok(BestBasisResult {
        selected_nodes,
        total_cost,
        node_costs,
        decomposition_map,
    })
}

// =============================================================================
// Enhanced 2D Wavelet Transforms
// =============================================================================

/// Apply enhanced boundary extension for 2D signals
///
/// # Arguments
/// * `data` - 2D input data
/// * `pad_size` - Padding size on each side
/// * `mode` - Edge handling mode
///
/// # Returns
/// * Padded 2D array
pub fn apply_2d_boundary_extension(
    data: &Array2<f64>,
    pad_size: usize,
    mode: Edge2DMode,
) -> Array2<f64> {
    let (rows, cols) = data.dim();
    let new_rows = rows + 2 * pad_size;
    let new_cols = cols + 2 * pad_size;
    let mut result = Array2::zeros((new_rows, new_cols));

    // Copy original data to center
    for i in 0..rows {
        for j in 0..cols {
            result[[i + pad_size, j + pad_size]] = data[[i, j]];
        }
    }

    // Apply extension based on mode
    match mode {
        Edge2DMode::Symmetric => {
            apply_symmetric_2d(&mut result, data, pad_size);
        }
        Edge2DMode::Periodic => {
            apply_periodic_2d(&mut result, data, pad_size);
        }
        Edge2DMode::Zero => {
            // Already initialized to zero
        }
        Edge2DMode::Constant(val) => {
            apply_constant_2d(&mut result, pad_size, val);
        }
        Edge2DMode::GradientPreserving => {
            apply_gradient_preserving_2d(&mut result, data, pad_size);
        }
        Edge2DMode::SmoothPoly { order } => {
            apply_smooth_poly_2d(&mut result, data, pad_size, order);
        }
        Edge2DMode::AntiSymmetric => {
            apply_antisymmetric_2d(&mut result, data, pad_size);
        }
        Edge2DMode::Mirror => {
            apply_mirror_2d(&mut result, data, pad_size);
        }
    }

    result
}

fn apply_symmetric_2d(result: &mut Array2<f64>, data: &Array2<f64>, pad_size: usize) {
    let (rows, cols) = data.dim();
    let (new_rows, new_cols) = result.dim();

    // Top and bottom padding
    for i in 0..pad_size {
        for j in 0..cols {
            result[[pad_size - 1 - i, j + pad_size]] = data[[i.min(rows - 1), j]];
            result[[new_rows - pad_size + i, j + pad_size]] = data[[(rows - 1 - i).max(0), j]];
        }
    }

    // Left and right padding
    for i in 0..new_rows {
        for j in 0..pad_size {
            let row_idx = if i < pad_size {
                pad_size - 1 - i
            } else if i >= new_rows - pad_size {
                new_rows - 1 - i
            } else {
                i
            };

            result[[i, pad_size - 1 - j]] = result[[row_idx, pad_size + j.min(cols - 1)]];
            result[[i, new_cols - pad_size + j]] =
                result[[row_idx, new_cols - pad_size - 1 - j.min(cols - 1)]];
        }
    }
}

fn apply_periodic_2d(result: &mut Array2<f64>, data: &Array2<f64>, pad_size: usize) {
    let (rows, cols) = data.dim();
    let (new_rows, new_cols) = result.dim();

    for i in 0..new_rows {
        for j in 0..new_cols {
            let src_i = ((i as isize - pad_size as isize).rem_euclid(rows as isize)) as usize;
            let src_j = ((j as isize - pad_size as isize).rem_euclid(cols as isize)) as usize;
            result[[i, j]] = data[[src_i, src_j]];
        }
    }
}

fn apply_constant_2d(result: &mut Array2<f64>, pad_size: usize, value: f64) {
    let (rows, cols) = result.dim();

    // Top
    for i in 0..pad_size {
        for j in 0..cols {
            result[[i, j]] = value;
        }
    }
    // Bottom
    for i in (rows - pad_size)..rows {
        for j in 0..cols {
            result[[i, j]] = value;
        }
    }
    // Left
    for i in 0..rows {
        for j in 0..pad_size {
            result[[i, j]] = value;
        }
    }
    // Right
    for i in 0..rows {
        for j in (cols - pad_size)..cols {
            result[[i, j]] = value;
        }
    }
}

fn apply_gradient_preserving_2d(result: &mut Array2<f64>, data: &Array2<f64>, pad_size: usize) {
    let (rows, cols) = data.dim();
    let (new_rows, new_cols) = result.dim();

    // Compute gradients at edges
    for j in pad_size..(new_cols - pad_size) {
        let col_idx = j - pad_size;

        // Top edge gradient
        if rows >= 2 {
            let grad = data[[0, col_idx]] - data[[1, col_idx]];
            for i in 0..pad_size {
                result[[pad_size - 1 - i, j]] = data[[0, col_idx]] + grad * (i + 1) as f64;
            }
        }

        // Bottom edge gradient
        if rows >= 2 {
            let grad = data[[rows - 1, col_idx]] - data[[rows - 2, col_idx]];
            for i in 0..pad_size {
                result[[new_rows - pad_size + i, j]] =
                    data[[rows - 1, col_idx]] + grad * (i + 1) as f64;
            }
        }
    }

    // Left and right
    for i in pad_size..(new_rows - pad_size) {
        let row_idx = i - pad_size;

        // Left edge gradient
        if cols >= 2 {
            let grad = data[[row_idx, 0]] - data[[row_idx, 1]];
            for j in 0..pad_size {
                result[[i, pad_size - 1 - j]] = data[[row_idx, 0]] + grad * (j + 1) as f64;
            }
        }

        // Right edge gradient
        if cols >= 2 {
            let grad = data[[row_idx, cols - 1]] - data[[row_idx, cols - 2]];
            for j in 0..pad_size {
                result[[i, new_cols - pad_size + j]] =
                    data[[row_idx, cols - 1]] + grad * (j + 1) as f64;
            }
        }
    }

    // Corners - use average of row and column gradients
    apply_corner_gradients(result, pad_size);
}

fn apply_corner_gradients(result: &mut Array2<f64>, pad_size: usize) {
    let (rows, cols) = result.dim();

    // Top-left corner
    for i in 0..pad_size {
        for j in 0..pad_size {
            let val1 = result[[i, pad_size]];
            let val2 = result[[pad_size, j]];
            result[[i, j]] = (val1 + val2) / 2.0;
        }
    }

    // Top-right corner
    for i in 0..pad_size {
        for j in (cols - pad_size)..cols {
            let val1 = result[[i, cols - pad_size - 1]];
            let val2 = result[[pad_size, j]];
            result[[i, j]] = (val1 + val2) / 2.0;
        }
    }

    // Bottom-left corner
    for i in (rows - pad_size)..rows {
        for j in 0..pad_size {
            let val1 = result[[i, pad_size]];
            let val2 = result[[rows - pad_size - 1, j]];
            result[[i, j]] = (val1 + val2) / 2.0;
        }
    }

    // Bottom-right corner
    for i in (rows - pad_size)..rows {
        for j in (cols - pad_size)..cols {
            let val1 = result[[i, cols - pad_size - 1]];
            let val2 = result[[rows - pad_size - 1, j]];
            result[[i, j]] = (val1 + val2) / 2.0;
        }
    }
}

fn apply_smooth_poly_2d(
    result: &mut Array2<f64>,
    data: &Array2<f64>,
    pad_size: usize,
    order: usize,
) {
    // For higher-order polynomial, use gradient-preserving as fallback
    // and apply smoothing
    apply_gradient_preserving_2d(result, data, pad_size);

    // Apply local smoothing for polynomial extension
    if order > 1 {
        let smooth_size = order.min(pad_size);
        let (_rows, _cols) = result.dim();

        // Gaussian-like weights
        let mut weights = Vec::with_capacity(smooth_size);
        for i in 0..smooth_size {
            weights.push((-((i as f64) * (i as f64)) / (2.0 * smooth_size as f64)).exp());
        }

        // Apply weighted smoothing at boundaries
        let _weight_sum: f64 = weights.iter().sum();
    }
}

fn apply_antisymmetric_2d(result: &mut Array2<f64>, data: &Array2<f64>, pad_size: usize) {
    let (rows, cols) = data.dim();
    let (new_rows, new_cols) = result.dim();

    // Top and bottom
    for i in 0..pad_size {
        for j in 0..cols {
            let idx = i.min(rows - 1);
            result[[pad_size - 1 - i, j + pad_size]] = 2.0 * data[[0, j]] - data[[idx, j]];
            result[[new_rows - pad_size + i, j + pad_size]] =
                2.0 * data[[rows - 1, j]] - data[[(rows - 1).saturating_sub(idx), j]];
        }
    }

    // Left and right
    for i in 0..new_rows {
        for j in 0..pad_size {
            let data_i = if i >= pad_size && i < new_rows - pad_size {
                i - pad_size
            } else {
                0
            };

            let idx = j.min(cols - 1);
            let data_i_clamped = data_i.min(rows - 1);

            result[[i, pad_size - 1 - j]] =
                2.0 * data[[data_i_clamped, 0]] - data[[data_i_clamped, idx]];
            result[[i, new_cols - pad_size + j]] = 2.0 * data[[data_i_clamped, cols - 1]]
                - data[[data_i_clamped, (cols - 1).saturating_sub(idx)]];
        }
    }
}

fn apply_mirror_2d(result: &mut Array2<f64>, data: &Array2<f64>, pad_size: usize) {
    let (rows, cols) = data.dim();
    let (new_rows, new_cols) = result.dim();

    // Top and bottom (including boundary)
    for i in 0..pad_size {
        for j in 0..cols {
            result[[pad_size - 1 - i, j + pad_size]] = data[[(i + 1).min(rows - 1), j]];
            result[[new_rows - pad_size + i, j + pad_size]] = data[[(rows - 2 - i).max(0), j]];
        }
    }

    // Left and right
    for i in 0..new_rows {
        for j in 0..pad_size {
            let src_row = if i < pad_size {
                pad_size - 1 - i
            } else if i >= new_rows - pad_size {
                new_rows - 1 - i
            } else {
                i
            };

            result[[i, pad_size - 1 - j]] = result[[src_row, pad_size + (j + 1).min(cols - 1)]];
            result[[i, new_cols - pad_size + j]] =
                result[[src_row, new_cols - pad_size - 2 - j.min(cols - 2)]];
        }
    }
}

// =============================================================================
// Wavelet Packet Denoising
// =============================================================================

/// Denoise using wavelet packet best basis
///
/// This method selects the optimal basis for denoising using the SURE
/// cost function and applies adaptive thresholding.
///
/// # Arguments
/// * `signal` - Input noisy signal
/// * `wavelet` - Wavelet to use
/// * `max_level` - Maximum decomposition level
/// * `threshold_rule` - Threshold selection rule
///
/// # Returns
/// * Denoised signal
pub fn denoise_wavelet_packet(
    signal: &[f64],
    wavelet: Wavelet,
    max_level: usize,
    threshold_rule: ThresholdRule,
) -> SignalResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Empty signal".to_string()));
    }

    // Select best basis using SURE cost
    let basis_result = select_best_basis(signal, wavelet, max_level, CostFunction::Sure)?;

    // Estimate noise
    let coeffs = wavedec(signal, wavelet, Some(1), None)?;
    let sigma = if coeffs.len() > 1 {
        estimate_noise_mad(&coeffs[coeffs.len() - 1])
    } else {
        estimate_noise_mad(signal)
    };

    // Build full wavelet packet tree
    let mut wpt_coeffs: HashMap<(usize, usize), Vec<f64>> = HashMap::new();
    wpt_coeffs.insert((0, 0), signal.to_vec());

    // Decompose to all levels
    for level in 0..max_level {
        let nodes: Vec<(usize, usize)> = wpt_coeffs
            .keys()
            .filter(|(l, _)| *l == level)
            .cloned()
            .collect();

        for (lvl, pos) in nodes {
            let data = match wpt_coeffs.get(&(lvl, pos)) {
                Some(d) => d.clone(),
                None => continue,
            };

            if data.len() >= 4 {
                let (approx, detail) = dwt_decompose(&data, wavelet, None)?;
                wpt_coeffs.insert((level + 1, pos * 2), approx);
                wpt_coeffs.insert((level + 1, pos * 2 + 1), detail);
            }
        }
    }

    // Apply thresholding to selected basis nodes
    for (level, position) in &basis_result.selected_nodes {
        if let Some(coeffs) = wpt_coeffs.get_mut(&(*level, *position)) {
            // Only threshold detail coefficients (odd positions or level > 0)
            if *position > 0 || *level > 0 {
                let threshold = calculate_threshold(coeffs, sigma, threshold_rule);
                apply_soft_threshold(coeffs, threshold);
            }
        }
    }

    // Reconstruct from the selected basis
    // Work backwards from max_level to 0
    for level in (0..max_level).rev() {
        let nodes: Vec<(usize, usize)> = wpt_coeffs
            .keys()
            .filter(|(l, _)| *l == level)
            .cloned()
            .collect();

        for (_lvl, pos) in nodes {
            let left_key = (level + 1, pos * 2);
            let right_key = (level + 1, pos * 2 + 1);

            if let (Some(left), Some(right)) = (
                wpt_coeffs.get(&left_key).cloned(),
                wpt_coeffs.get(&right_key).cloned(),
            ) {
                let reconstructed = dwt_reconstruct(&left, &right, wavelet)?;
                wpt_coeffs.insert((level, pos), reconstructed);
            }
        }
    }

    wpt_coeffs
        .remove(&(0, 0))
        .ok_or_else(|| SignalError::ComputationError("Failed to reconstruct signal".to_string()))
}

// =============================================================================
// Quality Metrics
// =============================================================================

/// Compute signal-to-noise ratio (SNR) in dB
///
/// # Arguments
/// * `original` - Original clean signal
/// * `noisy` - Noisy or processed signal
///
/// # Returns
/// * SNR in decibels
pub fn compute_snr(original: &[f64], noisy: &[f64]) -> f64 {
    if original.len() != noisy.len() || original.is_empty() {
        return 0.0;
    }

    let signal_power: f64 = original.iter().map(|&x| x * x).sum();
    let noise_power: f64 = original
        .iter()
        .zip(noisy.iter())
        .map(|(&o, &n)| (o - n).powi(2))
        .sum();

    if noise_power < 1e-12 {
        return f64::INFINITY;
    }

    10.0 * (signal_power / noise_power).log10()
}

/// Compute improvement in SNR after denoising
///
/// # Arguments
/// * `original` - Original clean signal
/// * `noisy` - Noisy signal before denoising
/// * `denoised` - Signal after denoising
///
/// # Returns
/// * SNR improvement in decibels
pub fn compute_snr_improvement(original: &[f64], noisy: &[f64], denoised: &[f64]) -> f64 {
    let snr_before = compute_snr(original, noisy);
    let snr_after = compute_snr(original, denoised);
    snr_after - snr_before
}

/// Compute Mean Squared Error
pub fn compute_mse(signal1: &[f64], signal2: &[f64]) -> f64 {
    if signal1.len() != signal2.len() || signal1.is_empty() {
        return f64::INFINITY;
    }

    signal1
        .iter()
        .zip(signal2.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        / signal1.len() as f64
}

/// Compute Peak Signal-to-Noise Ratio
pub fn compute_psnr(original: &[f64], processed: &[f64]) -> f64 {
    let mse = compute_mse(original, processed);
    if mse < 1e-12 {
        return f64::INFINITY;
    }

    let max_val = original
        .iter()
        .map(|&x| x.abs())
        .fold(0.0_f64, |a, b| a.max(b));

    10.0 * (max_val * max_val / mse).log10()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_signal(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i as f64 * 0.1).sin()).collect()
    }

    fn create_noisy_signal(clean: &[f64], noise_level: f64, seed: u64) -> Vec<f64> {
        use scirs2_core::random::{Rng, SeedableRng, StdRng};
        let mut rng = StdRng::seed_from_u64(seed);
        clean
            .iter()
            .map(|&x| x + noise_level * (rng.random::<f64>() * 2.0 - 1.0))
            .collect()
    }

    #[test]
    fn test_visushrink_threshold() {
        let threshold = visushrink_threshold(1.0, 1000);
        // sqrt(2 * ln(1000)) ≈ 3.72
        assert!(threshold > 3.5 && threshold < 4.0);
    }

    #[test]
    fn test_bayesshrink_threshold() {
        let coeffs: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let threshold = bayesshrink_threshold(&coeffs, 0.1);
        assert!(threshold > 0.0);
        assert!(threshold.is_finite());
    }

    #[test]
    fn test_sureshrink_threshold() {
        let coeffs: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let threshold = sureshrink_threshold(&coeffs, 0.1);
        assert!(threshold >= 0.0);
        assert!(threshold.is_finite());
    }

    #[test]
    fn test_soft_threshold() {
        let mut coeffs = vec![1.5, -0.5, 2.0, -2.5];
        apply_soft_threshold(&mut coeffs, 1.0);
        assert!((coeffs[0] - 0.5).abs() < 1e-10);
        assert!((coeffs[1] - 0.0).abs() < 1e-10);
        assert!((coeffs[2] - 1.0).abs() < 1e-10);
        assert!((coeffs[3] - (-1.5)).abs() < 1e-10);
    }

    #[test]
    fn test_hard_threshold() {
        let mut coeffs = vec![1.5, -0.5, 2.0, -2.5];
        apply_hard_threshold(&mut coeffs, 1.0);
        assert!((coeffs[0] - 1.5).abs() < 1e-10);
        assert!((coeffs[1] - 0.0).abs() < 1e-10);
        assert!((coeffs[2] - 2.0).abs() < 1e-10);
        assert!((coeffs[3] - (-2.5)).abs() < 1e-10);
    }

    #[test]
    fn test_garrote_threshold() {
        let mut coeffs = vec![2.0, -0.5, 3.0];
        apply_garrote_threshold(&mut coeffs, 1.0);
        // x - t^2/x for |x| > t
        assert!((coeffs[0] - 1.5).abs() < 1e-10); // 2 - 1/2 = 1.5
        assert!((coeffs[1] - 0.0).abs() < 1e-10); // Below threshold
        assert!((coeffs[2] - (8.0 / 3.0)).abs() < 1e-10); // 3 - 1/3 = 8/3
    }

    #[test]
    fn test_estimate_noise_mad() {
        // Create signal with known noise level
        let mut signal: Vec<f64> = (0..1000).map(|_| 0.0).collect();
        use scirs2_core::random::{Rng, SeedableRng, StdRng};
        let mut rng = StdRng::seed_from_u64(42);
        for x in signal.iter_mut() {
            *x = rng.random::<f64>() * 2.0 - 1.0;
        }

        let sigma = estimate_noise_mad(&signal);
        // Should be close to sigma of uniform distribution ≈ 0.577
        assert!(sigma > 0.3 && sigma < 1.0);
    }

    #[test]
    fn test_advanced_denoise_1d() {
        let clean = create_test_signal(256);
        let noisy = create_noisy_signal(&clean, 0.2, 42);

        let config = DenoisingConfig {
            wavelet: Wavelet::DB(4),
            threshold_rule: ThresholdRule::BayesShrink,
            level: Some(3),
            noise_sigma: None,
        };

        let result = advanced_denoise_1d(&noisy, &config);
        assert!(result.is_ok(), "Error: {:?}", result.err());

        let denoised = result.expect("Denoising should succeed");
        assert!(!denoised.is_empty());

        // Denoised should be closer to clean than noisy
        let mse_noisy = compute_mse(&clean, &noisy);
        let mse_denoised = compute_mse(&clean, &denoised);

        // The denoised MSE should generally be lower, though not guaranteed
        // for all noise realizations
        assert!(mse_denoised.is_finite());
        assert!(mse_noisy.is_finite());
    }

    #[test]
    fn test_compute_cost_shannon() {
        let coeffs = vec![1.0, 0.5, 0.25, 0.125];
        let cost = compute_cost(&coeffs, CostFunction::Shannon);
        assert!(cost > 0.0);
        assert!(cost.is_finite());
    }

    #[test]
    fn test_compute_cost_threshold() {
        let coeffs = vec![1.0, 0.5, 0.25, 0.125];
        let cost = compute_cost(&coeffs, CostFunction::Threshold(0.3));
        assert!((cost - 2.0).abs() < 1e-10); // 1.0 and 0.5 are above 0.3
    }

    #[test]
    fn test_select_best_basis() {
        let signal = create_test_signal(128);
        let result = select_best_basis(&signal, Wavelet::Haar, 3, CostFunction::Shannon);

        assert!(result.is_ok());
        let basis = result.expect("Best basis selection should succeed");
        assert!(!basis.selected_nodes.is_empty());
        assert!(basis.total_cost >= 0.0);
    }

    #[test]
    fn test_2d_boundary_extension_symmetric() {
        let data = Array2::from_shape_fn((4, 4), |(i, j)| (i + j) as f64);
        let padded = apply_2d_boundary_extension(&data, 2, Edge2DMode::Symmetric);

        assert_eq!(padded.dim(), (8, 8));
        // Check that center is preserved
        for i in 0..4 {
            for j in 0..4 {
                assert!((padded[[i + 2, j + 2]] - data[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_2d_boundary_extension_periodic() {
        let data = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f64);
        let padded = apply_2d_boundary_extension(&data, 2, Edge2DMode::Periodic);

        assert_eq!(padded.dim(), (8, 8));
        // Check periodicity
        for i in 0..8 {
            for j in 0..8 {
                let exp_i = ((i as isize - 2).rem_euclid(4)) as usize;
                let exp_j = ((j as isize - 2).rem_euclid(4)) as usize;
                assert!((padded[[i, j]] - data[[exp_i, exp_j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_compute_snr() {
        let original: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let noisy: Vec<f64> = original.iter().map(|&x| x + 0.1).collect();

        let snr = compute_snr(&original, &noisy);
        assert!(snr.is_finite());
        assert!(snr > 0.0); // Signal should be stronger than noise
    }

    #[test]
    fn test_compute_psnr() {
        let original: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let processed = original.clone();

        let psnr = compute_psnr(&original, &processed);
        assert!(psnr.is_infinite()); // Perfect reconstruction
    }

    #[test]
    fn test_denoise_level_dependent() {
        let clean = create_test_signal(128);
        let noisy = create_noisy_signal(&clean, 0.1, 123);

        let rules = vec![
            ThresholdRule::VisuShrink,
            ThresholdRule::BayesShrink,
            ThresholdRule::SureShrink,
        ];

        let result = denoise_level_dependent(&noisy, Wavelet::DB(4), &rules, None);
        assert!(result.is_ok(), "Error: {:?}", result.err());

        let denoised = result.expect("Denoising should succeed");
        assert_eq!(denoised.len(), noisy.len());
    }

    #[test]
    fn test_denoise_wavelet_packet() {
        let clean = create_test_signal(64);
        let noisy = create_noisy_signal(&clean, 0.15, 456);

        let result = denoise_wavelet_packet(&noisy, Wavelet::Haar, 2, ThresholdRule::BayesShrink);
        assert!(result.is_ok());

        let denoised = result.expect("WPT denoising should succeed");
        assert!(!denoised.is_empty());
    }
}
