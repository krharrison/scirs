//! Multi-channel signal processing
//!
//! Provides capabilities for processing multi-channel (array) signals:
//! - Channel mixing (mono, stereo, N-channel)
//! - Cross-channel correlation
//! - Independent Component Analysis (FastICA)
//! - Common Spatial Patterns (CSP) for EEG/BCI
//! - Channel selection and reordering

use crate::error::{SignalError, SignalResult};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A multi-channel signal represented as `channels x samples`.
/// Each inner `Vec<f64>` is one channel.
#[derive(Debug, Clone)]
pub struct MultiChannelSignal {
    /// Channel data: `channels[ch][sample]`
    pub channels: Vec<Vec<f64>>,
}

impl MultiChannelSignal {
    /// Create from a vector of channel data. All channels must have the same
    /// length.
    pub fn new(channels: Vec<Vec<f64>>) -> SignalResult<Self> {
        if channels.is_empty() {
            return Err(SignalError::ValueError(
                "At least one channel is required".into(),
            ));
        }
        let n = channels[0].len();
        if n == 0 {
            return Err(SignalError::ValueError("Channels must not be empty".into()));
        }
        for (i, ch) in channels.iter().enumerate() {
            if ch.len() != n {
                return Err(SignalError::DimensionMismatch(format!(
                    "Channel 0 has {} samples but channel {} has {}",
                    n,
                    i,
                    ch.len()
                )));
            }
        }
        Ok(Self { channels })
    }

    /// Number of channels.
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    /// Number of samples per channel.
    pub fn num_samples(&self) -> usize {
        self.channels.first().map_or(0, |c| c.len())
    }
}

// ---------------------------------------------------------------------------
// Channel mixing
// ---------------------------------------------------------------------------

/// Mix mode for channel conversion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MixMode {
    /// Simple average of all channels.
    Average,
    /// Weighted mix (weights must have one value per channel).
    Weighted,
    /// Keep only the specified channel.
    Select(usize),
}

/// Convert a multi-channel signal to mono via the given mix mode.
///
/// When `MixMode::Weighted` is used, `weights` must be `Some` with length
/// equal to the number of channels. Weights are normalised so they sum to 1.
pub fn mix_to_mono(
    signal: &MultiChannelSignal,
    mode: MixMode,
    weights: Option<&[f64]>,
) -> SignalResult<Vec<f64>> {
    let n_ch = signal.num_channels();
    let n_samp = signal.num_samples();

    match mode {
        MixMode::Average => {
            let inv = 1.0 / n_ch as f64;
            let mut out = vec![0.0; n_samp];
            for ch in &signal.channels {
                for (o, &s) in out.iter_mut().zip(ch.iter()) {
                    *o += s * inv;
                }
            }
            Ok(out)
        }
        MixMode::Weighted => {
            let w = weights.ok_or_else(|| {
                SignalError::InvalidArgument("Weights required for Weighted mode".into())
            })?;
            if w.len() != n_ch {
                return Err(SignalError::DimensionMismatch(format!(
                    "Expected {} weights, got {}",
                    n_ch,
                    w.len()
                )));
            }
            let wsum: f64 = w.iter().map(|v| v.abs()).sum();
            if wsum < f64::EPSILON {
                return Err(SignalError::ValueError("Weights sum to zero".into()));
            }
            let norm: Vec<f64> = w.iter().map(|v| v / wsum).collect();
            let mut out = vec![0.0; n_samp];
            for (ch, &nw) in signal.channels.iter().zip(norm.iter()) {
                for (o, &s) in out.iter_mut().zip(ch.iter()) {
                    *o += s * nw;
                }
            }
            Ok(out)
        }
        MixMode::Select(idx) => {
            if idx >= n_ch {
                return Err(SignalError::InvalidArgument(format!(
                    "Channel index {} out of range (0..{})",
                    idx, n_ch
                )));
            }
            Ok(signal.channels[idx].clone())
        }
    }
}

/// Upmix a mono signal to N channels by copying.
pub fn mono_to_multichannel(mono: &[f64], n_channels: usize) -> SignalResult<MultiChannelSignal> {
    if mono.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".into()));
    }
    if n_channels == 0 {
        return Err(SignalError::InvalidArgument(
            "n_channels must be >= 1".into(),
        ));
    }
    let channels = vec![mono.to_vec(); n_channels];
    MultiChannelSignal::new(channels)
}

/// Apply a mixing matrix `[out_ch x in_ch]` to a multi-channel signal.
///
/// `matrix` is row-major: `matrix[out_ch * in_ch + in]`.
pub fn apply_mixing_matrix(
    signal: &MultiChannelSignal,
    out_channels: usize,
    in_channels: usize,
    matrix: &[f64],
) -> SignalResult<MultiChannelSignal> {
    if in_channels != signal.num_channels() {
        return Err(SignalError::DimensionMismatch(format!(
            "Mixing matrix expects {} input channels but signal has {}",
            in_channels,
            signal.num_channels()
        )));
    }
    if matrix.len() != out_channels * in_channels {
        return Err(SignalError::DimensionMismatch(format!(
            "Mixing matrix size {} != {} x {}",
            matrix.len(),
            out_channels,
            in_channels
        )));
    }
    let n_samp = signal.num_samples();
    let mut out = vec![vec![0.0; n_samp]; out_channels];
    for oc in 0..out_channels {
        for ic in 0..in_channels {
            let w = matrix[oc * in_channels + ic];
            for (o, &s) in out[oc].iter_mut().zip(signal.channels[ic].iter()) {
                *o += w * s;
            }
        }
    }
    MultiChannelSignal::new(out)
}

// ---------------------------------------------------------------------------
// Cross-channel correlation
// ---------------------------------------------------------------------------

/// Compute the cross-correlation matrix at zero lag for a multi-channel signal.
///
/// Returns a flat `n_ch x n_ch` row-major matrix.
pub fn cross_channel_correlation(signal: &MultiChannelSignal) -> SignalResult<Vec<f64>> {
    let n_ch = signal.num_channels();
    let n = signal.num_samples() as f64;
    if n < 1.0 {
        return Err(SignalError::ValueError("Signal too short".into()));
    }

    // Compute means
    let means: Vec<f64> = signal
        .channels
        .iter()
        .map(|ch| ch.iter().sum::<f64>() / n)
        .collect();

    // Compute standard deviations
    let stds: Vec<f64> = signal
        .channels
        .iter()
        .zip(means.iter())
        .map(|(ch, &m)| {
            let var = ch.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / n;
            var.sqrt()
        })
        .collect();

    let mut corr = vec![0.0; n_ch * n_ch];
    for i in 0..n_ch {
        for j in i..n_ch {
            let val = if stds[i] < f64::EPSILON || stds[j] < f64::EPSILON {
                if i == j {
                    1.0
                } else {
                    0.0
                }
            } else {
                let cov: f64 = signal.channels[i]
                    .iter()
                    .zip(signal.channels[j].iter())
                    .map(|(&a, &b)| (a - means[i]) * (b - means[j]))
                    .sum::<f64>()
                    / n;
                cov / (stds[i] * stds[j])
            };
            corr[i * n_ch + j] = val;
            corr[j * n_ch + i] = val;
        }
    }
    Ok(corr)
}

/// Compute cross-correlation between two channels at given lags.
/// Returns correlation values for each lag in `lags`.
pub fn cross_correlation_lag(ch_a: &[f64], ch_b: &[f64], lags: &[i64]) -> SignalResult<Vec<f64>> {
    if ch_a.is_empty() || ch_b.is_empty() {
        return Err(SignalError::ValueError("Channels must not be empty".into()));
    }
    let na = ch_a.len() as i64;
    let nb = ch_b.len() as i64;

    let mean_a: f64 = ch_a.iter().sum::<f64>() / ch_a.len() as f64;
    let mean_b: f64 = ch_b.iter().sum::<f64>() / ch_b.len() as f64;
    let std_a =
        (ch_a.iter().map(|&x| (x - mean_a).powi(2)).sum::<f64>() / ch_a.len() as f64).sqrt();
    let std_b =
        (ch_b.iter().map(|&x| (x - mean_b).powi(2)).sum::<f64>() / ch_b.len() as f64).sqrt();

    let denom = std_a * std_b;
    if denom < f64::EPSILON {
        return Ok(vec![0.0; lags.len()]);
    }

    let mut result = Vec::with_capacity(lags.len());
    for &lag in lags {
        let mut sum = 0.0;
        let mut count = 0u64;
        for i in 0..na {
            let j = i + lag;
            if j >= 0 && j < nb {
                sum += (ch_a[i as usize] - mean_a) * (ch_b[j as usize] - mean_b);
                count += 1;
            }
        }
        if count > 0 {
            result.push(sum / (count as f64 * denom));
        } else {
            result.push(0.0);
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Channel selection & reordering
// ---------------------------------------------------------------------------

/// Select a subset of channels by index.
pub fn select_channels(
    signal: &MultiChannelSignal,
    indices: &[usize],
) -> SignalResult<MultiChannelSignal> {
    let n_ch = signal.num_channels();
    let mut out = Vec::with_capacity(indices.len());
    for &idx in indices {
        if idx >= n_ch {
            return Err(SignalError::InvalidArgument(format!(
                "Channel index {} out of range (0..{})",
                idx, n_ch
            )));
        }
        out.push(signal.channels[idx].clone());
    }
    MultiChannelSignal::new(out)
}

/// Reorder channels according to the given permutation.
pub fn reorder_channels(
    signal: &MultiChannelSignal,
    order: &[usize],
) -> SignalResult<MultiChannelSignal> {
    if order.len() != signal.num_channels() {
        return Err(SignalError::DimensionMismatch(format!(
            "Order length {} != number of channels {}",
            order.len(),
            signal.num_channels()
        )));
    }
    select_channels(signal, order)
}

// ---------------------------------------------------------------------------
// Independent Component Analysis (FastICA)
// ---------------------------------------------------------------------------

/// Configuration for FastICA.
#[derive(Debug, Clone)]
pub struct FastIcaConfig {
    /// Number of independent components to extract.
    pub n_components: usize,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Non-linearity function: "logcosh", "exp", or "cube".
    pub fun: String,
}

impl Default for FastIcaConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            max_iter: 200,
            tol: 1e-4,
            fun: "logcosh".into(),
        }
    }
}

/// Result of FastICA.
#[derive(Debug, Clone)]
pub struct FastIcaResult {
    /// Unmixing matrix `[n_components x n_channels]` (row-major).
    pub unmixing: Vec<f64>,
    /// Estimated independent sources `[n_components][n_samples]`.
    pub sources: Vec<Vec<f64>>,
    /// Number of iterations until convergence.
    pub n_iter: usize,
}

/// Non-linearity functions for FastICA.
fn g_logcosh(x: f64) -> (f64, f64) {
    let gx = x.tanh();
    let g_prime = 1.0 - gx * gx;
    (gx, g_prime)
}

fn g_exp(x: f64) -> (f64, f64) {
    let ex = (-x * x / 2.0).exp();
    let gx = x * ex;
    let g_prime = (1.0 - x * x) * ex;
    (gx, g_prime)
}

fn g_cube(x: f64) -> (f64, f64) {
    (x * x * x, 3.0 * x * x)
}

/// Run FastICA on a multi-channel signal.
///
/// This implements the deflation-based FastICA algorithm with optional
/// non-linearity choice. The signal is first centered and whitened.
pub fn fastica(signal: &MultiChannelSignal, config: &FastIcaConfig) -> SignalResult<FastIcaResult> {
    let n_ch = signal.num_channels();
    let n_samp = signal.num_samples();

    if config.n_components > n_ch {
        return Err(SignalError::InvalidArgument(format!(
            "n_components ({}) > number of channels ({})",
            config.n_components, n_ch
        )));
    }
    if n_samp < 2 {
        return Err(SignalError::ValueError(
            "Need at least 2 samples for ICA".into(),
        ));
    }

    let g_fn: fn(f64) -> (f64, f64) = match config.fun.as_str() {
        "logcosh" => g_logcosh,
        "exp" => g_exp,
        "cube" => g_cube,
        other => {
            return Err(SignalError::InvalidArgument(format!(
                "Unknown non-linearity: {}",
                other
            )));
        }
    };

    // Center data: data[ch][samp]
    let means: Vec<f64> = signal
        .channels
        .iter()
        .map(|ch| ch.iter().sum::<f64>() / n_samp as f64)
        .collect();

    let centered: Vec<Vec<f64>> = signal
        .channels
        .iter()
        .zip(means.iter())
        .map(|(ch, &m)| ch.iter().map(|&x| x - m).collect())
        .collect();

    // Covariance matrix (n_ch x n_ch) row-major
    let cov = compute_cov_matrix(&centered, n_ch, n_samp);

    // Eigendecomposition of covariance via Jacobi for whitening
    let (eigenvalues, eigenvectors) = symmetric_eigen_jacobi(&cov, n_ch, 100)?;

    // Whitening matrix: D^{-1/2} * E^T  (use top n_ch eigenvalues)
    // eigenvalues are sorted descending
    let mut whitening = vec![0.0; n_ch * n_ch]; // n_ch x n_ch
    for i in 0..n_ch {
        let d_inv_sqrt = if eigenvalues[i] > 1e-12 {
            1.0 / eigenvalues[i].sqrt()
        } else {
            0.0
        };
        for j in 0..n_ch {
            // W_white[i][j] = d_inv_sqrt * E[j][i] (E stored column-major as rows of eigenvectors)
            whitening[i * n_ch + j] = d_inv_sqrt * eigenvectors[j * n_ch + i];
        }
    }

    // Whitened data: X_w[i][t] = sum_j whitening[i][j] * centered[j][t]
    let mut x_white = vec![vec![0.0; n_samp]; n_ch];
    for i in 0..n_ch {
        for j in 0..n_ch {
            let w = whitening[i * n_ch + j];
            for t in 0..n_samp {
                x_white[i][t] += w * centered[j][t];
            }
        }
    }

    // FastICA deflation
    let n_comp = config.n_components;
    let mut w_all = vec![vec![0.0; n_ch]; n_comp]; // unmixing vectors

    // Use deterministic initialization
    let mut total_iter = 0usize;
    for p in 0..n_comp {
        // Initialize w_p deterministically using a spread-out direction
        let mut w = vec![0.0; n_ch];
        // Use a simple deterministic initialization: unit vector with rotation
        for i in 0..n_ch {
            let angle = PI * (p as f64 + 0.5) * (i as f64 + 1.0) / (n_ch as f64 + 1.0);
            w[i] = angle.sin();
        }
        let norm = vec_norm(&w);
        if norm > f64::EPSILON {
            for v in &mut w {
                *v /= norm;
            }
        }

        for iter in 0..config.max_iter {
            total_iter += 1;
            // wx[t] = w . x_white[:,t]
            let mut wx = vec![0.0; n_samp];
            for t in 0..n_samp {
                for i in 0..n_ch {
                    wx[t] += w[i] * x_white[i][t];
                }
            }

            // E{x * g(w^T x)} and E{g'(w^T x)}
            let mut w_new = vec![0.0; n_ch];
            let mut mean_g_prime = 0.0;
            for t in 0..n_samp {
                let (gval, gp) = g_fn(wx[t]);
                for i in 0..n_ch {
                    w_new[i] += x_white[i][t] * gval;
                }
                mean_g_prime += gp;
            }
            let inv_n = 1.0 / n_samp as f64;
            mean_g_prime *= inv_n;
            for i in 0..n_ch {
                w_new[i] = w_new[i] * inv_n - mean_g_prime * w[i];
            }

            // Deflation: orthogonalise w_new against already-found components
            for q in 0..p {
                let dot: f64 = w_new
                    .iter()
                    .zip(w_all[q].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                for i in 0..n_ch {
                    w_new[i] -= dot * w_all[q][i];
                }
            }

            // Normalise
            let norm = vec_norm(&w_new);
            if norm < f64::EPSILON {
                break;
            }
            for v in &mut w_new {
                *v /= norm;
            }

            // Check convergence (1 - |w . w_new| < tol)
            let dot: f64 = w.iter().zip(w_new.iter()).map(|(&a, &b)| a * b).sum();
            w = w_new;

            if (1.0 - dot.abs()) < config.tol {
                break;
            }
        }
        w_all[p] = w;
    }

    // Build unmixing matrix: W_ica = W_component * W_white
    let mut unmixing = vec![0.0; n_comp * n_ch];
    for p in 0..n_comp {
        for j in 0..n_ch {
            let mut val = 0.0;
            for k in 0..n_ch {
                val += w_all[p][k] * whitening[k * n_ch + j];
            }
            unmixing[p * n_ch + j] = val;
        }
    }

    // Compute sources: S[p][t] = sum_j unmixing[p][j] * centered[j][t]
    let mut sources = vec![vec![0.0; n_samp]; n_comp];
    for p in 0..n_comp {
        for j in 0..n_ch {
            let u = unmixing[p * n_ch + j];
            for t in 0..n_samp {
                sources[p][t] += u * centered[j][t];
            }
        }
    }

    Ok(FastIcaResult {
        unmixing,
        sources,
        n_iter: total_iter,
    })
}

// ---------------------------------------------------------------------------
// Common Spatial Patterns (CSP) for EEG / BCI
// ---------------------------------------------------------------------------

/// Configuration for CSP.
#[derive(Debug, Clone)]
pub struct CspConfig {
    /// Number of spatial filter pairs to extract (total filters = 2 * n_pairs).
    pub n_pairs: usize,
    /// Regularisation parameter (Tikhonov).  0 = no regularisation.
    pub regularisation: f64,
}

impl Default for CspConfig {
    fn default() -> Self {
        Self {
            n_pairs: 3,
            regularisation: 0.0,
        }
    }
}

/// Result of CSP.
#[derive(Debug, Clone)]
pub struct CspResult {
    /// Spatial filters `[2*n_pairs x n_channels]` row-major.
    pub filters: Vec<f64>,
    /// Eigenvalues corresponding to the filters (sorted).
    pub eigenvalues: Vec<f64>,
    /// Number of channels.
    pub n_channels: usize,
    /// Number of filter pairs.
    pub n_pairs: usize,
}

/// Compute Common Spatial Patterns from two-class EEG data.
///
/// `class1` and `class2` are collections of trials. Each trial is a
/// `MultiChannelSignal`. All trials must have the same number of channels.
///
/// CSP finds spatial filters that maximise the variance ratio between the
/// two classes.
pub fn csp(
    class1: &[MultiChannelSignal],
    class2: &[MultiChannelSignal],
    config: &CspConfig,
) -> SignalResult<CspResult> {
    if class1.is_empty() || class2.is_empty() {
        return Err(SignalError::ValueError(
            "Both classes must have at least one trial".into(),
        ));
    }
    let n_ch = class1[0].num_channels();
    // Validate all trials
    for (cls_idx, cls) in [class1, class2].iter().enumerate() {
        for (t, trial) in cls.iter().enumerate() {
            if trial.num_channels() != n_ch {
                return Err(SignalError::DimensionMismatch(format!(
                    "Class {} trial {} has {} channels, expected {}",
                    cls_idx + 1,
                    t,
                    trial.num_channels(),
                    n_ch
                )));
            }
        }
    }
    if config.n_pairs == 0 || 2 * config.n_pairs > n_ch {
        return Err(SignalError::InvalidArgument(format!(
            "n_pairs must be in [1, {}]",
            n_ch / 2
        )));
    }

    // Average covariance for each class
    let cov1 = average_class_covariance(class1, n_ch)?;
    let cov2 = average_class_covariance(class2, n_ch)?;

    // Composite covariance C = C1 + C2 + reg * I
    let mut c_composite = vec![0.0; n_ch * n_ch];
    for i in 0..n_ch * n_ch {
        c_composite[i] = cov1[i] + cov2[i];
    }
    if config.regularisation > 0.0 {
        for i in 0..n_ch {
            c_composite[i * n_ch + i] += config.regularisation;
        }
    }

    // Whitening: eigendecompose C_composite
    let (ev_comp, u_comp) = symmetric_eigen_jacobi(&c_composite, n_ch, 200)?;

    // P = D^{-1/2} * U^T
    let mut p_white = vec![0.0; n_ch * n_ch];
    for i in 0..n_ch {
        let d_inv = if ev_comp[i] > 1e-12 {
            1.0 / ev_comp[i].sqrt()
        } else {
            0.0
        };
        for j in 0..n_ch {
            p_white[i * n_ch + j] = d_inv * u_comp[j * n_ch + i];
        }
    }

    // S1 = P * C1 * P^T
    let s1 = triple_product(&p_white, &cov1, n_ch);

    // Eigendecompose S1
    let (eigenvalues, eigvecs) = symmetric_eigen_jacobi(&s1, n_ch, 200)?;

    // Full spatial filter = eigvecs^T * P
    let mut full_filters = vec![0.0; n_ch * n_ch];
    for i in 0..n_ch {
        for j in 0..n_ch {
            let mut val = 0.0;
            for k in 0..n_ch {
                val += eigvecs[k * n_ch + i] * p_white[k * n_ch + j];
            }
            full_filters[i * n_ch + j] = val;
        }
    }

    // Select top n_pairs and bottom n_pairs
    let np = config.n_pairs;
    let total = 2 * np;
    let mut filters = vec![0.0; total * n_ch];
    let mut selected_eigenvalues = Vec::with_capacity(total);
    // Top n_pairs (highest eigenvalues)
    for p in 0..np {
        for j in 0..n_ch {
            filters[p * n_ch + j] = full_filters[p * n_ch + j];
        }
        selected_eigenvalues.push(eigenvalues[p]);
    }
    // Bottom n_pairs (lowest eigenvalues)
    for p in 0..np {
        let src = n_ch - 1 - p;
        for j in 0..n_ch {
            filters[(np + p) * n_ch + j] = full_filters[src * n_ch + j];
        }
        selected_eigenvalues.push(eigenvalues[src]);
    }

    Ok(CspResult {
        filters,
        eigenvalues: selected_eigenvalues,
        n_channels: n_ch,
        n_pairs: np,
    })
}

/// Apply CSP filters to a multi-channel signal, returning filtered channels.
pub fn csp_apply(
    signal: &MultiChannelSignal,
    csp_result: &CspResult,
) -> SignalResult<MultiChannelSignal> {
    let n_ch = signal.num_channels();
    if n_ch != csp_result.n_channels {
        return Err(SignalError::DimensionMismatch(format!(
            "Signal has {} channels but CSP expects {}",
            n_ch, csp_result.n_channels
        )));
    }
    let n_filt = 2 * csp_result.n_pairs;
    let n_samp = signal.num_samples();

    let mut out = vec![vec![0.0; n_samp]; n_filt];
    for f in 0..n_filt {
        for c in 0..n_ch {
            let w = csp_result.filters[f * n_ch + c];
            for t in 0..n_samp {
                out[f][t] += w * signal.channels[c][t];
            }
        }
    }
    MultiChannelSignal::new(out)
}

// ---------------------------------------------------------------------------
// Helper: covariance, eigendecomposition, etc.
// ---------------------------------------------------------------------------

fn compute_cov_matrix(data: &[Vec<f64>], n_ch: usize, n_samp: usize) -> Vec<f64> {
    let inv_n = 1.0 / n_samp as f64;
    let mut cov = vec![0.0; n_ch * n_ch];
    for i in 0..n_ch {
        for j in i..n_ch {
            let val: f64 = data[i]
                .iter()
                .zip(data[j].iter())
                .map(|(&a, &b)| a * b)
                .sum::<f64>()
                * inv_n;
            cov[i * n_ch + j] = val;
            cov[j * n_ch + i] = val;
        }
    }
    cov
}

fn average_class_covariance(trials: &[MultiChannelSignal], n_ch: usize) -> SignalResult<Vec<f64>> {
    let mut avg = vec![0.0; n_ch * n_ch];
    let mut total_weight = 0.0;

    for trial in trials {
        let n_samp = trial.num_samples();
        // Center each trial
        let means: Vec<f64> = trial
            .channels
            .iter()
            .map(|ch| ch.iter().sum::<f64>() / n_samp as f64)
            .collect();
        let centered: Vec<Vec<f64>> = trial
            .channels
            .iter()
            .zip(means.iter())
            .map(|(ch, &m)| ch.iter().map(|&x| x - m).collect())
            .collect();
        let cov = compute_cov_matrix(&centered, n_ch, n_samp);
        // Trace-normalize
        let trace: f64 = (0..n_ch).map(|i| cov[i * n_ch + i]).sum();
        let inv_trace = if trace > f64::EPSILON {
            1.0 / trace
        } else {
            0.0
        };
        for i in 0..n_ch * n_ch {
            avg[i] += cov[i] * inv_trace;
        }
        total_weight += 1.0;
    }

    if total_weight > 0.0 {
        let inv_w = 1.0 / total_weight;
        for v in &mut avg {
            *v *= inv_w;
        }
    }
    Ok(avg)
}

/// Jacobi eigendecomposition for symmetric matrices.
/// Returns (eigenvalues_descending, eigenvectors_column_major_as_row_major).
fn symmetric_eigen_jacobi(
    matrix: &[f64],
    n: usize,
    max_sweeps: usize,
) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    if matrix.len() != n * n {
        return Err(SignalError::DimensionMismatch(format!(
            "Matrix size {} != {} x {}",
            matrix.len(),
            n,
            n
        )));
    }

    let mut a = matrix.to_vec();
    // Eigenvectors as identity
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    for _sweep in 0..max_sweeps {
        // Find max off-diagonal
        let mut max_off = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                max_off = max_off.max(a[i * n + j].abs());
            }
        }
        if max_off < 1e-15 {
            break;
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let aij = a[i * n + j];
                if aij.abs() < 1e-15 {
                    continue;
                }
                let diff = a[j * n + j] - a[i * n + i];
                let t = if diff.abs() < 1e-15 {
                    1.0
                } else {
                    let tau = diff / (2.0 * aij);
                    let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
                    sign / (tau.abs() + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update matrix
                let aii = a[i * n + i];
                let ajj = a[j * n + j];
                a[i * n + i] = aii - t * aij;
                a[j * n + j] = ajj + t * aij;
                a[i * n + j] = 0.0;
                a[j * n + i] = 0.0;

                for k in 0..n {
                    if k != i && k != j {
                        let aki = a[k * n + i];
                        let akj = a[k * n + j];
                        a[k * n + i] = c * aki - s * akj;
                        a[i * n + k] = a[k * n + i];
                        a[k * n + j] = s * aki + c * akj;
                        a[j * n + k] = a[k * n + j];
                    }
                }

                // Update eigenvectors
                for k in 0..n {
                    let vki = v[k * n + i];
                    let vkj = v[k * n + j];
                    v[k * n + i] = c * vki - s * vkj;
                    v[k * n + j] = s * vki + c * vkj;
                }
            }
        }
    }

    // Collect eigenvalues and sort descending
    let mut eig_pairs: Vec<(f64, usize)> = (0..n).map(|i| (a[i * n + i], i)).collect();
    eig_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let eigenvalues: Vec<f64> = eig_pairs.iter().map(|&(e, _)| e).collect();
    let mut sorted_v = vec![0.0; n * n];
    for (new_col, &(_, old_col)) in eig_pairs.iter().enumerate() {
        for row in 0..n {
            sorted_v[row * n + new_col] = v[row * n + old_col];
        }
    }

    Ok((eigenvalues, sorted_v))
}

fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Compute A * B * A^T for n x n matrices (all row-major).
fn triple_product(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    // tmp = A * B
    let mut tmp = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += a[i * n + k] * b[k * n + j];
            }
            tmp[i * n + j] = s;
        }
    }
    // result = tmp * A^T
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += tmp[i * n + k] * a[j * n + k];
            }
            result[i * n + j] = s;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_sine_channels(freqs: &[f64], n: usize, fs: f64) -> MultiChannelSignal {
        let channels: Vec<Vec<f64>> = freqs
            .iter()
            .map(|&f| {
                (0..n)
                    .map(|i| (2.0 * PI * f * i as f64 / fs).sin())
                    .collect()
            })
            .collect();
        MultiChannelSignal::new(channels).expect("test signal creation")
    }

    #[test]
    fn test_multichannel_new_valid() {
        let sig = MultiChannelSignal::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert!(sig.is_ok());
        let sig = sig.expect("valid signal");
        assert_eq!(sig.num_channels(), 2);
        assert_eq!(sig.num_samples(), 2);
    }

    #[test]
    fn test_multichannel_new_mismatch() {
        let sig = MultiChannelSignal::new(vec![vec![1.0, 2.0], vec![3.0]]);
        assert!(sig.is_err());
    }

    #[test]
    fn test_mix_to_mono_average() {
        let sig =
            MultiChannelSignal::new(vec![vec![2.0, 4.0], vec![6.0, 8.0]]).expect("test signal");
        let mono = mix_to_mono(&sig, MixMode::Average, None).expect("mono mix");
        assert!((mono[0] - 4.0).abs() < 1e-10);
        assert!((mono[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_mix_to_mono_weighted() {
        let sig =
            MultiChannelSignal::new(vec![vec![1.0, 1.0], vec![3.0, 3.0]]).expect("test signal");
        let mono = mix_to_mono(&sig, MixMode::Weighted, Some(&[1.0, 3.0])).expect("weighted mix");
        // weights normalised: [0.25, 0.75]  => 0.25*1 + 0.75*3 = 2.5
        assert!((mono[0] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_mix_to_mono_select() {
        let sig =
            MultiChannelSignal::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).expect("test signal");
        let mono = mix_to_mono(&sig, MixMode::Select(1), None).expect("select");
        assert!((mono[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mono_to_multichannel() {
        let mono = vec![1.0, 2.0, 3.0];
        let mc = mono_to_multichannel(&mono, 3).expect("upmix");
        assert_eq!(mc.num_channels(), 3);
        assert!((mc.channels[2][1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_mixing_matrix() {
        let sig =
            MultiChannelSignal::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]]).expect("test signal");
        // Swap channels
        let matrix = vec![0.0, 1.0, 1.0, 0.0];
        let mixed = apply_mixing_matrix(&sig, 2, 2, &matrix).expect("mix");
        assert!((mixed.channels[0][0] - 0.0).abs() < 1e-10);
        assert!((mixed.channels[0][1] - 1.0).abs() < 1e-10);
        assert!((mixed.channels[1][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_channel_correlation() {
        let sig = MultiChannelSignal::new(vec![vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 2.0, 3.0, 4.0]])
            .expect("test signal");
        let corr = cross_channel_correlation(&sig).expect("corr");
        // Identical channels => correlation 1.0
        assert!((corr[0] - 1.0).abs() < 1e-10);
        assert!((corr[1] - 1.0).abs() < 1e-10);
        assert!((corr[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_correlation_lag() {
        let a = vec![0.0, 1.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 1.0, 0.0];
        let lags: Vec<i64> = (-2..=2).collect();
        let result = cross_correlation_lag(&a, &b, &lags).expect("xcorr lag");
        // Peak should be at lag=1 (a shifted by +1 matches b)
        let max_idx = result
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(lags[max_idx], 1);
    }

    #[test]
    fn test_select_channels() {
        let sig = MultiChannelSignal::new(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]])
            .expect("test signal");
        let sel = select_channels(&sig, &[2, 0]).expect("select");
        assert_eq!(sel.num_channels(), 2);
        assert!((sel.channels[0][0] - 5.0).abs() < 1e-10);
        assert!((sel.channels[1][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reorder_channels() {
        let sig = MultiChannelSignal::new(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]])
            .expect("test signal");
        let reordered = reorder_channels(&sig, &[2, 0, 1]).expect("reorder");
        assert!((reordered.channels[0][0] - 5.0).abs() < 1e-10);
        assert!((reordered.channels[1][0] - 1.0).abs() < 1e-10);
        assert!((reordered.channels[2][0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_fastica_basic() {
        // Create two simple mixed sources
        let n = 512;
        let fs = 256.0;
        let s1: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 3.0 * i as f64 / fs).sin())
            .collect();
        let s2: Vec<f64> = (0..n)
            .map(|i| {
                let phase = 2.0 * PI * 7.0 * i as f64 / fs;
                if phase.sin() > 0.0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();

        // Mix: x1 = 0.7*s1 + 0.3*s2, x2 = 0.4*s1 + 0.6*s2
        let x1: Vec<f64> = s1
            .iter()
            .zip(s2.iter())
            .map(|(&a, &b)| 0.7 * a + 0.3 * b)
            .collect();
        let x2: Vec<f64> = s1
            .iter()
            .zip(s2.iter())
            .map(|(&a, &b)| 0.4 * a + 0.6 * b)
            .collect();

        let sig = MultiChannelSignal::new(vec![x1, x2]).expect("mixed signal");
        let config = FastIcaConfig {
            n_components: 2,
            max_iter: 200,
            tol: 1e-4,
            fun: "logcosh".into(),
        };
        let result = fastica(&sig, &config).expect("fastica");
        assert_eq!(result.sources.len(), 2);
        assert_eq!(result.sources[0].len(), n);
        // Check that sources are non-trivial (have nonzero variance)
        let var0: f64 = result.sources[0].iter().map(|&x| x * x).sum::<f64>() / n as f64;
        let var1: f64 = result.sources[1].iter().map(|&x| x * x).sum::<f64>() / n as f64;
        assert!(var0 > 0.01, "Source 0 variance too small: {}", var0);
        assert!(var1 > 0.01, "Source 1 variance too small: {}", var1);
    }

    #[test]
    fn test_fastica_nonlinearities() {
        let n = 256;
        let sig = make_sine_channels(&[5.0, 11.0], n, 256.0);

        for fun in &["logcosh", "exp", "cube"] {
            let config = FastIcaConfig {
                n_components: 2,
                max_iter: 100,
                tol: 1e-3,
                fun: fun.to_string(),
            };
            let result = fastica(&sig, &config);
            assert!(result.is_ok(), "FastICA failed with fun={}", fun);
        }
    }

    #[test]
    fn test_csp_basic() {
        // Create simple two-class data
        let n_ch = 4;
        let n_samp = 100;

        // Class 1: energy mostly in channel 0
        let mut ch1_data = vec![vec![0.0; n_samp]; n_ch];
        for i in 0..n_samp {
            ch1_data[0][i] = (2.0 * PI * 10.0 * i as f64 / 100.0).sin();
            ch1_data[1][i] = 0.1 * (2.0 * PI * 10.0 * i as f64 / 100.0).sin();
            ch1_data[2][i] = 0.05 * (2.0 * PI * 5.0 * i as f64 / 100.0).sin();
            ch1_data[3][i] = 0.02 * (2.0 * PI * 3.0 * i as f64 / 100.0).sin();
        }
        let trial1 = MultiChannelSignal::new(ch1_data).expect("trial1");

        // Class 2: energy mostly in channel 3
        let mut ch2_data = vec![vec![0.0; n_samp]; n_ch];
        for i in 0..n_samp {
            ch2_data[0][i] = 0.02 * (2.0 * PI * 10.0 * i as f64 / 100.0).sin();
            ch2_data[1][i] = 0.05 * (2.0 * PI * 10.0 * i as f64 / 100.0).sin();
            ch2_data[2][i] = 0.1 * (2.0 * PI * 5.0 * i as f64 / 100.0).sin();
            ch2_data[3][i] = (2.0 * PI * 3.0 * i as f64 / 100.0).sin();
        }
        let trial2 = MultiChannelSignal::new(ch2_data).expect("trial2");

        let config = CspConfig {
            n_pairs: 2,
            regularisation: 1e-6,
        };
        let result = csp(&[trial1.clone()], &[trial2.clone()], &config).expect("csp");
        assert_eq!(result.eigenvalues.len(), 4);
        assert_eq!(result.filters.len(), 4 * n_ch);

        // Apply CSP
        let filtered = csp_apply(&trial1, &result).expect("csp_apply");
        assert_eq!(filtered.num_channels(), 4);
    }

    #[test]
    fn test_symmetric_eigen_jacobi() {
        // 2x2 symmetric matrix: [[3, 1], [1, 3]] => eigenvalues 4, 2
        let m = vec![3.0, 1.0, 1.0, 3.0];
        let (evals, _evecs) = symmetric_eigen_jacobi(&m, 2, 50).expect("eigen");
        assert!(
            (evals[0] - 4.0).abs() < 1e-10,
            "First eigenvalue should be 4, got {}",
            evals[0]
        );
        assert!(
            (evals[1] - 2.0).abs() < 1e-10,
            "Second eigenvalue should be 2, got {}",
            evals[1]
        );
    }

    #[test]
    fn test_cross_channel_correlation_orthogonal() {
        // Orthogonal channels: sin and cos
        let n = 1024;
        let sig = MultiChannelSignal::new(vec![
            (0..n)
                .map(|i| (2.0 * PI * 4.0 * i as f64 / n as f64).sin())
                .collect(),
            (0..n)
                .map(|i| (2.0 * PI * 4.0 * i as f64 / n as f64).cos())
                .collect(),
        ])
        .expect("test");
        let corr = cross_channel_correlation(&sig).expect("corr");
        // Self-correlation should be 1
        assert!((corr[0] - 1.0).abs() < 1e-8);
        assert!((corr[3] - 1.0).abs() < 1e-8);
        // Cross-correlation should be near 0
        assert!(
            corr[1].abs() < 0.05,
            "Cross-corr should be near 0, got {}",
            corr[1]
        );
    }
}
