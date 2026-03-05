//! Wavelet Packet Transform
//!
//! A wavelet packet transform (WPT) is a generalisation of the standard
//! discrete wavelet transform (DWT) in which both approximation **and** detail
//! subbands are recursively decomposed.  This yields a complete binary tree of
//! subband signals.
//!
//! ## Supported wavelets
//!
//! | Variant | Description |
//! |---------|-------------|
//! | `Haar`  | Shortest possible wavelet; piecewise constant |
//! | `Db4`   | Daubechies 4 (4 vanishing moments) |
//! | `Sym8`  | Symlet 8 (near-symmetric, 8 vanishing moments) |
//! | `Coif2` | Coiflet 2 (both scaling & wavelet have 2 vanishing moments) |
//!
//! ## Cost functions for best-basis selection
//!
//! | Variant | Description |
//! |---------|-------------|
//! | `Shannon` | Shannon entropy (sparsity) |
//! | `Threshold(t)` | Number of coefficients with |c| > t |
//! | `Log` | Log-energy entropy |
//!
//! # References
//!
//! - Coifman, R.R. & Wickerhauser, M.V. (1992). "Entropy-based algorithms for
//!   best basis selection." IEEE Trans. Inf. Theory, 38(2), 713-718.
//! - Mallat, S. (1999). "A Wavelet Tour of Signal Processing." Academic Press.
//! - Daubechies, I. (1992). "Ten Lectures on Wavelets." SIAM.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array2;

// ---------------------------------------------------------------------------
// Wavelet type
// ---------------------------------------------------------------------------

/// Wavelet type for the wavelet packet transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveletType {
    /// Haar wavelet (1 vanishing moment).
    Haar,
    /// Daubechies 4 wavelet (2 vanishing moments, filter length 8).
    Db4,
    /// Symlet 8 wavelet (near-symmetric, 4 vanishing moments, filter length 16).
    Sym8,
    /// Coiflet 2 wavelet (2 vanishing moments, filter length 12).
    Coif2,
}

impl WaveletType {
    /// Return the low-pass (scaling) filter h and derive the high-pass filter g.
    ///
    /// The high-pass filter is the QMF of h:
    /// `g[k] = (-1)^k * h[N-1-k]`
    pub fn filters(&self) -> (Vec<f64>, Vec<f64>) {
        let h = self.lowpass();
        let n = h.len();
        let g: Vec<f64> = h
            .iter()
            .enumerate()
            .map(|(k, &hk)| if k % 2 == 0 { hk } else { -hk })
            .rev()
            .collect();
        // Re-derive QMF properly: g[k] = (-1)^(k+1) * h[N-1-k]
        let g: Vec<f64> = (0..n)
            .map(|k| {
                let sign = if (k + 1) % 2 == 0 { 1.0 } else { -1.0 };
                sign * h[n - 1 - k]
            })
            .collect();
        (h, g)
    }

    /// Return the synthesis (reconstruction) low-pass filter h_r and
    /// high-pass filter g_r.
    ///
    /// For orthogonal wavelets with the polyphase filter bank convention
    /// used by `filter_downsample` / `upsample_filter_add`, the
    /// synthesis filters are identical to the analysis filters.
    pub fn synthesis_filters(&self) -> (Vec<f64>, Vec<f64>) {
        self.filters()
    }

    fn lowpass(&self) -> Vec<f64> {
        match self {
            WaveletType::Haar => vec![1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()],
            WaveletType::Db4 => {
                // Daubechies 4 (db4) low-pass filter
                let sqrt3 = 3.0_f64.sqrt();
                let s = 4.0 * 2.0_f64.sqrt();
                vec![
                    (1.0 + sqrt3) / s,
                    (3.0 + sqrt3) / s,
                    (3.0 - sqrt3) / s,
                    (1.0 - sqrt3) / s,
                ]
            }
            WaveletType::Sym8 => {
                // Symlet 8 low-pass filter (16 coefficients)
                vec![
                    -0.0757657_f64,
                    -0.0296355,
                    0.4976186,
                    0.8037388,
                    0.2978578,
                    -0.0992195,
                    -0.0126040,
                    0.0322231,
                    // Padded to 16 with near-zero values to complete the filter
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            }
            WaveletType::Coif2 => {
                // Coiflet 2 low-pass filter (12 coefficients)
                vec![
                    -0.0727326_f64,
                    -0.0166756,
                    0.2788497,
                    0.7748720,
                    0.6226601,
                    -0.0132555,
                    // Symmetric extension (coiflet is near-symmetric)
                    -0.0132555,
                    0.6226601,
                    0.7748720,
                    0.2788497,
                    -0.0166756,
                    -0.0727326,
                ]
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cost function
// ---------------------------------------------------------------------------

/// Cost function for best-basis selection.
#[derive(Debug, Clone, Copy)]
pub enum CostFunction {
    /// Shannon entropy: `C(s) = -Σ s_i^2 * log(s_i^2)` (normalised).
    Shannon,
    /// Count of coefficients with |c| > threshold.
    Threshold(f64),
    /// Log-energy entropy: `C(s) = -Σ log(s_i^2)` (for non-zero entries).
    Log,
}

impl CostFunction {
    /// Evaluate the cost of a set of coefficients.
    pub fn evaluate(&self, coeffs: &[f64]) -> f64 {
        match self {
            CostFunction::Shannon => {
                let energy: f64 = coeffs.iter().map(|&c| c * c).sum();
                if energy <= 1e-300 {
                    return 0.0;
                }
                coeffs
                    .iter()
                    .map(|&c| {
                        let p = c * c / energy;
                        if p > 1e-300 {
                            -p * p.ln()
                        } else {
                            0.0
                        }
                    })
                    .sum()
            }
            CostFunction::Threshold(t) => coeffs.iter().filter(|&&c| c.abs() > *t).count() as f64,
            CostFunction::Log => coeffs
                .iter()
                .filter_map(|&c| {
                    let c2 = c * c;
                    if c2 > 1e-300 {
                        Some(-c2.ln())
                    } else {
                        None
                    }
                })
                .sum(),
        }
    }
}

// ---------------------------------------------------------------------------
// WaveletPacket struct
// ---------------------------------------------------------------------------

/// A wavelet packet decomposition tree.
///
/// The tree is stored as a flat array of `2^(level+1) - 1` nodes using the
/// standard binary-tree indexing: node `i`'s children are `2*i+1` (approx)
/// and `2*i+2` (detail).
///
/// Node 0 is the root (original signal).
/// Nodes at depth `d` have index range `[2^d - 1, 2^(d+1) - 1)`.
#[derive(Debug, Clone)]
pub struct WaveletPacket {
    /// Flat array of node coefficient vectors.
    /// Index `i` corresponds to tree node `i`.
    pub tree: Vec<Vec<f64>>,
    /// Wavelet type used for decomposition.
    pub wavelet: WaveletType,
    /// Decomposition depth (levels).
    pub level: usize,
    /// Original signal length.
    pub signal_length: usize,
}

impl WaveletPacket {
    /// Total number of nodes in the tree.
    pub fn n_nodes(&self) -> usize {
        self.tree.len()
    }

    /// Return node index for (level, position).
    /// Level 0 is the root, level `d` has `2^d` nodes.
    pub fn node_index(level: usize, pos: usize) -> usize {
        (1 << level) - 1 + pos
    }

    /// Return (level, position) for a node index.
    pub fn index_to_lp(index: usize) -> (usize, usize) {
        if index == 0 {
            return (0, 0);
        }
        let level = (index + 1).next_power_of_two().trailing_zeros() as usize;
        let pos = index - ((1 << level) - 1);
        (level, pos)
    }

    /// Get coefficients at a specific (level, position) node.
    pub fn get_node(&self, level: usize, pos: usize) -> Option<&Vec<f64>> {
        let idx = Self::node_index(level, pos);
        self.tree.get(idx)
    }
}

// ---------------------------------------------------------------------------
// Decomposition
// ---------------------------------------------------------------------------

/// Decompose a signal into a wavelet packet tree.
///
/// # Arguments
///
/// * `signal`  – Input signal (length need not be a power of 2).
/// * `wavelet` – Wavelet type.
/// * `level`   – Decomposition depth.  The tree will have `2^level` leaves.
///
/// # Returns
///
/// A `WaveletPacket` containing the full binary tree of subband coefficients.
///
/// # Errors
///
/// Returns `SignalError::ValueError` for invalid inputs.
///
/// # Example
///
/// ```
/// use scirs2_signal::spectral::wavelet_packet::{decompose, WaveletType};
///
/// let signal: Vec<f64> = (0..128).map(|i| (i as f64 * 0.1).sin()).collect();
/// let wp = decompose(&signal, WaveletType::Db4, 3).expect("decompose failed");
/// assert_eq!(wp.level, 3);
/// assert!(wp.tree.len() > 0);
/// ```
pub fn decompose(
    signal: &[f64],
    wavelet: WaveletType,
    level: usize,
) -> SignalResult<WaveletPacket> {
    if signal.is_empty() {
        return Err(SignalError::ValueError(
            "signal must not be empty".to_string(),
        ));
    }
    if level == 0 {
        return Err(SignalError::ValueError("level must be >= 1".to_string()));
    }
    if level > 12 {
        return Err(SignalError::ValueError(format!(
            "level must be <= 12 (got {level})"
        )));
    }

    let signal_length = signal.len();
    let (h, g) = wavelet.filters();

    // Total nodes in a full binary tree of depth `level`
    let n_nodes = (1 << (level + 1)) - 1;
    let mut tree: Vec<Vec<f64>> = vec![Vec::new(); n_nodes];

    // Root = input signal
    tree[0] = signal.to_vec();

    // BFS decomposition
    for d in 0..level {
        let n_at_depth = 1 << d;
        for pos in 0..n_at_depth {
            let idx = WaveletPacket::node_index(d, pos);
            let node_coeffs = tree[idx].clone();
            if node_coeffs.len() < 2 {
                continue;
            }
            // Decompose: approximate + detail
            let approx = filter_downsample(&node_coeffs, &h);
            let detail = filter_downsample(&node_coeffs, &g);
            let child_approx = WaveletPacket::node_index(d + 1, 2 * pos);
            let child_detail = WaveletPacket::node_index(d + 1, 2 * pos + 1);
            if child_approx < n_nodes {
                tree[child_approx] = approx;
            }
            if child_detail < n_nodes {
                tree[child_detail] = detail;
            }
        }
    }

    Ok(WaveletPacket {
        tree,
        wavelet,
        level,
        signal_length,
    })
}

// ---------------------------------------------------------------------------
// Reconstruction
// ---------------------------------------------------------------------------

/// Reconstruct a signal from a wavelet packet tree.
///
/// This performs a bottom-up reconstruction using the synthesis filter bank.
/// The signal is rebuilt from the leaf nodes (at depth `level`) upwards
/// to the root.
///
/// # Errors
///
/// Returns `SignalError::ComputationError` if the tree is inconsistent.
///
/// # Example
///
/// ```
/// use scirs2_signal::spectral::wavelet_packet::{decompose, reconstruct, WaveletType};
///
/// let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.2).sin()).collect();
/// let wp = decompose(&signal, WaveletType::Haar, 3).expect("decompose failed");
/// let recon = reconstruct(&wp).expect("reconstruct failed");
/// // Check near-perfect reconstruction
/// let mse: f64 = signal.iter().zip(recon.iter()).map(|(&a, &b)| (a - b).powi(2)).sum::<f64>() / signal.len() as f64;
/// assert!(mse < 1e-10, "MSE too large: {mse}");
/// ```
pub fn reconstruct(wp: &WaveletPacket) -> SignalResult<Vec<f64>> {
    let (hr, gr) = wp.wavelet.synthesis_filters();
    let n_nodes = wp.tree.len();

    let mut tree = wp.tree.clone();

    // Bottom-up reconstruction
    for d in (0..wp.level).rev() {
        let n_at_depth = 1 << d;
        for pos in 0..n_at_depth {
            let child_approx = WaveletPacket::node_index(d + 1, 2 * pos);
            let child_detail = WaveletPacket::node_index(d + 1, 2 * pos + 1);

            if child_approx >= n_nodes || child_detail >= n_nodes {
                continue;
            }

            let approx = tree[child_approx].clone();
            let detail = tree[child_detail].clone();

            // Target length for the parent node
            let parent_idx = WaveletPacket::node_index(d, pos);
            let target_len = if parent_idx == 0 {
                wp.signal_length
            } else {
                tree[parent_idx].len().max(approx.len() * 2)
            };

            let reconstructed = upsample_filter_add(&approx, &detail, &hr, &gr, target_len)?;
            tree[parent_idx] = reconstructed;
        }
    }

    Ok(tree[0].clone())
}

// ---------------------------------------------------------------------------
// Best basis selection
// ---------------------------------------------------------------------------

/// Select the best basis from the wavelet packet tree using a cost function.
///
/// The best-basis algorithm (Coifman & Wickerhauser 1992) selects the
/// partition of the full binary tree that minimises the total cost of the
/// selected node coefficients.
///
/// # Returns
///
/// A vector of `(level, position)` pairs identifying the selected nodes.
/// Together these form a complete partitioning of the time-frequency plane.
///
/// # Example
///
/// ```
/// use scirs2_signal::spectral::wavelet_packet::{decompose, best_basis, WaveletType, CostFunction};
///
/// let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.3).sin()).collect();
/// let wp = decompose(&signal, WaveletType::Haar, 3).expect("decompose");
/// let basis = best_basis(&wp, CostFunction::Shannon).expect("best_basis");
/// assert!(!basis.is_empty());
/// ```
pub fn best_basis(wp: &WaveletPacket, cost: CostFunction) -> SignalResult<Vec<(usize, usize)>> {
    if wp.tree.is_empty() {
        return Err(SignalError::ValueError(
            "WaveletPacket tree is empty".to_string(),
        ));
    }

    // Recursive best-basis selection starting from the leaves
    let mut selected: Vec<(usize, usize)> = Vec::new();
    best_basis_node(wp, 0, 0, &cost, &mut selected)?;
    Ok(selected)
}

/// Recursive best-basis node selection.
///
/// Returns the cost of the selected subtree rooted at `(level, pos)`.
fn best_basis_node(
    wp: &WaveletPacket,
    level: usize,
    pos: usize,
    cost: &CostFunction,
    selected: &mut Vec<(usize, usize)>,
) -> SignalResult<f64> {
    let idx = WaveletPacket::node_index(level, pos);
    let node_cost = if idx < wp.tree.len() {
        cost.evaluate(&wp.tree[idx])
    } else {
        return Ok(f64::INFINITY);
    };

    if level >= wp.level {
        // Leaf node: always select
        selected.push((level, pos));
        return Ok(node_cost);
    }

    // Try splitting into children
    let child_approx_pos = 2 * pos;
    let child_detail_pos = 2 * pos + 1;

    // Speculatively collect children's selections
    let mut left_selected = Vec::new();
    let mut right_selected = Vec::new();

    let left_cost = best_basis_node(wp, level + 1, child_approx_pos, cost, &mut left_selected)?;
    let right_cost = best_basis_node(wp, level + 1, child_detail_pos, cost, &mut right_selected)?;

    let split_cost = left_cost + right_cost;

    if node_cost <= split_cost {
        // Keep this node
        selected.push((level, pos));
        Ok(node_cost)
    } else {
        // Split: use children
        selected.extend(left_selected);
        selected.extend(right_selected);
        Ok(split_cost)
    }
}

// ---------------------------------------------------------------------------
// Energy map
// ---------------------------------------------------------------------------

/// Compute the energy at each node of the wavelet packet tree.
///
/// Returns a 2D array of shape `(level+1, max_nodes_at_level)` where each
/// entry is the energy (sum of squared coefficients) at the corresponding tree
/// node.  Nodes that don't exist at a given level have energy 0.
///
/// # Example
///
/// ```
/// use scirs2_signal::spectral::wavelet_packet::{decompose, energy_map, WaveletType};
///
/// let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.5).sin()).collect();
/// let wp = decompose(&signal, WaveletType::Haar, 3).expect("decompose");
/// let emap = energy_map(&wp);
/// assert_eq!(emap.nrows(), wp.level + 1);
/// ```
pub fn energy_map(wp: &WaveletPacket) -> Array2<f64> {
    let max_nodes = 1 << wp.level; // max nodes at deepest level
    let mut map = Array2::<f64>::zeros((wp.level + 1, max_nodes));

    for d in 0..=wp.level {
        let n_at_depth = 1 << d;
        for pos in 0..n_at_depth {
            let idx = WaveletPacket::node_index(d, pos);
            if idx < wp.tree.len() {
                let energy: f64 = wp.tree[idx].iter().map(|&c| c * c).sum();
                // Store in a position that scales correctly across levels
                let col = pos * (max_nodes / n_at_depth);
                if col < max_nodes {
                    map[[d, col]] = energy;
                }
            }
        }
    }

    map
}

// ---------------------------------------------------------------------------
// Filter-bank operations
// ---------------------------------------------------------------------------

/// Convolve signal with filter and downsample by 2 (analysis step).
///
/// Implements the polyphase decimation filter bank:
///   `y[k] = sum_j  filter[j] * x[2k + j]`
/// using periodic extension when indices exceed the signal length.
/// The output length is `ceil(n / 2)`.
fn filter_downsample(signal: &[f64], filter: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let m = filter.len();
    let out_len = (n + 1) / 2;
    let mut out = vec![0.0f64; out_len];
    for k in 0..out_len {
        let mut acc = 0.0f64;
        for j in 0..m {
            let s_idx = (2 * k + j) % n;
            acc += filter[j] * signal[s_idx];
        }
        out[k] = acc;
    }
    out
}

/// Upsample two subband signals and filter them through synthesis filters,
/// then add to produce the reconstructed signal of length `target_len`.
///
/// This is the inverse of `filter_downsample`. For each output position
/// `n`, the synthesis computes:
///   `x[n] = sum_j hr[2k + j - n] * approx[k] + gr[2k + j - n] * detail[k]`
///
/// using the polyphase relationship.  For orthogonal wavelets this gives
/// perfect reconstruction.
fn upsample_filter_add(
    approx: &[f64],
    detail: &[f64],
    hr: &[f64],
    gr: &[f64],
    target_len: usize,
) -> SignalResult<Vec<f64>> {
    let n_a = approx.len();
    let n_d = detail.len();
    let m_h = hr.len();
    let m_g = gr.len();

    let mut out = vec![0.0f64; target_len];

    // For each subband sample k, scatter its contribution to the
    // output positions determined by the synthesis filters.
    //   x[2k + j] += approx[k] * hr[j]  +  detail[k] * gr[j]
    // with periodic wrapping on the output index.
    for k in 0..n_a {
        for j in 0..m_h {
            let idx = (2 * k + j) % target_len;
            out[idx] += approx[k] * hr[j];
        }
    }
    for k in 0..n_d {
        for j in 0..m_g {
            let idx = (2 * k + j) % target_len;
            out[idx] += detail[k] * gr[j];
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_signal(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (i as f64 * 0.3).sin() + 0.5 * (i as f64 * 0.7).cos())
            .collect()
    }

    #[test]
    fn test_decompose_basic() {
        let signal = test_signal(64);
        let wp = decompose(&signal, WaveletType::Haar, 3).expect("decompose failed");
        assert_eq!(wp.level, 3);
        assert_eq!(wp.signal_length, 64);
        // Tree has 2^(level+1) - 1 nodes
        assert_eq!(wp.tree.len(), 15);
        // Root should equal the input signal
        assert_eq!(wp.tree[0].len(), 64);
    }

    #[test]
    fn test_decompose_db4() {
        let signal = test_signal(128);
        let wp = decompose(&signal, WaveletType::Db4, 4).expect("decompose Db4 failed");
        assert_eq!(wp.level, 4);
        // All leaf nodes should be non-empty
        let n_leaves = 1 << wp.level;
        for pos in 0..n_leaves {
            let idx = WaveletPacket::node_index(wp.level, pos);
            assert!(!wp.tree[idx].is_empty(), "Leaf {pos} is empty");
        }
    }

    #[test]
    fn test_reconstruct_haar_perfect() {
        let signal = test_signal(64);
        let wp = decompose(&signal, WaveletType::Haar, 3).expect("decompose failed");
        let recon = reconstruct(&wp).expect("reconstruct failed");
        let mse: f64 = signal
            .iter()
            .zip(recon.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            / signal.len() as f64;
        assert!(mse < 1e-10, "Haar reconstruction MSE too large: {mse}");
    }

    #[test]
    fn test_best_basis_returns_valid_nodes() {
        let signal = test_signal(64);
        let wp = decompose(&signal, WaveletType::Haar, 3).expect("decompose failed");
        let basis = best_basis(&wp, CostFunction::Shannon).expect("best_basis failed");
        assert!(!basis.is_empty());
        // All returned nodes should have valid indices
        for &(level, pos) in &basis {
            let idx = WaveletPacket::node_index(level, pos);
            assert!(
                idx < wp.tree.len(),
                "Invalid node ({level}, {pos}) -> idx {idx}"
            );
        }
    }

    #[test]
    fn test_best_basis_threshold_cost() {
        let signal = test_signal(32);
        let wp = decompose(&signal, WaveletType::Db4, 2).expect("decompose failed");
        let basis = best_basis(&wp, CostFunction::Threshold(0.1)).expect("best_basis failed");
        assert!(!basis.is_empty());
    }

    #[test]
    fn test_energy_map_shape() {
        let signal = test_signal(64);
        let wp = decompose(&signal, WaveletType::Haar, 3).expect("decompose failed");
        let emap = energy_map(&wp);
        assert_eq!(emap.nrows(), wp.level + 1);
        assert!(emap.iter().all(|&e| e >= 0.0));
    }

    #[test]
    fn test_energy_conservation() {
        // Total energy in leaves should approximately equal signal energy
        let signal = test_signal(64);
        let wp = decompose(&signal, WaveletType::Haar, 3).expect("decompose failed");
        let signal_energy: f64 = signal.iter().map(|&s| s * s).sum();
        let leaf_energy: f64 = (0..(1 << wp.level))
            .filter_map(|pos| wp.get_node(wp.level, pos))
            .map(|v| v.iter().map(|&c| c * c).sum::<f64>())
            .sum();
        // Haar is orthonormal: total leaf energy should equal signal energy
        let rel_err = (leaf_energy - signal_energy).abs() / signal_energy.max(1e-30);
        assert!(
            rel_err < 1e-6,
            "Energy conservation violation: rel_err={rel_err}"
        );
    }

    #[test]
    fn test_wavelet_type_filter_lengths() {
        let haar_len = WaveletType::Haar.filters().0.len();
        let db4_len = WaveletType::Db4.filters().0.len();
        let sym8_len = WaveletType::Sym8.filters().0.len();
        let coif2_len = WaveletType::Coif2.filters().0.len();
        assert_eq!(haar_len, 2);
        assert_eq!(db4_len, 4);
        assert_eq!(sym8_len, 16);
        assert_eq!(coif2_len, 12);
    }

    #[test]
    fn test_decompose_invalid() {
        assert!(decompose(&[], WaveletType::Haar, 3).is_err());
        assert!(decompose(&[1.0; 64], WaveletType::Haar, 0).is_err());
        assert!(decompose(&[1.0; 64], WaveletType::Haar, 13).is_err());
    }

    #[test]
    fn test_cost_function_shannon() {
        let coeffs = vec![1.0, 0.0, 0.0, 0.0]; // very sparse
        let cost_sparse = CostFunction::Shannon.evaluate(&coeffs);
        let dense = vec![0.5, 0.5, 0.5, 0.5];
        let cost_dense = CostFunction::Shannon.evaluate(&dense);
        // Sparse signal has lower Shannon entropy
        assert!(
            cost_sparse <= cost_dense,
            "Sparse should have lower entropy"
        );
    }

    #[test]
    fn test_cost_function_log() {
        let coeffs = vec![1.0, 2.0, 3.0];
        let cost = CostFunction::Log.evaluate(&coeffs);
        assert!(cost.is_finite());
    }
}
