//! Wavelet Packet Transform (WPT)
//!
//! This module implements the full binary-tree wavelet packet decomposition.
//! Unlike the standard Discrete Wavelet Transform (DWT), which only recurses on
//! approximation coefficients, the WPT decomposes **both** approximation and
//! detail branches at every level, giving `2^max_level` terminal subbands that
//! together tile the time-frequency plane with uniform resolution.
//!
//! # Algorithm Overview
//!
//! Given a signal `x[n]` and a filter bank `{h[n], g[n]}`:
//!
//! ```text
//!          x ──┬── H ─↓2─ a   (approximation)
//!              └── G ─↓2─ d   (detail)
//! ```
//!
//! The WPT applies this recursively to both `a` and `d` up to `max_level`.
//! Reconstruction uses the synthesis bank `{h̃, g̃}` with upsampling.
//!
//! # Best-Basis Selection (Coifman–Wickerhauser)
//!
//! Given a cost function `C(·)` that is *additive* (i.e. `C(a ∪ b) = C(a) + C(b)`)
//! and *subadditive* with respect to splitting (`C(node) <= C(left) + C(right)`
//! holds in the best case), we use a bottom-up dynamic programme:
//!
//! 1. Initialise every leaf with its cost.
//! 2. For each internal node, set its cost to `min(node_cost, left + right)`.
//! 3. Reconstruct the optimal partition by a single top-down pass.

use crate::dwt::{dwt_decompose, dwt_reconstruct, Wavelet};
use crate::error::{SignalError, SignalResult};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Cost measures
// ---------------------------------------------------------------------------

/// A cost (information / complexity) measure used for best-basis selection.
///
/// All measures are designed so that *lower* cost is *better* (more
/// concentrated / sparser representation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CostMeasure {
    /// Shannon entropy: `−∑ p_i log₂(p_i)` where `p_i = x_i²/‖x‖²`.
    ///
    /// Zero coefficients contribute zero (lim_{p→0} p log p = 0).
    Shannon,

    /// Log-energy entropy: `∑ log₂(x_i²)` for `x_i ≠ 0`.
    ///
    /// More sensitive to near-zero coefficients than Shannon.
    LogEnergy,

    /// ℓ^p quasi-norm: `∑ |x_i|^p` for `0 < p ≤ 2`.
    ///
    /// Smaller p emphasises sparsity. The parameter is stored in the variant.
    LpNorm(f64),

    /// Threshold concentration: count of coefficients with `|x_i| > thresh`.
    ///
    /// Penalises bases that require many large coefficients.
    Threshold(f64),

    /// Gini-like concentration index (1 − Gini coefficient).
    ///
    /// Ranges from 0 (maximally sparse) to 1 (uniform energy).
    Concentration,

    /// SURE (Stein's Unbiased Risk Estimate) for soft-thresholding.
    ///
    /// `sigma` is the noise standard deviation.
    Sure(f64),
}

// ---------------------------------------------------------------------------
// Single-node cost computation
// ---------------------------------------------------------------------------

/// Compute the cost of a coefficient vector according to `measure`.
///
/// Returns `f64::INFINITY` when the vector is empty so that empty subbands
/// are never preferred over non-empty ones.
pub fn wpt_cost_function(coeffs: &[f64], measure: CostMeasure) -> f64 {
    if coeffs.is_empty() {
        return f64::INFINITY;
    }

    match measure {
        CostMeasure::Shannon => {
            let energy: f64 = coeffs.iter().map(|&c| c * c).sum();
            if energy < f64::EPSILON {
                return 0.0;
            }
            coeffs.iter().fold(0.0, |acc, &c| {
                let p = c * c / energy;
                if p > f64::EPSILON {
                    acc - p * p.log2()
                } else {
                    acc
                }
            })
        }

        CostMeasure::LogEnergy => coeffs.iter().fold(0.0, |acc, &c| {
            let c2 = c * c;
            if c2 > f64::EPSILON {
                acc + c2.log2()
            } else {
                acc
            }
        }),

        CostMeasure::LpNorm(p) => {
            let p = p.clamp(f64::EPSILON, 2.0);
            coeffs.iter().map(|&c| c.abs().powf(p)).sum()
        }

        CostMeasure::Threshold(thresh) => {
            coeffs.iter().filter(|&&c| c.abs() > thresh).count() as f64
        }

        CostMeasure::Concentration => {
            // 1 − Gini coefficient (lower = sparser = better)
            let mut abs_sorted: Vec<f64> = coeffs.iter().map(|&c| c.abs()).collect();
            abs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = abs_sorted.len() as f64;
            let total: f64 = abs_sorted.iter().sum();
            if total < f64::EPSILON {
                return 0.0;
            }
            let gini_num: f64 = abs_sorted
                .iter()
                .enumerate()
                .map(|(i, &v)| (2.0 * (i as f64 + 1.0) - n - 1.0) * v)
                .sum::<f64>();
            let gini = gini_num / (n * total);
            1.0 - gini.abs()
        }

        CostMeasure::Sure(sigma) => {
            // SURE for soft-thresholding with universal threshold t = σ√(2 log n)
            let n = coeffs.len() as f64;
            let thresh = sigma * (2.0 * n.ln()).sqrt();
            let risk: f64 = coeffs.iter().map(|&c| {
                let c_abs = c.abs();
                if c_abs <= thresh {
                    -sigma * sigma + c * c
                } else {
                    sigma * sigma * (2.0 * (c_abs / thresh).ln() + 1.0)
                }
            }).sum();
            risk
        }
    }
}

// ---------------------------------------------------------------------------
// WaveletPacketNode
// ---------------------------------------------------------------------------

/// A single node in the wavelet packet tree.
///
/// Nodes are addressed by `(level, position)` where:
/// - `level = 0` is the root (contains the full signal),
/// - `position = 0` at `level` is the leftmost (all-approximation) node,
/// - Left child of `(l, p)` is `(l+1, 2*p)` (approximation branch),
/// - Right child of `(l, p)` is `(l+1, 2*p+1)` (detail branch).
#[derive(Debug, Clone)]
pub struct WaveletPacketNode {
    /// Decomposition level (0 = root).
    pub level: usize,
    /// Position within the level.
    pub position: usize,
    /// Subband coefficients.
    pub coeffs: Vec<f64>,
    /// Pre-computed cost (lazily filled by [`WaveletPacketTree`]).
    pub cost: f64,
}

impl WaveletPacketNode {
    /// Construct a new node.
    pub fn new(level: usize, position: usize, coeffs: Vec<f64>) -> Self {
        WaveletPacketNode { level, position, coeffs, cost: f64::INFINITY }
    }

    /// Binary path from root to this node.
    ///
    /// Each element is `0` (approximation) or `1` (detail).
    pub fn path(&self) -> Vec<u8> {
        if self.level == 0 {
            return vec![];
        }
        let mut path = vec![0u8; self.level];
        let mut pos = self.position;
        for i in (0..self.level).rev() {
            path[i] = (pos & 1) as u8;
            pos >>= 1;
        }
        path
    }

    /// Human-readable path string, e.g. `"aad"` for approx-approx-detail.
    pub fn path_string(&self) -> String {
        self.path()
            .iter()
            .map(|&b| if b == 0 { 'a' } else { 'd' })
            .collect()
    }

    /// Energy (sum of squared coefficients).
    pub fn energy(&self) -> f64 {
        self.coeffs.iter().map(|&c| c * c).sum()
    }

    /// Address of the left child `(l+1, 2p)` (approximation subband).
    pub fn left_child(&self) -> (usize, usize) {
        (self.level + 1, self.position * 2)
    }

    /// Address of the right child `(l+1, 2p+1)` (detail subband).
    pub fn right_child(&self) -> (usize, usize) {
        (self.level + 1, self.position * 2 + 1)
    }

    /// Address of the parent node, or `None` for the root.
    pub fn parent(&self) -> Option<(usize, usize)> {
        if self.level == 0 {
            None
        } else {
            Some((self.level - 1, self.position / 2))
        }
    }
}

// ---------------------------------------------------------------------------
// WaveletPacketTree
// ---------------------------------------------------------------------------

/// Full binary wavelet packet tree.
///
/// After construction all `2^(max_level+1) − 1` nodes are populated.
/// The `build` method performs a level-by-level decomposition; `best_basis`
/// then uses the Coifman–Wickerhauser algorithm to select the optimal
/// sub-tree partition.
#[derive(Debug)]
pub struct WaveletPacketTree {
    /// All nodes keyed by `(level, position)`.
    pub nodes: HashMap<(usize, usize), WaveletPacketNode>,
    /// Wavelet used for all decompositions.
    pub wavelet: Wavelet,
    /// Maximum level (depth of leaves).
    pub max_level: usize,
    /// Signal extension mode (e.g. `"symmetric"`, `"periodic"`).
    pub mode: String,
    /// Length of the original input signal.
    pub original_length: usize,
}

impl WaveletPacketTree {
    /// Build a full wavelet packet tree from `signal`.
    ///
    /// # Arguments
    /// - `signal` – Input time-domain signal (any length ≥ 2 · filter_len).
    /// - `wavelet` – Wavelet family; must support `dwt_decompose`.
    /// - `max_level` – Number of decomposition levels.
    /// - `mode` – Boundary extension mode (`"symmetric"`, `"periodic"`, `"zero"`).
    ///
    /// # Errors
    /// Returns [`SignalError::ValueError`] if the signal is too short or
    /// `max_level` is zero.
    pub fn build(
        signal: &[f64],
        wavelet: Wavelet,
        max_level: usize,
        mode: &str,
    ) -> SignalResult<Self> {
        if signal.is_empty() {
            return Err(SignalError::ValueError(
                "WaveletPacketTree: signal must not be empty".to_string(),
            ));
        }
        if max_level == 0 {
            return Err(SignalError::ValueError(
                "WaveletPacketTree: max_level must be at least 1".to_string(),
            ));
        }

        let filter_len = wavelet.get_filter_length()?;
        let min_len = 2 * filter_len;
        if signal.len() < min_len {
            return Err(SignalError::ValueError(format!(
                "WaveletPacketTree: signal length {} is less than the minimum {} \
                 (2 × filter length) for wavelet {:?}",
                signal.len(),
                min_len,
                wavelet
            )));
        }

        let mut nodes: HashMap<(usize, usize), WaveletPacketNode> =
            HashMap::with_capacity((1 << (max_level + 1)) - 1);

        // Insert root
        nodes.insert(
            (0, 0),
            WaveletPacketNode::new(0, 0, signal.to_vec()),
        );

        // Decompose level by level
        for level in 0..max_level {
            let count = 1usize << level; // 2^level nodes at this level
            for pos in 0..count {
                let coeffs = nodes
                    .get(&(level, pos))
                    .map(|n| n.coeffs.clone())
                    .ok_or_else(|| {
                        SignalError::ComputationError(format!(
                            "WaveletPacketTree: missing node ({}, {})",
                            level, pos
                        ))
                    })?;

                if coeffs.len() < min_len {
                    // Too short to decompose — insert copies as leaves and stop
                    let left = WaveletPacketNode::new(level + 1, 2 * pos, coeffs.clone());
                    let right = WaveletPacketNode::new(level + 1, 2 * pos + 1, coeffs);
                    nodes.insert((level + 1, 2 * pos), left);
                    nodes.insert((level + 1, 2 * pos + 1), right);
                    continue;
                }

                let (approx, detail) =
                    dwt_decompose::<f64>(&coeffs, wavelet, Some(mode))?;

                nodes.insert(
                    (level + 1, 2 * pos),
                    WaveletPacketNode::new(level + 1, 2 * pos, approx),
                );
                nodes.insert(
                    (level + 1, 2 * pos + 1),
                    WaveletPacketNode::new(level + 1, 2 * pos + 1, detail),
                );
            }
        }

        Ok(WaveletPacketTree {
            nodes,
            wavelet,
            max_level,
            mode: mode.to_string(),
            original_length: signal.len(),
        })
    }

    /// Return a reference to the node at `(level, position)`.
    pub fn node(&self, level: usize, position: usize) -> Option<&WaveletPacketNode> {
        self.nodes.get(&(level, position))
    }

    /// Return all leaf nodes (nodes at `max_level`), sorted by position.
    pub fn leaves(&self) -> Vec<&WaveletPacketNode> {
        let mut leaves: Vec<&WaveletPacketNode> = self
            .nodes
            .values()
            .filter(|n| n.level == self.max_level)
            .collect();
        leaves.sort_by_key(|n| n.position);
        leaves
    }

    /// Select the best basis using the Coifman–Wickerhauser algorithm.
    ///
    /// Returns a vector of `(level, position)` pairs that form the optimal
    /// partition minimising the total cost under `measure`.
    ///
    /// # Algorithm
    /// 1. For each leaf, the cost is `wpt_cost_function(leaf.coeffs, measure)`.
    /// 2. For each internal node, we compare `node_cost` vs `left_cost + right_cost`;
    ///    we keep whichever is smaller.
    /// 3. A top-down pass collects the selected nodes.
    pub fn best_basis(&self, measure: CostMeasure) -> SignalResult<Vec<(usize, usize)>> {
        // Bottom-up cost computation stored in a HashMap
        let mut cost_map: HashMap<(usize, usize), f64> =
            HashMap::with_capacity(self.nodes.len());

        // Initialise all nodes with their own cost
        for (&key, node) in &self.nodes {
            cost_map.insert(key, wpt_cost_function(&node.coeffs, measure));
        }

        // Bottom-up: process from max_level - 1 down to 0
        for level in (0..self.max_level).rev() {
            let count = 1usize << level;
            for pos in 0..count {
                if !self.nodes.contains_key(&(level, pos)) {
                    continue;
                }
                let left_cost = cost_map
                    .get(&(level + 1, 2 * pos))
                    .copied()
                    .unwrap_or(f64::INFINITY);
                let right_cost = cost_map
                    .get(&(level + 1, 2 * pos + 1))
                    .copied()
                    .unwrap_or(f64::INFINITY);
                let children_cost = left_cost + right_cost;
                let node_cost = *cost_map
                    .get(&(level, pos))
                    .unwrap_or(&f64::INFINITY);

                // If children together cost less, use them
                if children_cost < node_cost {
                    cost_map.insert((level, pos), children_cost);
                }
            }
        }

        // Top-down collection: start at the root
        let mut result = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((0usize, 0usize));

        while let Some((level, pos)) = queue.pop_front() {
            // Is splitting beneficial?
            let node_cost = *cost_map.get(&(level, pos)).unwrap_or(&f64::INFINITY);
            let left_cost = cost_map
                .get(&(level + 1, 2 * pos))
                .copied()
                .unwrap_or(f64::INFINITY);
            let right_cost = cost_map
                .get(&(level + 1, 2 * pos + 1))
                .copied()
                .unwrap_or(f64::INFINITY);
            let children_cost = left_cost + right_cost;

            let left_exists = self.nodes.contains_key(&(level + 1, 2 * pos));
            let right_exists = self.nodes.contains_key(&(level + 1, 2 * pos + 1));

            if level < self.max_level && left_exists && right_exists
                && children_cost < node_cost
            {
                queue.push_back((level + 1, 2 * pos));
                queue.push_back((level + 1, 2 * pos + 1));
            } else {
                result.push((level, pos));
            }
        }

        result.sort();
        Ok(result)
    }

    /// Total energy across all leaf nodes.
    pub fn total_energy(&self) -> f64 {
        self.leaves().iter().map(|n| n.energy()).sum()
    }
}

// ---------------------------------------------------------------------------
// Standalone functional API
// ---------------------------------------------------------------------------

/// Decompose `signal` into a full wavelet packet tree.
///
/// This is a convenience wrapper around [`WaveletPacketTree::build`].
pub fn wpt_decompose(
    signal: &[f64],
    wavelet: Wavelet,
    max_level: usize,
    mode: &str,
) -> SignalResult<WaveletPacketTree> {
    WaveletPacketTree::build(signal, wavelet, max_level, mode)
}

/// Select the best basis from a pre-built tree.
///
/// Returns a list of `(level, position)` pairs.  See
/// [`WaveletPacketTree::best_basis`] for details.
pub fn best_basis_selection(
    tree: &WaveletPacketTree,
    measure: CostMeasure,
) -> SignalResult<Vec<(usize, usize)>> {
    tree.best_basis(measure)
}

/// Reconstruct a signal from a subset of wavelet packet nodes.
///
/// `selected` is a list of `(level, position)` addresses that form a valid
/// partition of the time-frequency plane (e.g. the output of
/// [`best_basis_selection`]).  The function collects all selected coefficient
/// arrays and synthesises the signal bottom-up.
///
/// # Arguments
/// - `tree` – The wavelet packet tree produced by [`wpt_decompose`].
/// - `selected` – The node addresses to reconstruct from.
/// - `target_length` – Desired output length (the original signal length).
///
/// # Errors
/// Returns [`SignalError::ValueError`] if a requested node is absent from the
/// tree.
pub fn wpt_reconstruct(
    tree: &WaveletPacketTree,
    selected: &[(usize, usize)],
    target_length: usize,
) -> SignalResult<Vec<f64>> {
    if selected.is_empty() {
        return Err(SignalError::ValueError(
            "wpt_reconstruct: selected node list is empty".to_string(),
        ));
    }

    // Collect selected nodes into a mutable map keyed by (level, position)
    let mut working: HashMap<(usize, usize), Vec<f64>> = HashMap::new();
    for &(level, pos) in selected {
        let node = tree.nodes.get(&(level, pos)).ok_or_else(|| {
            SignalError::ValueError(format!(
                "wpt_reconstruct: node ({}, {}) not found in tree",
                level, pos
            ))
        })?;
        working.insert((level, pos), node.coeffs.clone());
    }

    // Determine the maximum level present in the selected set
    let max_selected_level = selected.iter().map(|&(l, _)| l).max().unwrap_or(0);

    // Bottom-up reconstruction: from max_selected_level down to 0
    for level in (1..=max_selected_level).rev() {
        // Find pairs of siblings to merge
        let positions_at_level: Vec<usize> = working
            .keys()
            .filter(|&&(l, _)| l == level)
            .map(|&(_, p)| p)
            .collect();

        // Group siblings: (2k, 2k+1) → parent at level-1
        let mut parent_positions: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        for &pos in &positions_at_level {
            parent_positions.insert(pos / 2);
        }

        for parent_pos in parent_positions {
            let left_key = (level, 2 * parent_pos);
            let right_key = (level, 2 * parent_pos + 1);

            // Only merge when both siblings are present
            if let (Some(approx), Some(detail)) =
                (working.get(&left_key), working.get(&right_key))
            {
                let approx = approx.clone();
                let detail = detail.clone();
                let parent_level = level - 1;

                // Determine reconstruction length: use the length stored in
                // the parent node of the original tree if available; otherwise
                // estimate it as 2 * approx.len()
                let rec_len = tree
                    .nodes
                    .get(&(parent_level, parent_pos))
                    .map(|n| n.coeffs.len())
                    .unwrap_or_else(|| approx.len() * 2);

                let reconstructed = dwt_reconstruct(
                    &approx,
                    &detail,
                    tree.wavelet,
                    Some(rec_len),
                )?;

                working.remove(&left_key);
                working.remove(&right_key);
                working.insert((parent_level, parent_pos), reconstructed);
            }
        }
    }

    // The root should now be present
    let root = working
        .remove(&(0, 0))
        .ok_or_else(|| {
            SignalError::ComputationError(
                "wpt_reconstruct: failed to reconstruct root node — \
                 ensure selected nodes form a valid partition"
                    .to_string(),
            )
        })?;

    // Trim or pad to the target length
    let mut out = root;
    out.resize(target_length, 0.0);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dwt::Wavelet;

    fn sine_signal(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i as f64 * std::f64::consts::TAU / 32.0).sin()).collect()
    }

    #[test]
    fn test_cost_function_shannon() {
        // Concentrated energy → low entropy
        let sparse = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let uniform: Vec<f64> = vec![0.35355339; 8]; // 1/√8 each
        let c_sparse = wpt_cost_function(&sparse, CostMeasure::Shannon);
        let c_uniform = wpt_cost_function(&uniform, CostMeasure::Shannon);
        assert!(c_sparse < c_uniform, "sparse should have lower Shannon entropy");
    }

    #[test]
    fn test_cost_function_lp_norm() {
        let coeffs = vec![1.0, 0.5, 0.25, 0.0];
        let cost = wpt_cost_function(&coeffs, CostMeasure::LpNorm(1.0));
        let expected = 1.0 + 0.5 + 0.25;
        assert!((cost - expected).abs() < 1e-12);
    }

    #[test]
    fn test_build_tree() {
        let signal = sine_signal(128);
        let tree = WaveletPacketTree::build(&signal, Wavelet::DB(4), 3, "symmetric")
            .expect("tree build should succeed");

        // At level 3 there should be 2^3 = 8 nodes
        assert_eq!(tree.leaves().len(), 8);
        // Root should be present
        assert!(tree.node(0, 0).is_some());
    }

    #[test]
    fn test_best_basis_completeness() {
        let signal = sine_signal(64);
        let tree = WaveletPacketTree::build(&signal, Wavelet::Haar, 3, "periodic")
            .expect("tree build should succeed");
        let basis = tree.best_basis(CostMeasure::Shannon).expect("best basis failed");

        // Every sample must be covered exactly once.
        // We verify that the leaves form a partition by checking energy preservation
        // (under an orthogonal wavelet like Haar, energy is preserved exactly).
        let basis_energy: f64 = basis
            .iter()
            .map(|&(l, p)| {
                tree.nodes
                    .get(&(l, p))
                    .map(|n| n.energy())
                    .unwrap_or(0.0)
            })
            .sum();
        let signal_energy: f64 = signal.iter().map(|&x| x * x).sum();
        // Allow 5 % relative error due to boundary effects
        assert!(
            (basis_energy - signal_energy).abs() / signal_energy < 0.05,
            "best basis energy {} should be close to signal energy {}",
            basis_energy,
            signal_energy
        );
    }

    #[test]
    fn test_wpt_decompose_reconstruct_roundtrip() {
        let signal = sine_signal(64);
        let tree = wpt_decompose(&signal, Wavelet::Haar, 2, "periodic")
            .expect("decompose should succeed");

        // Use all leaves for reconstruction
        let leaves: Vec<(usize, usize)> =
            tree.leaves().iter().map(|n| (n.level, n.position)).collect();

        let reconstructed = wpt_reconstruct(&tree, &leaves, signal.len())
            .expect("reconstruct should succeed");

        let err: f64 = signal
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(f64::NEG_INFINITY, f64::max);

        assert!(err < 0.05, "max reconstruction error {} should be small", err);
    }

    #[test]
    fn test_node_path() {
        // Node at (3, 5) should have path [1, 0, 1] (binary 101 reversed = aad)
        let n = WaveletPacketNode::new(3, 5, vec![]);
        assert_eq!(n.path(), vec![1, 0, 1]);
        assert_eq!(n.path_string(), "dad");
    }

    #[test]
    fn test_tree_too_short_signal() {
        let short = vec![1.0, 2.0];
        let result = WaveletPacketTree::build(&short, Wavelet::DB(4), 3, "symmetric");
        assert!(result.is_err());
    }

    #[test]
    fn test_concentration_cost() {
        let all_in_one = vec![1.0, 0.0, 0.0, 0.0];
        let spread = vec![0.5, 0.5, 0.5, 0.5];
        let c1 = wpt_cost_function(&all_in_one, CostMeasure::Concentration);
        let c2 = wpt_cost_function(&spread, CostMeasure::Concentration);
        assert!(c1 <= c2, "concentrated signal should have lower concentration cost");
    }
}
