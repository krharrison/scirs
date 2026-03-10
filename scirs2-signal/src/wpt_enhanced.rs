//! Enhanced Wavelet Packet Transform Module
//!
//! This module provides comprehensive wavelet packet transform capabilities with:
//!
//! - Full wavelet packet decomposition and reconstruction
//! - Multiple cost functions for best basis selection
//! - Bottom-up and top-down basis optimization
//! - Energy-preserving transforms
//! - Comprehensive validation and quality metrics
//!
//! ## Features
//!
//! ### Cost Functions
//! - Shannon entropy
//! - Log-energy entropy
//! - Threshold cost (sparsity)
//! - Lp norm cost
//! - SURE (Stein's Unbiased Risk Estimate)
//!
//! ### Best Basis Selection
//! - Coifman-Wickerhauser algorithm
//! - Bottom-up cost minimization
//! - Adaptive level selection
//!
//! ## Example
//!
//! ```rust
//! use scirs2_signal::wpt_enhanced::{
//!     WaveletPacketTree, CostFunction
//! };
//! use scirs2_signal::dwt::Wavelet;
//!
//! // Create a test signal
//! let signal: Vec<f64> = (0..128).map(|i| (i as f64 * 0.1).sin()).collect();
//!
//! // Build wavelet packet tree
//! let tree = WaveletPacketTree::new(&signal, Wavelet::DB(4), 4, "symmetric");
//!
//! match tree {
//!     Ok(t) => {
//!         // Select best basis using Shannon entropy
//!         let best = t.select_best_basis(CostFunction::Shannon);
//!         println!("Selected {} basis nodes", best.len());
//!     },
//!     Err(e) => eprintln!("Failed: {}", e),
//! }
//! ```

use crate::dwt::{dwt_decompose, dwt_reconstruct, Wavelet};
use crate::error::{SignalError, SignalResult};
use std::collections::HashMap;

// =============================================================================
// Types and Enums
// =============================================================================

/// Cost function for best basis selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CostFunction {
    /// Shannon entropy: -sum(p * log(p))
    Shannon,
    /// Log-energy entropy: sum(log(x^2 + epsilon))
    LogEnergy,
    /// Threshold cost: count of |x| > threshold
    Threshold(f64),
    /// Lp norm: sum(|x|^p)
    Norm(f64),
    /// SURE cost for denoising
    Sure(f64), // sigma parameter
    /// Concentration measure (Gini-like)
    Concentration,
}

/// Node in the wavelet packet tree
#[derive(Debug, Clone)]
pub struct WptNode {
    /// Level in the tree (0 = root)
    pub level: usize,
    /// Position within the level (0 = leftmost)
    pub position: usize,
    /// Coefficient values
    pub coeffs: Vec<f64>,
    /// Pre-computed cost
    cost: Option<f64>,
    /// Flag indicating if children exist
    has_children: bool,
}

impl WptNode {
    /// Create a new WPT node
    pub fn new(level: usize, position: usize, coeffs: Vec<f64>) -> Self {
        WptNode {
            level,
            position,
            coeffs,
            cost: None,
            has_children: false,
        }
    }

    /// Get the binary path from root to this node
    pub fn path(&self) -> Vec<u8> {
        if self.level == 0 {
            return vec![];
        }

        let mut path = Vec::with_capacity(self.level);
        let mut pos = self.position;

        for _ in 0..self.level {
            path.push((pos & 1) as u8);
            pos >>= 1;
        }

        path.reverse();
        path
    }

    /// Get path as string (e.g., "aad" for approximation-approximation-detail)
    pub fn path_string(&self) -> String {
        self.path()
            .iter()
            .map(|&b| if b == 0 { 'a' } else { 'd' })
            .collect()
    }

    /// Compute energy of this node
    pub fn energy(&self) -> f64 {
        self.coeffs.iter().map(|&x| x * x).sum()
    }

    /// Left child position
    pub fn left_child_pos(&self) -> usize {
        self.position * 2
    }

    /// Right child position
    pub fn right_child_pos(&self) -> usize {
        self.position * 2 + 1
    }

    /// Parent position
    pub fn parent_pos(&self) -> Option<usize> {
        if self.level == 0 {
            None
        } else {
            Some(self.position / 2)
        }
    }
}

/// Complete wavelet packet tree
#[derive(Debug)]
pub struct WaveletPacketTree {
    /// All nodes indexed by (level, position)
    nodes: HashMap<(usize, usize), WptNode>,
    /// Wavelet used
    wavelet: Wavelet,
    /// Maximum decomposition level
    max_level: usize,
    /// Extension mode
    mode: String,
    /// Original signal length
    original_length: usize,
}

impl WaveletPacketTree {
    /// Create a new wavelet packet tree with full decomposition
    ///
    /// # Arguments
    /// * `signal` - Input signal
    /// * `wavelet` - Wavelet to use
    /// * `max_level` - Maximum decomposition level
    /// * `mode` - Signal extension mode
    ///
    /// # Returns
    /// * WaveletPacketTree with full decomposition
    pub fn new(
        signal: &[f64],
        wavelet: Wavelet,
        max_level: usize,
        mode: &str,
    ) -> SignalResult<Self> {
        if signal.is_empty() {
            return Err(SignalError::ValueError("Empty signal".to_string()));
        }

        let mut tree = WaveletPacketTree {
            nodes: HashMap::new(),
            wavelet,
            max_level,
            mode: mode.to_string(),
            original_length: signal.len(),
        };

        // Create root node
        let root = WptNode::new(0, 0, signal.to_vec());
        tree.nodes.insert((0, 0), root);

        // Decompose level by level
        for level in 0..max_level {
            let nodes_to_decompose: Vec<(usize, usize)> = tree
                .nodes
                .keys()
                .filter(|(l, _)| *l == level)
                .cloned()
                .collect();

            for (lvl, pos) in nodes_to_decompose {
                // Get coefficients from parent
                let parent_coeffs = match tree.nodes.get(&(lvl, pos)) {
                    Some(node) => node.coeffs.clone(),
                    None => continue,
                };

                // Skip if too short
                if parent_coeffs.len() < 4 {
                    continue;
                }

                // Decompose
                let (approx, detail) = dwt_decompose(&parent_coeffs, wavelet, Some(&tree.mode))?;

                // Create child nodes
                let left = WptNode::new(level + 1, pos * 2, approx);
                let right = WptNode::new(level + 1, pos * 2 + 1, detail);

                // Mark parent as having children
                if let Some(parent) = tree.nodes.get_mut(&(lvl, pos)) {
                    parent.has_children = true;
                }

                // Insert children
                tree.nodes.insert((level + 1, pos * 2), left);
                tree.nodes.insert((level + 1, pos * 2 + 1), right);
            }
        }

        Ok(tree)
    }

    /// Get a node at specified level and position
    pub fn get_node(&self, level: usize, position: usize) -> Option<&WptNode> {
        self.nodes.get(&(level, position))
    }

    /// Get all nodes at a specific level
    pub fn get_level(&self, level: usize) -> Vec<&WptNode> {
        let mut nodes: Vec<&WptNode> = self
            .nodes
            .iter()
            .filter(|((l, _), _)| *l == level)
            .map(|(_, node)| node)
            .collect();

        nodes.sort_by_key(|n| n.position);
        nodes
    }

    /// Get all leaf nodes (nodes without children)
    pub fn get_leaves(&self) -> Vec<&WptNode> {
        self.nodes
            .values()
            .filter(|node| !node.has_children)
            .collect()
    }

    /// Total number of nodes
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if tree is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Compute cost for a node using specified cost function
    pub fn compute_node_cost(&self, level: usize, position: usize, cost_fn: CostFunction) -> f64 {
        match self.nodes.get(&(level, position)) {
            Some(node) => compute_cost(&node.coeffs, cost_fn),
            None => f64::INFINITY,
        }
    }

    /// Select best basis using specified cost function
    ///
    /// Implements the Coifman-Wickerhauser algorithm for best basis selection
    /// using bottom-up cost comparison.
    ///
    /// # Arguments
    /// * `cost_fn` - Cost function to minimize
    ///
    /// # Returns
    /// * Vector of (level, position) pairs representing the best basis
    pub fn select_best_basis(&self, cost_fn: CostFunction) -> Vec<(usize, usize)> {
        // Compute costs for all nodes
        let mut costs: HashMap<(usize, usize), f64> = HashMap::new();
        let mut keep: HashMap<(usize, usize), bool> = HashMap::new();

        // Initialize costs for all nodes
        for ((level, position), node) in &self.nodes {
            costs.insert((*level, *position), compute_cost(&node.coeffs, cost_fn));
        }

        // Bottom-up optimization
        for level in (0..self.max_level).rev() {
            let nodes_at_level: Vec<(usize, usize)> = self
                .nodes
                .keys()
                .filter(|(l, _)| *l == level)
                .cloned()
                .collect();

            for (lvl, pos) in nodes_at_level {
                let parent_cost = costs.get(&(lvl, pos)).copied().unwrap_or(f64::INFINITY);

                let left_key = (level + 1, pos * 2);
                let right_key = (level + 1, pos * 2 + 1);

                let left_cost = costs.get(&left_key).copied().unwrap_or(f64::INFINITY);
                let right_cost = costs.get(&right_key).copied().unwrap_or(f64::INFINITY);

                let children_cost = left_cost + right_cost;

                // Decide: keep parent or children
                if parent_cost <= children_cost || !self.nodes.contains_key(&left_key) {
                    keep.insert((lvl, pos), false); // Don't split, use parent
                    costs.insert((lvl, pos), parent_cost);
                } else {
                    keep.insert((lvl, pos), true); // Split, use children
                    costs.insert((lvl, pos), children_cost);
                }
            }
        }

        // Collect best basis nodes
        let mut result = Vec::new();
        let mut queue = vec![(0usize, 0usize)];

        while let Some((level, position)) = queue.pop() {
            let should_split = keep.get(&(level, position)).copied().unwrap_or(false);

            if should_split && level < self.max_level {
                queue.push((level + 1, position * 2));
                queue.push((level + 1, position * 2 + 1));
            } else {
                result.push((level, position));
            }
        }

        // Sort by level then position for consistent ordering
        result.sort();
        result
    }

    /// Reconstruct signal from specified nodes
    ///
    /// # Arguments
    /// * `nodes` - List of (level, position) pairs to use for reconstruction
    ///
    /// # Returns
    /// * Reconstructed signal
    pub fn reconstruct(&self, nodes: &[(usize, usize)]) -> SignalResult<Vec<f64>> {
        if nodes.is_empty() {
            return Err(SignalError::ValueError("No nodes specified".to_string()));
        }

        // Special case: single root node
        if nodes.len() == 1 && nodes[0] == (0, 0) {
            if let Some(root) = self.nodes.get(&(0, 0)) {
                return Ok(root.coeffs.clone());
            }
        }

        // Build reconstruction tree
        let mut recon_coeffs: HashMap<(usize, usize), Vec<f64>> = HashMap::new();

        // Initialize with selected nodes
        for &(level, position) in nodes {
            if let Some(node) = self.nodes.get(&(level, position)) {
                recon_coeffs.insert((level, position), node.coeffs.clone());
            }
        }

        // Work backwards from max level
        for level in (0..self.max_level).rev() {
            let positions: Vec<usize> = recon_coeffs
                .keys()
                .filter(|(l, _)| *l == level + 1)
                .map(|(_, p)| *p)
                .collect();

            // Process pairs
            for pos in positions.iter() {
                let left_pos = pos - (pos % 2);
                let right_pos = left_pos + 1;
                let parent_pos = left_pos / 2;

                // Check if both children exist
                let left = recon_coeffs.get(&(level + 1, left_pos)).cloned();
                let right = recon_coeffs.get(&(level + 1, right_pos)).cloned();

                if let (Some(l), Some(r)) = (left, right) {
                    let parent = dwt_reconstruct(&l, &r, self.wavelet)?;
                    recon_coeffs.insert((level, parent_pos), parent);

                    // Remove children
                    recon_coeffs.remove(&(level + 1, left_pos));
                    recon_coeffs.remove(&(level + 1, right_pos));
                }
            }
        }

        // Get root and trim to original length
        let mut result = recon_coeffs
            .remove(&(0, 0))
            .ok_or_else(|| SignalError::ComputationError("Reconstruction failed".to_string()))?;
        result.truncate(self.original_length);
        Ok(result)
    }

    /// Validate the wavelet packet tree
    ///
    /// Checks energy conservation, reconstruction accuracy, and tree structure.
    pub fn validate(&self) -> WptValidationResult {
        let mut result = WptValidationResult {
            energy_ratios: Vec::new(),
            max_reconstruction_error: 0.0,
            is_complete: true,
            issues: Vec::new(),
        };

        // Check energy conservation at each level
        let root_energy: f64 = self.nodes.get(&(0, 0)).map(|n| n.energy()).unwrap_or(0.0);

        for level in 1..=self.max_level {
            let level_energy: f64 = self.get_level(level).iter().map(|n| n.energy()).sum();

            if root_energy > 1e-12 {
                let ratio = level_energy / root_energy;
                result.energy_ratios.push(ratio);

                if (ratio - 1.0).abs() > 0.05 {
                    result.issues.push(format!(
                        "Energy ratio at level {} is {:.4} (expected ~1.0)",
                        level, ratio
                    ));
                }
            }
        }

        // Check tree completeness
        for level in 0..self.max_level {
            let expected_nodes = 1 << level; // 2^level
            let actual_nodes = self.get_level(level).len();

            if actual_nodes < expected_nodes {
                result.is_complete = false;
                result.issues.push(format!(
                    "Level {} incomplete: {} of {} nodes",
                    level, actual_nodes, expected_nodes
                ));
            }
        }

        // Test reconstruction from leaves
        let leaves: Vec<(usize, usize)> = self
            .get_leaves()
            .iter()
            .map(|n| (n.level, n.position))
            .collect();

        if let Ok(reconstructed) = self.reconstruct(&leaves) {
            if let Some(root) = self.nodes.get(&(0, 0)) {
                let min_len = reconstructed.len().min(root.coeffs.len());
                let error: f64 = reconstructed[..min_len]
                    .iter()
                    .zip(root.coeffs[..min_len].iter())
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0.0_f64, |a, b| a.max(b));

                result.max_reconstruction_error = error;

                if error > 1e-10 {
                    result
                        .issues
                        .push(format!("Reconstruction error: {:.2e}", error));
                }
            }
        }

        result
    }
}

/// Result of WPT validation
#[derive(Debug, Clone)]
pub struct WptValidationResult {
    /// Energy ratio at each level (should be close to 1.0)
    pub energy_ratios: Vec<f64>,
    /// Maximum reconstruction error
    pub max_reconstruction_error: f64,
    /// Whether the tree is complete
    pub is_complete: bool,
    /// Issues found during validation
    pub issues: Vec<String>,
}

impl WptValidationResult {
    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        self.issues.is_empty()
    }
}

// =============================================================================
// Cost Functions
// =============================================================================

/// Compute cost of coefficients using specified cost function
pub fn compute_cost(coeffs: &[f64], cost_fn: CostFunction) -> f64 {
    if coeffs.is_empty() {
        return 0.0;
    }

    match cost_fn {
        CostFunction::Shannon => shannon_entropy(coeffs),
        CostFunction::LogEnergy => log_energy_entropy(coeffs),
        CostFunction::Threshold(t) => threshold_cost(coeffs, t),
        CostFunction::Norm(p) => norm_cost(coeffs, p),
        CostFunction::Sure(sigma) => sure_cost(coeffs, sigma),
        CostFunction::Concentration => concentration_cost(coeffs),
    }
}

/// Shannon entropy: -sum(p * log(p))
fn shannon_entropy(coeffs: &[f64]) -> f64 {
    let total_energy: f64 = coeffs.iter().map(|&x| x * x).sum();

    if total_energy < 1e-12 {
        return 0.0;
    }

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

/// Log-energy entropy: sum(log(x^2 + epsilon))
fn log_energy_entropy(coeffs: &[f64]) -> f64 {
    let epsilon = 1e-12;
    coeffs.iter().map(|&x| (x * x + epsilon).ln()).sum()
}

/// Threshold cost: count of |x| > threshold
fn threshold_cost(coeffs: &[f64], threshold: f64) -> f64 {
    coeffs.iter().filter(|&&x| x.abs() > threshold).count() as f64
}

/// Lp norm: sum(|x|^p)
fn norm_cost(coeffs: &[f64], p: f64) -> f64 {
    coeffs.iter().map(|&x| x.abs().powf(p)).sum()
}

/// SURE cost for soft thresholding
fn sure_cost(coeffs: &[f64], sigma: f64) -> f64 {
    if sigma < 1e-12 {
        return 0.0;
    }

    let n = coeffs.len() as f64;
    let sigma_sq = sigma * sigma;

    // Universal threshold
    let threshold = sigma * (2.0 * n.ln()).sqrt();

    // SURE formula
    let mut cost = n * sigma_sq;

    for &x in coeffs {
        let x_sq = x * x;
        let t_sq = threshold * threshold;

        if x_sq <= t_sq {
            cost += x_sq - sigma_sq;
        } else {
            cost += t_sq - sigma_sq;
        }
    }

    cost
}

/// Concentration measure (Gini-like)
fn concentration_cost(coeffs: &[f64]) -> f64 {
    if coeffs.is_empty() {
        return 0.0;
    }

    let mut sorted: Vec<f64> = coeffs.iter().map(|&x| x.abs()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len() as f64;
    let total: f64 = sorted.iter().sum();

    if total < 1e-12 {
        return 0.0;
    }

    // Gini coefficient
    let mut sum = 0.0;
    for (i, &val) in sorted.iter().enumerate() {
        sum += (2.0 * (i + 1) as f64 - n - 1.0) * val;
    }

    sum / (n * total)
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Perform best basis wavelet packet analysis
///
/// A convenience function that builds the tree and selects the best basis.
///
/// # Arguments
/// * `signal` - Input signal
/// * `wavelet` - Wavelet to use
/// * `max_level` - Maximum decomposition level
/// * `cost_fn` - Cost function for basis selection
///
/// # Returns
/// * (best_basis_nodes, tree)
pub fn best_basis_analysis(
    signal: &[f64],
    wavelet: Wavelet,
    max_level: usize,
    cost_fn: CostFunction,
) -> SignalResult<(Vec<(usize, usize)>, WaveletPacketTree)> {
    let tree = WaveletPacketTree::new(signal, wavelet, max_level, "symmetric")?;
    let best_basis = tree.select_best_basis(cost_fn);
    Ok((best_basis, tree))
}

/// Denoise using wavelet packets with best basis selection
///
/// # Arguments
/// * `signal` - Input noisy signal
/// * `wavelet` - Wavelet to use
/// * `max_level` - Maximum decomposition level
/// * `threshold_fn` - Function to compute threshold for each node
///
/// # Returns
/// * Denoised signal
pub fn wpt_denoise<F>(
    signal: &[f64],
    wavelet: Wavelet,
    max_level: usize,
    threshold_fn: F,
) -> SignalResult<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    // Build tree
    let tree = WaveletPacketTree::new(signal, wavelet, max_level, "symmetric")?;

    // Get all leaves
    let leaves: Vec<(usize, usize)> = tree
        .get_leaves()
        .iter()
        .map(|n| (n.level, n.position))
        .collect();

    // Apply thresholding and reconstruct
    // For simplicity, we'll threshold all detail nodes
    let mut modified_coeffs: HashMap<(usize, usize), Vec<f64>> = HashMap::new();

    for (level, position) in &leaves {
        if let Some(node) = tree.get_node(*level, *position) {
            let threshold = threshold_fn(&node.coeffs);
            let thresholded: Vec<f64> = node
                .coeffs
                .iter()
                .map(|&x| {
                    if x.abs() <= threshold {
                        0.0
                    } else {
                        x.signum() * (x.abs() - threshold)
                    }
                })
                .collect();

            modified_coeffs.insert((*level, *position), thresholded);
        }
    }

    // Reconstruct from thresholded leaves
    let mut recon = modified_coeffs;

    for level in (0..max_level).rev() {
        let positions: Vec<usize> = recon
            .keys()
            .filter(|(l, _)| *l == level + 1)
            .map(|(_, p)| *p)
            .collect();

        for pos in positions {
            let left_pos = pos - (pos % 2);
            let right_pos = left_pos + 1;
            let parent_pos = left_pos / 2;

            let left = recon.get(&(level + 1, left_pos)).cloned();
            let right = recon.get(&(level + 1, right_pos)).cloned();

            if let (Some(l), Some(r)) = (left, right) {
                let parent = dwt_reconstruct(&l, &r, wavelet)?;
                recon.insert((level, parent_pos), parent);
                recon.remove(&(level + 1, left_pos));
                recon.remove(&(level + 1, right_pos));
            }
        }
    }

    let mut result = recon
        .remove(&(0, 0))
        .ok_or_else(|| SignalError::ComputationError("Reconstruction failed".to_string()))?;
    result.truncate(signal.len());
    Ok(result)
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

    #[test]
    fn test_wpt_creation() {
        let signal = create_test_signal(128);
        let tree = WaveletPacketTree::new(&signal, Wavelet::Haar, 3, "symmetric");

        assert!(tree.is_ok());
        let tree = tree.expect("Tree creation should succeed");

        // Check nodes exist at each level
        assert!(tree.get_node(0, 0).is_some());
        assert!(tree.get_node(1, 0).is_some());
        assert!(tree.get_node(1, 1).is_some());
    }

    #[test]
    fn test_node_path() {
        let node = WptNode::new(3, 5, vec![1.0, 2.0]);
        // Position 5 in binary is 101
        // Path should be [1, 0, 1]
        assert_eq!(node.path(), vec![1, 0, 1]);
        assert_eq!(node.path_string(), "dad");
    }

    #[test]
    fn test_best_basis_selection() {
        let signal = create_test_signal(64);
        let tree = WaveletPacketTree::new(&signal, Wavelet::Haar, 3, "symmetric")
            .expect("Tree creation should succeed");

        let best_basis = tree.select_best_basis(CostFunction::Shannon);

        // Best basis should not be empty
        assert!(!best_basis.is_empty());

        // All nodes in best basis should exist
        for (level, position) in &best_basis {
            assert!(tree.get_node(*level, *position).is_some());
        }
    }

    #[test]
    fn test_reconstruction() {
        let signal = create_test_signal(64);
        // Use periodic mode for perfect reconstruction with power-of-2 signal
        let tree = WaveletPacketTree::new(&signal, Wavelet::Haar, 2, "periodic")
            .expect("Tree creation should succeed");

        // Reconstruct from all leaves
        let leaves: Vec<(usize, usize)> = tree
            .get_leaves()
            .iter()
            .map(|n| (n.level, n.position))
            .collect();

        let reconstructed = tree.reconstruct(&leaves);
        assert!(reconstructed.is_ok());

        let recon = reconstructed.expect("Reconstruction should succeed");

        // Check length matches
        assert_eq!(recon.len(), signal.len());

        // Check values are close (periodic mode gives near-perfect reconstruction)
        let max_error: f64 = signal
            .iter()
            .zip(recon.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, |a, b| a.max(b));

        assert!(max_error < 1e-10, "Max error: {}", max_error);
    }

    #[test]
    fn test_validation() {
        let signal = create_test_signal(64);
        // Use periodic mode for accurate energy preservation and reconstruction
        let tree = WaveletPacketTree::new(&signal, Wavelet::Haar, 3, "periodic")
            .expect("Tree creation should succeed");

        let validation = tree.validate();

        // Energy ratios should be close to 1.0
        for &ratio in &validation.energy_ratios {
            assert!((ratio - 1.0).abs() < 0.1, "Energy ratio off: {}", ratio);
        }

        // Reconstruction error should be small
        assert!(
            validation.max_reconstruction_error < 1e-9,
            "Reconstruction error: {}",
            validation.max_reconstruction_error
        );
    }

    #[test]
    fn test_cost_functions() {
        let coeffs = vec![1.0, 0.5, 0.25, 0.125, 0.0625];

        // Shannon entropy should be positive
        let shannon = compute_cost(&coeffs, CostFunction::Shannon);
        assert!(shannon > 0.0);

        // Threshold cost should count values above threshold
        let threshold_cost = compute_cost(&coeffs, CostFunction::Threshold(0.2));
        assert!((threshold_cost - 3.0).abs() < 1e-10); // 1.0, 0.5, 0.25 are above 0.2

        // Norm should be positive
        let norm = compute_cost(&coeffs, CostFunction::Norm(2.0));
        assert!(norm > 0.0);
    }

    #[test]
    fn test_wpt_denoise() {
        let clean = create_test_signal(64);

        // Add noise
        use scirs2_core::random::{Rng, SeedableRng, StdRng};
        let mut rng = StdRng::seed_from_u64(42);
        let noisy: Vec<f64> = clean
            .iter()
            .map(|&x| x + 0.1 * (rng.random::<f64>() * 2.0 - 1.0))
            .collect();

        // Denoise with universal threshold
        let threshold_fn = |coeffs: &[f64]| {
            let sigma: f64 =
                coeffs.iter().map(|&x| x.abs()).sum::<f64>() / coeffs.len() as f64 / 0.6745;

            sigma * (2.0 * (coeffs.len() as f64).ln()).sqrt()
        };

        let result = wpt_denoise(&noisy, Wavelet::DB(4), 3, threshold_fn);
        assert!(result.is_ok());

        let denoised = result.expect("Denoising should succeed");
        assert_eq!(denoised.len(), noisy.len());
    }

    #[test]
    fn test_best_basis_analysis() {
        let signal = create_test_signal(128);

        let result = best_basis_analysis(&signal, Wavelet::DB(4), 4, CostFunction::Shannon);
        assert!(result.is_ok());

        let (best_basis, tree) = result.expect("Analysis should succeed");
        assert!(!best_basis.is_empty());
        assert!(tree.len() > 1);
    }
}
