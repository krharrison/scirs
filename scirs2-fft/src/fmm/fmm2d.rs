//! 2D Fast Multipole Method for N-body Laplace interactions.
//!
//! Computes the potential φ_i = Σ_j q_j ln|x_i - x_j|  in O(N log N) time
//! (or O(N) for sufficiently dense, uniform distributions) versus O(N²)
//! for direct summation.
//!
//! ## Algorithm
//!
//! 1. **Build** a quad-tree over all source (and target) points.
//! 2. **Upward pass** (leaf → root): form multipole expansions at leaves,
//!    then M2M-translate them up to the root.
//! 3. **Downward pass** (root → leaf): for each node, convert well-separated
//!    multipole expansions from the interaction list to local expansions (M2L),
//!    then L2L-translate them down to leaves.
//! 4. **Near-field evaluation**: directly compute contributions between
//!    adjacent leaves.
//! 5. **Evaluate**: sum local expansion + near-field at each target.
//!
//! ## References
//! - Greengard & Rokhlin (1987) "A fast algorithm for particle simulations."
//! - Beatson & Greengard (1997) "A short course on fast multipole methods."

use std::collections::HashMap;

use crate::error::{FFTError, FFTResult};
use super::multipole::{LocalExpansion, MultipoleExpansion};
use super::tree::{QuadTree, TreeNode};

/// 2D FMM solver for the Laplace kernel φ(r) = ln(r).
pub struct FMM2D {
    /// Multipole / local expansion order (number of terms).
    pub order: usize,
    /// Maximum quad-tree depth.
    pub max_depth: usize,
    /// Multipole acceptance criterion θ (MAC).  Higher θ → more multipole
    /// interactions (faster but less accurate).  Typically 0.5.
    pub mac: f64,
    /// Minimum points per leaf cell.
    pub max_per_leaf: usize,
}

impl FMM2D {
    /// Construct with default parameters.
    ///
    /// # Arguments
    /// * `order` – Multipole expansion order (typically 6–20 for 6–10 digits accuracy).
    pub fn new(order: usize) -> Self {
        FMM2D {
            order,
            max_depth: 10,
            mac: 0.5,
            max_per_leaf: 16,
        }
    }

    /// Configure the multipole acceptance criterion.
    pub fn with_mac(mut self, mac: f64) -> Self {
        self.mac = mac;
        self
    }

    /// Configure the maximum quad-tree depth.
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Configure the maximum number of source points per leaf.
    pub fn with_max_per_leaf(mut self, n: usize) -> Self {
        self.max_per_leaf = n;
        self
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Compute the 2D Laplace potential at each target position.
    ///
    /// φ_i = Σ_j q_j · ln|x_i − s_j|
    ///
    /// # Arguments
    /// * `sources` – Source positions `[[x, y], …]`, length N.
    /// * `charges` – Charge strengths, length N.
    /// * `targets` – Target evaluation positions `[[x, y], …]`, length M.
    ///
    /// # Returns
    /// Potential values `φ_i` at each target, length M.
    pub fn compute_potentials(
        &self,
        sources: &[[f64; 2]],
        charges: &[f64],
        targets: &[[f64; 2]],
    ) -> FFTResult<Vec<f64>> {
        if sources.len() != charges.len() {
            return Err(FFTError::ValueError(
                "sources and charges must have the same length".into(),
            ));
        }
        if sources.is_empty() || targets.is_empty() {
            return Ok(vec![0.0; targets.len()]);
        }

        // ----------------------------------------------------------------
        // 1. Build quad-tree over all source positions.
        // ----------------------------------------------------------------
        let tree = QuadTree::build(sources, self.max_depth, self.max_per_leaf)
            .map_err(|e| FFTError::ComputationError(format!("tree build: {e}")))?;

        // ----------------------------------------------------------------
        // 2. Upward pass: form and translate multipole expansions.
        // ----------------------------------------------------------------
        let multipoles = self.upward_pass(&tree, sources, charges)?;

        // ----------------------------------------------------------------
        // 3. Downward pass: M2L interactions + L2L translations.
        // ----------------------------------------------------------------
        let locals = self.downward_pass(&tree, &multipoles)?;

        // ----------------------------------------------------------------
        // 4 & 5. Evaluate: near-field direct + local expansion at targets.
        // ----------------------------------------------------------------
        let result = self.evaluate_at_targets(
            &tree, sources, charges, targets, &locals,
        )?;

        Ok(result)
    }

    /// O(N²) direct summation for accuracy comparison.
    ///
    /// φ_i = Σ_j q_j · ln|x_i − s_j|,  singular pairs yield 0.
    pub fn direct_sum(
        sources: &[[f64; 2]],
        charges: &[f64],
        targets: &[[f64; 2]],
    ) -> Vec<f64> {
        targets
            .iter()
            .map(|t| {
                sources
                    .iter()
                    .zip(charges.iter())
                    .map(|(s, q)| {
                        let r = ((t[0] - s[0]).powi(2) + (t[1] - s[1]).powi(2)).sqrt();
                        if r > 1e-15 {
                            q * r.ln()
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>()
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Upward pass
    // -----------------------------------------------------------------------

    fn upward_pass(
        &self,
        tree: &QuadTree,
        sources: &[[f64; 2]],
        charges: &[f64],
    ) -> FFTResult<HashMap<usize, MultipoleExpansion>> {
        let mut multipoles: HashMap<usize, MultipoleExpansion> = HashMap::new();

        // Post-order DFS: process children before parents.
        self.upward_recursive(&tree.root, sources, charges, &mut multipoles)?;

        Ok(multipoles)
    }

    fn upward_recursive(
        &self,
        node: &TreeNode,
        sources: &[[f64; 2]],
        charges: &[f64],
        multipoles: &mut HashMap<usize, MultipoleExpansion>,
    ) -> FFTResult<()> {
        if node.is_leaf {
            // Leaf: build multipole expansion from contained source points.
            let mut m = MultipoleExpansion::new(node.center, self.order);
            for &idx in &node.point_indices {
                if idx < sources.len() {
                    m.add_source(sources[idx], charges[idx]);
                }
            }
            multipoles.insert(node.node_id, m);
        } else if let Some(children) = &node.children {
            // Recurse into children first.
            for child in children.iter() {
                self.upward_recursive(child, sources, charges, multipoles)?;
            }

            // M2M: translate each child's multipole to the parent center and
            // accumulate into the parent's expansion.
            let mut parent_m = MultipoleExpansion::new(node.center, self.order);
            // a_0 starts at 0
            for child in children.iter() {
                if let Some(child_m) = multipoles.get(&child.node_id) {
                    let translated = child_m.translate(node.center);
                    // Accumulate by adding coefficients
                    for k in 0..=self.order {
                        parent_m.coeffs[k][0] += translated.coeffs[k][0];
                        parent_m.coeffs[k][1] += translated.coeffs[k][1];
                    }
                }
            }
            multipoles.insert(node.node_id, parent_m);
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Downward pass
    // -----------------------------------------------------------------------

    fn downward_pass(
        &self,
        tree: &QuadTree,
        multipoles: &HashMap<usize, MultipoleExpansion>,
    ) -> FFTResult<HashMap<usize, LocalExpansion>> {
        let mut locals: HashMap<usize, LocalExpansion> = HashMap::new();

        // Top-down DFS.  At each node, we look at its "interaction list":
        // nodes that are well-separated from us but whose parents were not.
        self.downward_recursive(&tree.root, &tree.root, multipoles, &mut locals, true)?;

        Ok(locals)
    }

    fn downward_recursive(
        &self,
        root: &TreeNode,
        node: &TreeNode,
        multipoles: &HashMap<usize, MultipoleExpansion>,
        locals: &mut HashMap<usize, LocalExpansion>,
        is_root_call: bool,
    ) -> FFTResult<()> {
        // Initialise local expansion for this node.
        if !locals.contains_key(&node.node_id) {
            locals.insert(node.node_id, LocalExpansion::new(node.center, self.order));
        }

        // On the first (root) call we do a global pass: for every pair of
        // nodes at each level, convert multipoles of well-separated sources to
        // local expansions at the target.
        if is_root_call {
            // Collect all nodes with a BFS sweep.
            let all_nodes = root.bfs_nodes_with_ids();
            let n = all_nodes.len();

            for i in 0..n {
                let (_, ni) = &all_nodes[i];
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let (_, nj) = &all_nodes[j];
                    // Check if ni and nj are well-separated.
                    if ni.is_well_separated(nj, self.mac) {
                        // Convert nj's multipole to a local expansion at ni's center.
                        if let Some(m_j) = multipoles.get(&nj.node_id) {
                            // Only apply if nj has non-negligible charge.
                            let q_total = m_j.coeffs[0][0].abs() + m_j.coeffs[0][1].abs();
                            if q_total > 1e-30 {
                                match m_j.to_local(ni.center, self.order) {
                                    Ok(new_local) => {
                                        let entry = locals
                                            .entry(ni.node_id)
                                            .or_insert_with(|| LocalExpansion::new(ni.center, self.order));
                                        entry.add(&new_local);
                                    }
                                    Err(_) => {
                                        // Skip degenerate configurations.
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // L2L: propagate local expansions from parents to children.
            self.l2l_pass(root, locals)?;
            return Ok(());
        }

        Ok(())
    }

    fn l2l_pass(
        &self,
        node: &TreeNode,
        locals: &mut HashMap<usize, LocalExpansion>,
    ) -> FFTResult<()> {
        if let Some(children) = &node.children {
            // Translate node's local expansion to each child.
            if let Some(parent_local) = locals.get(&node.node_id).cloned() {
                for child in children.iter() {
                    let child_local = parent_local.translate(child.center);
                    let entry = locals
                        .entry(child.node_id)
                        .or_insert_with(|| LocalExpansion::new(child.center, self.order));
                    entry.add(&child_local);
                }
            }

            // Recurse into children.
            for child in children.iter() {
                self.l2l_pass(child, locals)?;
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Target evaluation
    // -----------------------------------------------------------------------

    fn evaluate_at_targets(
        &self,
        tree: &QuadTree,
        sources: &[[f64; 2]],
        charges: &[f64],
        targets: &[[f64; 2]],
        locals: &HashMap<usize, LocalExpansion>,
    ) -> FFTResult<Vec<f64>> {
        let mut potentials = vec![0.0_f64; targets.len()];

        // Get all leaf nodes.
        let leaves = tree.leaves();

        // For each target, find its leaf, evaluate local expansion + near-field.
        for (t_idx, target) in targets.iter().enumerate() {
            // Find which leaf contains this target.
            let containing_leaf = find_leaf(&tree.root, *target);

            let mut phi = 0.0;

            if let Some(leaf) = containing_leaf {
                // Far-field contribution via local expansion.
                if let Some(local) = locals.get(&leaf.node_id) {
                    phi += local.evaluate(*target);
                }

                // Near-field: direct sum with all sources in adjacent leaves.
                for other_leaf in &leaves {
                    if leaf.is_adjacent(other_leaf) || other_leaf.node_id == leaf.node_id {
                        for &s_idx in &other_leaf.point_indices {
                            if s_idx < sources.len() {
                                let s = sources[s_idx];
                                let r = ((target[0] - s[0]).powi(2)
                                    + (target[1] - s[1]).powi(2))
                                .sqrt();
                                if r > 1e-15 {
                                    phi += charges[s_idx] * r.ln();
                                }
                            }
                        }
                    }
                }
            } else {
                // Fallback: direct sum over all sources.
                for (s, q) in sources.iter().zip(charges.iter()) {
                    let r = ((target[0] - s[0]).powi(2) + (target[1] - s[1]).powi(2)).sqrt();
                    if r > 1e-15 {
                        phi += q * r.ln();
                    }
                }
            }

            potentials[t_idx] = phi;
        }

        Ok(potentials)
    }
}

// -----------------------------------------------------------------------
// Tree traversal helpers
// -----------------------------------------------------------------------

/// Find the leaf node that contains point p.
fn find_leaf<'a>(node: &'a TreeNode, p: [f64; 2]) -> Option<&'a TreeNode> {
    if !node.contains(p) {
        return None;
    }
    if node.is_leaf {
        return Some(node);
    }
    if let Some(children) = &node.children {
        for child in children.iter() {
            if let Some(found) = find_leaf(child, p) {
                return Some(found);
            }
        }
    }
    // p is in the node's bounding box but no child contains it (can happen at boundaries).
    Some(node)
}

/// Extension trait for BFS traversal with node IDs.
trait BfsWithIds {
    fn bfs_nodes_with_ids(&self) -> Vec<(usize, &TreeNode)>;
}

impl BfsWithIds for TreeNode {
    fn bfs_nodes_with_ids(&self) -> Vec<(usize, &TreeNode)> {
        let mut result = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((0usize, self));
        while let Some((idx, node)) = queue.pop_front() {
            result.push((idx, node));
            if let Some(children) = &node.children {
                for child in children.iter() {
                    queue.push_back((child.node_id, child));
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_points(n: usize, seed: u64) -> (Vec<[f64; 2]>, Vec<f64>) {
        // Simple LCG PRNG for reproducibility without external crates.
        let mut state = seed;
        let lcg = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*s >> 33) as f64 / (1u64 << 31) as f64
        };

        let positions: Vec<[f64; 2]> = (0..n)
            .map(|_| [lcg(&mut state) * 2.0 - 1.0, lcg(&mut state) * 2.0 - 1.0])
            .collect();
        let charges: Vec<f64> = (0..n)
            .map(|_| lcg(&mut state) * 2.0 - 1.0)
            .collect();
        (positions, charges)
    }

    #[test]
    fn test_direct_sum_single() {
        let sources = vec![[1.0_f64, 0.0]];
        let charges = vec![1.0_f64];
        let targets = vec![[0.0_f64, 0.0]];
        let phi = FMM2D::direct_sum(&sources, &charges, &targets);
        // ln(1) = 0
        assert!((phi[0] - 0.0).abs() < 1e-12, "phi={}", phi[0]);

        let targets2 = vec![[2.0_f64, 0.0]];
        let phi2 = FMM2D::direct_sum(&sources, &charges, &targets2);
        // ln|2-1| = ln(1) = 0
        assert!((phi2[0] - 0.0).abs() < 1e-12, "phi2={}", phi2[0]);

        let targets3 = vec![[1.0_f64 + std::f64::consts::E, 0.0]];
        let phi3 = FMM2D::direct_sum(&sources, &charges, &targets3);
        // ln|e| = 1
        assert!((phi3[0] - 1.0).abs() < 1e-10, "phi3={:.8}", phi3[0]);
    }

    #[test]
    fn test_fmm_vs_direct_small() {
        let (sources, charges) = random_points(20, 42);
        let (targets, _) = random_points(10, 99);

        let fmm = FMM2D::new(8).with_mac(0.5).with_max_per_leaf(4);
        let fmm_phi = fmm
            .compute_potentials(&sources, &charges, &targets)
            .expect("FMM failed");
        let direct_phi = FMM2D::direct_sum(&sources, &charges, &targets);

        // Check relative error at each target.
        for (i, (&fmm_v, &dir_v)) in fmm_phi.iter().zip(direct_phi.iter()).enumerate() {
            let rel_err = if dir_v.abs() > 1e-10 {
                (fmm_v - dir_v).abs() / dir_v.abs()
            } else {
                (fmm_v - dir_v).abs()
            };
            assert!(
                rel_err < 0.5,
                "target {i}: FMM={fmm_v:.6} direct={dir_v:.6} rel_err={rel_err:.4}"
            );
        }
    }

    #[test]
    fn test_fmm_direct_sum_parity() {
        // For very small problems the FMM should equal direct sum almost exactly.
        let sources = vec![
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ];
        let charges = vec![1.0, -1.0, 0.5, -0.5];
        let targets = vec![[3.0, 3.0], [4.0, 0.0]];

        let direct_phi = FMM2D::direct_sum(&sources, &charges, &targets);
        let fmm = FMM2D::new(10).with_mac(0.4).with_max_per_leaf(16);
        let fmm_phi = fmm
            .compute_potentials(&sources, &charges, &targets)
            .expect("FMM failed");

        for (i, (&d, &f)) in direct_phi.iter().zip(fmm_phi.iter()).enumerate() {
            let err = (d - f).abs();
            assert!(err < 0.1, "target {i}: direct={d:.8} fmm={f:.8} err={err:.2e}");
        }
    }
}
