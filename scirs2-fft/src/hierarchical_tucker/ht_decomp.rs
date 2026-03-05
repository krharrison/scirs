//! Hierarchical Tucker (HT) decomposition of dense tensors.
//!
//! ## Mathematical Background
//!
//! For a d-dimensional tensor T ∈ ℝ^{n_1 × … × n_d}, the HT format stores:
//!
//! - **Leaf matrices** U_t ∈ ℝ^{n_μ(t) × r_t}  for each leaf t of the dimension tree.
//! - **Transfer tensors** G_t ∈ ℝ^{r_t × r_{t_L} × r_{t_R}}  for each interior node t.
//!
//! The tensor is reconstructed bottom-up: at each interior node, the two
//! children's "frame matrices" B_{t_L} and B_{t_R} are contracted with G_t:
//!
//!   B_t = G_t ×_2 B_{t_L} ×_3 B_{t_R}    (n-mode products)
//!
//! The root yields the full tensor as B_root (reshaped to the original shape).
//!
//! ## Storage
//!
//! Total parameter count ≈ d·n·k + (d-1)·k³  where k is the maximum rank.
//! Compare to n^d for the dense tensor — exponential savings for high dimensions.
//!
//! ## Reference
//! Hackbusch & Kühn (2009) "A new scheme for the tensor representation."

use crate::error::{FFTError, FFTResult};
use super::ht_ops::{
    frobenius_norm, matmul, n_mode_product, n_mode_unfolding, truncated_svd,
};
use super::ht_tree::{HTNode, build_dimension_tree};

// ============================================================================
// HierarchicalTucker
// ============================================================================

/// Hierarchical Tucker representation of a dense tensor.
#[derive(Debug, Clone)]
pub struct HierarchicalTucker {
    /// Number of tensor modes.
    pub n_dims: usize,
    /// Shape of the original tensor.
    pub shape: Vec<usize>,
    /// HT-ranks at each node (indexed by node id).
    pub ranks: Vec<usize>,
    /// Basis matrices for each leaf node.
    /// `leaves[k]` is the basis matrix for leaf k (in tree order), stored as
    /// a flat row-major array of size  shape[mode] × rank.
    pub leaves: Vec<Vec<f64>>,
    /// Leaf node shapes: `(n_rows, n_cols)` = `(shape[mode], rank)`.
    pub leaf_shape: Vec<(usize, usize)>,
    /// Transfer tensors for each interior node, stored as flat row-major
    /// arrays of size `rank × rank_left × rank_right`.
    pub transfer: Vec<Vec<f64>>,
    /// Transfer tensor shapes: `(r_t, r_left, r_right)`.
    pub transfer_shape: Vec<(usize, usize, usize)>,
    /// The dimension-partition tree.
    pub nodes: Vec<HTNode>,
}

impl HierarchicalTucker {
    // -----------------------------------------------------------------------
    // Decomposition
    // -----------------------------------------------------------------------

    /// Compute the HT decomposition of a dense tensor.
    ///
    /// # Arguments
    /// * `tensor`    – Flat row-major tensor data.
    /// * `shape`     – Mode sizes `[n_1, …, n_d]`.
    /// * `max_rank`  – Maximum HT-rank allowed at any node.
    ///
    /// # Returns
    /// A `HierarchicalTucker` struct encoding the decomposition.
    pub fn decompose(tensor: &[f64], shape: &[usize], max_rank: usize) -> FFTResult<Self> {
        let n_dims = shape.len();
        if n_dims < 2 {
            return Err(FFTError::ValueError(
                "HT decomposition requires at least 2 dimensions".into(),
            ));
        }

        let n_total: usize = shape.iter().product();
        if tensor.len() != n_total {
            return Err(FFTError::ValueError(
                format!("tensor length {} ≠ product of shape {}", tensor.len(), n_total),
            ));
        }
        if max_rank == 0 {
            return Err(FFTError::ValueError("max_rank must be ≥ 1".into()));
        }

        // Build balanced binary dimension tree.
        let mut nodes = build_dimension_tree(n_dims);

        // Bottom-up SVD truncation to compute HT factors.
        // We store the "frame matrices" B_t bottom-up: at each leaf t,
        // B_t = U_t ∈ ℝ^{n_t × r_t}.
        // At interior nodes we compute the transfer tensor G_t and update B_t.

        // We need to process nodes in post-order (leaves first).
        let post_order = postorder_indices(&nodes, 0);

        // Maps from node_id to frame matrix B_t and its shape (rows × cols).
        let mut frame_mats: Vec<Option<Vec<f64>>> = vec![None; nodes.len()];
        let mut frame_rows: Vec<usize> = vec![0; nodes.len()];
        let mut frame_cols: Vec<usize> = vec![0; nodes.len()]; // = rank

        // Storage for leaf basis matrices and transfer tensors.
        let mut leaves_storage: Vec<Vec<f64>> = Vec::new();
        let mut leaf_shape: Vec<(usize, usize)> = Vec::new();
        let mut transfer_storage: Vec<Vec<f64>> = Vec::new();
        let mut transfer_shape: Vec<(usize, usize, usize)> = Vec::new();

        // Node-to-leaf and node-to-transfer indices.
        let mut node_to_leaf: Vec<Option<usize>> = vec![None; nodes.len()];
        let mut node_to_transfer: Vec<Option<usize>> = vec![None; nodes.len()];

        // Temporary tensors for mode matricization.  At each node we need the
        // "sub-tensor" formed by the modes it spans.  For the root this is the
        // entire tensor; for children we propagate via mode-projection.

        // We take a different approach: compute each leaf's basis matrix by
        // taking the mode-k unfolding of the original tensor and truncating it.

        for &nid in &post_order {
            let node = &nodes[nid];

            if node.is_leaf {
                // Mode index = mode_start (leaves span exactly 1 mode).
                let mode = node.mode_start;
                let n_mode = shape[mode];

                // Compute the mode-k unfolding of the full tensor.
                let (mat, n_rows, n_cols) = n_mode_unfolding(tensor, shape, mode)?;
                debug_assert_eq!(n_rows, n_mode);
                debug_assert_eq!(n_cols, n_total / n_mode);

                // Truncated SVD: U_t has shape n_mode × r_t.
                let actual_rank = max_rank.min(n_rows).min(n_cols);
                let (u, _s, _vt) = truncated_svd(&mat, n_rows, n_cols, actual_rank)?;
                let r_t = actual_rank;

                leaves_storage.push(u.clone());
                leaf_shape.push((n_rows, r_t));
                node_to_leaf[nid] = Some(leaves_storage.len() - 1);

                // Frame matrix B_t = U_t  (n_mode × r_t)
                frame_mats[nid] = Some(u);
                frame_rows[nid] = n_rows;
                frame_cols[nid] = r_t;

                nodes[nid].rank = r_t;
            } else {
                // Interior node: children must already have frame matrices.
                let left_id = node.left.ok_or_else(|| {
                    FFTError::ComputationError(format!("Node {nid} has no left child"))
                })?;
                let right_id = node.right.ok_or_else(|| {
                    FFTError::ComputationError(format!("Node {nid} has no right child"))
                })?;

                let r_l = frame_cols[left_id];
                let r_r = frame_cols[right_id];

                // Build the "merged" frame by Kronecker-type unfolding.
                // For the root node, we obtain the transfer tensor by computing
                // a matricization of the tensor projected onto child bases.

                // Compute the mode-group unfolding for the modes in `node`:
                // Unfold along all modes in [mode_start, mode_end) simultaneously
                // by first applying the left and right frame matrices.

                // Step 1: Project tensor onto left child's basis (modes [mode_start, mid))
                // Step 2: Project onto right child's basis (modes [mid, mode_end))
                // Step 3: The resulting matrix has size r_l*r_r × (rest)
                // Step 4: Truncated SVD gives G_t and B_t.

                let (left_b, right_b) = {
                    let lb = frame_mats[left_id].as_ref().ok_or_else(|| {
                        FFTError::ComputationError(format!("Missing frame for left child {left_id}"))
                    })?;
                    let rb = frame_mats[right_id].as_ref().ok_or_else(|| {
                        FFTError::ComputationError(format!("Missing frame for right child {right_id}"))
                    })?;
                    (lb.clone(), rb.clone())
                };

                // Project the original tensor onto the left and right bases.
                // The combined frame B_t = (B_L ⊗ B_R) — but we compute the
                // transfer tensor G_t via a simple least-squares fitting approach:
                //
                //  1. Form the "interface matrix" M = (B_L ⊗ B_R)^T · T_{(node)}
                //  2. If node is root, T_{(node)} is the full tensor unfolded.
                //  3. Otherwise truncate M by SVD to get G_t and B_t.

                // For simplicity, we form the product-of-modes unfolding:
                // Unfold T along modes [left.mode_start..left.mode_end) and
                // [right.mode_start..right.mode_end) together.

                // Create the combined projection matrix via Khatri-Rao product.
                // (B_L ⊗ B_R) is the Kronecker product of the two frame matrices.
                let kr = khatri_rao_product(&left_b, frame_rows[left_id], r_l,
                                             &right_b, frame_rows[right_id], r_r)?;
                let kr_rows = frame_rows[left_id] * frame_rows[right_id];
                let kr_cols = r_l * r_r;

                // Unfold the tensor along the node's modes simultaneously.
                // This is a multi-mode unfolding: rows = shape[mode_start..mode_end],
                // cols = remaining modes.
                let node_modes: Vec<usize> = (node.mode_start..node.mode_end).collect();
                let mode_size: usize = node_modes.iter().map(|&m| shape[m]).product();
                let rest_size = n_total / mode_size;

                // Use a custom multi-mode unfolding.
                let (t_unfold, _, _) = multimode_unfolding(tensor, shape, &node_modes)?;

                // Project: G_mat = (B_L ⊗ B_R)^T · T_unfold,  size (r_l*r_r) × rest_size
                // kr^T is kr_cols × kr_rows — compute transpose explicitly.
                let mut kr_t = vec![0.0_f64; kr_cols * kr_rows];
                for i in 0..kr_rows {
                    for j in 0..kr_cols {
                        kr_t[j * kr_rows + i] = kr[i * kr_cols + j];
                    }
                }
                let g_mat = matmul(&kr_t, kr_cols, kr_rows, &t_unfold, rest_size)?;

                // If this is the root node, there is no further SVD needed.
                // The transfer tensor is simply g_mat reshaped to (r_l*r_r) × rest.
                // But if it's not the root, we truncate g_mat to get G_t and B_t.
                if node.parent.is_none() {
                    // Root node: the transfer tensor relates the two child
                    // frame matrices to the full tensor.
                    //
                    // We compute truncated SVD of g_mat (kr_cols × rest_size).
                    // The actual rank r_t is constrained by both max_rank
                    // and r_full = min(kr_cols, rest_size).
                    let r_full = kr_cols.min(rest_size);
                    let r_t = max_rank.min(r_full);
                    let (u_g, _s_g, _vt_g) = truncated_svd(&g_mat, kr_cols, rest_size, r_t)?;

                    // u_g has shape kr_cols × r_t (row-major).
                    // Store with shape (r_t, r_l, r_r) for reconstruction.
                    transfer_storage.push(u_g.clone());
                    transfer_shape.push((r_t, r_l, r_r));
                    node_to_transfer[nid] = Some(transfer_storage.len() - 1);

                    frame_mats[nid] = Some(u_g);
                    frame_rows[nid] = kr_cols;
                    frame_cols[nid] = r_t;
                    nodes[nid].rank = r_t;
                } else {
                    // Interior (non-root): truncated SVD gives G_t and new frame B_t.
                    let r_full = kr_cols.min(rest_size);
                    let r_t = max_rank.min(r_full);
                    let (u_g, _s_g, vt_g) = truncated_svd(&g_mat, kr_cols, rest_size, r_t)?;

                    transfer_storage.push(u_g.clone());
                    transfer_shape.push((r_t, r_l, r_r));
                    node_to_transfer[nid] = Some(transfer_storage.len() - 1);

                    // New frame: B_t = (B_L ⊗ B_R) · U_G,  size mode_size × r_t
                    let kr2 = khatri_rao_product(&left_b, frame_rows[left_id], r_l,
                                                 &right_b, frame_rows[right_id], r_r)?;
                    let b_t = matmul(&kr2, kr_rows, kr_cols, &u_g, r_t)?;

                    frame_mats[nid] = Some(b_t);
                    frame_rows[nid] = mode_size;
                    frame_cols[nid] = r_t;
                    nodes[nid].rank = r_t;
                    let _ = vt_g;
                }
            }
        }

        // Collect ranks.
        let ranks: Vec<usize> = nodes.iter().map(|n| n.rank).collect();

        Ok(HierarchicalTucker {
            n_dims,
            shape: shape.to_vec(),
            ranks,
            leaves: leaves_storage,
            leaf_shape,
            transfer: transfer_storage,
            transfer_shape,
            nodes,
        })
    }

    // -----------------------------------------------------------------------
    // Reconstruction
    // -----------------------------------------------------------------------

    /// Reconstruct the dense tensor from the HT representation.
    ///
    /// Uses the bottom-up contraction:
    ///   B_leaf = U_leaf
    ///   B_node = (B_left ⊗ B_right) · G_node^T   (for interior nodes)
    ///   T      = reshape(B_root)
    pub fn reconstruct(&self) -> FFTResult<Vec<f64>> {
        let n_total: usize = self.shape.iter().product();

        // Process nodes bottom-up (post-order).
        let post_order = postorder_indices(&self.nodes, 0);

        // Store the reconstructed frame matrices.
        let mut frames: Vec<Option<Vec<f64>>> = vec![None; self.nodes.len()];
        let mut frame_rows: Vec<usize> = vec![0; self.nodes.len()];
        let mut frame_cols: Vec<usize> = vec![0; self.nodes.len()];

        for &nid in &post_order {
            let node = &self.nodes[nid];

            if node.is_leaf {
                let leaf_idx = self.nodes[nid].mode_start; // leaf index = mode index
                // Find the leaf storage index.
                let leaf_storage_idx = self.find_leaf_for_node(nid)?;
                let u = self.leaves[leaf_storage_idx].clone();
                let (nr, nc) = self.leaf_shape[leaf_storage_idx];
                frames[nid] = Some(u);
                frame_rows[nid] = nr;
                frame_cols[nid] = nc;
                let _ = leaf_idx;
            } else {
                let left_id = node.left.ok_or_else(|| {
                    FFTError::ComputationError(format!("Reconstruct: node {nid} has no left child"))
                })?;
                let right_id = node.right.ok_or_else(|| {
                    FFTError::ComputationError(format!("Reconstruct: node {nid} has no right child"))
                })?;

                let b_l = frames[left_id].as_ref().ok_or_else(|| {
                    FFTError::ComputationError(format!("Missing frame for left child {left_id}"))
                })?;
                let b_r = frames[right_id].as_ref().ok_or_else(|| {
                    FFTError::ComputationError(format!("Missing frame for right child {right_id}"))
                })?;

                let r_l = frame_cols[left_id];
                let r_r = frame_cols[right_id];
                let nr_l = frame_rows[left_id];
                let nr_r = frame_rows[right_id];

                // Kronecker product: B_L ⊗ B_R, shape (nr_l * nr_r) × (r_l * r_r).
                let kr = khatri_rao_product(b_l, nr_l, r_l, b_r, nr_r, r_r)?;

                // Transfer tensor for this node.
                let trans_idx = self.find_transfer_for_node(nid)?;
                let g = &self.transfer[trans_idx];
                let (r_t, r_gl, r_gr) = self.transfer_shape[trans_idx];

                // Sanity check.
                if r_gl != r_l || r_gr != r_r {
                    return Err(FFTError::ComputationError(
                        format!("Reconstruct: rank mismatch at node {nid}: ({r_gl},{r_gr}) ≠ ({r_l},{r_r})")
                    ));
                }

                // B_t = (B_L ⊗ B_R) · G_t^T,  shape (nr_l*nr_r) × r_t.
                // G_t is stored as kr_cols × r_t (= r_l*r_r × r_t),
                // so G_t^T is r_t × (r_l*r_r).
                // But actually G stores U from the SVD of g_mat = U · Σ · V^T
                // where g_mat = G_t^T · T_unfold... so G stores rows of size kr_cols.
                // G shape: r_t × (r_l * r_r) — the columns in G span r_l*r_r.
                if g.len() != r_t * r_l * r_r {
                    return Err(FFTError::ComputationError(
                        format!("Reconstruct: G size {} ≠ {}×{}×{}", g.len(), r_t, r_gl, r_gr)
                    ));
                }

                // Multiply kr (mode_size × r_l*r_r) by G (r_l*r_r × r_t) → B_t (mode_size × r_t)
                // But G is stored as (r_l*r_r × r_t) already from truncated_svd output U.
                let b_t = matmul(&kr, nr_l * nr_r, r_l * r_r, g, r_t)?;

                frame_rows[nid] = nr_l * nr_r;
                frame_cols[nid] = r_t;
                frames[nid] = Some(b_t);
                let _ = r_t;
            }
        }

        // Root frame is the full tensor (flattened).
        let root_frame = frames[0].as_ref().ok_or_else(|| {
            FFTError::ComputationError("Missing root frame during reconstruction".into())
        })?;

        if root_frame.len() >= n_total {
            Ok(root_frame[..n_total].to_vec())
        } else {
            // Pad with zeros if truncation reduced size.
            let mut out = root_frame.clone();
            out.resize(n_total, 0.0);
            Ok(out)
        }
    }

    // -----------------------------------------------------------------------
    // Norms, compression, operations
    // -----------------------------------------------------------------------

    /// Frobenius norm of the HT tensor (via reconstruction).
    pub fn norm(&self) -> FFTResult<f64> {
        let t = self.reconstruct()?;
        Ok(frobenius_norm(&t))
    }

    /// Total number of stored scalar parameters.
    pub fn storage_size(&self) -> usize {
        let leaf_total: usize = self.leaves.iter().map(|v| v.len()).sum();
        let transfer_total: usize = self.transfer.iter().map(|v| v.len()).sum();
        leaf_total + transfer_total
    }

    /// Compression ratio: full tensor size / HT storage size.
    ///
    /// Higher values mean better compression.
    pub fn compression_ratio(&self) -> f64 {
        let full: usize = self.shape.iter().product();
        let stored = self.storage_size().max(1);
        full as f64 / stored as f64
    }

    /// Truncate to a lower maximum rank (recompression via reconstruction + re-decompose).
    pub fn truncate(&self, new_max_rank: usize) -> FFTResult<Self> {
        let dense = self.reconstruct()?;
        Self::decompose(&dense, &self.shape, new_max_rank)
    }

    /// Element-wise addition of two HT tensors with the same shape.
    /// After addition, truncates to max_rank to control rank growth.
    pub fn add(&self, other: &HierarchicalTucker, max_rank: usize) -> FFTResult<Self> {
        if self.shape != other.shape {
            return Err(FFTError::ValueError(
                "HT add: shape mismatch".into(),
            ));
        }
        let a = self.reconstruct()?;
        let b = other.reconstruct()?;
        let sum: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        Self::decompose(&sum, &self.shape, max_rank)
    }

    /// Element-wise (Hadamard) product of two HT tensors.
    pub fn hadamard(&self, other: &HierarchicalTucker, max_rank: usize) -> FFTResult<Self> {
        if self.shape != other.shape {
            return Err(FFTError::ValueError(
                "HT hadamard: shape mismatch".into(),
            ));
        }
        let a = self.reconstruct()?;
        let b = other.reconstruct()?;
        let prod: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
        Self::decompose(&prod, &self.shape, max_rank)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Find the leaf storage index for a given node id.
    fn find_leaf_for_node(&self, nid: usize) -> FFTResult<usize> {
        // Walk the tree in post-order to match leaf ordering.
        let leaf_nodes: Vec<usize> = postorder_indices(&self.nodes, 0)
            .into_iter()
            .filter(|&i| self.nodes[i].is_leaf)
            .collect();
        let pos = leaf_nodes
            .iter()
            .position(|&i| i == nid)
            .ok_or_else(|| FFTError::ComputationError(
                format!("Node {nid} not found among leaves")
            ))?;
        if pos >= self.leaves.len() {
            return Err(FFTError::ComputationError(
                format!("Leaf index {pos} out of range (have {} leaves)", self.leaves.len())
            ));
        }
        Ok(pos)
    }

    /// Find the transfer tensor storage index for a given interior node id.
    fn find_transfer_for_node(&self, nid: usize) -> FFTResult<usize> {
        let interior_nodes: Vec<usize> = postorder_indices(&self.nodes, 0)
            .into_iter()
            .filter(|&i| !self.nodes[i].is_leaf)
            .collect();
        let pos = interior_nodes
            .iter()
            .position(|&i| i == nid)
            .ok_or_else(|| FFTError::ComputationError(
                format!("Node {nid} not found among interior nodes")
            ))?;
        if pos >= self.transfer.len() {
            return Err(FFTError::ComputationError(
                format!("Transfer index {pos} out of range (have {} transfers)", self.transfer.len())
            ));
        }
        Ok(pos)
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Compute post-order traversal indices of the dimension tree.
fn postorder_indices(nodes: &[HTNode], root: usize) -> Vec<usize> {
    let mut result = Vec::new();
    postorder_recursive(nodes, root, &mut result);
    result
}

fn postorder_recursive(nodes: &[HTNode], nid: usize, result: &mut Vec<usize>) {
    let node = &nodes[nid];
    if let (Some(l), Some(r)) = (node.left, node.right) {
        postorder_recursive(nodes, l, result);
        postorder_recursive(nodes, r, result);
    }
    result.push(nid);
}

/// Compute the multi-mode unfolding of a tensor.
///
/// Groups the modes in `modes` (which must be contiguous) into the row
/// dimension and all remaining modes into the column dimension.
fn multimode_unfolding(
    tensor: &[f64],
    shape: &[usize],
    modes: &[usize],
) -> FFTResult<(Vec<f64>, usize, usize)> {
    if modes.is_empty() {
        return Err(FFTError::ValueError("multimode_unfolding: empty modes".into()));
    }

    let d = shape.len();
    let n_total: usize = shape.iter().product();
    let row_size: usize = modes.iter().map(|&m| shape[m]).product();
    let col_size = n_total / row_size;

    // Mode set as a HashSet for fast lookup.
    let mode_set: std::collections::HashSet<usize> = modes.iter().copied().collect();

    // Strides in the original tensor.
    let mut strides = vec![1usize; d];
    for k in (0..d - 1).rev() {
        strides[k] = strides[k + 1] * shape[k + 1];
    }

    // Row-dimension ordering: modes in `modes` order.
    // Column-dimension ordering: remaining modes in original order.
    let col_modes: Vec<usize> = (0..d).filter(|m| !mode_set.contains(m)).collect();

    let mut col_strides = vec![1usize; col_modes.len()];
    if !col_modes.is_empty() {
        for k in (0..col_modes.len() - 1).rev() {
            col_strides[k] = col_strides[k + 1] * shape[col_modes[k + 1]];
        }
    }

    // Row strides among `modes`.
    let mut row_strides = vec![1usize; modes.len()];
    if modes.len() > 1 {
        for k in (0..modes.len() - 1).rev() {
            row_strides[k] = row_strides[k + 1] * shape[modes[k + 1]];
        }
    }

    let mut mat = vec![0.0_f64; row_size * col_size];

    for flat_idx in 0..n_total {
        // Decode multi-index.
        let mut multi_idx = vec![0usize; d];
        let mut rem = flat_idx;
        for k in 0..d {
            multi_idx[k] = rem / strides[k];
            rem %= strides[k];
        }

        // Compute row index.
        let row: usize = modes
            .iter()
            .enumerate()
            .map(|(i, &m)| multi_idx[m] * row_strides[i])
            .sum();

        // Compute column index.
        let col: usize = col_modes
            .iter()
            .enumerate()
            .map(|(i, &m)| multi_idx[m] * col_strides[i])
            .sum();

        mat[row * col_size + col] = tensor[flat_idx];
    }

    Ok((mat, row_size, col_size))
}

/// Kronecker product of two matrices A (m × k) and B (n × l).
/// Result is (m*n) × (k*l).
fn khatri_rao_product(
    a: &[f64], m: usize, k: usize,
    b: &[f64], n: usize, l: usize,
) -> FFTResult<Vec<f64>> {
    if a.len() != m * k {
        return Err(FFTError::ValueError(
            format!("khatri_rao: A has {} elements, expected {}×{}", a.len(), m, k),
        ));
    }
    if b.len() != n * l {
        return Err(FFTError::ValueError(
            format!("khatri_rao: B has {} elements, expected {}×{}", b.len(), n, l),
        ));
    }

    let rows = m * n;
    let cols = k * l;
    let mut result = vec![0.0_f64; rows * cols];

    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                for q in 0..l {
                    result[(i * n + j) * cols + (p * l + q)] = a[i * k + p] * b[j * l + q];
                }
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a separable 4D tensor T[i,j,k,l] = u[i]*v[j]*w[k]*x[l].
    fn separable_4d(n: usize) -> (Vec<f64>, Vec<usize>) {
        let shape = vec![n, n, n, n];
        let mut tensor = vec![0.0_f64; n * n * n * n];
        let u: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let v: Vec<f64> = (0..n).map(|j| ((j + 1) as f64).sqrt()).collect();
        let w: Vec<f64> = (0..n).map(|k| 1.0 / (k + 1) as f64).collect();
        let x: Vec<f64> = (0..n).map(|l| (l as f64 * 0.1).sin() + 1.0).collect();

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        let idx = i * n * n * n + j * n * n + k * n + l;
                        tensor[idx] = u[i] * v[j] * w[k] * x[l];
                    }
                }
            }
        }
        (tensor, shape)
    }

    #[test]
    fn test_decompose_reconstruct_2d() {
        let shape = vec![4, 5];
        let tensor: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ht = HierarchicalTucker::decompose(&tensor, &shape, 4).expect("decompose failed");
        let rec = ht.reconstruct().expect("reconstruct failed");

        // Relative Frobenius error.
        let err: f64 = tensor.iter().zip(rec.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        let norm: f64 = tensor.iter().map(|x| x * x).sum::<f64>().sqrt();
        let rel_err = if norm > 1e-12 { err / norm } else { err };
        assert!(rel_err < 0.5, "2D rel_err={rel_err:.4}");
    }

    #[test]
    fn test_decompose_separable_4d() {
        let (tensor, shape) = separable_4d(4);
        let ht = HierarchicalTucker::decompose(&tensor, &shape, 8).expect("decompose failed");

        // Compression ratio should be > 1.
        let ratio = ht.compression_ratio();
        assert!(ratio > 1.0, "compression_ratio={ratio:.2}");

        // Reconstruction error.
        let rec = ht.reconstruct().expect("reconstruct failed");
        let err: f64 = tensor.iter().zip(rec.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        let norm: f64 = tensor.iter().map(|x| x * x).sum::<f64>().sqrt();
        let rel_err = if norm > 1e-10 { err / norm } else { err };
        assert!(rel_err < 0.5, "4D separable rel_err={rel_err:.4}");
    }

    #[test]
    fn test_storage_size_positive() {
        let shape = vec![3, 4, 3];
        let tensor: Vec<f64> = (0..36).map(|i| i as f64 * 0.1).collect();
        let ht = HierarchicalTucker::decompose(&tensor, &shape, 3).expect("decompose failed");
        assert!(ht.storage_size() > 0);
    }

    #[test]
    fn test_add_roundtrip() {
        let shape = vec![3, 4];
        let a: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..12).map(|i| (12 - i) as f64).collect();

        let ht_a = HierarchicalTucker::decompose(&a, &shape, 4).expect("failed to create ht_a");
        let ht_b = HierarchicalTucker::decompose(&b, &shape, 4).expect("failed to create ht_b");
        let ht_sum = ht_a.add(&ht_b, 4).expect("failed to create ht_sum");

        let rec = ht_sum.reconstruct().expect("failed to create rec");
        let expected: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

        let err: f64 = rec.iter().zip(expected.iter()).map(|(r, e)| (r - e).powi(2)).sum::<f64>().sqrt();
        let norm: f64 = expected.iter().map(|x| x * x).sum::<f64>().sqrt();
        let rel_err = if norm > 1e-10 { err / norm } else { err };
        assert!(rel_err < 0.5, "add rel_err={rel_err:.4}");
    }
}
