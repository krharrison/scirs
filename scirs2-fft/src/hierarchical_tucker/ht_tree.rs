//! Binary dimension-partition tree for Hierarchical Tucker (HT) decomposition.
//!
//! An HT decomposition of a d-dimensional tensor T stores:
//! - At each **leaf** node t (representing a single mode μ(t)):
//!   a basis matrix  U_t ∈ ℝ^{n_{μ(t)} × r_t}
//! - At each **interior** node t (with children t_L, t_R):
//!   a transfer tensor G_t ∈ ℝ^{r_t × r_{t_L} × r_{t_R}}
//!
//! The dimension-partition tree determines which modes are grouped together.
//! For balanced binary trees the total storage is O(d · k · r²  +  d · n · r).

/// A node in the HT dimension-partition tree.
///
/// Each node owns a contiguous range of tensor modes [`mode_start`, `mode_end`).
#[derive(Debug, Clone)]
pub struct HTNode {
    /// Index of this node in the flat node array.
    pub id: usize,
    /// First mode index (inclusive).
    pub mode_start: usize,
    /// Last mode index (exclusive).
    pub mode_end: usize,
    /// Left child index in the node array (None for leaves).
    pub left: Option<usize>,
    /// Right child index in the node array (None for leaves).
    pub right: Option<usize>,
    /// Parent index (None for root).
    pub parent: Option<usize>,
    /// HT-rank for this node.
    pub rank: usize,
    /// Whether this is a leaf node.
    pub is_leaf: bool,
}

impl HTNode {
    /// Create a new node.
    pub fn new(id: usize, mode_start: usize, mode_end: usize, parent: Option<usize>) -> Self {
        HTNode {
            id,
            mode_start,
            mode_end,
            left: None,
            right: None,
            parent,
            rank: 0,
            is_leaf: mode_end - mode_start == 1,
        }
    }

    /// Number of modes this node spans.
    pub fn n_modes(&self) -> usize {
        self.mode_end - self.mode_start
    }
}

/// Build a balanced binary dimension-partition tree for `n_dims` modes.
///
/// Returns a flat vector of `HTNode`s where index 0 is always the root.
pub fn build_dimension_tree(n_dims: usize) -> Vec<HTNode> {
    let mut nodes: Vec<HTNode> = Vec::new();
    build_recursive(&mut nodes, 0, n_dims, None);
    nodes
}

fn build_recursive(
    nodes: &mut Vec<HTNode>,
    mode_start: usize,
    mode_end: usize,
    parent: Option<usize>,
) -> usize {
    let id = nodes.len();
    let node = HTNode::new(id, mode_start, mode_end, parent);
    nodes.push(node);

    let n = mode_end - mode_start;
    if n > 1 {
        let mid = mode_start + n / 2;
        let left_id = build_recursive(nodes, mode_start, mid, Some(id));
        let right_id = build_recursive(nodes, mid, mode_end, Some(id));
        nodes[id].left = Some(left_id);
        nodes[id].right = Some(right_id);
        nodes[id].is_leaf = false;
    }

    id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_4d_tree() {
        let nodes = build_dimension_tree(4);
        // Root should span all 4 modes.
        assert_eq!(nodes[0].mode_start, 0);
        assert_eq!(nodes[0].mode_end, 4);
        // Root should have two children.
        assert!(nodes[0].left.is_some());
        assert!(nodes[0].right.is_some());
        // Leaves should each span 1 mode.
        for node in &nodes {
            if node.is_leaf {
                assert_eq!(node.n_modes(), 1);
            }
        }
    }

    #[test]
    fn test_leaf_count() {
        for d in 2..=8 {
            let nodes = build_dimension_tree(d);
            let leaves: Vec<_> = nodes.iter().filter(|n| n.is_leaf).collect();
            assert_eq!(leaves.len(), d, "d={d}: expected {d} leaves, got {}", leaves.len());
        }
    }
}
