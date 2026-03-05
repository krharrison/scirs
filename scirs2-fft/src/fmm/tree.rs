//! Quad-tree and oct-tree data structures for FMM.
//!
//! Provides hierarchical spatial decomposition for organizing particles
//! in 2D (QuadTree) and 3D (OctTree) for the Fast Multipole Method.

use crate::error::{FFTError, FFTResult};

/// A node in the quad-tree spatial decomposition.
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Center of this node's bounding box.
    pub center: [f64; 2],
    /// Half-width of the bounding box (axis-aligned square).
    pub half_width: f64,
    /// Depth of this node in the tree (root = 0).
    pub depth: usize,
    /// Indices into the source/target array for points in this node.
    pub point_indices: Vec<usize>,
    /// Child nodes (SW, SE, NW, NE), or None if leaf.
    pub children: Option<Box<[TreeNode; 4]>>,
    /// Whether this is a leaf node (no children).
    pub is_leaf: bool,
    /// Node index in a flat traversal order (set during build).
    pub node_id: usize,
}

impl TreeNode {
    /// Create a new empty tree node.
    pub fn new(center: [f64; 2], half_width: f64, depth: usize) -> Self {
        TreeNode {
            center,
            half_width,
            depth,
            point_indices: Vec::new(),
            children: None,
            is_leaf: true,
            node_id: 0,
        }
    }

    /// Test whether point p lies within this node's bounding box.
    /// Uses inclusive boundaries on all sides.
    pub fn contains(&self, p: [f64; 2]) -> bool {
        let hw = self.half_width;
        p[0] >= self.center[0] - hw
            && p[0] <= self.center[0] + hw
            && p[1] >= self.center[1] - hw
            && p[1] <= self.center[1] + hw
    }

    /// Return the centers of the four child quadrants in order:
    /// index 0 = SW, 1 = SE, 2 = NW, 3 = NE.
    pub fn child_centers(&self) -> [[f64; 2]; 4] {
        let qw = self.half_width * 0.5;
        let cx = self.center[0];
        let cy = self.center[1];
        [
            [cx - qw, cy - qw], // SW
            [cx + qw, cy - qw], // SE
            [cx - qw, cy + qw], // NW
            [cx + qw, cy + qw], // NE
        ]
    }

    /// Determine the child quadrant index (0–3) that contains point p.
    /// Returns None if p is outside this node entirely.
    pub fn child_index_for(&self, p: [f64; 2]) -> Option<usize> {
        if !self.contains(p) {
            return None;
        }
        let east = p[0] > self.center[0];
        let north = p[1] > self.center[1];
        let idx = match (east, north) {
            (false, false) => 0, // SW
            (true, false) => 1,  // SE
            (false, true) => 2,  // NW
            (true, true) => 3,   // NE
        };
        Some(idx)
    }

    /// Test whether this node is spatially adjacent to another node.
    /// Two nodes are adjacent if they share a face, edge, or corner.
    pub fn is_adjacent(&self, other: &TreeNode) -> bool {
        let tol = 1e-10;
        let hw_sum = self.half_width + other.half_width + tol;
        let dx = (self.center[0] - other.center[0]).abs();
        let dy = (self.center[1] - other.center[1]).abs();
        dx < hw_sum && dy < hw_sum
    }

    /// Multipole Acceptance Criterion (MAC):
    /// Returns true when the interaction between this node and another
    /// can be represented by a multipole approximation.
    ///
    /// Standard criterion: r > (hw_self + hw_other) / mac_theta
    /// where r is the center-to-center distance.
    pub fn is_well_separated(&self, other: &TreeNode, mac: f64) -> bool {
        let dx = self.center[0] - other.center[0];
        let dy = self.center[1] - other.center[1];
        let r = (dx * dx + dy * dy).sqrt();
        let max_hw = self.half_width.max(other.half_width);
        r > max_hw / mac
    }

    /// Compute the bounding box as (min_x, min_y, max_x, max_y).
    pub fn bbox(&self) -> [f64; 4] {
        [
            self.center[0] - self.half_width,
            self.center[1] - self.half_width,
            self.center[0] + self.half_width,
            self.center[1] + self.half_width,
        ]
    }
}

/// A 2D quad-tree for organizing point sets.
///
/// Used by FMM2D to spatially partition sources and targets.
#[derive(Debug, Clone)]
pub struct QuadTree {
    /// Root node of the tree.
    pub root: TreeNode,
    /// Maximum allowed depth (limits memory use).
    pub max_depth: usize,
    /// Maximum number of points per leaf before subdivision.
    pub max_points_per_leaf: usize,
    /// Total number of nodes.
    pub node_count: usize,
}

impl QuadTree {
    /// Build a quad-tree from a set of 2D points.
    ///
    /// # Arguments
    /// * `points`       – Array of 2D positions `[x, y]`.
    /// * `max_depth`    – Maximum recursion depth (root = depth 0).
    /// * `max_per_leaf` – Subdivision threshold.
    pub fn build(
        points: &[[f64; 2]],
        max_depth: usize,
        max_per_leaf: usize,
    ) -> FFTResult<Self> {
        if points.is_empty() {
            return Err(FFTError::ValueError("QuadTree: empty point set".into()));
        }

        // Compute bounding box with a small margin to avoid boundary issues.
        let mut min_x = points[0][0];
        let mut max_x = points[0][0];
        let mut min_y = points[0][1];
        let mut max_y = points[0][1];

        for p in points.iter() {
            if p[0] < min_x { min_x = p[0]; }
            if p[0] > max_x { max_x = p[0]; }
            if p[1] < min_y { min_y = p[1]; }
            if p[1] > max_y { max_y = p[1]; }
        }

        let margin = 1e-6;
        let center_x = (min_x + max_x) * 0.5;
        let center_y = (min_y + max_y) * 0.5;
        let half_width = ((max_x - min_x).max(max_y - min_y)) * 0.5 + margin;

        let root = TreeNode::new([center_x, center_y], half_width, 0);
        let mut tree = QuadTree {
            root,
            max_depth,
            max_points_per_leaf: max_per_leaf,
            node_count: 1,
        };

        // Insert all points by index.
        let all_indices: Vec<usize> = (0..points.len()).collect();
        insert_recursive(
            &mut tree.root,
            points,
            &all_indices,
            0,
            max_depth,
            max_per_leaf,
            &mut tree.node_count,
        );

        Ok(tree)
    }

    /// Return references to all leaf nodes via DFS traversal.
    pub fn leaves(&self) -> Vec<&TreeNode> {
        let mut result = Vec::new();
        collect_leaves(&self.root, &mut result);
        result
    }

    /// Return references to all nodes at a given depth level.
    pub fn level_nodes(&self, level: usize) -> Vec<&TreeNode> {
        let mut result = Vec::new();
        collect_level(&self.root, level, &mut result);
        result
    }

    /// Return all nodes in BFS order.
    pub fn all_nodes_bfs(&self) -> Vec<&TreeNode> {
        let mut result = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(&self.root);
        while let Some(node) = queue.pop_front() {
            result.push(node);
            if let Some(children) = &node.children {
                for child in children.iter() {
                    queue.push_back(child);
                }
            }
        }
        result
    }

    /// Maximum depth actually reached in the tree.
    pub fn actual_depth(&self) -> usize {
        max_depth_recursive(&self.root)
    }
}

// ---------------------------------------------------------------------------
// Private recursive helpers
// ---------------------------------------------------------------------------

fn insert_recursive(
    node: &mut TreeNode,
    points: &[[f64; 2]],
    indices: &[usize],
    depth: usize,
    max_depth: usize,
    max_per_leaf: usize,
    node_count: &mut usize,
) {
    node.point_indices.extend_from_slice(indices);

    // Stop subdividing if at max depth or few enough points.
    if depth >= max_depth || indices.len() <= max_per_leaf {
        node.is_leaf = true;
        return;
    }

    node.is_leaf = false;

    // Build four empty children.
    let child_centers = node.child_centers();
    let child_hw = node.half_width * 0.5;
    let child_depth = depth + 1;

    let mut children = Box::new([
        TreeNode::new(child_centers[0], child_hw, child_depth),
        TreeNode::new(child_centers[1], child_hw, child_depth),
        TreeNode::new(child_centers[2], child_hw, child_depth),
        TreeNode::new(child_centers[3], child_hw, child_depth),
    ]);

    // Assign IDs and partition points into child quadrants.
    for i in 0..4usize {
        *node_count += 1;
        children[i].node_id = *node_count;
    }

    let mut child_indices: [Vec<usize>; 4] = [
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ];

    for &idx in indices.iter() {
        let p = points[idx];
        // Use parent center to determine quadrant for stability.
        let east = p[0] > node.center[0];
        let north = p[1] > node.center[1];
        let q = match (east, north) {
            (false, false) => 0,
            (true, false) => 1,
            (false, true) => 2,
            (true, true) => 3,
        };
        child_indices[q].push(idx);
    }

    // Recurse into children.
    for i in 0..4usize {
        if !child_indices[i].is_empty() {
            insert_recursive(
                &mut children[i],
                points,
                &child_indices[i],
                child_depth,
                max_depth,
                max_per_leaf,
                node_count,
            );
        } else {
            children[i].is_leaf = true;
        }
    }

    node.children = Some(children);
}

fn collect_leaves<'a>(node: &'a TreeNode, result: &mut Vec<&'a TreeNode>) {
    if node.is_leaf {
        result.push(node);
    } else if let Some(children) = &node.children {
        for child in children.iter() {
            collect_leaves(child, result);
        }
    }
}

fn collect_level<'a>(node: &'a TreeNode, level: usize, result: &mut Vec<&'a TreeNode>) {
    if node.depth == level {
        result.push(node);
        return;
    }
    if let Some(children) = &node.children {
        for child in children.iter() {
            collect_level(child, level, result);
        }
    }
}

fn max_depth_recursive(node: &TreeNode) -> usize {
    if node.is_leaf {
        return node.depth;
    }
    if let Some(children) = &node.children {
        children
            .iter()
            .map(max_depth_recursive)
            .max()
            .unwrap_or(node.depth)
    } else {
        node.depth
    }
}

/// Placeholder 3D oct-tree (structure only; FMM3D is out of scope for this module).
#[derive(Debug, Clone)]
pub struct OctTree {
    /// Bounding box center.
    pub center: [f64; 3],
    /// Half-width of the cubic bounding box.
    pub half_width: f64,
    /// Maximum depth.
    pub max_depth: usize,
}

impl OctTree {
    /// Create a stub OctTree (full 3D FMM would extend this).
    pub fn new(center: [f64; 3], half_width: f64, max_depth: usize) -> Self {
        OctTree { center, half_width, max_depth }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_node_contains() {
        let node = TreeNode::new([0.0, 0.0], 1.0, 0);
        assert!(node.contains([0.0, 0.0]));
        assert!(node.contains([0.9, 0.9]));
        assert!(!node.contains([1.1, 0.0]));
    }

    #[test]
    fn test_child_centers_quadrants() {
        let node = TreeNode::new([0.0, 0.0], 1.0, 0);
        let centers = node.child_centers();
        // SW
        assert!((centers[0][0] + 0.5).abs() < 1e-12);
        assert!((centers[0][1] + 0.5).abs() < 1e-12);
        // NE
        assert!((centers[3][0] - 0.5).abs() < 1e-12);
        assert!((centers[3][1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_quad_tree_build() {
        let points: Vec<[f64; 2]> = (0..100)
            .map(|i| {
                let t = i as f64 / 100.0;
                [t.cos(), t.sin()]
            })
            .collect();

        let tree = QuadTree::build(&points, 5, 4).expect("build failed");
        let leaves = tree.leaves();
        assert!(!leaves.is_empty());

        // Every source index should appear in exactly one leaf.
        let mut seen = vec![false; 100];
        for leaf in &leaves {
            for &idx in &leaf.point_indices {
                assert!(!seen[idx], "duplicate index {idx}");
                seen[idx] = true;
            }
        }
        // All points covered.
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn test_adjacency() {
        let n1 = TreeNode::new([0.0, 0.0], 0.5, 0);
        let n2 = TreeNode::new([1.0, 0.0], 0.5, 0);
        let n3 = TreeNode::new([2.0, 0.0], 0.5, 0);
        assert!(n1.is_adjacent(&n2));
        assert!(!n1.is_adjacent(&n3));
    }

    #[test]
    fn test_well_separated() {
        let n1 = TreeNode::new([0.0, 0.0], 0.5, 0);
        let n2 = TreeNode::new([10.0, 0.0], 0.5, 0);
        assert!(n1.is_well_separated(&n2, 0.5));
        let n3 = TreeNode::new([1.0, 0.0], 0.5, 0);
        assert!(!n1.is_well_separated(&n3, 0.5));
    }
}
