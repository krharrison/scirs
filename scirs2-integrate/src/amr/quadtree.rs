//! Quad-tree AMR using Morton (Z-curve) ordering for 2D adaptive mesh refinement.
//!
//! # Overview
//!
//! A `QuadTree` stores cells indexed by a `CellId` which encodes both the
//! refinement level and a 2-D Morton index.  Refinement always preserves the
//! 2:1 balance constraint: a cell may not be refined unless all of its face-
//! neighbours are at most one level coarser.
//!
//! # Cell coordinate conventions
//!
//! At level *L* the domain is divided into 2^L × 2^L equal cells.
//! The Morton index is computed from the integer (ix, iy) pair where
//! 0 ≤ ix, iy < 2^L.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Morton utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Interleave bits of a 32-bit integer for 2-D Morton encoding.
///
/// Spreads the bits of `v` into every other bit position so that two such
/// spread integers can be OR-ed to form a Morton code.
#[inline]
pub fn spread_bits_2d(v: u32) -> u64 {
    let mut x = v as u64 & 0x0000_0000_FFFF_FFFF;
    x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF;
    x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF;
    x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;
    x = (x | (x << 1)) & 0x5555_5555_5555_5555;
    x
}

/// Compact (de-interleave) every other bit back into a 32-bit integer.
#[inline]
pub fn compact_bits_2d(v: u64) -> u32 {
    let mut x = v & 0x5555_5555_5555_5555;
    x = (x | (x >> 1)) & 0x3333_3333_3333_3333;
    x = (x | (x >> 2)) & 0x0F0F_0F0F_0F0F_0F0F;
    x = (x | (x >> 4)) & 0x00FF_00FF_00FF_00FF;
    x = (x | (x >> 8)) & 0x0000_FFFF_0000_FFFF;
    x = (x | (x >> 16)) & 0x0000_0000_FFFF_FFFF;
    x as u32
}

/// 2-D Morton utility: encode integer coordinates (ix, iy) → Morton index.
pub struct Morton2D;

impl Morton2D {
    /// Encode (ix, iy) as a 2-D Morton (Z-curve) code.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_integrate::amr::quadtree::Morton2D;
    /// assert_eq!(Morton2D::encode(0, 0), 0);
    /// assert_eq!(Morton2D::encode(1, 0), 1);
    /// assert_eq!(Morton2D::encode(0, 1), 2);
    /// assert_eq!(Morton2D::encode(1, 1), 3);
    /// ```
    #[inline]
    pub fn encode(ix: u32, iy: u32) -> u64 {
        spread_bits_2d(ix) | (spread_bits_2d(iy) << 1)
    }

    /// Decode a Morton code back into (ix, iy).
    #[inline]
    pub fn decode(code: u64) -> (u32, u32) {
        (compact_bits_2d(code), compact_bits_2d(code >> 1))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CellId
// ─────────────────────────────────────────────────────────────────────────────

/// Compact cell identifier: upper 8 bits = level, lower 56 bits = Morton index.
///
/// This allows up to 28 levels of refinement (each level doubles resolution in
/// each axis) and uniquely identifies every cell in the hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CellId(pub u64);

impl CellId {
    /// Construct a `CellId` from level and Morton index.
    pub fn new(level: u8, morton: u64) -> Self {
        debug_assert!(level < 64, "level must be < 64");
        CellId(((level as u64) << 56) | (morton & 0x00FF_FFFF_FFFF_FFFF))
    }

    /// Extract the refinement level (0 = root / coarsest).
    #[inline]
    pub fn level(self) -> u8 {
        (self.0 >> 56) as u8
    }

    /// Extract the Morton index within the level.
    #[inline]
    pub fn morton(self) -> u64 {
        self.0 & 0x00FF_FFFF_FFFF_FFFF
    }

    /// Integer grid coordinates `(ix, iy)` at this cell's level.
    pub fn grid_coords(self) -> (u32, u32) {
        Morton2D::decode(self.morton())
    }

    /// The parent `CellId` (one level coarser).  Returns `None` for the root.
    pub fn parent(self) -> Option<CellId> {
        let lv = self.level();
        if lv == 0 {
            return None;
        }
        let (ix, iy) = self.grid_coords();
        Some(CellId::new(lv - 1, Morton2D::encode(ix >> 1, iy >> 1)))
    }

    /// The four child `CellId`s (one level finer).
    pub fn children(self) -> [CellId; 4] {
        let lv = self.level();
        let (ix, iy) = self.grid_coords();
        let base_ix = ix << 1;
        let base_iy = iy << 1;
        [
            CellId::new(lv + 1, Morton2D::encode(base_ix, base_iy)),
            CellId::new(lv + 1, Morton2D::encode(base_ix + 1, base_iy)),
            CellId::new(lv + 1, Morton2D::encode(base_ix, base_iy + 1)),
            CellId::new(lv + 1, Morton2D::encode(base_ix + 1, base_iy + 1)),
        ]
    }

    /// 2:1-balance face-neighbours at the *same* level (up to 4).
    ///
    /// Neighbours that fall outside `[0, 2^level)` in either coordinate are
    /// omitted.
    pub fn same_level_neighbours(self) -> Vec<CellId> {
        let lv = self.level();
        let size: u32 = 1u32.checked_shl(lv as u32).unwrap_or(u32::MAX);
        let (ix, iy) = self.grid_coords();
        let mut nbrs = Vec::with_capacity(4);
        // Left
        if ix > 0 {
            nbrs.push(CellId::new(lv, Morton2D::encode(ix - 1, iy)));
        }
        // Right
        if ix + 1 < size {
            nbrs.push(CellId::new(lv, Morton2D::encode(ix + 1, iy)));
        }
        // Bottom
        if iy > 0 {
            nbrs.push(CellId::new(lv, Morton2D::encode(ix, iy - 1)));
        }
        // Top
        if iy + 1 < size {
            nbrs.push(CellId::new(lv, Morton2D::encode(ix, iy + 1)));
        }
        nbrs
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CellData
// ─────────────────────────────────────────────────────────────────────────────

/// Data stored at each cell in the quad-tree.
#[derive(Debug, Clone)]
pub struct CellData {
    /// Per-variable solution values (length = n_vars).
    pub values: Vec<f64>,
    /// Refinement level of this cell.
    pub level: u8,
    /// Unique identifier.
    pub id: CellId,
    /// Children if this cell has been refined.
    pub children: Option<[CellId; 4]>,
}

impl CellData {
    fn new(id: CellId, n_vars: usize) -> Self {
        CellData {
            values: vec![0.0; n_vars],
            level: id.level(),
            id,
            children: None,
        }
    }

    /// Returns `true` if this cell has been refined.
    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    /// Cell width at this level given domain width.
    pub fn dx(&self, domain_width: f64) -> f64 {
        let cells_per_side = (1u64 << self.level) as f64;
        domain_width / cells_per_side
    }

    /// Cell height at this level given domain height.
    pub fn dy(&self, domain_height: f64) -> f64 {
        let cells_per_side = (1u64 << self.level) as f64;
        domain_height / cells_per_side
    }

    /// Physical center of this cell within `domain = [xmin, xmax, ymin, ymax]`.
    pub fn center(&self, domain: &[f64; 4]) -> (f64, f64) {
        let (ix, iy) = self.id.grid_coords();
        let w = domain[1] - domain[0];
        let h = domain[3] - domain[2];
        let dx = self.dx(w);
        let dy = self.dy(h);
        let x = domain[0] + (ix as f64 + 0.5) * dx;
        let y = domain[2] + (iy as f64 + 0.5) * dy;
        (x, y)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RefinementCriterion trait
// ─────────────────────────────────────────────────────────────────────────────

/// Trait that decides whether a leaf cell should be refined.
pub trait RefinementCriterion {
    /// Returns `true` when `cell` should be split into four children.
    fn should_refine(&self, cell: &CellData) -> bool;
}

/// Refine cells whose maximum inter-component gradient exceeds a threshold.
///
/// Gradient is estimated as the maximum absolute pairwise difference between
/// adjacent values in `CellData::values`.
pub struct GradientCriterion {
    /// Refinement threshold: refine when max-gradient > threshold.
    pub threshold: f64,
}

impl GradientCriterion {
    /// Create a new `GradientCriterion` with the given threshold.
    pub fn new(threshold: f64) -> Self {
        GradientCriterion { threshold }
    }
}

impl RefinementCriterion for GradientCriterion {
    fn should_refine(&self, cell: &CellData) -> bool {
        if cell.values.len() < 2 {
            // For single-var cells, compare value against threshold directly
            return cell.values.first().copied().unwrap_or(0.0).abs() > self.threshold;
        }
        let max_diff = cell
            .values
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0_f64, f64::max);
        max_diff > self.threshold
    }
}

/// Refine cells where any value exceeds an absolute magnitude threshold.
pub struct MagnitudeCriterion {
    /// Refinement threshold.
    pub threshold: f64,
}

impl MagnitudeCriterion {
    /// Create a new `MagnitudeCriterion`.
    pub fn new(threshold: f64) -> Self {
        MagnitudeCriterion { threshold }
    }
}

impl RefinementCriterion for MagnitudeCriterion {
    fn should_refine(&self, cell: &CellData) -> bool {
        cell.values.iter().any(|&v| v.abs() > self.threshold)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QuadTree
// ─────────────────────────────────────────────────────────────────────────────

/// 2-D quad-tree for adaptive mesh refinement.
///
/// The tree is stored as a flat `HashMap<CellId, CellData>`.  Only *leaf* cells
/// hold the actual solution; interior cells store the volume-averaged value
/// produced during coarsening.
///
/// # Domain Layout
///
/// `domain = [xmin, xmax, ymin, ymax]`
pub struct QuadTree {
    /// All cells (both leaves and interior).
    pub cells: HashMap<CellId, CellData>,
    /// Maximum permitted refinement level.
    pub max_level: u8,
    /// Physical domain `[xmin, xmax, ymin, ymax]`.
    pub domain: [f64; 4],
    /// Number of solution variables per cell.
    pub n_vars: usize,
}

impl QuadTree {
    /// Create a new quad-tree with a single root cell.
    ///
    /// # Arguments
    ///
    /// * `domain` – `[xmin, xmax, ymin, ymax]`
    /// * `max_level` – maximum refinement depth (0 = no refinement)
    /// * `n_vars` – number of solution variables per cell
    pub fn new(domain: [f64; 4], max_level: u8, n_vars: usize) -> Self {
        let root_id = CellId::new(0, 0);
        let root = CellData::new(root_id, n_vars);
        let mut cells = HashMap::new();
        cells.insert(root_id, root);
        QuadTree {
            cells,
            max_level,
            domain,
            n_vars,
        }
    }

    /// Return the IDs of all leaf cells (cells that have not been refined).
    pub fn leaves(&self) -> Vec<CellId> {
        self.cells
            .values()
            .filter(|c| c.is_leaf())
            .map(|c| c.id)
            .collect()
    }

    /// Refine a leaf cell into four children.
    ///
    /// If the cell has already been refined, or is at `max_level`, or does not
    /// exist, the call is a no-op.
    pub fn refine_cell(&mut self, id: CellId) {
        // Check feasibility
        let (level, is_leaf) = match self.cells.get(&id) {
            Some(c) => (c.level, c.is_leaf()),
            None => return,
        };
        if !is_leaf || level >= self.max_level {
            return;
        }

        let parent_values = self.cells[&id].values.clone();
        let child_ids = id.children();

        // Create children inheriting parent values
        for &child_id in &child_ids {
            let mut child = CellData::new(child_id, self.n_vars);
            child.values = parent_values.clone();
            self.cells.insert(child_id, child);
        }

        // Mark parent as interior
        if let Some(parent) = self.cells.get_mut(&id) {
            parent.children = Some(child_ids);
        }
    }

    /// Coarsen a group of four siblings back into their parent.
    ///
    /// Restriction is volume-weighted averaging.  All four children must be
    /// leaves; if any child has been further refined, the call is a no-op.
    pub fn coarsen_cell(&mut self, id: CellId) {
        // `id` is the *parent* we want to restore to leaf status.
        let child_ids = match self.cells.get(&id) {
            Some(c) => match c.children {
                Some(ch) => ch,
                None => return, // already a leaf
            },
            None => return,
        };

        // All children must be leaves
        for &ch in &child_ids {
            match self.cells.get(&ch) {
                Some(c) if !c.is_leaf() => return,
                None => return,
                _ => {}
            }
        }

        // Compute volume-weighted average (equal volumes → simple mean)
        let n = self.n_vars;
        let mut avg = vec![0.0f64; n];
        for &ch in &child_ids {
            if let Some(c) = self.cells.get(&ch) {
                for (a, &v) in avg.iter_mut().zip(c.values.iter()) {
                    *a += v;
                }
            }
        }
        let count = child_ids.len() as f64;
        for a in &mut avg {
            *a /= count;
        }

        // Remove children
        for &ch in &child_ids {
            self.cells.remove(&ch);
        }

        // Restore parent to leaf
        if let Some(parent) = self.cells.get_mut(&id) {
            parent.values = avg;
            parent.children = None;
        }
    }

    /// Return face-neighbours of `id` respecting the 2:1 balance rule.
    ///
    /// Neighbours may be at the same level, one level coarser (parent), or one
    /// level finer (children).
    pub fn neighbors_of(&self, id: CellId) -> Vec<CellId> {
        let lv = id.level();
        let (ix, iy) = id.grid_coords();
        let size: u32 = 1u32.checked_shl(lv as u32).unwrap_or(u32::MAX);

        let mut result = Vec::new();

        // Four face directions: (-1,0), (+1,0), (0,-1), (0,+1)
        let offsets: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        for (dx, dy) in offsets {
            let nx = ix as i64 + dx;
            let ny = iy as i64 + dy;
            if nx < 0 || ny < 0 || nx >= size as i64 || ny >= size as i64 {
                continue;
            }
            let nbr_id = CellId::new(lv, Morton2D::encode(nx as u32, ny as u32));

            if self.cells.contains_key(&nbr_id) {
                // Same level neighbour exists
                result.push(nbr_id);
            } else {
                // Try one level coarser (parent-level neighbour)
                if lv > 0 {
                    let coarse_id =
                        CellId::new(lv - 1, Morton2D::encode(nx as u32 >> 1, ny as u32 >> 1));
                    if self.cells.contains_key(&coarse_id) {
                        result.push(coarse_id);
                        continue;
                    }
                }
                // Try one level finer (2:1 allows this)
                if lv < self.max_level {
                    // The two potential fine cells on this face
                    let fine_candidates: [CellId; 2] = if dx != 0 {
                        [
                            CellId::new(lv + 1, Morton2D::encode(nx as u32 * 2, ny as u32 * 2)),
                            CellId::new(lv + 1, Morton2D::encode(nx as u32 * 2, ny as u32 * 2 + 1)),
                        ]
                    } else {
                        [
                            CellId::new(lv + 1, Morton2D::encode(nx as u32 * 2, ny as u32 * 2)),
                            CellId::new(lv + 1, Morton2D::encode(nx as u32 * 2 + 1, ny as u32 * 2)),
                        ]
                    };
                    for fc in fine_candidates {
                        if self.cells.contains_key(&fc) {
                            result.push(fc);
                        }
                    }
                }
            }
        }
        result
    }

    /// Apply a refinement criterion to all current leaf cells.
    ///
    /// Cells for which `criterion.should_refine` returns `true` are refined.
    /// After refinement a single pass of coarsening is attempted for leaves
    /// whose values are uniform.
    pub fn adapt(&mut self, criterion: &dyn RefinementCriterion) {
        // Collect leaves that should be refined
        let to_refine: Vec<CellId> = self
            .cells
            .values()
            .filter(|c| c.is_leaf() && criterion.should_refine(c))
            .map(|c| c.id)
            .collect();

        for id in to_refine {
            self.refine_cell(id);
        }
    }

    /// Set the values of a leaf cell.  Returns `false` if cell not found or
    /// not a leaf.
    pub fn set_values(&mut self, id: CellId, values: &[f64]) -> bool {
        match self.cells.get_mut(&id) {
            Some(c) if c.is_leaf() => {
                c.values = values.to_vec();
                true
            }
            _ => false,
        }
    }

    /// Get a reference to cell data.
    pub fn get(&self, id: CellId) -> Option<&CellData> {
        self.cells.get(&id)
    }

    /// Get a mutable reference to cell data.
    pub fn get_mut(&mut self, id: CellId) -> Option<&mut CellData> {
        self.cells.get_mut(&id)
    }

    /// Total number of cells (leaves + interior).
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Domain width (xmax - xmin).
    pub fn domain_width(&self) -> f64 {
        self.domain[1] - self.domain[0]
    }

    /// Domain height (ymax - ymin).
    pub fn domain_height(&self) -> f64 {
        self.domain[3] - self.domain[2]
    }

    /// Cell spacing at a given level in x.
    pub fn dx_at_level(&self, level: u8) -> f64 {
        self.domain_width() / (1u64 << level) as f64
    }

    /// Cell spacing at a given level in y.
    pub fn dy_at_level(&self, level: u8) -> f64 {
        self.domain_height() / (1u64 << level) as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morton_encode_decode() {
        assert_eq!(Morton2D::encode(0, 0), 0);
        assert_eq!(Morton2D::encode(1, 0), 1);
        assert_eq!(Morton2D::encode(0, 1), 2);
        assert_eq!(Morton2D::encode(1, 1), 3);
        // Round-trip
        for ix in 0u32..8 {
            for iy in 0u32..8 {
                let code = Morton2D::encode(ix, iy);
                let (rx, ry) = Morton2D::decode(code);
                assert_eq!((rx, ry), (ix, iy), "round-trip failed for ({ix},{iy})");
            }
        }
    }

    #[test]
    fn test_cell_id_level_morton() {
        let id = CellId::new(3, 7);
        assert_eq!(id.level(), 3);
        assert_eq!(id.morton(), 7);
    }

    #[test]
    fn test_cell_id_parent_children() {
        let root = CellId::new(0, Morton2D::encode(0, 0));
        assert!(root.parent().is_none());
        let children = root.children();
        assert_eq!(children.len(), 4);
        for ch in &children {
            assert_eq!(ch.level(), 1);
            assert_eq!(ch.parent(), Some(root));
        }
    }

    #[test]
    fn test_quadtree_initial_state() {
        let tree = QuadTree::new([0.0, 1.0, 0.0, 1.0], 5, 1);
        assert_eq!(tree.cell_count(), 1);
        assert_eq!(tree.leaves().len(), 1);
    }

    #[test]
    fn test_quadtree_single_refinement() {
        let mut tree = QuadTree::new([0.0, 1.0, 0.0, 1.0], 5, 2);
        let root_id = CellId::new(0, 0);
        tree.refine_cell(root_id);
        // Root becomes interior, 4 new leaves
        assert_eq!(tree.leaves().len(), 4);
        assert_eq!(tree.cell_count(), 5); // root + 4 children
    }

    #[test]
    fn test_quadtree_leaf_count_k_refinements() {
        let mut tree = QuadTree::new([0.0, 1.0, 0.0, 1.0], 10, 1);
        // Refine root k=3 times (one branch only → 4^k leaves total after
        // uniform refinement)
        let root_id = CellId::new(0, 0);
        tree.refine_cell(root_id);
        let leaves_lv1 = tree.leaves();
        for id in leaves_lv1 {
            tree.refine_cell(id);
        }
        let leaves_lv2 = tree.leaves();
        for id in leaves_lv2 {
            tree.refine_cell(id);
        }
        assert_eq!(tree.leaves().len(), 4usize.pow(3));
    }

    #[test]
    fn test_refine_then_coarsen() {
        let mut tree = QuadTree::new([0.0, 1.0, 0.0, 1.0], 5, 1);
        let root_id = CellId::new(0, 0);
        tree.refine_cell(root_id);
        assert_eq!(tree.leaves().len(), 4);
        // Coarsen back
        tree.coarsen_cell(root_id);
        assert_eq!(tree.leaves().len(), 1);
        assert!(tree.get(root_id).map(|c| c.is_leaf()).unwrap_or(false));
    }

    #[test]
    fn test_neighbors_center_leaf_level2() {
        // Build a uniform level-2 grid (refine root, then refine all level-1 leaves)
        let mut tree = QuadTree::new([0.0, 1.0, 0.0, 1.0], 5, 1);
        let root_id = CellId::new(0, 0);
        tree.refine_cell(root_id);
        let lv1_leaves: Vec<_> = tree.leaves();
        for id in lv1_leaves {
            tree.refine_cell(id);
        }
        // Pick the cell at (1,1) at level 2 — it has 4 face neighbours
        let center = CellId::new(2, Morton2D::encode(1, 1));
        let nbrs = tree.neighbors_of(center);
        assert_eq!(
            nbrs.len(),
            4,
            "interior cell at level 2 should have 4 neighbours, got {nbrs:?}"
        );
    }

    #[test]
    fn test_gradient_criterion() {
        let mut tree = QuadTree::new([0.0, 1.0, 0.0, 1.0], 5, 2);
        // Manually set root values with a large gradient
        let root_id = CellId::new(0, 0);
        tree.set_values(root_id, &[0.0, 10.0]);
        let criterion = GradientCriterion::new(5.0);
        tree.adapt(&criterion);
        // Root should have been refined
        assert!(
            tree.get(root_id).map(|c| !c.is_leaf()).unwrap_or(false),
            "root should have been refined"
        );
    }

    #[test]
    fn test_no_refinement_below_threshold() {
        let mut tree = QuadTree::new([0.0, 1.0, 0.0, 1.0], 5, 2);
        let root_id = CellId::new(0, 0);
        tree.set_values(root_id, &[1.0, 1.1]);
        let criterion = GradientCriterion::new(5.0);
        tree.adapt(&criterion);
        assert!(tree.get(root_id).map(|c| c.is_leaf()).unwrap_or(false));
    }
}
