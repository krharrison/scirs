//! Oct-tree AMR using 3-D Morton (Z-curve) ordering.
//!
//! Follows the same design as `quadtree` but extends it to three spatial
//! dimensions.  Each cell at level *L* covers 1/2^L of the domain along each
//! axis, and refinement splits a cell into 8 equally-sized children.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Morton utilities – 3-D
// ─────────────────────────────────────────────────────────────────────────────

/// Spread a 21-bit integer (x ≤ 2097151) into every third bit position.
///
/// This allows three such spread integers to be OR-ed to form a 3-D Morton
/// code (maximum 63-bit index, comfortably < 2^56).
#[inline]
pub fn spread_bits_3d(v: u32) -> u64 {
    let mut x = v as u64 & 0x001F_FFFF; // 21 bits
    x = (x | (x << 32)) & 0x001F_0000_0000_FFFF;
    x = (x | (x << 16)) & 0x001F_0000_FF00_00FF;
    x = (x | (x << 8)) & 0x100F_00F0_0F00_F00F;
    x = (x | (x << 4)) & 0x10C3_0C30_C30C_30C3;
    x = (x | (x << 2)) & 0x1249_2492_4924_9249;
    x
}

/// Compact every third bit of a 3-D Morton code back into a 21-bit integer.
#[inline]
pub fn compact_bits_3d(v: u64) -> u32 {
    let mut x = v & 0x1249_2492_4924_9249;
    x = (x | (x >> 2)) & 0x10C3_0C30_C30C_30C3;
    x = (x | (x >> 4)) & 0x100F_00F0_0F00_F00F;
    x = (x | (x >> 8)) & 0x001F_0000_FF00_00FF;
    x = (x | (x >> 16)) & 0x001F_0000_0000_FFFF;
    x = (x | (x >> 32)) & 0x001F_FFFF;
    x as u32
}

/// 3-D Morton utility: encode integer coordinates (ix, iy, iz) → Morton index.
pub struct Morton3D;

impl Morton3D {
    /// Encode (ix, iy, iz) as a 3-D Morton (Z-curve) code.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_integrate::amr::octree::Morton3D;
    /// assert_eq!(Morton3D::encode(0, 0, 0), 0);
    /// assert_eq!(Morton3D::encode(1, 0, 0), 1);
    /// assert_eq!(Morton3D::encode(0, 1, 0), 2);
    /// assert_eq!(Morton3D::encode(0, 0, 1), 4);
    /// ```
    #[inline]
    pub fn encode(ix: u32, iy: u32, iz: u32) -> u64 {
        spread_bits_3d(ix) | (spread_bits_3d(iy) << 1) | (spread_bits_3d(iz) << 2)
    }

    /// Decode a Morton code back into (ix, iy, iz).
    #[inline]
    pub fn decode(code: u64) -> (u32, u32, u32) {
        (
            compact_bits_3d(code),
            compact_bits_3d(code >> 1),
            compact_bits_3d(code >> 2),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OctCellId
// ─────────────────────────────────────────────────────────────────────────────

/// Compact cell identifier for the oct-tree: upper 8 bits = level, lower 56 bits = Morton index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OctCellId(pub u64);

impl OctCellId {
    /// Construct from level and Morton index.
    pub fn new(level: u8, morton: u64) -> Self {
        debug_assert!(level < 18, "max 18 levels for 3-D Morton (21-bit coords)");
        OctCellId(((level as u64) << 56) | (morton & 0x00FF_FFFF_FFFF_FFFF))
    }

    /// Refinement level.
    #[inline]
    pub fn level(self) -> u8 {
        (self.0 >> 56) as u8
    }

    /// Morton index within the level.
    #[inline]
    pub fn morton(self) -> u64 {
        self.0 & 0x00FF_FFFF_FFFF_FFFF
    }

    /// Integer grid coordinates `(ix, iy, iz)`.
    pub fn grid_coords(self) -> (u32, u32, u32) {
        Morton3D::decode(self.morton())
    }

    /// Parent cell (one level coarser).  `None` for the root.
    pub fn parent(self) -> Option<OctCellId> {
        let lv = self.level();
        if lv == 0 {
            return None;
        }
        let (ix, iy, iz) = self.grid_coords();
        Some(OctCellId::new(
            lv - 1,
            Morton3D::encode(ix >> 1, iy >> 1, iz >> 1),
        ))
    }

    /// Eight children at one level finer.
    pub fn children(self) -> [OctCellId; 8] {
        let lv = self.level();
        let (ix, iy, iz) = self.grid_coords();
        let bx = ix << 1;
        let by = iy << 1;
        let bz = iz << 1;
        [
            OctCellId::new(lv + 1, Morton3D::encode(bx, by, bz)),
            OctCellId::new(lv + 1, Morton3D::encode(bx + 1, by, bz)),
            OctCellId::new(lv + 1, Morton3D::encode(bx, by + 1, bz)),
            OctCellId::new(lv + 1, Morton3D::encode(bx + 1, by + 1, bz)),
            OctCellId::new(lv + 1, Morton3D::encode(bx, by, bz + 1)),
            OctCellId::new(lv + 1, Morton3D::encode(bx + 1, by, bz + 1)),
            OctCellId::new(lv + 1, Morton3D::encode(bx, by + 1, bz + 1)),
            OctCellId::new(lv + 1, Morton3D::encode(bx + 1, by + 1, bz + 1)),
        ]
    }

    /// Face neighbours at the same level (up to 6 in 3-D).
    pub fn same_level_neighbours(self) -> Vec<OctCellId> {
        let lv = self.level();
        let size: u32 = 1u32.checked_shl(lv as u32).unwrap_or(u32::MAX);
        let (ix, iy, iz) = self.grid_coords();
        let mut nbrs = Vec::with_capacity(6);
        let offsets: [(i64, i64, i64); 6] = [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ];
        for (dx, dy, dz) in offsets {
            let nx = ix as i64 + dx;
            let ny = iy as i64 + dy;
            let nz = iz as i64 + dz;
            if nx < 0
                || ny < 0
                || nz < 0
                || nx >= size as i64
                || ny >= size as i64
                || nz >= size as i64
            {
                continue;
            }
            nbrs.push(OctCellId::new(
                lv,
                Morton3D::encode(nx as u32, ny as u32, nz as u32),
            ));
        }
        nbrs
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OctCellData
// ─────────────────────────────────────────────────────────────────────────────

/// Data stored at each cell of the oct-tree.
#[derive(Debug, Clone)]
pub struct OctCellData {
    /// Solution values (length = n_vars).
    pub values: Vec<f64>,
    /// Refinement level.
    pub level: u8,
    /// Unique identifier.
    pub id: OctCellId,
    /// Children if this cell has been refined.
    pub children: Option<[OctCellId; 8]>,
}

impl OctCellData {
    fn new(id: OctCellId, n_vars: usize) -> Self {
        OctCellData {
            values: vec![0.0; n_vars],
            level: id.level(),
            id,
            children: None,
        }
    }

    /// Returns `true` if this cell has not been refined.
    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    /// Cell width at this level given domain width along x.
    pub fn dx(&self, domain_width: f64) -> f64 {
        domain_width / (1u64 << self.level) as f64
    }

    /// Cell height at this level given domain width along y.
    pub fn dy(&self, domain_height: f64) -> f64 {
        domain_height / (1u64 << self.level) as f64
    }

    /// Cell depth at this level given domain width along z.
    pub fn dz(&self, domain_depth: f64) -> f64 {
        domain_depth / (1u64 << self.level) as f64
    }

    /// Physical center within `domain = [xmin, xmax, ymin, ymax, zmin, zmax]`.
    pub fn center(&self, domain: &[f64; 6]) -> (f64, f64, f64) {
        let (ix, iy, iz) = self.id.grid_coords();
        let wx = domain[1] - domain[0];
        let wy = domain[3] - domain[2];
        let wz = domain[5] - domain[4];
        let dx = self.dx(wx);
        let dy = self.dy(wy);
        let dz = self.dz(wz);
        let x = domain[0] + (ix as f64 + 0.5) * dx;
        let y = domain[2] + (iy as f64 + 0.5) * dy;
        let z = domain[4] + (iz as f64 + 0.5) * dz;
        (x, y, z)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RefinementCriterion3D
// ─────────────────────────────────────────────────────────────────────────────

/// Trait for deciding whether a 3-D oct-tree leaf should be refined.
pub trait RefinementCriterion3D {
    /// Returns `true` when `cell` should be split into eight children.
    fn should_refine(&self, cell: &OctCellData) -> bool;
}

/// Refine 3-D cells when the maximum inter-component gradient exceeds the threshold.
pub struct GradientCriterion3D {
    /// Refinement threshold.
    pub threshold: f64,
}

impl GradientCriterion3D {
    /// Create a new criterion.
    pub fn new(threshold: f64) -> Self {
        GradientCriterion3D { threshold }
    }
}

impl RefinementCriterion3D for GradientCriterion3D {
    fn should_refine(&self, cell: &OctCellData) -> bool {
        if cell.values.len() < 2 {
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

// ─────────────────────────────────────────────────────────────────────────────
// OctTree
// ─────────────────────────────────────────────────────────────────────────────

/// 3-D oct-tree for adaptive mesh refinement.
///
/// # Domain Layout
///
/// `domain = [xmin, xmax, ymin, ymax, zmin, zmax]`
pub struct OctTree {
    /// All cells (leaves + interior).
    pub cells: HashMap<OctCellId, OctCellData>,
    /// Maximum permitted refinement level.
    pub max_level: u8,
    /// Physical domain `[xmin, xmax, ymin, ymax, zmin, zmax]`.
    pub domain: [f64; 6],
    /// Number of solution variables per cell.
    pub n_vars: usize,
}

impl OctTree {
    /// Create a new oct-tree with a single root cell.
    pub fn new(domain: [f64; 6], max_level: u8, n_vars: usize) -> Self {
        let root_id = OctCellId::new(0, 0);
        let root = OctCellData::new(root_id, n_vars);
        let mut cells = HashMap::new();
        cells.insert(root_id, root);
        OctTree {
            cells,
            max_level,
            domain,
            n_vars,
        }
    }

    /// All leaf cells.
    pub fn leaves(&self) -> Vec<OctCellId> {
        self.cells
            .values()
            .filter(|c| c.is_leaf())
            .map(|c| c.id)
            .collect()
    }

    /// Refine a leaf cell into eight children.
    pub fn refine_cell(&mut self, id: OctCellId) {
        let (level, is_leaf) = match self.cells.get(&id) {
            Some(c) => (c.level, c.is_leaf()),
            None => return,
        };
        if !is_leaf || level >= self.max_level {
            return;
        }

        let parent_values = self.cells[&id].values.clone();
        let child_ids = id.children();

        for &child_id in &child_ids {
            let mut child = OctCellData::new(child_id, self.n_vars);
            child.values = parent_values.clone();
            self.cells.insert(child_id, child);
        }

        if let Some(parent) = self.cells.get_mut(&id) {
            parent.children = Some(child_ids);
        }
    }

    /// Coarsen eight sibling leaves back into their parent.
    pub fn coarsen_cell(&mut self, id: OctCellId) {
        let child_ids = match self.cells.get(&id) {
            Some(c) => match c.children {
                Some(ch) => ch,
                None => return,
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

        for &ch in &child_ids {
            self.cells.remove(&ch);
        }

        if let Some(parent) = self.cells.get_mut(&id) {
            parent.values = avg;
            parent.children = None;
        }
    }

    /// Face neighbours of `id` respecting 2:1 balance.
    pub fn neighbors_of(&self, id: OctCellId) -> Vec<OctCellId> {
        let lv = id.level();
        let (ix, iy, iz) = id.grid_coords();
        let size: u32 = 1u32.checked_shl(lv as u32).unwrap_or(u32::MAX);

        let mut result = Vec::new();

        let offsets: [(i64, i64, i64); 6] = [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ];
        for (dx, dy, dz) in offsets {
            let nx = ix as i64 + dx;
            let ny = iy as i64 + dy;
            let nz = iz as i64 + dz;
            if nx < 0
                || ny < 0
                || nz < 0
                || nx >= size as i64
                || ny >= size as i64
                || nz >= size as i64
            {
                continue;
            }
            let nbr_id = OctCellId::new(lv, Morton3D::encode(nx as u32, ny as u32, nz as u32));

            if self.cells.contains_key(&nbr_id) {
                result.push(nbr_id);
            } else if lv > 0 {
                // Try coarser neighbour
                let coarse_id = OctCellId::new(
                    lv - 1,
                    Morton3D::encode(nx as u32 >> 1, ny as u32 >> 1, nz as u32 >> 1),
                );
                if self.cells.contains_key(&coarse_id) {
                    result.push(coarse_id);
                }
            }
        }
        result
    }

    /// Apply a refinement criterion to all leaf cells.
    pub fn adapt(&mut self, criterion: &dyn RefinementCriterion3D) {
        let to_refine: Vec<OctCellId> = self
            .cells
            .values()
            .filter(|c| c.is_leaf() && criterion.should_refine(c))
            .map(|c| c.id)
            .collect();

        for id in to_refine {
            self.refine_cell(id);
        }
    }

    /// Set the values of a leaf cell.
    pub fn set_values(&mut self, id: OctCellId, values: &[f64]) -> bool {
        match self.cells.get_mut(&id) {
            Some(c) if c.is_leaf() => {
                c.values = values.to_vec();
                true
            }
            _ => false,
        }
    }

    /// Get a reference to cell data.
    pub fn get(&self, id: OctCellId) -> Option<&OctCellData> {
        self.cells.get(&id)
    }

    /// Get a mutable reference to cell data.
    pub fn get_mut(&mut self, id: OctCellId) -> Option<&mut OctCellData> {
        self.cells.get_mut(&id)
    }

    /// Total cell count.
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Spacing at a given level in x.
    pub fn dx_at_level(&self, level: u8) -> f64 {
        (self.domain[1] - self.domain[0]) / (1u64 << level) as f64
    }

    /// Spacing at a given level in y.
    pub fn dy_at_level(&self, level: u8) -> f64 {
        (self.domain[3] - self.domain[2]) / (1u64 << level) as f64
    }

    /// Spacing at a given level in z.
    pub fn dz_at_level(&self, level: u8) -> f64 {
        (self.domain[5] - self.domain[4]) / (1u64 << level) as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morton_3d_encode_decode() {
        assert_eq!(Morton3D::encode(0, 0, 0), 0);
        assert_eq!(Morton3D::encode(1, 0, 0), 1);
        assert_eq!(Morton3D::encode(0, 1, 0), 2);
        assert_eq!(Morton3D::encode(0, 0, 1), 4);
        // Round-trip
        for ix in 0u32..4 {
            for iy in 0u32..4 {
                for iz in 0u32..4 {
                    let code = Morton3D::encode(ix, iy, iz);
                    let (rx, ry, rz) = Morton3D::decode(code);
                    assert_eq!(
                        (rx, ry, rz),
                        (ix, iy, iz),
                        "round-trip failed for ({ix},{iy},{iz})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_octtree_initial_state() {
        let tree = OctTree::new([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 5, 1);
        assert_eq!(tree.cell_count(), 1);
        assert_eq!(tree.leaves().len(), 1);
    }

    #[test]
    fn test_octtree_single_refinement() {
        let mut tree = OctTree::new([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 5, 1);
        let root_id = OctCellId::new(0, 0);
        tree.refine_cell(root_id);
        assert_eq!(tree.leaves().len(), 8);
        assert_eq!(tree.cell_count(), 9);
    }

    #[test]
    fn test_octtree_refine_coarsen() {
        let mut tree = OctTree::new([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 5, 2);
        let root_id = OctCellId::new(0, 0);
        tree.refine_cell(root_id);
        assert_eq!(tree.leaves().len(), 8);
        tree.coarsen_cell(root_id);
        assert_eq!(tree.leaves().len(), 1);
        assert!(tree.get(root_id).map(|c| c.is_leaf()).unwrap_or(false));
    }

    #[test]
    fn test_oct_cell_id_parent_children() {
        let root = OctCellId::new(0, Morton3D::encode(0, 0, 0));
        assert!(root.parent().is_none());
        let children = root.children();
        assert_eq!(children.len(), 8);
        for ch in &children {
            assert_eq!(ch.level(), 1);
            assert_eq!(ch.parent(), Some(root));
        }
    }

    #[test]
    fn test_gradient_criterion_3d() {
        let mut tree = OctTree::new([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 5, 2);
        let root_id = OctCellId::new(0, 0);
        tree.set_values(root_id, &[0.0, 10.0]);
        let criterion = GradientCriterion3D::new(5.0);
        tree.adapt(&criterion);
        assert!(
            tree.get(root_id).map(|c| !c.is_leaf()).unwrap_or(false),
            "root should have been refined"
        );
    }
}
