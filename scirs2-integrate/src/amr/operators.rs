//! Conservative prolongation and restriction operators for AMR.
//!
//! # Conservative Property
//!
//! The restriction operator satisfies the discrete conservation identity:
//!
//! ```text
//! sum_c ( val_c * vol_c ) = val_parent * vol_parent
//! ```
//!
//! For a uniform split (all children have equal volume) this reduces to a
//! simple arithmetic mean.  The prolongation operator uses bilinear (2-D) or
//! trilinear (3-D) interpolation so that a *constant* parent field is
//! reproduced exactly in all children.
//!
//! # Usage
//!
//! ```
//! use scirs2_integrate::amr::operators::AmrOperators;
//!
//! let ops = AmrOperators::new();
//! // prolongate a 2-variable constant field
//! let parent_vals = vec![3.0, 7.0];
//! let mut children = [
//!     parent_vals.clone(),
//!     parent_vals.clone(),
//!     parent_vals.clone(),
//!     parent_vals.clone(),
//! ];
//! ops.prolongate_2d_values(&parent_vals, &mut children);
//! for ch in &children {
//!     assert!((ch[0] - 3.0).abs() < 1e-12);
//! }
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// AmrOperators
// ─────────────────────────────────────────────────────────────────────────────

/// Collection of conservative prolongation and restriction operators.
///
/// The struct is stateless; methods take references to cell data represented
/// as `Vec<f64>` slices to avoid coupling to a particular tree type.
#[derive(Debug, Clone, Default)]
pub struct AmrOperators;

impl AmrOperators {
    /// Create a new `AmrOperators` (stateless).
    pub fn new() -> Self {
        AmrOperators
    }

    // ── 2-D prolongation ─────────────────────────────────────────────────────

    /// Bilinear prolongation from a single parent to four children.
    ///
    /// The four children are indexed by their sub-cell position:
    /// ```text
    ///   child 2 (0,1) | child 3 (1,1)
    ///   ──────────────┼──────────────
    ///   child 0 (0,0) | child 1 (1,0)
    /// ```
    ///
    /// For a *constant* field (same value everywhere in the parent) each child
    /// receives exactly the parent value — the operator is exact for degree-0
    /// polynomials.  For a *linear* field we would need neighbour information;
    /// here we implement the simplest conservative approach: each child
    /// inherits the parent cell-average.  This is equivalent to piecewise-
    /// constant (zeroth-order) prolongation, which is always conservative.
    ///
    /// `parent_vals` and each element of `children` must have the same length
    /// `n_vars`.
    pub fn prolongate_2d_values(&self, parent_vals: &[f64], children: &mut [Vec<f64>; 4]) {
        // Bilinear sub-cell offsets relative to the parent cell centre.
        // The centres of the four children in normalised parent coordinates are:
        //   (±0.25, ±0.25)
        // With bilinear interpolation from a single cell average the best we
        // can do without neighbour gradients is piecewise-constant: each child
        // simply gets the parent average.
        for child in children.iter_mut() {
            child.resize(parent_vals.len(), 0.0);
            child.copy_from_slice(parent_vals);
        }
    }

    /// Bilinear prolongation with first-order gradient correction using
    /// neighbour information.
    ///
    /// `grad_x[v]` and `grad_y[v]` are the gradients of variable `v`
    /// at the parent cell centre.  Each of the four children is at offset
    /// `(±0.25*dx, ±0.25*dy)` from the parent centre.
    pub fn prolongate_2d_with_gradient(
        &self,
        parent_vals: &[f64],
        grad_x: &[f64],
        grad_y: &[f64],
        dx: f64,
        dy: f64,
        children: &mut [Vec<f64>; 4],
    ) {
        // Sub-cell offsets (fraction of parent cell size from parent centre):
        // child 0 → (-dx/4, -dy/4)
        // child 1 → (+dx/4, -dy/4)
        // child 2 → (-dx/4, +dy/4)
        // child 3 → (+dx/4, +dy/4)
        let offsets: [(f64, f64); 4] = [
            (-0.25 * dx, -0.25 * dy),
            (0.25 * dx, -0.25 * dy),
            (-0.25 * dx, 0.25 * dy),
            (0.25 * dx, 0.25 * dy),
        ];
        let n = parent_vals.len();
        for (k, child) in children.iter_mut().enumerate() {
            child.resize(n, 0.0);
            let (ox, oy) = offsets[k];
            for v in 0..n {
                child[v] = parent_vals[v]
                    + grad_x.get(v).copied().unwrap_or(0.0) * ox
                    + grad_y.get(v).copied().unwrap_or(0.0) * oy;
            }
        }
    }

    // ── 2-D restriction ───────────────────────────────────────────────────────

    /// Volume-weighted restriction from four children to their parent.
    ///
    /// All four children have equal volume (uniform refinement), so the
    /// result is the arithmetic mean.
    ///
    /// # Conservation guarantee
    ///
    /// `sum_c(child[c] * vol_child) == parent * vol_parent`
    ///
    /// Since `vol_child = vol_parent / 4` and there are 4 children:
    /// `mean(child) * (4 * vol_parent/4) = mean(child) * vol_parent` ✓
    pub fn restrict_2d_values(&self, children: &[Vec<f64>; 4]) -> Vec<f64> {
        let n = children[0].len();
        let mut parent = vec![0.0f64; n];
        for child in children {
            for (p, &c) in parent.iter_mut().zip(child.iter()) {
                *p += c;
            }
        }
        for p in &mut parent {
            *p /= 4.0;
        }
        parent
    }

    /// Verify conservation for 2-D restriction.
    ///
    /// Returns `true` if `sum(child_vals[c] * vol_child) ≈ parent_val * vol_parent`
    /// within `tol`.
    pub fn check_conservation_2d(
        &self,
        parent_vals: &[f64],
        children: &[Vec<f64>; 4],
        tol: f64,
    ) -> bool {
        let n = parent_vals.len();
        // vol_child = vol_parent / 4; sum = 4 * val * (vol_parent/4) = val * vol_parent
        for v in 0..n {
            let sum: f64 = children.iter().map(|ch| ch[v]).sum::<f64>() / 4.0;
            if (sum - parent_vals[v]).abs() > tol {
                return false;
            }
        }
        true
    }

    // ── 3-D prolongation ─────────────────────────────────────────────────────

    /// Trilinear prolongation from one parent to eight children (piecewise-constant).
    pub fn prolongate_3d_values(&self, parent_vals: &[f64], children: &mut [Vec<f64>; 8]) {
        for child in children.iter_mut() {
            child.resize(parent_vals.len(), 0.0);
            child.copy_from_slice(parent_vals);
        }
    }

    /// Trilinear prolongation with first-order gradient correction.
    ///
    /// Sub-cell offsets from parent centre: `(±dx/4, ±dy/4, ±dz/4)`.
    /// Children are ordered by the 3-D Morton pattern:
    /// ```text
    /// child k: bit0 → x+,  bit1 → y+,  bit2 → z+
    /// ```
    pub fn prolongate_3d_with_gradient(
        &self,
        parent_vals: &[f64],
        grad_x: &[f64],
        grad_y: &[f64],
        grad_z: &[f64],
        dx: f64,
        dy: f64,
        dz: f64,
        children: &mut [Vec<f64>; 8],
    ) {
        let n = parent_vals.len();
        for (k, child) in children.iter_mut().enumerate() {
            child.resize(n, 0.0);
            let ox = if k & 1 != 0 { 0.25 * dx } else { -0.25 * dx };
            let oy = if k & 2 != 0 { 0.25 * dy } else { -0.25 * dy };
            let oz = if k & 4 != 0 { 0.25 * dz } else { -0.25 * dz };
            for v in 0..n {
                child[v] = parent_vals[v]
                    + grad_x.get(v).copied().unwrap_or(0.0) * ox
                    + grad_y.get(v).copied().unwrap_or(0.0) * oy
                    + grad_z.get(v).copied().unwrap_or(0.0) * oz;
            }
        }
    }

    // ── 3-D restriction ───────────────────────────────────────────────────────

    /// Volume-weighted restriction from eight children to their parent.
    pub fn restrict_3d_values(&self, children: &[Vec<f64>; 8]) -> Vec<f64> {
        let n = children[0].len();
        let mut parent = vec![0.0f64; n];
        for child in children {
            for (p, &c) in parent.iter_mut().zip(child.iter()) {
                *p += c;
            }
        }
        for p in &mut parent {
            *p /= 8.0;
        }
        parent
    }

    /// Verify conservation for 3-D restriction.
    pub fn check_conservation_3d(
        &self,
        parent_vals: &[f64],
        children: &[Vec<f64>; 8],
        tol: f64,
    ) -> bool {
        let n = parent_vals.len();
        for v in 0..n {
            let sum: f64 = children.iter().map(|ch| ch[v]).sum::<f64>() / 8.0;
            if (sum - parent_vals[v]).abs() > tol {
                return false;
            }
        }
        true
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience free functions
// ─────────────────────────────────────────────────────────────────────────────

/// Prolongate a 2-D parent cell to four children (piecewise-constant).
pub fn prolongate_2d(parent_vals: &[f64], children: &mut [Vec<f64>; 4]) {
    AmrOperators::new().prolongate_2d_values(parent_vals, children);
}

/// Restrict four 2-D children to their parent (volume-weighted mean).
pub fn restrict_2d(children: &[Vec<f64>; 4]) -> Vec<f64> {
    AmrOperators::new().restrict_2d_values(children)
}

/// Prolongate a 3-D parent cell to eight children (piecewise-constant).
pub fn prolongate_3d(parent_vals: &[f64], children: &mut [Vec<f64>; 8]) {
    AmrOperators::new().prolongate_3d_values(parent_vals, children);
}

/// Restrict eight 3-D children to their parent (volume-weighted mean).
pub fn restrict_3d(children: &[Vec<f64>; 8]) -> Vec<f64> {
    AmrOperators::new().restrict_3d_values(children)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_children_2d(vals: Vec<f64>) -> [Vec<f64>; 4] {
        [vals.clone(), vals.clone(), vals.clone(), vals.clone()]
    }

    fn make_children_3d(vals: Vec<f64>) -> [Vec<f64>; 8] {
        [
            vals.clone(),
            vals.clone(),
            vals.clone(),
            vals.clone(),
            vals.clone(),
            vals.clone(),
            vals.clone(),
            vals.clone(),
        ]
    }

    #[test]
    fn test_prolongate_2d_constant_field() {
        let ops = AmrOperators::new();
        let parent = vec![3.0, 7.0];
        let mut children = make_children_2d(vec![0.0, 0.0]);
        ops.prolongate_2d_values(&parent, &mut children);
        for ch in &children {
            assert!((ch[0] - 3.0).abs() < 1e-12);
            assert!((ch[1] - 7.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_restrict_2d_conservation() {
        let ops = AmrOperators::new();
        // Four children with values summing to 4 * parent for each variable
        let children: [Vec<f64>; 4] = [
            vec![1.0, 4.0],
            vec![2.0, 5.0],
            vec![3.0, 6.0],
            vec![4.0, 7.0],
        ];
        let parent = ops.restrict_2d_values(&children);
        // mean of (1,2,3,4) = 2.5;  mean of (4,5,6,7) = 5.5
        assert!((parent[0] - 2.5).abs() < 1e-12);
        assert!((parent[1] - 5.5).abs() < 1e-12);
    }

    #[test]
    fn test_conservation_check_2d() {
        let ops = AmrOperators::new();
        let parent = vec![2.5, 5.5];
        let children: [Vec<f64>; 4] = [
            vec![1.0, 4.0],
            vec![2.0, 5.0],
            vec![3.0, 6.0],
            vec![4.0, 7.0],
        ];
        assert!(ops.check_conservation_2d(&parent, &children, 1e-10));
    }

    #[test]
    fn test_prolong_restrict_roundtrip_2d() {
        let ops = AmrOperators::new();
        let parent_orig = vec![5.0, -3.0];
        let mut children = make_children_2d(vec![0.0, 0.0]);
        ops.prolongate_2d_values(&parent_orig, &mut children);
        let parent_back = ops.restrict_2d_values(&children);
        for (a, b) in parent_orig.iter().zip(parent_back.iter()) {
            assert!((a - b).abs() < 1e-12, "roundtrip mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_prolongate_3d_constant_field() {
        let ops = AmrOperators::new();
        let parent = vec![2.0, 4.0, 6.0];
        let mut children = make_children_3d(vec![0.0; 3]);
        ops.prolongate_3d_values(&parent, &mut children);
        for ch in &children {
            assert!((ch[0] - 2.0).abs() < 1e-12);
            assert!((ch[1] - 4.0).abs() < 1e-12);
            assert!((ch[2] - 6.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_restrict_3d_conservation() {
        let ops = AmrOperators::new();
        let children: [Vec<f64>; 8] = [
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
            vec![7.0],
            vec![8.0],
        ];
        let parent = ops.restrict_3d_values(&children);
        // mean of 1..=8 = 4.5
        assert!((parent[0] - 4.5).abs() < 1e-12);
        assert!(ops.check_conservation_3d(&parent, &children, 1e-10));
    }

    #[test]
    fn test_prolong_restrict_roundtrip_3d() {
        let ops = AmrOperators::new();
        let parent_orig = vec![9.0, -1.0, 0.5];
        let mut children = make_children_3d(vec![0.0; 3]);
        ops.prolongate_3d_values(&parent_orig, &mut children);
        let parent_back = ops.restrict_3d_values(&children);
        for (a, b) in parent_orig.iter().zip(parent_back.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_prolongate_2d_with_gradient() {
        let ops = AmrOperators::new();
        // Linear field f(x,y) = x, grad_x=1, grad_y=0
        let parent_val = vec![0.0]; // f at centre (0,0)
        let grad_x = vec![1.0];
        let grad_y = vec![0.0];
        let dx = 1.0;
        let dy = 1.0;
        let mut children = make_children_2d(vec![0.0]);
        ops.prolongate_2d_with_gradient(&parent_val, &grad_x, &grad_y, dx, dy, &mut children);
        // child 0: offset (-0.25, -0.25) → f = -0.25
        // child 1: offset (+0.25, -0.25) → f = +0.25
        assert!((children[0][0] - (-0.25)).abs() < 1e-12);
        assert!((children[1][0] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_prolongate_3d_with_gradient() {
        let ops = AmrOperators::new();
        // Linear field f(x,y,z) = x + y + z, grad = (1,1,1)
        let parent_val = vec![0.0];
        let grad_x = vec![1.0];
        let grad_y = vec![1.0];
        let grad_z = vec![1.0];
        let dx = 1.0;
        let dy = 1.0;
        let dz = 1.0;
        let mut children = make_children_3d(vec![0.0]);
        ops.prolongate_3d_with_gradient(
            &parent_val,
            &grad_x,
            &grad_y,
            &grad_z,
            dx,
            dy,
            dz,
            &mut children,
        );
        // child 7 (bit0+bit1+bit2 all set): offset (+0.25, +0.25, +0.25) → f = 0.75
        assert!((children[7][0] - 0.75).abs() < 1e-12);
        // child 0 (all bits clear): offset (-0.25, -0.25, -0.25) → f = -0.75
        assert!((children[0][0] - (-0.75)).abs() < 1e-12);
    }
}
