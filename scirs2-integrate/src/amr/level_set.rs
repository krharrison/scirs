//! Level-set interface tracking on an AMR quad-tree.
//!
//! # Level-Set Method
//!
//! The level-set method represents a moving interface Γ as the zero contour of
//! a smooth signed-distance function φ:
//!
//! ```text
//! Γ(t) = { x : φ(x,t) = 0 }
//! φ(x,t) > 0  →  x is outside the interface
//! φ(x,t) < 0  →  x is inside
//! |∇φ| = 1    →  signed-distance property
//! ```
//!
//! The level-set equation is:
//! ```text
//! ∂φ/∂t + v·∇φ = 0
//! ```
//!
//! # Reinitialization
//!
//! After advection the signed-distance property degrades.  Reinitialization
//! solves the Eikonal equation via the Sussman (1994) PDE:
//! ```text
//! ∂φ/∂τ + sign(φ₀)(|∇φ| − 1) = 0
//! ```
//! using pseudo-time iterations τ on a narrow band around the interface.

use crate::amr::quadtree::{CellData, CellId, Morton2D, QuadTree};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Smoothed sign function (Sussman 1994):
///
/// ```text
/// sign_ε(φ) = φ / √(φ² + ε²)
/// ```
#[inline]
fn sign_smooth(phi: f64, eps: f64) -> f64 {
    phi / (phi * phi + eps * eps).sqrt()
}

/// First-order upwind gradient magnitude |∇φ| using Godunov's scheme.
///
/// For a scalar φ and upwind differences:
/// ```text
/// a⁺ = max(D⁻φ,  0)
/// a⁻ = min(D⁺φ,  0)
/// b⁺ = max(D⁻φy, 0)
/// b⁻ = min(D⁺φy, 0)
/// |∇φ|² = max(a⁺², a⁻²) + max(b⁺², b⁻²)
/// ```
fn godunov_gradient_magnitude(
    phi_c: f64,
    phi_xm: f64,
    phi_xp: f64,
    phi_ym: f64,
    phi_yp: f64,
    dx: f64,
    dy: f64,
    sign: f64,
) -> f64 {
    let dm_x = (phi_c - phi_xm) / dx;
    let dp_x = (phi_xp - phi_c) / dx;
    let dm_y = (phi_c - phi_ym) / dy;
    let dp_y = (phi_yp - phi_c) / dy;

    // Godunov numerical Hamiltonian
    let ax = if sign > 0.0 {
        f64::max(f64::max(dm_x, 0.0).powi(2), f64::min(dp_x, 0.0).powi(2))
    } else {
        f64::max(f64::min(dm_x, 0.0).powi(2), f64::max(dp_x, 0.0).powi(2))
    };
    let ay = if sign > 0.0 {
        f64::max(f64::max(dm_y, 0.0).powi(2), f64::min(dp_y, 0.0).powi(2))
    } else {
        f64::max(f64::min(dm_y, 0.0).powi(2), f64::max(dp_y, 0.0).powi(2))
    };
    (ax + ay).sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// LevelSet
// ─────────────────────────────────────────────────────────────────────────────

/// Signed-distance level-set function stored on a `QuadTree`.
///
/// Variable index 0 of each leaf cell contains the level-set value φ.  The
/// tree must have been constructed with `n_vars >= 1`.
///
/// The *narrow band* is the set of cells where `|φ| < delta`.  Only narrow-
/// band cells are updated during reinitialization and advection, which makes
/// these algorithms O(N_interface) rather than O(N_total).
pub struct LevelSet {
    /// The underlying quad-tree (n_vars == 1, storing φ).
    pub tree: QuadTree,
    /// Narrow-band half-width in physical units.
    pub delta: f64,
}

impl LevelSet {
    /// Create a new level-set by evaluating `phi0(x, y)` on every leaf of
    /// an *already-constructed* quad-tree (must have `n_vars >= 1`).
    ///
    /// `phi0` should return a **signed distance** (positive outside, negative
    /// inside) but any smooth function is accepted.
    pub fn new(tree: QuadTree, phi0: impl Fn(f64, f64) -> f64) -> Self {
        let mut ls = LevelSet { tree, delta: 3.0 };
        // Evaluate phi0 on all leaf cells
        let leaves: Vec<CellId> = ls.tree.leaves();
        for id in leaves {
            if let Some(cell) = ls.tree.cells.get(&id) {
                let (cx, cy) = cell.center(&ls.tree.domain);
                let phi = phi0(cx, cy);
                let n_vars = ls.tree.n_vars;
                let mut vals = vec![0.0f64; n_vars];
                vals[0] = phi;
                // Store back
                if let Some(cell) = ls.tree.cells.get_mut(&id) {
                    cell.values = vals;
                }
            }
        }
        // Set delta to 3 * coarsest cell size
        let coarsest_dx = ls.tree.dx_at_level(0);
        ls.delta = 3.0 * coarsest_dx;
        ls
    }

    /// Override the narrow-band half-width (default: 3 × coarsest cell size).
    pub fn set_delta(&mut self, delta: f64) {
        self.delta = delta;
    }

    /// Return the level-set value φ at a leaf cell.
    pub fn phi(&self, id: CellId) -> Option<f64> {
        self.tree.cells.get(&id).map(|c| c.values[0])
    }

    /// Build a snapshot map `CellId → φ` for all leaf cells.
    fn phi_snapshot(&self) -> HashMap<CellId, f64> {
        self.tree
            .leaves()
            .into_iter()
            .filter_map(|id| {
                let phi = self.tree.cells.get(&id)?.values[0];
                Some((id, phi))
            })
            .collect()
    }

    /// Look up φ in a snapshot or fall back to the nearest ancestor in the tree.
    fn get_phi_or_ancestor(snap: &HashMap<CellId, f64>, tree: &QuadTree, id: CellId) -> f64 {
        if let Some(&phi) = snap.get(&id) {
            return phi;
        }
        // Walk up to find the first ancestor that is in the snapshot
        let mut cur = id;
        loop {
            match cur.parent() {
                None => return 0.0,
                Some(p) => {
                    if let Some(&phi) = snap.get(&p) {
                        return phi;
                    }
                    cur = p;
                }
            }
        }
    }

    /// Return the φ value of a face-neighbour of `id` in the direction given
    /// by `(dx, dy) ∈ {±1} × {0}` or `{0} × {±1}`.
    ///
    /// If the neighbour does not exist (boundary), the current cell's φ is
    /// returned (zero-flux Neumann condition).
    fn neighbour_phi(&self, id: CellId, snap: &HashMap<CellId, f64>, di: i64, dj: i64) -> f64 {
        let lv = id.level();
        let size = 1u32.checked_shl(lv as u32).unwrap_or(u32::MAX);
        let (ix, iy) = id.grid_coords();
        let ni = ix as i64 + di;
        let nj = iy as i64 + dj;
        if ni < 0 || nj < 0 || ni >= size as i64 || nj >= size as i64 {
            // Boundary: Neumann
            return snap.get(&id).copied().unwrap_or(0.0);
        }
        let nbr_id = CellId::new(lv, Morton2D::encode(ni as u32, nj as u32));
        Self::get_phi_or_ancestor(snap, &self.tree, nbr_id)
    }

    /// Sussman reinitialization: solve `∂φ/∂τ + sign(φ₀)(|∇φ|−1) = 0`
    /// for `n_iters` pseudo-time steps using first-order Godunov upwind.
    ///
    /// Only narrow-band cells (`|φ| < delta`) are updated.
    pub fn reinitialize(&mut self, n_iters: usize) {
        for _ in 0..n_iters {
            let snap = self.phi_snapshot();
            let leaves: Vec<CellId> = self.tree.leaves().into_iter().collect();

            let mut updates: Vec<(CellId, f64)> = Vec::new();

            for &id in &leaves {
                let phi_c = match snap.get(&id) {
                    Some(&v) => v,
                    None => continue,
                };
                // Only update narrow-band cells
                if phi_c.abs() >= self.delta {
                    continue;
                }

                let lv = id.level();
                let dx = self.tree.dx_at_level(lv);
                let dy = self.tree.dy_at_level(lv);
                let eps = 0.5 * dx.min(dy);

                let sign0 = sign_smooth(phi_c, eps);

                let phi_xm = self.neighbour_phi(id, &snap, -1, 0);
                let phi_xp = self.neighbour_phi(id, &snap, 1, 0);
                let phi_ym = self.neighbour_phi(id, &snap, 0, -1);
                let phi_yp = self.neighbour_phi(id, &snap, 0, 1);

                let grad_mag = godunov_gradient_magnitude(
                    phi_c, phi_xm, phi_xp, phi_ym, phi_yp, dx, dy, sign0,
                );

                // CFL: dt = 0.5 * min(dx, dy)
                let dt = 0.5 * dx.min(dy);
                let new_phi = phi_c - dt * sign0 * (grad_mag - 1.0);
                updates.push((id, new_phi));
            }

            // Apply updates
            for (id, new_phi) in updates {
                if let Some(cell) = self.tree.cells.get_mut(&id) {
                    cell.values[0] = new_phi;
                }
            }
        }
    }

    /// First-order upwind advection: `∂φ/∂t + v·∇φ = 0`.
    ///
    /// `vx(x, y, t)` and `vy(x, y, t)` are the advecting velocity components.
    /// Only narrow-band cells are updated.  The time step `dt` must satisfy the
    /// CFL condition `dt * max(|v|) / dx < 1`.
    pub fn advect(
        &mut self,
        vx: &dyn Fn(f64, f64, f64) -> f64,
        vy: &dyn Fn(f64, f64, f64) -> f64,
        t: f64,
        dt: f64,
    ) {
        let snap = self.phi_snapshot();
        let leaves: Vec<CellId> = self.tree.leaves();
        let mut updates: Vec<(CellId, f64)> = Vec::new();

        for &id in &leaves {
            let phi_c = match snap.get(&id) {
                Some(&v) => v,
                None => continue,
            };
            if phi_c.abs() >= self.delta {
                continue;
            }

            let lv = id.level();
            let dx = self.tree.dx_at_level(lv);
            let dy = self.tree.dy_at_level(lv);

            let cell = match self.tree.cells.get(&id) {
                Some(c) => c,
                None => continue,
            };
            let (cx, cy) = cell.center(&self.tree.domain);
            let u = vx(cx, cy, t);
            let v_y = vy(cx, cy, t);

            // First-order upwind differences
            let dphi_x = if u >= 0.0 {
                let phi_xm = self.neighbour_phi(id, &snap, -1, 0);
                (phi_c - phi_xm) / dx
            } else {
                let phi_xp = self.neighbour_phi(id, &snap, 1, 0);
                (phi_xp - phi_c) / dx
            };
            let dphi_y = if v_y >= 0.0 {
                let phi_ym = self.neighbour_phi(id, &snap, 0, -1);
                (phi_c - phi_ym) / dy
            } else {
                let phi_yp = self.neighbour_phi(id, &snap, 0, 1);
                (phi_yp - phi_c) / dy
            };

            let new_phi = phi_c - dt * (u * dphi_x + v_y * dphi_y);
            updates.push((id, new_phi));
        }

        for (id, new_phi) in updates {
            if let Some(cell) = self.tree.cells.get_mut(&id) {
                cell.values[0] = new_phi;
            }
        }
    }

    /// Return the IDs of all *interface cells*: leaf cells where φ changes
    /// sign among the cell and its face-neighbours.
    ///
    /// Equivalently: cells where `|φ| < delta` **and** at least one neighbour
    /// has the opposite sign of φ.
    pub fn interface_cells(&self) -> Vec<CellId> {
        let snap = self.phi_snapshot();
        let mut result = Vec::new();

        for (&id, &phi_c) in &snap {
            if phi_c.abs() >= self.delta {
                continue;
            }
            let nbrs = self.tree.neighbors_of(id);
            let crosses_zero = nbrs.iter().any(|&nbr| {
                let phi_n = Self::get_phi_or_ancestor(&snap, &self.tree, nbr);
                phi_c * phi_n < 0.0
            });
            if crosses_zero {
                result.push(id);
            }
        }
        result
    }

    /// Compute the first-order centred gradient `(∂φ/∂x, ∂φ/∂y)` at a leaf cell.
    pub fn gradient(&self, id: CellId) -> Option<(f64, f64)> {
        let snap = self.phi_snapshot();
        let phi_c = *snap.get(&id)?;
        let lv = id.level();
        let dx = self.tree.dx_at_level(lv);
        let dy = self.tree.dy_at_level(lv);

        let phi_xm = self.neighbour_phi(id, &snap, -1, 0);
        let phi_xp = self.neighbour_phi(id, &snap, 1, 0);
        let phi_ym = self.neighbour_phi(id, &snap, 0, -1);
        let phi_yp = self.neighbour_phi(id, &snap, 0, 1);

        // Use one-sided differences at the boundary (where Neumann returns phi_c)
        let gx = if (phi_xm - phi_c).abs() < 1e-100 && (phi_xp - phi_c).abs() < 1e-100 {
            0.0
        } else {
            (phi_xp - phi_xm) / (2.0 * dx)
        };
        let gy = if (phi_ym - phi_c).abs() < 1e-100 && (phi_yp - phi_c).abs() < 1e-100 {
            0.0
        } else {
            (phi_yp - phi_ym) / (2.0 * dy)
        };
        Some((gx, gy))
    }

    /// Return the minimum cell spacing (finest leaf) in x.
    fn min_dx(&self) -> f64 {
        self.tree
            .leaves()
            .iter()
            .map(|id| self.tree.dx_at_level(id.level()))
            .fold(f64::INFINITY, f64::min)
    }

    /// Statistics: number of narrow-band leaf cells.
    pub fn narrow_band_count(&self) -> usize {
        self.tree
            .leaves()
            .iter()
            .filter(|&&id| {
                self.tree
                    .cells
                    .get(&id)
                    .map(|c| c.values[0].abs() < self.delta)
                    .unwrap_or(false)
            })
            .count()
    }

    /// Retrieve the `CellData` of a leaf.
    pub fn cell_data(&self, id: CellId) -> Option<&CellData> {
        self.tree.cells.get(&id)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a uniform-level-k quad-tree over [−1,1]×[−1,1].
    fn uniform_tree(k: u8, n_vars: usize) -> QuadTree {
        let mut tree = QuadTree::new([-1.0, 1.0, -1.0, 1.0], k, n_vars);
        for _ in 0..k {
            let current_leaves: Vec<_> = tree.leaves();
            for id in current_leaves {
                tree.refine_cell(id);
            }
        }
        tree
    }

    #[test]
    fn test_level_set_circle_signs() {
        // Circle of radius 0.5 centered at (0,0): φ = |r| − 0.5
        let tree = uniform_tree(2, 1);
        let ls = LevelSet::new(tree, |x, y| (x * x + y * y).sqrt() - 0.5);
        // Cell at the origin should have φ < 0 (inside)
        let near_origin = ls
            .tree
            .leaves()
            .iter()
            .find(|&&id| {
                ls.tree
                    .cells
                    .get(&id)
                    .map(|c| {
                        let (cx, cy) = c.center(&ls.tree.domain);
                        cx.abs() < 0.25 && cy.abs() < 0.25
                    })
                    .unwrap_or(false)
            })
            .copied();
        if let Some(id) = near_origin {
            assert!(ls.phi(id).unwrap_or(0.0) < 0.0, "inside circle → φ < 0");
        }
        // Cell at corner (0.9,0.9) should have φ > 0 (outside)
        let far_corner = ls
            .tree
            .leaves()
            .iter()
            .find(|&&id| {
                ls.tree
                    .cells
                    .get(&id)
                    .map(|c| {
                        let (cx, cy) = c.center(&ls.tree.domain);
                        cx > 0.7 && cy > 0.7
                    })
                    .unwrap_or(false)
            })
            .copied();
        if let Some(id) = far_corner {
            assert!(ls.phi(id).unwrap_or(0.0) > 0.0, "outside circle → φ > 0");
        }
    }

    #[test]
    fn test_reinitialize_preserves_zero_set() {
        let tree = uniform_tree(3, 1);
        let mut ls = LevelSet::new(tree, |x, y| (x * x + y * y).sqrt() - 0.5);
        // Record interface cell positions before reinitialization
        let ifaces_before: Vec<_> = ls.interface_cells();
        assert!(!ifaces_before.is_empty(), "should have interface cells");

        ls.reinitialize(5);

        // After reinitialization the zero level-set should not move much
        // (check that interface cells still have small |φ|)
        let ifaces_after = ls.interface_cells();
        for id in &ifaces_after {
            let phi = ls.phi(*id).unwrap_or(f64::INFINITY);
            // Interface cells should have |φ| < 2*dx at the finest level
            let dx = ls.tree.dx_at_level(id.level());
            assert!(
                phi.abs() < 2.0 * dx + 1e-6,
                "interface cell has |φ|={} > 2*dx={} after reinit",
                phi.abs(),
                2.0 * dx
            );
        }
    }

    #[test]
    fn test_interface_cells_small_phi() {
        let tree = uniform_tree(3, 1);
        let ls = LevelSet::new(tree, |x, y| (x * x + y * y).sqrt() - 0.5);
        let ifaces = ls.interface_cells();
        assert!(!ifaces.is_empty(), "circle should produce interface cells");
        for id in &ifaces {
            let phi = ls.phi(*id).unwrap_or(f64::INFINITY);
            let dx = ls.tree.dx_at_level(id.level());
            assert!(
                phi.abs() < 2.0 * dx + 1e-6,
                "interface cell |φ|={} should be < 2*dx={}",
                phi.abs(),
                2.0 * dx
            );
        }
    }

    #[test]
    fn test_advect_translates_interface() {
        // Constant velocity field u=1, v=0: interface should translate right
        let tree = uniform_tree(3, 1);
        let mut ls = LevelSet::new(tree, |x, _y| x); // vertical interface at x=0
        let leaves = ls.tree.leaves();
        let dt = 0.05;
        // After 1 step with u=1 the zero-set should move from x=0 to x≈dt
        ls.advect(&|_, _, _| 1.0, &|_, _, _| 0.0, 0.0, dt);
        // Check that cells near x=dt have φ near 0
        let near_dt: Vec<_> = leaves
            .iter()
            .filter(|&&id| {
                ls.tree
                    .cells
                    .get(&id)
                    .map(|c| {
                        let (cx, _) = c.center(&ls.tree.domain);
                        (cx - dt).abs() < 0.15
                    })
                    .unwrap_or(false)
            })
            .collect();
        let any_small: bool = near_dt
            .iter()
            .any(|&&id| ls.phi(id).map(|phi| phi.abs() < 0.25).unwrap_or(false));
        assert!(any_small, "interface should have moved towards x=dt");
    }

    #[test]
    fn test_narrow_band_count() {
        // Use a deeper tree (level 4 → 16×16 = 256 leaves) and a small delta
        // so the narrow band does not span the entire domain.
        let tree = uniform_tree(4, 1);
        let mut ls = LevelSet::new(tree, |x, y| (x * x + y * y).sqrt() - 0.5);
        // Override delta to 3 × finest cell size (level-4 dx = 2/16 = 0.125)
        let fine_dx = ls.tree.dx_at_level(4);
        ls.set_delta(3.0 * fine_dx);
        let nb = ls.narrow_band_count();
        assert!(nb > 0, "should have narrow-band cells around the circle");
        let total = ls.tree.leaves().len();
        assert!(
            nb < total,
            "narrow band ({nb}) should be smaller than total leaves ({total})"
        );
    }

    #[test]
    fn test_sign_smooth_properties() {
        // sign(+) > 0, sign(-) < 0, sign(0) = 0
        assert!(sign_smooth(1.0, 0.1) > 0.0);
        assert!(sign_smooth(-1.0, 0.1) < 0.0);
        assert_eq!(sign_smooth(0.0, 0.1), 0.0);
    }
}
