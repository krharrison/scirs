//! Ordinary state-based peridynamics (Silling et al. 2007).
//!
//! Unlike the bond-based formulation the state-based approach replaces pairwise
//! scalar forces with *force states* — functions that map a bond direction to a
//! scalar force density.  This removes the Poisson-ratio restriction and allows
//! general isotropic elastic materials with arbitrary bulk modulus κ and shear
//! modulus μ.
//!
//! # Ordinary state-based constitutive model
//!
//! ## Influence function
//!
//! ```text
//! ω(ξ) = δ / |ξ|      (conical weighting)
//! ```
//!
//! ## Weighted volume
//!
//! ```text
//! m_i = Σ_j ω(ξ_{ij}) |ξ_{ij}|² V_j
//! ```
//!
//! ## Dilatation
//!
//! ```text
//! θ_i = (3 / m_i) Σ_j ω(ξ_{ij}) |ξ_{ij}| e_{ij} V_j
//! ```
//!
//! where `e_{ij} = |y_j − y_i| − |ξ_{ij}|` is the scalar extension state.
//!
//! ## Force state (ordinary state-based)
//!
//! ```text
//! t_{ij} = [3 κ θ_i / m_i · ω(ξ_{ij}) |ξ_{ij}|]
//!        + [15 μ / m_i · ω(ξ_{ij}) (e_{ij} − θ_i |ξ_{ij}| / 3)]
//! ```
//!
//! The force density on particle `i` from bond `(i,j)` is:
//!
//! ```text
//! f_{ij} = (t_{ij} − t_{ji}) · n̂_{ij} · V_j
//! ```
//!
//! where the antisymmetry `t_{ij} − t_{ji}` ensures linear and angular momentum
//! conservation.

use super::{Bond, PeridynamicState};
use crate::pde::peridynamics::neighbor_list::NeighborList;

/// Configuration for the ordinary state-based peridynamic solver.
#[derive(Debug, Clone)]
pub struct StateBasedConfig {
    /// Peridynamic horizon δ.
    pub horizon: f64,
    /// Bulk modulus κ.
    pub bulk_modulus: f64,
    /// Shear modulus μ.
    pub shear_modulus: f64,
    /// Mass density ρ.
    pub density: f64,
    /// Critical scalar extension (relative to reference length) for bond failure.
    pub critical_stretch: f64,
}

impl Default for StateBasedConfig {
    fn default() -> Self {
        Self {
            horizon: 0.1,
            bulk_modulus: 1.0,
            shear_modulus: 0.5,
            density: 1.0,
            critical_stretch: 0.05,
        }
    }
}

/// Ordinary state-based peridynamic solver.
#[derive(Debug)]
pub struct StateBasedSolver {
    /// Current state of the particle system.
    pub state: PeridynamicState,
    /// Volume of each particle.
    pub volumes: Vec<f64>,
    /// Mass of each particle.
    pub masses: Vec<f64>,
    /// All bonds (stored once with i < j).
    pub bonds: Vec<Bond>,
    /// Initial bond count per particle.
    pub initial_bond_counts: Vec<usize>,
    /// Active bond count per particle.
    pub active_bond_counts: Vec<usize>,
    /// Solver configuration.
    pub config: StateBasedConfig,
    /// Neighbor list in the reference configuration.
    pub neighbor_list: NeighborList,
    /// Cached forces from the previous sub-step (Velocity-Verlet).
    prev_forces: Vec<[f64; 3]>,
}

impl StateBasedSolver {
    /// Create a solver from reference positions, volumes, and configuration.
    pub fn new(positions: Vec<[f64; 3]>, volumes: Vec<f64>, config: StateBasedConfig) -> Self {
        let n = positions.len();
        assert_eq!(volumes.len(), n);

        let neighbor_list = NeighborList::build(&positions, config.horizon);

        let mut bonds: Vec<Bond> = Vec::new();
        for i in 0..n {
            for &j in &neighbor_list.neighbors[i] {
                if j > i {
                    let xi = [
                        positions[j][0] - positions[i][0],
                        positions[j][1] - positions[i][1],
                        positions[j][2] - positions[i][2],
                    ];
                    bonds.push(Bond {
                        i,
                        j,
                        xi,
                        active: true,
                    });
                }
            }
        }

        let mut initial_bond_counts = vec![0usize; n];
        for b in &bonds {
            initial_bond_counts[b.i] += 1;
            initial_bond_counts[b.j] += 1;
        }
        let active_bond_counts = initial_bond_counts.clone();

        let masses: Vec<f64> = volumes.iter().map(|&v| config.density * v).collect();
        let state = PeridynamicState::new(positions);
        let prev_forces = vec![[0.0; 3]; n];

        Self {
            state,
            volumes,
            masses,
            bonds,
            initial_bond_counts,
            active_bond_counts,
            config,
            neighbor_list,
            prev_forces,
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Influence function ω(ξ) = δ / |ξ|.
    #[inline]
    fn influence(&self, xi_len: f64) -> f64 {
        if xi_len < f64::EPSILON {
            0.0
        } else {
            self.config.horizon / xi_len
        }
    }

    /// Compute weighted volumes m_i for all particles.
    fn weighted_volumes(&self) -> Vec<f64> {
        let n = self.state.n_particles;
        let mut m = vec![0.0_f64; n];
        for bond in &self.bonds {
            if !bond.active {
                continue;
            }
            let xi_len = bond.reference_length();
            let omega = self.influence(xi_len);
            let contrib = omega * xi_len * xi_len;
            m[bond.i] += contrib * self.volumes[bond.j];
            m[bond.j] += contrib * self.volumes[bond.i];
        }
        m
    }

    /// Compute the dilatation θ_i for all particles given current positions.
    ///
    /// ```text
    /// θ_i = (3 / m_i) Σ_j ω(ξ_{ij}) |ξ_{ij}| e_{ij} V_j
    /// ```
    fn dilatations(&self, cur_pos: &[[f64; 3]], weighted_vol: &[f64]) -> Vec<f64> {
        let n = self.state.n_particles;
        let mut theta = vec![0.0_f64; n];

        for bond in &self.bonds {
            if !bond.active {
                continue;
            }
            let i = bond.i;
            let j = bond.j;

            let xi_len = bond.reference_length();
            if xi_len < f64::EPSILON {
                continue;
            }
            let omega = self.influence(xi_len);

            let cy = [
                cur_pos[j][0] - cur_pos[i][0],
                cur_pos[j][1] - cur_pos[i][1],
                cur_pos[j][2] - cur_pos[i][2],
            ];
            let cy_len = (cy[0] * cy[0] + cy[1] * cy[1] + cy[2] * cy[2]).sqrt();
            let extension = cy_len - xi_len; // e_{ij}

            // Contribution to dilatation of i and j
            let base = omega * xi_len * extension;
            if weighted_vol[i] > f64::EPSILON {
                theta[i] += 3.0 * base * self.volumes[j] / weighted_vol[i];
            }
            if weighted_vol[j] > f64::EPSILON {
                theta[j] += 3.0 * base * self.volumes[i] / weighted_vol[j];
            }
        }

        theta
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Compute peridynamic body-force densities for all particles.
    ///
    /// Bond failure is checked using the relative extension criterion:
    ///
    /// ```text
    /// e_{ij} / |ξ_{ij}| > s_crit
    /// ```
    pub fn compute_forces(&mut self) -> Vec<[f64; 3]> {
        let n = self.state.n_particles;
        let cur_pos: Vec<[f64; 3]> = (0..n).map(|i| self.state.current_position(i)).collect();

        let m = self.weighted_volumes();
        let theta = self.dilatations(&cur_pos, &m);

        let mut forces = vec![[0.0_f64; 3]; n];

        // Per-bond extension states (for antisymmetric force computation)
        // We iterate mutably to break bonds
        for bond in self.bonds.iter_mut() {
            if !bond.active {
                continue;
            }
            let i = bond.i;
            let j = bond.j;

            let xi_len = bond.reference_length();
            if xi_len < f64::EPSILON {
                continue;
            }

            let cy = [
                cur_pos[j][0] - cur_pos[i][0],
                cur_pos[j][1] - cur_pos[i][1],
                cur_pos[j][2] - cur_pos[i][2],
            ];
            let cy_len = (cy[0] * cy[0] + cy[1] * cy[1] + cy[2] * cy[2]).sqrt();
            if cy_len < f64::EPSILON {
                continue;
            }

            let extension = cy_len - xi_len; // e_{ij}
            let stretch = extension / xi_len;

            // Check failure criterion
            if stretch > self.config.critical_stretch {
                bond.active = false;
                if self.active_bond_counts[i] > 0 {
                    self.active_bond_counts[i] -= 1;
                }
                if self.active_bond_counts[j] > 0 {
                    self.active_bond_counts[j] -= 1;
                }
                let init_i = self.initial_bond_counts[i];
                let init_j = self.initial_bond_counts[j];
                self.state.damage[i] = Self::compute_damage(self.active_bond_counts[i], init_i);
                self.state.damage[j] = Self::compute_damage(self.active_bond_counts[j], init_j);
                continue;
            }

            // Influence function ω(ξ) = δ / |ξ| — inlined to avoid borrow conflict
            let omega_ij = self.config.horizon / xi_len;

            // Force state scalars t_{ij} and t_{ji}
            // Using ordinary state-based formula
            let kappa = self.config.bulk_modulus;
            let mu = self.config.shear_modulus;

            let deviatoric_i = extension - theta[i] * xi_len / 3.0;
            let deviatoric_j = extension - theta[j] * xi_len / 3.0;

            let t_ij = if m[i] > f64::EPSILON {
                3.0 * kappa * theta[i] / m[i] * omega_ij * xi_len
                    + 15.0 * mu / m[i] * omega_ij * deviatoric_i
            } else {
                0.0
            };

            let t_ji = if m[j] > f64::EPSILON {
                3.0 * kappa * theta[j] / m[j] * omega_ij * xi_len
                    + 15.0 * mu / m[j] * omega_ij * deviatoric_j
            } else {
                0.0
            };

            // Unit bond direction in the deformed configuration
            let n_dir = [cy[0] / cy_len, cy[1] / cy_len, cy[2] / cy_len];

            // Antisymmetric force density: f_{ij} = (t_ij − t_ji) · n̂ · V_j
            let f_scalar_i = (t_ij - t_ji) * self.volumes[j];
            let f_scalar_j = -(t_ij - t_ji) * self.volumes[i];

            for d in 0..3 {
                forces[i][d] += f_scalar_i * n_dir[d];
                forces[j][d] += f_scalar_j * n_dir[d];
            }
        }

        forces
    }

    /// Advance the simulation by one time step using velocity-Verlet integration.
    pub fn step(&mut self, dt: f64) {
        let n = self.state.n_particles;

        let forces_old = self.prev_forces.clone();
        for i in 0..n {
            let inv_m = 1.0 / self.masses[i];
            for d in 0..3 {
                self.state.velocities[i][d] += 0.5 * forces_old[i][d] * inv_m * dt;
            }
        }

        for i in 0..n {
            for d in 0..3 {
                self.state.displacements[i][d] += self.state.velocities[i][d] * dt;
            }
        }

        let forces_new = self.compute_forces();

        for i in 0..n {
            let inv_m = 1.0 / self.masses[i];
            for d in 0..3 {
                self.state.velocities[i][d] += 0.5 * forces_new[i][d] * inv_m * dt;
            }
        }

        self.prev_forces = forces_new;
    }

    /// Return the damage field φ_i ∈ [0, 1].
    pub fn damage_field(&self) -> Vec<f64> {
        (0..self.state.n_particles)
            .map(|i| Self::compute_damage(self.active_bond_counts[i], self.initial_bond_counts[i]))
            .collect()
    }

    #[inline]
    fn compute_damage(active: usize, initial: usize) -> f64 {
        if initial == 0 {
            0.0
        } else {
            1.0 - (active as f64 / initial as f64)
        }
    }

    /// Compute the dilatation field θ_i using the current state.
    pub fn dilatation_field(&self) -> Vec<f64> {
        let n = self.state.n_particles;
        let cur_pos: Vec<[f64; 3]> = (0..n).map(|i| self.state.current_position(i)).collect();
        let m = self.weighted_volumes();
        self.dilatations(&cur_pos, &m)
    }

    /// Expose the (mutable) raw force-state scalar t_{ij} for bond (i,j).
    ///
    /// Returns `(t_ij, t_ji)` — useful for Newton's-3rd-law validation.
    pub fn force_state_pair(&self, bond_idx: usize) -> Option<(f64, f64)> {
        let bond = self.bonds.get(bond_idx)?;
        if !bond.active {
            return None;
        }
        let n = self.state.n_particles;
        let cur_pos: Vec<[f64; 3]> = (0..n).map(|k| self.state.current_position(k)).collect();
        let m = self.weighted_volumes();
        let theta = self.dilatations(&cur_pos, &m);

        let i = bond.i;
        let j = bond.j;
        let xi_len = bond.reference_length();
        if xi_len < f64::EPSILON {
            return None;
        }

        let cy = [
            cur_pos[j][0] - cur_pos[i][0],
            cur_pos[j][1] - cur_pos[i][1],
            cur_pos[j][2] - cur_pos[i][2],
        ];
        let cy_len = (cy[0] * cy[0] + cy[1] * cy[1] + cy[2] * cy[2]).sqrt();
        if cy_len < f64::EPSILON {
            return None;
        }

        let extension = cy_len - xi_len;
        let omega = self.influence(xi_len);
        let kappa = self.config.bulk_modulus;
        let mu = self.config.shear_modulus;

        let dev_i = extension - theta[i] * xi_len / 3.0;
        let dev_j = extension - theta[j] * xi_len / 3.0;

        let t_ij = if m[i] > f64::EPSILON {
            3.0 * kappa * theta[i] / m[i] * omega * xi_len + 15.0 * mu / m[i] * omega * dev_i
        } else {
            0.0
        };

        let t_ji = if m[j] > f64::EPSILON {
            3.0 * kappa * theta[j] / m[j] * omega * xi_len + 15.0 * mu / m[j] * omega * dev_j
        } else {
            0.0
        };

        Some((t_ij, t_ji))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regular_3d_grid(n: usize, dx: f64) -> (Vec<[f64; 3]>, Vec<f64>) {
        let mut pos = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    pos.push([i as f64 * dx, j as f64 * dx, k as f64 * dx]);
                }
            }
        }
        let vol = vec![dx * dx * dx; pos.len()];
        (pos, vol)
    }

    #[test]
    fn test_dilatation_zero_for_rigid_translation() {
        let (pos, vol) = regular_3d_grid(3, 0.05);
        let config = StateBasedConfig {
            horizon: 0.08,
            ..Default::default()
        };
        let mut solver = StateBasedSolver::new(pos, vol, config);
        // Apply a rigid body translation: u_i = (0.01, 0.02, 0.03) for all i
        for i in 0..solver.state.n_particles {
            solver.state.displacements[i] = [0.01, 0.02, 0.03];
        }
        let theta = solver.dilatation_field();
        for (i, &t) in theta.iter().enumerate() {
            assert!(
                t.abs() < 1e-10,
                "dilatation[{i}] = {t:.3e} should be zero for rigid translation"
            );
        }
    }

    #[test]
    fn test_dilatation_proportional_to_volumetric_strain() {
        // Uniform volumetric strain ε: u_i = ε * x_i
        // The dilatation should be ~ 3ε (volumetric strain in 3-D)
        let (pos, vol) = regular_3d_grid(3, 0.05);
        let config = StateBasedConfig {
            horizon: 0.08,
            critical_stretch: 1.0, // disable breakage
            ..Default::default()
        };
        let eps = 0.01_f64;
        let mut solver = StateBasedSolver::new(pos.clone(), vol, config);
        for i in 0..solver.state.n_particles {
            solver.state.displacements[i] = [eps * pos[i][0], eps * pos[i][1], eps * pos[i][2]];
        }
        let theta = solver.dilatation_field();
        // Interior particles should have θ ≈ 3ε (small deviations at boundaries)
        let interior_idx = pos.iter().position(|p| {
            (p[0] - 0.05).abs() < 1e-9 && (p[1] - 0.05).abs() < 1e-9 && (p[2] - 0.05).abs() < 1e-9
        });
        if let Some(idx) = interior_idx {
            let expected = 3.0 * eps;
            assert!(
                (theta[idx] - expected).abs() < 0.3 * expected.abs() + 1e-12,
                "interior dilatation {:.4e} not close to 3ε = {:.4e}",
                theta[idx],
                expected
            );
        }
    }

    #[test]
    fn test_force_state_antisymmetric() {
        // For an un-deformed state with uniform extension, t_ij and t_ji should
        // both be non-zero and the force should balance (momentum conservation)
        let (pos, vol) = regular_3d_grid(3, 0.05);
        let config = StateBasedConfig {
            horizon: 0.08,
            critical_stretch: 1.0,
            ..Default::default()
        };
        let mut solver = StateBasedSolver::new(pos, vol, config);
        // Apply a small stretching to particle 0
        solver.state.displacements[0][0] = 0.005;
        let forces = solver.compute_forces();
        // Total force on the system should be zero (linear momentum conservation)
        let total: [f64; 3] = forces.iter().fold([0.0; 3], |acc, f| {
            [acc[0] + f[0], acc[1] + f[1], acc[2] + f[2]]
        });
        let tol = 1e-8;
        assert!(
            total[0].abs() < tol && total[1].abs() < tol && total[2].abs() < tol,
            "sum of forces not zero: {total:?}"
        );
    }

    #[test]
    fn test_step_produces_non_nan_positions() {
        let (pos, vol) = regular_3d_grid(3, 0.05);
        let config = StateBasedConfig {
            horizon: 0.08,
            ..Default::default()
        };
        let mut solver = StateBasedSolver::new(pos, vol, config);
        let f0 = solver.compute_forces();
        solver.prev_forces = f0;

        for _ in 0..10 {
            solver.step(1e-4);
        }

        for i in 0..solver.state.n_particles {
            let p = solver.state.current_position(i);
            for &c in &p {
                assert!(!c.is_nan(), "NaN detected in position after 10 steps");
            }
        }
    }
}
