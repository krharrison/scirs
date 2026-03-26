//! Bond-based peridynamics (Silling 2000).
//!
//! In the bond-based formulation every pair of particles within the horizon
//! interacts through a central pairwise force.  The spring micro-modulus `c`
//! is chosen to reproduce the bulk elastic response of an isotropic linear
//! elastic solid with Young's modulus `E`, subject to the constraint that
//! Poisson's ratio is fixed at ν = 1/3 in 3-D.
//!
//! # Bond force
//!
//! For a bond with reference vector `ξ = x_j − x_i` the stretch is
//!
//! ```text
//! s = (|y_j − y_i| − |ξ|) / |ξ|
//! ```
//!
//! and the pairwise force density acting on particle `i` due to particle `j` is
//!
//! ```text
//! f_{ij} = c · s · (y_j − y_i) / |y_j − y_i| · V_j
//! ```
//!
//! A bond breaks irreversibly when `s > s₀`.

use std::f64::consts::PI;

use super::{Bond, FailureMode, PeridynamicMaterial, PeridynamicState};
use crate::pde::peridynamics::neighbor_list::NeighborList;

/// Configuration for the bond-based peridynamic solver.
#[derive(Debug, Clone)]
pub struct BondBasedConfig {
    /// Peridynamic horizon radius δ (metres).
    pub horizon: f64,
    /// Mass density ρ (kg/m³).
    pub density: f64,
    /// Young's modulus E (Pa).
    pub youngs_modulus: f64,
    /// Poisson's ratio (informational only; bond-based is fixed at 1/3 in 3-D).
    pub poissons_ratio: f64,
    /// Critical stretch s₀ at which bonds break.
    pub critical_stretch: f64,
}

impl Default for BondBasedConfig {
    fn default() -> Self {
        Self {
            horizon: 0.1,
            density: 1.0,
            youngs_modulus: 1.0,
            poissons_ratio: 1.0 / 3.0,
            critical_stretch: 0.04,
        }
    }
}

/// Isotropic linear elastic material for bond-based peridynamics.
///
/// The micro-modulus is calibrated so that the stored elastic energy in the
/// peridynamic model matches that of classical linear elasticity for a uniform
/// dilatational deformation:
///
/// ```text
/// c = 18 E / (π δ⁴)    (3-D, ν = 1/3)
/// ```
#[derive(Debug, Clone)]
pub struct IsotropicMaterial {
    /// Micro-modulus c (N/m⁶ in SI).
    pub micro_modulus: f64,
    /// Critical stretch s₀.
    pub critical_stretch: f64,
}

impl IsotropicMaterial {
    /// Construct from Young's modulus and horizon.
    ///
    /// Assumes 3-D geometry and ν = 1/3.
    pub fn new(youngs_modulus: f64, horizon: f64, critical_stretch: f64) -> Self {
        let delta4 = horizon * horizon * horizon * horizon;
        let micro_modulus = 18.0 * youngs_modulus / (PI * delta4);
        Self {
            micro_modulus,
            critical_stretch,
        }
    }

    /// Construct from fracture energy `G_c` (J/m²) using the energy-release
    /// calibration of Silling & Askari (2005):
    ///
    /// ```text
    /// s₀ = sqrt(5 π G_c / (9 E δ))
    /// ```
    pub fn from_fracture_energy(youngs_modulus: f64, horizon: f64, fracture_energy: f64) -> Self {
        let critical_stretch =
            (5.0 * PI * fracture_energy / (9.0 * youngs_modulus * horizon)).sqrt();
        Self::new(youngs_modulus, horizon, critical_stretch)
    }
}

impl PeridynamicMaterial for IsotropicMaterial {
    fn bond_force(&self, stretch: f64, _xi: [f64; 3]) -> f64 {
        self.micro_modulus * stretch
    }

    fn critical_stretch(&self) -> f64 {
        self.critical_stretch
    }
}

/// Bond-based peridynamic solver using velocity-Verlet time integration.
#[derive(Debug)]
pub struct BondBasedSolver {
    /// Current state of the particle system.
    pub state: PeridynamicState,
    /// Volume of each particle (assumed uniform or explicitly provided).
    pub volumes: Vec<f64>,
    /// Mass of each particle.
    pub masses: Vec<f64>,
    /// All bonds (pairs within horizon).
    pub bonds: Vec<Bond>,
    /// Number of initial (intact) bonds per particle.
    pub initial_bond_counts: Vec<usize>,
    /// Active bond counts per particle (updated after breakage).
    pub active_bond_counts: Vec<usize>,
    /// Material model.
    pub material: IsotropicMaterial,
    /// Solver configuration.
    pub config: BondBasedConfig,
    /// Neighbor list built from reference positions.
    pub neighbor_list: NeighborList,
    /// Forces from the previous sub-step (used for Velocity-Verlet).
    pub prev_forces: Vec<[f64; 3]>,
}

impl BondBasedSolver {
    /// Create a solver from a set of reference positions, particle volumes, and configuration.
    ///
    /// # Arguments
    ///
    /// * `positions` - Reference (initial) positions of all particles.
    /// * `volumes`   - Representative volume of each particle (e.g. Δx³ for a regular grid).
    /// * `config`    - Solver configuration (material parameters, horizon, etc.).
    pub fn new(positions: Vec<[f64; 3]>, volumes: Vec<f64>, config: BondBasedConfig) -> Self {
        let n = positions.len();
        assert_eq!(
            volumes.len(),
            n,
            "volumes must have the same length as positions"
        );

        let material = IsotropicMaterial::new(
            config.youngs_modulus,
            config.horizon,
            config.critical_stretch,
        );

        let neighbor_list = NeighborList::build(&positions, config.horizon);

        // Build bond list (each bond stored once with i < j)
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

        // Count initial bonds per particle (including both i and j contributions)
        let mut initial_bond_counts = vec![0usize; n];
        for bond in &bonds {
            initial_bond_counts[bond.i] += 1;
            initial_bond_counts[bond.j] += 1;
        }
        let active_bond_counts = initial_bond_counts.clone();

        let masses: Vec<f64> = volumes.iter().map(|&v| config.density * v).collect();
        let state = PeridynamicState::new(positions);
        let prev_forces = vec![[0.0, 0.0, 0.0]; n];

        Self {
            state,
            volumes,
            masses,
            bonds,
            initial_bond_counts,
            active_bond_counts,
            material,
            config,
            neighbor_list,
            prev_forces,
        }
    }

    /// Compute the peridynamic body-force density vector for each particle.
    ///
    /// For each active bond (i, j) the current stretch `s` is computed.
    /// If `s > s₀` the bond is broken permanently and the damage indices are
    /// updated.  Otherwise the force density contribution is accumulated.
    ///
    /// Returns a `Vec<[f64; 3]>` of force densities (force per unit volume)
    /// for each particle.
    pub fn compute_forces(&mut self) -> Vec<[f64; 3]> {
        let n = self.state.n_particles;
        let mut forces = vec![[0.0_f64; 3]; n];

        // Collect current positions (reference + displacement)
        let cur_pos: Vec<[f64; 3]> = (0..n).map(|i| self.state.current_position(i)).collect();

        for bond in self.bonds.iter_mut() {
            if !bond.active {
                continue;
            }

            let i = bond.i;
            let j = bond.j;

            // Current bond vector y_j − y_i
            let cy = [
                cur_pos[j][0] - cur_pos[i][0],
                cur_pos[j][1] - cur_pos[i][1],
                cur_pos[j][2] - cur_pos[i][2],
            ];
            let cy_len = (cy[0] * cy[0] + cy[1] * cy[1] + cy[2] * cy[2]).sqrt();

            if cy_len < f64::EPSILON {
                // Degenerate bond; skip without breaking (may be numerical artifact)
                continue;
            }

            let xi_len = bond.reference_length();
            if xi_len < f64::EPSILON {
                continue;
            }

            let stretch = (cy_len - xi_len) / xi_len;

            // Check failure criterion
            if stretch > self.material.critical_stretch() {
                bond.active = false;
                // Update active bond counts
                if self.active_bond_counts[i] > 0 {
                    self.active_bond_counts[i] -= 1;
                }
                if self.active_bond_counts[j] > 0 {
                    self.active_bond_counts[j] -= 1;
                }
                // Update damage field
                self.state.damage[i] =
                    Self::compute_damage(self.active_bond_counts[i], self.initial_bond_counts[i]);
                self.state.damage[j] =
                    Self::compute_damage(self.active_bond_counts[j], self.initial_bond_counts[j]);
                continue;
            }

            // Force density scalar
            let f_scalar = self.material.bond_force(stretch, bond.xi);

            // Unit direction vector
            let n_dir = [cy[0] / cy_len, cy[1] / cy_len, cy[2] / cy_len];

            // Volume of the *other* particle is the weighting factor
            let vj = self.volumes[j];
            let vi = self.volumes[i];

            // Force on particle i (from j): positive stretch → attractive
            forces[i][0] += f_scalar * n_dir[0] * vj;
            forces[i][1] += f_scalar * n_dir[1] * vj;
            forces[i][2] += f_scalar * n_dir[2] * vj;

            // Newton's 3rd law: force on j from i is opposite
            forces[j][0] -= f_scalar * n_dir[0] * vi;
            forces[j][1] -= f_scalar * n_dir[1] * vi;
            forces[j][2] -= f_scalar * n_dir[2] * vi;
        }

        forces
    }

    /// Advance the simulation by one time step `dt` using velocity-Verlet integration.
    pub fn step(&mut self, dt: f64) {
        let n = self.state.n_particles;

        // --- Half-step velocity update ---
        let forces_old = self.prev_forces.clone();
        for i in 0..n {
            let inv_m = 1.0 / self.masses[i];
            for d in 0..3 {
                self.state.velocities[i][d] += 0.5 * forces_old[i][d] * inv_m * dt;
            }
        }

        // --- Full displacement update ---
        for i in 0..n {
            for d in 0..3 {
                self.state.displacements[i][d] += self.state.velocities[i][d] * dt;
            }
        }

        // --- Compute new forces ---
        let forces_new = self.compute_forces();

        // --- Second half-step velocity update ---
        for i in 0..n {
            let inv_m = 1.0 / self.masses[i];
            for d in 0..3 {
                self.state.velocities[i][d] += 0.5 * forces_new[i][d] * inv_m * dt;
            }
        }

        self.prev_forces = forces_new;
    }

    /// Compute the damage index for a single particle.
    ///
    /// φ_i = 1 - (active bonds) / (initial bonds)
    #[inline]
    fn compute_damage(active: usize, initial: usize) -> f64 {
        if initial == 0 {
            0.0
        } else {
            1.0 - (active as f64 / initial as f64)
        }
    }

    /// Return the damage field: φ_i ∈ [0, 1] for each particle.
    pub fn damage_field(&self) -> Vec<f64> {
        (0..self.state.n_particles)
            .map(|i| Self::compute_damage(self.active_bond_counts[i], self.initial_bond_counts[i]))
            .collect()
    }

    /// Estimate the stable CFL time step.
    ///
    /// The critical time step for explicit integration of bond-based peridynamics
    /// is given by:
    ///
    /// ```text
    /// Δt_crit = sqrt(ρ / Σ_j c V_j / V_i)
    /// ```
    ///
    /// The factor 0.8 provides a safety margin.
    pub fn cfl_timestep(&self) -> f64 {
        let n = self.state.n_particles;
        let mut min_dt = f64::MAX;

        for i in 0..n {
            let mut sum_c_vj = 0.0_f64;
            // Accumulate spring contributions from all bonds containing particle i
            for bond in &self.bonds {
                if !bond.active {
                    continue;
                }
                let j = if bond.i == i {
                    Some(bond.j)
                } else if bond.j == i {
                    Some(bond.i)
                } else {
                    None
                };
                if let Some(jj) = j {
                    sum_c_vj += self.material.micro_modulus * self.volumes[jj];
                }
            }

            if sum_c_vj < f64::EPSILON {
                continue;
            }

            // ρ V_i / (c · ΣV_j)
            let rho_vi = self.config.density * self.volumes[i];
            let dt_i = (rho_vi / sum_c_vj).sqrt();
            if dt_i < min_dt {
                min_dt = dt_i;
            }
        }

        if min_dt == f64::MAX {
            0.0
        } else {
            0.8 * min_dt
        }
    }

    /// Check whether bond `(i, j)` is currently active.
    pub fn bond_active(&self, i: usize, j: usize) -> Option<bool> {
        self.bonds
            .iter()
            .find(|b| (b.i == i && b.j == j) || (b.i == j && b.j == i))
            .map(|b| b.active)
    }

    /// Classify the failure mode for bond `(i, j)`.
    pub fn failure_mode(&self, i: usize, j: usize) -> FailureMode {
        match self
            .bonds
            .iter()
            .find(|b| (b.i == i && b.j == j) || (b.i == j && b.j == i))
        {
            None => FailureMode::NoBond,
            Some(b) if !b.active => FailureMode::StretchExceeded,
            _ => {
                // Bond still active — no failure
                FailureMode::StretchExceeded // unreachable arm kept for completeness
            }
        }
    }

    /// Total number of active bonds remaining in the system.
    pub fn active_bond_count(&self) -> usize {
        self.bonds.iter().filter(|b| b.active).count()
    }

    /// Total kinetic energy of the system: Σ ½ m_i |v_i|².
    pub fn kinetic_energy(&self) -> f64 {
        (0..self.state.n_particles)
            .map(|i| {
                let v = &self.state.velocities[i];
                let v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                0.5 * self.masses[i] * v2
            })
            .sum()
    }

    /// Total strain energy stored in all active bonds.
    ///
    /// Each bond stores energy ½ c s² |ξ| V_i V_j (split symmetrically).
    pub fn strain_energy(&self) -> f64 {
        let cur_pos: Vec<[f64; 3]> = (0..self.state.n_particles)
            .map(|i| self.state.current_position(i))
            .collect();

        self.bonds
            .iter()
            .filter(|b| b.active)
            .map(|bond| {
                let cy = [
                    cur_pos[bond.j][0] - cur_pos[bond.i][0],
                    cur_pos[bond.j][1] - cur_pos[bond.i][1],
                    cur_pos[bond.j][2] - cur_pos[bond.i][2],
                ];
                let cy_len = (cy[0] * cy[0] + cy[1] * cy[1] + cy[2] * cy[2]).sqrt();
                let xi_len = bond.reference_length();
                if xi_len < f64::EPSILON || cy_len < f64::EPSILON {
                    return 0.0;
                }
                let stretch = (cy_len - xi_len) / xi_len;
                let vi = self.volumes[bond.i];
                let vj = self.volumes[bond.j];
                0.5 * self.material.micro_modulus * stretch * stretch * xi_len * vi * vj
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Construct a simple 2×2×2 regular grid of particles.
    fn simple_grid() -> (Vec<[f64; 3]>, Vec<f64>) {
        let mut pos = Vec::new();
        let mut vol = Vec::new();
        let dx = 0.05_f64;
        for i in 0_usize..2 {
            for j in 0_usize..2 {
                for k in 0_usize..2 {
                    pos.push([i as f64 * dx, j as f64 * dx, k as f64 * dx]);
                    vol.push(dx * dx * dx);
                }
            }
        }
        (pos, vol)
    }

    #[test]
    fn test_zero_displacement_zero_forces() {
        let (pos, vol) = simple_grid();
        let config = BondBasedConfig {
            horizon: 0.08,
            ..Default::default()
        };
        let mut solver = BondBasedSolver::new(pos, vol, config);
        let forces = solver.compute_forces();
        for f in &forces {
            for &component in f {
                assert!(
                    component.abs() < 1e-12,
                    "expected zero force, got {component}"
                );
            }
        }
    }

    #[test]
    fn test_uniform_stretch_consistent() {
        // Apply non-uniform displacement to particle 0 only.
        // All bonds involving particle 0 should be stretched and produce non-zero forces.
        let (pos, vol) = simple_grid();
        let config = BondBasedConfig {
            horizon: 0.08,
            ..Default::default()
        };
        let mut solver = BondBasedSolver::new(pos, vol, config);
        // Displace particle 0 by a moderate amount in x
        solver.state.displacements[0][0] = 0.005; // well below critical stretch
        let forces = solver.compute_forces();
        // Particle 0 should experience a restoring force in -x direction
        assert!(
            forces[0][0].abs() > 1e-12,
            "expected non-zero force on displaced particle, got {}",
            forces[0][0]
        );
        // Total force must still sum to zero (momentum conservation)
        let total_fx: f64 = forces.iter().map(|f| f[0]).sum();
        assert!(total_fx.abs() < 1e-10, "total force not zero: {total_fx}");
    }

    #[test]
    fn test_bond_breaks_when_stretch_exceeds_critical() {
        let (pos, vol) = simple_grid();
        let config = BondBasedConfig {
            horizon: 0.08,
            critical_stretch: 0.01, // very small threshold
            ..Default::default()
        };
        let mut solver = BondBasedSolver::new(pos, vol, config);
        // Apply a large displacement to particle 0 in x
        solver.state.displacements[0][0] = 0.1;
        let _ = solver.compute_forces();
        // Some bonds connected to particle 0 should now be broken
        let broken = solver
            .bonds
            .iter()
            .any(|b| !b.active && (b.i == 0 || b.j == 0));
        assert!(
            broken,
            "bonds connected to over-stretched particle should break"
        );
    }

    #[test]
    fn test_damage_increases_after_bond_breakage() {
        let (pos, vol) = simple_grid();
        let config = BondBasedConfig {
            horizon: 0.08,
            critical_stretch: 0.001, // extremely small
            ..Default::default()
        };
        let mut solver = BondBasedSolver::new(pos, vol, config);
        solver.state.displacements[0][0] = 0.05;
        let _ = solver.compute_forces();
        let damage = solver.damage_field();
        assert!(
            damage[0] > 0.0,
            "damage of over-stretched particle should be > 0, got {}",
            damage[0]
        );
    }

    #[test]
    fn test_momentum_conservation() {
        let (pos, vol) = simple_grid();
        let config = BondBasedConfig {
            horizon: 0.08,
            ..Default::default()
        };
        let mut solver = BondBasedSolver::new(pos, vol, config);
        // Give all particles a small random-ish displacement
        solver.state.displacements[0][0] = 0.002;
        solver.state.displacements[3][1] = -0.001;
        let forces = solver.compute_forces();
        let total: [f64; 3] = forces.iter().fold([0.0; 3], |acc, f| {
            [acc[0] + f[0], acc[1] + f[1], acc[2] + f[2]]
        });
        let tol = 1e-10;
        assert!(
            total[0].abs() < tol && total[1].abs() < tol && total[2].abs() < tol,
            "sum of forces not zero: {total:?}"
        );
    }

    #[test]
    fn test_cfl_timestep_positive_and_reasonable() {
        let (pos, vol) = simple_grid();
        let config = BondBasedConfig {
            horizon: 0.08,
            ..Default::default()
        };
        let solver = BondBasedSolver::new(pos, vol, config);
        let dt = solver.cfl_timestep();
        assert!(dt > 0.0, "CFL timestep should be positive, got {dt}");
        assert!(
            dt < 1.0,
            "CFL timestep should be < 1.0 for unit params, got {dt}"
        );
    }

    #[test]
    fn test_velocity_verlet_step_preserves_energy_approximately() {
        // With no bond breakage and a sufficiently small dt the total mechanical
        // energy should be nearly conserved over a single step.
        let (pos, vol) = simple_grid();
        let config = BondBasedConfig {
            horizon: 0.08,
            critical_stretch: 1.0, // disable breakage
            ..Default::default()
        };
        let mut solver = BondBasedSolver::new(pos, vol, config);
        // Give particle 0 a very small initial velocity so that we have KE
        solver.state.velocities[0][0] = 1e-4;
        let f0 = solver.compute_forces();
        solver.prev_forces = f0;

        // Use a very small dt (1 % of CFL or 1e-6 whichever is smaller)
        let cfl = solver.cfl_timestep();
        let dt = if cfl > 0.0 { cfl * 0.01_f64 } else { 1e-6_f64 };

        let ke0 = solver.kinetic_energy();
        let se0 = solver.strain_energy();
        let e0 = ke0 + se0;

        // Only one step — Velocity-Verlet is 2nd-order so per-step drift is O(dt³)
        solver.step(dt);

        let e1 = solver.kinetic_energy() + solver.strain_energy();
        // For a single tiny step the relative energy change should be negligible
        let abs_err = (e1 - e0).abs();
        // e0 may be tiny (only KE from tiny velocity); use absolute tolerance
        assert!(
            abs_err < (1e-15_f64).max(e0.abs() * 0.05),
            "energy changed by {abs_err:.3e} after one micro-step (e0 = {e0:.3e})"
        );
    }

    #[test]
    fn test_damage_field_bounded() {
        let (pos, vol) = simple_grid();
        let config = BondBasedConfig {
            horizon: 0.08,
            critical_stretch: 0.005,
            ..Default::default()
        };
        let mut solver = BondBasedSolver::new(pos, vol, config);
        solver.state.displacements[0][0] = 0.1;
        let _ = solver.compute_forces();
        let damage = solver.damage_field();
        for (i, &d) in damage.iter().enumerate() {
            assert!((0.0..=1.0).contains(&d), "damage[{i}] = {d} out of [0,1]");
        }
    }
}
