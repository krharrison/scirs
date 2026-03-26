//! Time integration utilities and boundary condition handling for peridynamics.
//!
//! This module provides:
//!
//! * `velocity_verlet_step` — a standalone velocity-Verlet integrator for use
//!   with any particle state and force vector.
//! * `adaptive_timestep` — CFL-based time-step selector.
//! * `DirichletBC` — fixed-displacement boundary condition applied each step.
//! * `BodyForce` — uniform body acceleration applied to all (or a subset of) particles.
//! * `PdIntegrator` — a convenience wrapper that orchestrates boundary conditions,
//!   body forces, and time integration for the bond-based solver.
//! * Surface correction factors to account for boundary truncation of the
//!   peridynamic neighborhood.

use super::{bond_based::BondBasedSolver, PeridynamicState};

// ============================================================================
// Low-level velocity-Verlet integrator
// ============================================================================

/// Advance a `PeridynamicState` by one time step using the velocity-Verlet
/// (Störmer-Verlet) algorithm.
///
/// The algorithm is second-order accurate and time-reversible:
///
/// ```text
/// v_{n+½} = v_n + ½ Δt a_n
/// x_{n+1} = x_n + Δt v_{n+½}
/// a_{n+1} = f(x_{n+1}) / m      [caller must supply]
/// v_{n+1} = v_{n+½} + ½ Δt a_{n+1}
/// ```
///
/// # Arguments
///
/// * `state`      - Mutable reference to the particle state (updated in-place).
/// * `forces_old` - Force densities at the *start* of the step (f_n).
/// * `forces_new` - Force densities at the *end* of the step (f_{n+1}).
/// * `dt`         - Time step size.
/// * `masses`     - Particle masses (length = n_particles).
pub fn velocity_verlet_step(
    state: &mut PeridynamicState,
    forces_old: &[[f64; 3]],
    forces_new: &[[f64; 3]],
    dt: f64,
    masses: &[f64],
) {
    let n = state.n_particles;
    debug_assert_eq!(forces_old.len(), n);
    debug_assert_eq!(forces_new.len(), n);
    debug_assert_eq!(masses.len(), n);

    for i in 0..n {
        let inv_m = 1.0 / masses[i];

        // Half-step velocity update with old forces
        for d in 0..3 {
            state.velocities[i][d] += 0.5 * forces_old[i][d] * inv_m * dt;
        }

        // Full displacement update
        for d in 0..3 {
            state.displacements[i][d] += state.velocities[i][d] * dt;
        }

        // Half-step velocity update with new forces
        for d in 0..3 {
            state.velocities[i][d] += 0.5 * forces_new[i][d] * inv_m * dt;
        }
    }
}

// ============================================================================
// CFL-based adaptive timestep
// ============================================================================

/// Estimate the stable CFL timestep for a bond-based solver.
///
/// Delegates to `BondBasedSolver::cfl_timestep`.
pub fn adaptive_timestep(solver: &BondBasedSolver) -> f64 {
    solver.cfl_timestep()
}

// ============================================================================
// Boundary conditions
// ============================================================================

/// Dirichlet (fixed-displacement) boundary condition.
///
/// After each displacement update the listed particles are *projected* back to
/// their prescribed displacement.  Velocities of constrained particles are
/// zeroed to prevent drift.
#[derive(Debug, Clone)]
pub struct DirichletBC {
    /// Indices of constrained particles.
    pub particles: Vec<usize>,
    /// Prescribed displacement `[ux, uy, uz]`.
    pub displacement: [f64; 3],
}

impl DirichletBC {
    /// Construct a new Dirichlet BC.
    pub fn new(particles: Vec<usize>, displacement: [f64; 3]) -> Self {
        Self {
            particles,
            displacement,
        }
    }

    /// Apply the constraint to the given state.
    ///
    /// Constrained particles have their displacements overwritten and velocities
    /// zeroed.
    pub fn apply(&self, state: &mut PeridynamicState) {
        for &i in &self.particles {
            state.displacements[i] = self.displacement;
            state.velocities[i] = [0.0, 0.0, 0.0];
        }
    }
}

/// Uniform body-force density applied to a set of particles.
///
/// This models, e.g., gravity or a prescribed body acceleration.  The force
/// density vector is added to the peridynamic internal force after each
/// computation of `compute_forces`.
#[derive(Debug, Clone)]
pub struct BodyForce {
    /// Acceleration vector `[ax, ay, az]` (m/s²).
    pub acceleration: [f64; 3],
    /// Particle indices that receive this body force, or `None` for all particles.
    pub target_particles: Option<Vec<usize>>,
}

impl BodyForce {
    /// Apply to all particles.
    pub fn new_uniform(acceleration: [f64; 3]) -> Self {
        Self {
            acceleration,
            target_particles: None,
        }
    }

    /// Apply only to a subset of particles.
    pub fn new_targeted(acceleration: [f64; 3], particles: Vec<usize>) -> Self {
        Self {
            acceleration,
            target_particles: Some(particles),
        }
    }

    /// Add body force contributions to a force vector (in-place).
    ///
    /// # Arguments
    ///
    /// * `forces`  - Mutable force density vector to modify.
    /// * `masses`  - Particle masses.
    pub fn apply(&self, forces: &mut [[f64; 3]], masses: &[f64]) {
        let n = forces.len();
        match &self.target_particles {
            None => {
                for i in 0..n {
                    for d in 0..3 {
                        // Body force density = ρ * a → we add m_i * a to the force
                        forces[i][d] += masses[i] * self.acceleration[d];
                    }
                }
            }
            Some(indices) => {
                for &i in indices {
                    if i < n {
                        for d in 0..3 {
                            forces[i][d] += masses[i] * self.acceleration[d];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Surface correction factors
// ============================================================================

/// Compute surface correction factors β_i for each particle.
///
/// Near a free surface the peridynamic neighborhood is truncated, which
/// reduces the effective stiffness.  A first-order correction scales the
/// force contribution of each bond by the ratio of the full sphere volume to
/// the actual neighborhood volume:
///
/// ```text
/// β_i = V_sphere(δ) / Σ_j V_j       (sum over j in horizon of i)
/// ```
///
/// Interior particles have β_i ≈ 1; boundary particles have β_i > 1.
pub fn surface_correction_factors(
    neighbor_list: &crate::pde::peridynamics::neighbor_list::NeighborList,
    volumes: &[f64],
    horizon: f64,
) -> Vec<f64> {
    let n = neighbor_list.n_particles();
    let v_sphere = (4.0 / 3.0) * std::f64::consts::PI * horizon.powi(3);
    let mut beta = vec![1.0_f64; n];

    for i in 0..n {
        let v_actual: f64 = neighbor_list.neighbors[i].iter().map(|&j| volumes[j]).sum();
        if v_actual > f64::EPSILON {
            beta[i] = v_sphere / v_actual;
        }
    }

    beta
}

// ============================================================================
// PdIntegrator — high-level convenience wrapper
// ============================================================================

/// High-level time integrator for bond-based peridynamic simulations.
///
/// Combines the solver, optional boundary conditions, and optional body forces
/// into a single `step` method that advances the simulation while correctly
/// applying all constraints.
pub struct PdIntegrator {
    /// The underlying bond-based solver.
    pub solver: BondBasedSolver,
    /// Dirichlet boundary conditions applied each step.
    pub dirichlet_bcs: Vec<DirichletBC>,
    /// Body forces applied each step.
    pub body_forces: Vec<BodyForce>,
    /// Current simulation time.
    pub time: f64,
    /// Number of steps taken.
    pub step_count: usize,
}

impl PdIntegrator {
    /// Construct from a solver.
    pub fn new(solver: BondBasedSolver) -> Self {
        // Initialise prev_forces by computing forces at the initial configuration
        let mut integrator = Self {
            solver,
            dirichlet_bcs: Vec::new(),
            body_forces: Vec::new(),
            time: 0.0,
            step_count: 0,
        };
        // Bootstrap: compute initial forces so the first Verlet half-step is correct
        let f0 = integrator.solver.compute_forces();
        integrator.solver.prev_forces = f0;
        integrator
    }

    /// Add a Dirichlet boundary condition.
    pub fn add_dirichlet_bc(&mut self, bc: DirichletBC) {
        self.dirichlet_bcs.push(bc);
    }

    /// Add a body force.
    pub fn add_body_force(&mut self, bf: BodyForce) {
        self.body_forces.push(bf);
    }

    /// Advance the simulation by one time step `dt`.
    ///
    /// The integration sequence is:
    /// 1. Half-step velocity update (old forces).
    /// 2. Full displacement update.
    /// 3. Apply Dirichlet BCs (overwrite constrained displacements + zero velocities).
    /// 4. Compute new internal forces.
    /// 5. Add body forces.
    /// 6. Half-step velocity update (new forces).
    /// 7. Re-apply Dirichlet BCs (ensure constrained particles have zero velocity).
    pub fn step(&mut self, dt: f64) {
        let n = self.solver.state.n_particles;

        // --- 1. Half-step velocity with old forces ---
        {
            let forces_old = self.solver.prev_forces.clone();
            for i in 0..n {
                let inv_m = 1.0 / self.solver.masses[i];
                for d in 0..3 {
                    self.solver.state.velocities[i][d] += 0.5 * forces_old[i][d] * inv_m * dt;
                }
            }
        }

        // --- 2. Displacement update ---
        for i in 0..n {
            for d in 0..3 {
                self.solver.state.displacements[i][d] += self.solver.state.velocities[i][d] * dt;
            }
        }

        // --- 3. Apply Dirichlet BCs after displacement ---
        for bc in &self.dirichlet_bcs {
            bc.apply(&mut self.solver.state);
        }

        // --- 4. Compute new internal forces ---
        let mut forces_new = self.solver.compute_forces();

        // --- 5. Add body forces ---
        for bf in &self.body_forces {
            bf.apply(&mut forces_new, &self.solver.masses);
        }

        // --- 6. Second half-step velocity ---
        for i in 0..n {
            let inv_m = 1.0 / self.solver.masses[i];
            for d in 0..3 {
                self.solver.state.velocities[i][d] += 0.5 * forces_new[i][d] * inv_m * dt;
            }
        }

        // --- 7. Re-enforce Dirichlet BCs on velocities ---
        for bc in &self.dirichlet_bcs {
            bc.apply(&mut self.solver.state);
        }

        self.solver.prev_forces = forces_new;
        self.time += dt;
        self.step_count += 1;
    }

    /// Return the suggested stable timestep from the CFL condition.
    pub fn suggested_timestep(&self) -> f64 {
        adaptive_timestep(&self.solver)
    }

    /// Return the current damage field.
    pub fn damage_field(&self) -> Vec<f64> {
        self.solver.damage_field()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pde::peridynamics::{
        bond_based::{BondBasedConfig, BondBasedSolver},
        neighbor_list::NeighborList,
    };

    fn simple_grid_2x2x2(dx: f64) -> (Vec<[f64; 3]>, Vec<f64>) {
        let mut pos = Vec::new();
        for i in 0_usize..2 {
            for j in 0_usize..2 {
                for k in 0_usize..2 {
                    pos.push([i as f64 * dx, j as f64 * dx, k as f64 * dx]);
                }
            }
        }
        let vol = vec![dx * dx * dx; pos.len()];
        (pos, vol)
    }

    #[test]
    fn test_ten_steps_no_nan() {
        let (pos, vol) = simple_grid_2x2x2(0.05);
        let config = BondBasedConfig {
            horizon: 0.08,
            ..Default::default()
        };
        let solver = BondBasedSolver::new(pos, vol, config);
        let mut integrator = PdIntegrator::new(solver);
        let dt = integrator.suggested_timestep() * 0.5;
        let dt = if dt > 0.0 { dt } else { 1e-5 };

        for _ in 0..10 {
            integrator.step(dt);
        }

        for i in 0..integrator.solver.state.n_particles {
            let p = integrator.solver.state.current_position(i);
            for &c in &p {
                assert!(!c.is_nan(), "NaN in position after 10 steps");
            }
        }
    }

    #[test]
    fn test_dirichlet_bc_particles_dont_move() {
        let (pos, vol) = simple_grid_2x2x2(0.05);
        let config = BondBasedConfig {
            horizon: 0.08,
            ..Default::default()
        };
        let solver = BondBasedSolver::new(pos, vol, config);
        let mut integrator = PdIntegrator::new(solver);

        // Fix particle 0 at zero displacement
        integrator.add_dirichlet_bc(DirichletBC::new(vec![0], [0.0, 0.0, 0.0]));
        // Give particle 1 a body force to drive dynamics
        integrator.add_body_force(BodyForce::new_targeted([1.0, 0.0, 0.0], vec![1]));

        let dt = 1e-5;
        for _ in 0..20 {
            integrator.step(dt);
        }

        let u0 = integrator.solver.state.displacements[0];
        for (d, &c) in u0.iter().enumerate() {
            assert!(
                c.abs() < 1e-14,
                "constrained particle 0 displacement[{d}] = {c} should be 0"
            );
        }
    }

    #[test]
    fn test_surface_correction_factors_interior_close_to_one() {
        let dx = 0.1_f64;
        let horizon = 0.25_f64; // covers ~2.5 cell-widths
        let mut pos = Vec::new();
        let n = 5_usize;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    pos.push([i as f64 * dx, j as f64 * dx, k as f64 * dx]);
                }
            }
        }
        let vol = vec![dx * dx * dx; pos.len()];
        let nl = NeighborList::build(&pos, horizon);
        let beta = surface_correction_factors(&nl, &vol, horizon);

        // Interior particle at (0.2, 0.2, 0.2) → index 2*25+2*5+2 = 62
        let center = 2 * 25 + 2 * 5 + 2;
        assert!(
            beta[center] >= 0.5 && beta[center] <= 2.5,
            "interior beta[{center}] = {} seems unreasonable",
            beta[center]
        );
        // All factors must be positive
        for (i, &b) in beta.iter().enumerate() {
            assert!(b > 0.0, "beta[{i}] = {b} must be positive");
        }
    }

    #[test]
    fn test_damage_field_all_in_unit_interval() {
        let (pos, vol) = simple_grid_2x2x2(0.05);
        let config = BondBasedConfig {
            horizon: 0.08,
            critical_stretch: 0.005,
            ..Default::default()
        };
        let solver = BondBasedSolver::new(pos, vol, config);
        let mut integrator = PdIntegrator::new(solver);
        integrator.solver.state.displacements[0][0] = 0.05;
        let _ = integrator.solver.compute_forces();
        let damage = integrator.damage_field();
        for (i, &d) in damage.iter().enumerate() {
            assert!((0.0..=1.0).contains(&d), "damage[{i}] = {d} out of [0,1]");
        }
    }
}
