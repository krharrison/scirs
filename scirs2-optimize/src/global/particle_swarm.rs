//! Particle Swarm Optimization (PSO) algorithm for global optimization
//!
//! PSO is a population-based stochastic optimization algorithm inspired by
//! the social behavior of bird flocking or fish schooling. Each particle
//! moves through the search space with a velocity influenced by its own
//! best position and the global best position found by the swarm.
//!
//! Features:
//! - Standard PSO with inertia weight
//! - Constriction coefficient variant (Clerc & Kennedy)
//! - Ring and star topology for neighborhood communication
//! - Adaptive parameter scheduling (inertia weight, c1/c2)
//! - Stagnation detection with random re-initialization
//! - Velocity clamping with dynamic bounds

use crate::error::OptimizeError;
use crate::unconstrained::OptimizeResult;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{rng, Rng, RngExt, SeedableRng};

/// Swarm topology for neighborhood communication
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Topology {
    /// Star (global best): all particles communicate with all others
    Star,
    /// Ring: each particle communicates with its immediate neighbors
    Ring,
}

impl Default for Topology {
    fn default() -> Self {
        Topology::Star
    }
}

/// Velocity update strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VelocityStrategy {
    /// Standard inertia weight formulation
    InertiaWeight,
    /// Clerc's constriction coefficient
    Constriction,
}

impl Default for VelocityStrategy {
    fn default() -> Self {
        VelocityStrategy::InertiaWeight
    }
}

/// Options for Particle Swarm Optimization
#[derive(Debug, Clone)]
pub struct ParticleSwarmOptions {
    /// Number of particles in the swarm
    pub swarm_size: usize,
    /// Maximum number of iterations
    pub maxiter: usize,
    /// Cognitive parameter (attraction to personal best)
    pub c1: f64,
    /// Social parameter (attraction to global best)
    pub c2: f64,
    /// Inertia weight (used in InertiaWeight strategy)
    pub w: f64,
    /// Minimum velocity fraction of range
    pub vmin: f64,
    /// Maximum velocity fraction of range
    pub vmax: f64,
    /// Tolerance for convergence
    pub tol: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Whether to use adaptive parameters (linearly decrease w, adapt c1/c2)
    pub adaptive: bool,
    /// Swarm topology
    pub topology: Topology,
    /// Velocity update strategy
    pub velocity_strategy: VelocityStrategy,
    /// Number of stagnation iterations before re-initialization of worst particles
    pub stagnation_limit: usize,
    /// Fraction of particles to re-initialize on stagnation
    pub reinit_fraction: f64,
    /// Ring neighborhood size (for Ring topology, must be odd)
    pub ring_neighbors: usize,
}

impl Default for ParticleSwarmOptions {
    fn default() -> Self {
        Self {
            swarm_size: 50,
            maxiter: 500,
            c1: 2.0,
            c2: 2.0,
            w: 0.9,
            vmin: -0.5,
            vmax: 0.5,
            tol: 1e-8,
            seed: None,
            adaptive: false,
            topology: Topology::Star,
            velocity_strategy: VelocityStrategy::InertiaWeight,
            stagnation_limit: 20,
            reinit_fraction: 0.2,
            ring_neighbors: 3,
        }
    }
}

/// Bounds for variables
pub type Bounds = Vec<(f64, f64)>;

/// Particle in the swarm
#[derive(Debug, Clone)]
struct Particle {
    /// Current position
    position: Array1<f64>,
    /// Current velocity
    velocity: Array1<f64>,
    /// Personal best position
    best_position: Array1<f64>,
    /// Personal best value
    best_value: f64,
}

/// Particle Swarm Optimization solver
pub struct ParticleSwarm<F>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
{
    func: F,
    bounds: Bounds,
    options: ParticleSwarmOptions,
    ndim: usize,
    particles: Vec<Particle>,
    global_best_position: Array1<f64>,
    global_best_value: f64,
    rng: StdRng,
    nfev: usize,
    iteration: usize,
    inertia_weight: f64,
    /// Constriction coefficient (chi)
    constriction_chi: f64,
    /// Counter for stagnation detection (iterations without global improvement)
    stagnation_counter: usize,
    /// Previous global best for stagnation detection
    prev_global_best: f64,
}

impl<F> ParticleSwarm<F>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
{
    /// Create new Particle Swarm Optimization solver
    pub fn new(func: F, bounds: Bounds, options: ParticleSwarmOptions) -> Self {
        let ndim = bounds.len();
        let seed = options.seed.unwrap_or_else(|| rng().random());
        let mut rng_gen = StdRng::seed_from_u64(seed);

        // Compute constriction coefficient (Clerc & Kennedy 2002)
        let phi = options.c1 + options.c2;
        let constriction_chi = if phi > 4.0 {
            2.0 / (phi - 2.0 + (phi * phi - 4.0 * phi).sqrt()).abs()
        } else {
            0.7298 // Default constriction factor
        };

        // Initialize particles
        let mut particles = Vec::with_capacity(options.swarm_size);
        let mut global_best_position = Array1::zeros(ndim);
        let mut global_best_value = f64::INFINITY;
        let mut nfev = 0;

        for _ in 0..options.swarm_size {
            // Random initial position within bounds
            let mut position = Array1::zeros(ndim);
            let mut velocity = Array1::zeros(ndim);

            for j in 0..ndim {
                let (lb, ub) = bounds[j];
                position[j] = rng_gen.random_range(lb..ub);
                velocity[j] = rng_gen.random_range(options.vmin..options.vmax) * (ub - lb);
            }

            // Evaluate initial position
            let value = func(&position.view());
            nfev += 1;

            // Update global best if necessary
            if value < global_best_value {
                global_best_value = value;
                global_best_position = position.clone();
            }

            particles.push(Particle {
                position: position.clone(),
                velocity,
                best_position: position,
                best_value: value,
            });
        }

        Self {
            func,
            bounds,
            options: options.clone(),
            ndim,
            particles,
            global_best_position,
            global_best_value,
            rng: rng_gen,
            nfev,
            iteration: 0,
            inertia_weight: options.w,
            constriction_chi,
            stagnation_counter: 0,
            prev_global_best: global_best_value,
        }
    }

    /// Update the inertia weight adaptively
    fn update_inertia_weight(&mut self) {
        if self.options.adaptive {
            // Linear decrease from w to 0.4
            let w_max = self.options.w;
            let w_min = 0.4;
            let progress = self.iteration as f64 / self.options.maxiter as f64;
            self.inertia_weight = w_max - (w_max - w_min) * progress;
        }
    }

    /// Get adaptive cognitive/social parameters
    fn adaptive_coefficients(&self) -> (f64, f64) {
        if !self.options.adaptive {
            return (self.options.c1, self.options.c2);
        }

        // Adaptive: c1 decreases from 2.5 to 0.5, c2 increases from 0.5 to 2.5
        let progress = self.iteration as f64 / self.options.maxiter as f64;
        let c1 = 2.5 - 2.0 * progress;
        let c2 = 0.5 + 2.0 * progress;
        (c1, c2)
    }

    /// Get neighborhood best for a particle using the configured topology
    fn neighborhood_best(&self, particle_idx: usize) -> &Array1<f64> {
        match self.options.topology {
            Topology::Star => &self.global_best_position,
            Topology::Ring => {
                let n = self.particles.len();
                let half_neighbors = self.options.ring_neighbors / 2;
                let mut best_idx = particle_idx;
                let mut best_val = self.particles[particle_idx].best_value;

                for offset in 1..=half_neighbors {
                    // Left neighbor (wrapping)
                    let left = if particle_idx >= offset {
                        particle_idx - offset
                    } else {
                        n - (offset - particle_idx)
                    };
                    if self.particles[left].best_value < best_val {
                        best_val = self.particles[left].best_value;
                        best_idx = left;
                    }

                    // Right neighbor (wrapping)
                    let right = (particle_idx + offset) % n;
                    if self.particles[right].best_value < best_val {
                        best_val = self.particles[right].best_value;
                        best_idx = right;
                    }
                }

                &self.particles[best_idx].best_position
            }
        }
    }

    /// Update particle velocity and position
    fn update_particle(&mut self, idx: usize) {
        let (c1, c2) = self.adaptive_coefficients();
        let nbest = self.neighborhood_best(idx).clone();

        let particle = &mut self.particles[idx];

        // Update velocity
        match self.options.velocity_strategy {
            VelocityStrategy::InertiaWeight => {
                for j in 0..self.ndim {
                    let r1 = self.rng.random_range(0.0..1.0);
                    let r2 = self.rng.random_range(0.0..1.0);

                    let cognitive = c1 * r1 * (particle.best_position[j] - particle.position[j]);
                    let social = c2 * r2 * (nbest[j] - particle.position[j]);

                    particle.velocity[j] =
                        self.inertia_weight * particle.velocity[j] + cognitive + social;
                }
            }
            VelocityStrategy::Constriction => {
                for j in 0..self.ndim {
                    let r1 = self.rng.random_range(0.0..1.0);
                    let r2 = self.rng.random_range(0.0..1.0);

                    let cognitive = c1 * r1 * (particle.best_position[j] - particle.position[j]);
                    let social = c2 * r2 * (nbest[j] - particle.position[j]);

                    particle.velocity[j] =
                        self.constriction_chi * (particle.velocity[j] + cognitive + social);
                }
            }
        }

        // Clamp velocity
        for j in 0..self.ndim {
            let (lb, ub) = self.bounds[j];
            let range = ub - lb;
            let v_max = self.options.vmax * range;
            let v_min = self.options.vmin * range;
            particle.velocity[j] = particle.velocity[j].max(v_min).min(v_max);
        }

        // Update position
        for j in 0..self.ndim {
            particle.position[j] += particle.velocity[j];

            // Enforce bounds with reflection
            let (lb, ub) = self.bounds[j];
            if particle.position[j] < lb {
                particle.position[j] = lb + (lb - particle.position[j]).min(ub - lb);
                particle.velocity[j] *= -0.5; // Partial reflection
            } else if particle.position[j] > ub {
                particle.position[j] = ub - (particle.position[j] - ub).min(ub - lb);
                particle.velocity[j] *= -0.5; // Partial reflection
            }

            // Final clamping to ensure bounds
            particle.position[j] = particle.position[j].max(lb).min(ub);
        }

        // Evaluate new position
        let value = (self.func)(&particle.position.view());
        self.nfev += 1;

        // Update personal best
        if value < particle.best_value {
            particle.best_value = value;
            particle.best_position = particle.position.clone();

            // Update global best
            if value < self.global_best_value {
                self.global_best_value = value;
                self.global_best_position = particle.position.clone();
            }
        }
    }

    /// Check convergence criterion
    fn check_convergence(&self) -> bool {
        // Check if all particles have converged to the same region
        let mut max_distance: f64 = 0.0;

        for particle in &self.particles {
            let distance = (&particle.position - &self.global_best_position)
                .mapv(|x| x.abs())
                .sum();
            max_distance = max_distance.max(distance);
        }

        max_distance < self.options.tol
    }

    /// Detect and handle stagnation by re-initializing worst particles
    fn handle_stagnation(&mut self) {
        // Check if global best improved
        let improvement = self.prev_global_best - self.global_best_value;
        if improvement.abs() < 1e-15 * self.global_best_value.abs().max(1.0) {
            self.stagnation_counter += 1;
        } else {
            self.stagnation_counter = 0;
        }
        self.prev_global_best = self.global_best_value;

        // If stagnated too long, re-initialize a fraction of worst particles
        if self.stagnation_counter >= self.options.stagnation_limit {
            self.stagnation_counter = 0;

            let n_reinit =
                (self.options.reinit_fraction * self.particles.len() as f64).ceil() as usize;
            let n_reinit = n_reinit.max(1).min(self.particles.len());

            // Sort particles by personal best value (worst first)
            let mut indices: Vec<usize> = (0..self.particles.len()).collect();
            indices.sort_by(|&a, &b| {
                self.particles[b]
                    .best_value
                    .partial_cmp(&self.particles[a].best_value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Re-initialize worst particles
            for &idx in indices.iter().take(n_reinit) {
                let particle = &mut self.particles[idx];
                for j in 0..self.ndim {
                    let (lb, ub) = self.bounds[j];
                    particle.position[j] = self.rng.random_range(lb..ub);
                    particle.velocity[j] =
                        self.rng.random_range(self.options.vmin..self.options.vmax) * (ub - lb);
                }
                let value = (self.func)(&particle.position.view());
                self.nfev += 1;
                particle.best_position = particle.position.clone();
                particle.best_value = value;

                if value < self.global_best_value {
                    self.global_best_value = value;
                    self.global_best_position = particle.position.clone();
                }
            }
        }
    }

    /// Run one iteration of the algorithm
    fn step(&mut self) -> bool {
        self.iteration += 1;
        self.update_inertia_weight();

        // Update all particles
        for i in 0..self.options.swarm_size {
            self.update_particle(i);
        }

        // Handle stagnation
        self.handle_stagnation();

        self.check_convergence()
    }

    /// Run the particle swarm optimization algorithm
    pub fn run(&mut self) -> OptimizeResult<f64> {
        let mut converged = false;

        for _ in 0..self.options.maxiter {
            converged = self.step();

            if converged {
                break;
            }
        }

        OptimizeResult {
            x: self.global_best_position.clone(),
            fun: self.global_best_value,
            nfev: self.nfev,
            func_evals: self.nfev,
            nit: self.iteration,
            success: converged,
            message: if converged {
                "Optimization converged successfully"
            } else {
                "Maximum number of iterations reached"
            }
            .to_string(),
            ..Default::default()
        }
    }

    /// Get the current global best value
    pub fn best_value(&self) -> f64 {
        self.global_best_value
    }

    /// Get the current global best position
    pub fn best_position(&self) -> &Array1<f64> {
        &self.global_best_position
    }

    /// Get the number of function evaluations
    pub fn function_evaluations(&self) -> usize {
        self.nfev
    }
}

/// Perform global optimization using particle swarm optimization
///
/// # Arguments
///
/// * `func` - Objective function to minimize
/// * `bounds` - Variable bounds as Vec<(lower, upper)>
/// * `options` - PSO configuration options
///
/// # Returns
///
/// `OptimizeResult<f64>` with the optimization result
#[allow(dead_code)]
pub fn particle_swarm<F>(
    func: F,
    bounds: Bounds,
    options: Option<ParticleSwarmOptions>,
) -> Result<OptimizeResult<f64>, OptimizeError>
where
    F: Fn(&ArrayView1<f64>) -> f64 + Clone,
{
    if bounds.is_empty() {
        return Err(OptimizeError::InvalidInput(
            "Bounds must not be empty".to_string(),
        ));
    }
    for (i, &(lb, ub)) in bounds.iter().enumerate() {
        if lb >= ub {
            return Err(OptimizeError::InvalidInput(format!(
                "Lower bound must be less than upper bound for dimension {} ({} >= {})",
                i, lb, ub
            )));
        }
    }

    let options = options.unwrap_or_default();
    if options.swarm_size == 0 {
        return Err(OptimizeError::InvalidInput(
            "Swarm size must be > 0".to_string(),
        ));
    }

    let mut solver = ParticleSwarm::new(func, bounds, options);
    Ok(solver.run())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sphere function: sum(x_i^2), minimum at origin
    fn sphere(x: &ArrayView1<f64>) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    /// Rastrigin function: multimodal test function
    fn rastrigin(x: &ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;
        10.0 * n
            + x.iter()
                .map(|xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                .sum::<f64>()
    }

    /// Rosenbrock function
    fn rosenbrock(x: &ArrayView1<f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() - 1 {
            sum += 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2);
        }
        sum
    }

    #[test]
    fn test_pso_sphere_2d() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let mut opts = ParticleSwarmOptions::default();
        opts.seed = Some(42);
        opts.swarm_size = 30;
        opts.maxiter = 200;

        let result = particle_swarm(sphere, bounds, Some(opts));
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert!(
            res.fun < 0.01,
            "Sphere function should be near 0, got {}",
            res.fun
        );
    }

    #[test]
    fn test_pso_sphere_adaptive() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let mut opts = ParticleSwarmOptions::default();
        opts.seed = Some(123);
        opts.swarm_size = 40;
        opts.maxiter = 300;
        opts.adaptive = true;

        let result = particle_swarm(sphere, bounds, Some(opts));
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert!(
            res.fun < 0.01,
            "Adaptive PSO should find sphere min, got {}",
            res.fun
        );
    }

    #[test]
    fn test_pso_constriction_coefficient() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let mut opts = ParticleSwarmOptions::default();
        opts.seed = Some(99);
        opts.swarm_size = 40;
        opts.maxiter = 300;
        opts.velocity_strategy = VelocityStrategy::Constriction;

        let result = particle_swarm(sphere, bounds, Some(opts));
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert!(
            res.fun < 0.1,
            "Constriction PSO should find near-optimal, got {}",
            res.fun
        );
    }

    #[test]
    fn test_pso_ring_topology() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let mut opts = ParticleSwarmOptions::default();
        opts.seed = Some(55);
        opts.swarm_size = 30;
        opts.maxiter = 300;
        opts.topology = Topology::Ring;
        opts.ring_neighbors = 5;

        let result = particle_swarm(sphere, bounds, Some(opts));
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert!(
            res.fun < 0.1,
            "Ring topology should find near-optimal, got {}",
            res.fun
        );
    }

    #[test]
    fn test_pso_rastrigin() {
        // Rastrigin has many local minima; PSO should find near-global
        let bounds = vec![(-5.12, 5.12), (-5.12, 5.12)];
        let mut opts = ParticleSwarmOptions::default();
        opts.seed = Some(42);
        opts.swarm_size = 60;
        opts.maxiter = 500;
        opts.adaptive = true;

        let result = particle_swarm(rastrigin, bounds, Some(opts));
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        // Rastrigin global min is 0 at origin
        assert!(
            res.fun < 5.0,
            "PSO should find reasonable Rastrigin value, got {}",
            res.fun
        );
    }

    #[test]
    fn test_pso_rosenbrock() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let mut opts = ParticleSwarmOptions::default();
        opts.seed = Some(42);
        opts.swarm_size = 50;
        opts.maxiter = 500;
        opts.adaptive = true;

        let result = particle_swarm(rosenbrock, bounds, Some(opts));
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        // Rosenbrock min is 0 at (1, 1)
        assert!(
            res.fun < 5.0,
            "PSO should find reasonable Rosenbrock value, got {}",
            res.fun
        );
    }

    #[test]
    fn test_pso_invalid_bounds() {
        // Empty bounds
        let result = particle_swarm(sphere, vec![], None);
        assert!(result.is_err());

        // Lower >= upper
        let bounds = vec![(5.0, 2.0)];
        let result = particle_swarm(sphere, bounds, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_pso_invalid_swarm_size() {
        let bounds = vec![(-5.0, 5.0)];
        let mut opts = ParticleSwarmOptions::default();
        opts.swarm_size = 0;
        let result = particle_swarm(sphere, bounds, Some(opts));
        assert!(result.is_err());
    }

    #[test]
    fn test_pso_convergence_detection() {
        // Simple 1D function with very tight tolerance
        let bounds = vec![(-1.0, 1.0)];
        let mut opts = ParticleSwarmOptions::default();
        opts.seed = Some(42);
        opts.swarm_size = 20;
        opts.maxiter = 1000;
        opts.tol = 1e-3; // Relaxed tolerance for convergence detection

        let func = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] };
        let result = particle_swarm(func, bounds, Some(opts));
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert!(res.nit > 0);
        assert!(res.nfev > 0);
    }

    #[test]
    fn test_pso_stagnation_restart() {
        // Function with flat region that triggers stagnation
        let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];
        let mut opts = ParticleSwarmOptions::default();
        opts.seed = Some(42);
        opts.swarm_size = 20;
        opts.maxiter = 100;
        opts.stagnation_limit = 5;
        opts.reinit_fraction = 0.3;

        let result = particle_swarm(sphere, bounds, Some(opts));
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        // Should still find a reasonable solution
        assert!(res.fun < 1.0);
    }

    #[test]
    fn test_pso_high_dimensional() {
        let n = 10;
        let bounds: Vec<(f64, f64)> = vec![(-5.0, 5.0); n];
        let mut opts = ParticleSwarmOptions::default();
        opts.seed = Some(42);
        opts.swarm_size = 100;
        opts.maxiter = 500;
        opts.adaptive = true;

        let result = particle_swarm(sphere, bounds, Some(opts));
        assert!(result.is_ok());
        let res = result.expect("should succeed");
        assert!(
            res.fun < 1.0,
            "High-dim PSO should converge reasonably, got {}",
            res.fun
        );
    }

    #[test]
    fn test_pso_options_default() {
        let opts = ParticleSwarmOptions::default();
        assert_eq!(opts.swarm_size, 50);
        assert_eq!(opts.maxiter, 500);
        assert!((opts.c1 - 2.0).abs() < 1e-10);
        assert!((opts.c2 - 2.0).abs() < 1e-10);
        assert!((opts.w - 0.9).abs() < 1e-10);
        assert_eq!(opts.topology, Topology::Star);
        assert_eq!(opts.velocity_strategy, VelocityStrategy::InertiaWeight);
    }

    #[test]
    fn test_pso_solver_accessors() {
        let bounds = vec![(-5.0, 5.0)];
        let mut opts = ParticleSwarmOptions::default();
        opts.seed = Some(42);
        opts.swarm_size = 10;

        let solver = ParticleSwarm::new(sphere, bounds, opts);
        assert!(solver.best_value().is_finite());
        assert_eq!(solver.best_position().len(), 1);
        assert!(solver.function_evaluations() > 0);
    }
}
