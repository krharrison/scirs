//! Port-Hamiltonian System Discretization
//!
//! Port-Hamiltonian (PH) systems describe open physical systems that exchange energy
//! with their environment. The continuous-time model is:
//!
//!   ẋ = (J - R) ∂H/∂x + Bu
//!   y = Bᵀ ∂H/∂x
//!
//! where J is the skew-symmetric interconnection matrix, R ≥ 0 is the dissipation matrix,
//! H is the Hamiltonian (energy), B is the input matrix, u is the input, and y is the output.
//!
//! This module provides structure-preserving discretization methods for mechanical PH systems
//! of the form:
//!   H(q, p) = pᵀ M⁻¹ p / 2 + V(q)
//!   q̇ = M⁻¹ p
//!   ṗ = -∂V/∂q - D M⁻¹ p + Bu

use crate::error::{IntegrateError, IntegrateResult};

/// Integration methods for Port-Hamiltonian systems
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhIntegrator {
    /// Störmer-Verlet: symplectic, 2nd order (recommended for conservative systems)
    StormerVerlet,
    /// Implicit midpoint rule: preserves quadratic invariants
    MidpointRule,
    /// Discrete gradient method: exactly energy-preserving
    DiscreteGradient,
    /// Cayley transform method: suitable for linear PH systems
    CayleyTransform,
}

/// Configuration for Port-Hamiltonian system integration
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct PortHamiltonianConfig {
    /// Integration method (default: StormerVerlet)
    pub integrator: PhIntegrator,
    /// Time step size (default: 0.01)
    pub dt: f64,
    /// Number of integration steps (default: 100)
    pub n_steps: usize,
    /// Whether to track Hamiltonian value at each step (default: true)
    pub energy_tracking: bool,
    /// Maximum fixed-point iterations for implicit methods (default: 10)
    pub max_iter: usize,
    /// Tolerance for implicit solver convergence (default: 1e-10)
    pub tol: f64,
}

impl Default for PortHamiltonianConfig {
    fn default() -> Self {
        Self {
            integrator: PhIntegrator::StormerVerlet,
            dt: 0.01,
            n_steps: 100,
            energy_tracking: true,
            max_iter: 10,
            tol: 1e-10,
        }
    }
}

/// Result of Port-Hamiltonian system integration
#[derive(Debug, Clone)]
pub struct PhResult {
    /// Time points
    pub times: Vec<f64>,
    /// State trajectory: `states[i]` = \[q_0,...,q_{n-1}, p_0,...,p_{n-1}\] at time i
    pub states: Vec<Vec<f64>>,
    /// Hamiltonian values at each time point
    pub energies: Vec<f64>,
    /// System outputs y = Bᵀ ∂H/∂x at each time
    pub outputs: Vec<Vec<f64>>,
    /// Energy drift |H(T) - H(0)|
    pub energy_drift: f64,
}

/// Port-Hamiltonian mechanical system
///
/// Models the system:
///   H(q, p) = pᵀ M⁻¹ p / 2 + V(q)
///   q̇ = M⁻¹ p
///   ṗ = -∂V/∂q - D M⁻¹ p + Bu
///   y = Bᵀ M⁻¹ p
pub struct PortHamiltonianSystem {
    /// Number of degrees of freedom
    pub n_dof: usize,
    /// Mass matrix M (n_dof × n_dof, row-major)
    pub mass: Vec<f64>,
    /// Potential energy function V(q)
    potential: Box<dyn Fn(&[f64]) -> f64 + Send>,
    /// Gradient of potential energy ∂V/∂q
    grad_potential: Box<dyn Fn(&[f64]) -> Vec<f64> + Send>,
    /// Damping matrix D (n_dof × n_dof, row-major)
    pub damping: Vec<f64>,
    /// Input matrix B (n_dof × n_inputs, row-major)
    pub input_matrix: Vec<f64>,
    /// Number of inputs
    pub n_inputs: usize,
}

impl PortHamiltonianSystem {
    /// Create a new mechanical Port-Hamiltonian system with no inputs
    ///
    /// # Arguments
    /// * `n_dof` - number of degrees of freedom
    /// * `mass` - mass matrix M (n_dof × n_dof, row-major)
    /// * `potential` - potential energy V(q)
    /// * `grad_potential` - gradient ∂V/∂q
    /// * `damping` - damping matrix D (n_dof × n_dof, row-major)
    pub fn new_mechanical(
        n_dof: usize,
        mass: Vec<f64>,
        potential: Box<dyn Fn(&[f64]) -> f64 + Send>,
        grad_potential: Box<dyn Fn(&[f64]) -> Vec<f64> + Send>,
        damping: Vec<f64>,
    ) -> Self {
        Self {
            n_dof,
            mass,
            potential,
            grad_potential,
            damping,
            input_matrix: vec![],
            n_inputs: 0,
        }
    }

    /// Create system with input port
    pub fn new_mechanical_with_input(
        n_dof: usize,
        mass: Vec<f64>,
        potential: Box<dyn Fn(&[f64]) -> f64 + Send>,
        grad_potential: Box<dyn Fn(&[f64]) -> Vec<f64> + Send>,
        damping: Vec<f64>,
        input_matrix: Vec<f64>,
        n_inputs: usize,
    ) -> Self {
        Self {
            n_dof,
            mass,
            potential,
            grad_potential,
            damping,
            input_matrix,
            n_inputs,
        }
    }

    /// Evaluate the Hamiltonian H(q, p) = pᵀ M⁻¹ p / 2 + V(q)
    pub fn hamiltonian(&self, q: &[f64], p: &[f64]) -> f64 {
        let kinetic = self.kinetic_energy(p);
        let potential = (self.potential)(q);
        kinetic + potential
    }

    /// Kinetic energy T(p) = pᵀ M⁻¹ p / 2
    fn kinetic_energy(&self, p: &[f64]) -> f64 {
        let minv_p = self.mass_inv_vec(p);
        let mut ke = 0.0;
        for i in 0..self.n_dof {
            ke += p[i] * minv_p[i];
        }
        ke * 0.5
    }

    /// Compute M⁻¹ v using Gauss elimination (handles diagonal and general cases)
    fn mass_inv_vec(&self, v: &[f64]) -> Vec<f64> {
        let n = self.n_dof;
        if n == 0 {
            return vec![];
        }
        // Check if diagonal
        let is_diagonal = self.is_diagonal_mass();
        if is_diagonal {
            let mut result = vec![0.0; n];
            for i in 0..n {
                let m_ii = self.mass[i * n + i];
                if m_ii.abs() > f64::EPSILON {
                    result[i] = v[i] / m_ii;
                }
            }
            return result;
        }
        // General case: Gaussian elimination with partial pivoting
        self.solve_linear_system(&self.mass, v)
    }

    /// Check if mass matrix is diagonal
    fn is_diagonal_mass(&self) -> bool {
        let n = self.n_dof;
        for i in 0..n {
            for j in 0..n {
                if i != j && self.mass[i * n + j].abs() > f64::EPSILON * 100.0 {
                    return false;
                }
            }
        }
        true
    }

    /// Solve Ax = b via Gaussian elimination with partial pivoting
    fn solve_linear_system(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        let n = self.n_dof;
        // Augmented matrix [A | b]
        let mut aug: Vec<f64> = vec![0.0; n * (n + 1)];
        for i in 0..n {
            for j in 0..n {
                aug[i * (n + 1) + j] = a[i * n + j];
            }
            aug[i * (n + 1) + n] = b[i];
        }
        // Forward elimination
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = aug[col * (n + 1) + col].abs();
            for row in (col + 1)..n {
                let val = aug[row * (n + 1) + col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }
            // Swap rows
            if max_row != col {
                for k in 0..=(n) {
                    aug.swap(col * (n + 1) + k, max_row * (n + 1) + k);
                }
            }
            let pivot = aug[col * (n + 1) + col];
            if pivot.abs() < f64::EPSILON {
                continue;
            }
            for row in (col + 1)..n {
                let factor = aug[row * (n + 1) + col] / pivot;
                for k in col..=(n) {
                    let val = aug[col * (n + 1) + k];
                    aug[row * (n + 1) + k] -= factor * val;
                }
            }
        }
        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = aug[i * (n + 1) + n];
            for j in (i + 1)..n {
                sum -= aug[i * (n + 1) + j] * x[j];
            }
            let diag = aug[i * (n + 1) + i];
            if diag.abs() > f64::EPSILON {
                x[i] = sum / diag;
            }
        }
        x
    }

    /// Matrix-vector product: y = A * x where A is n×n row-major
    fn mat_vec(&self, a: &[f64], x: &[f64]) -> Vec<f64> {
        let n = self.n_dof;
        let mut y = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                y[i] += a[i * n + j] * x[j];
            }
        }
        y
    }

    /// Integrate the Port-Hamiltonian system
    ///
    /// # Arguments
    /// * `q0` - initial generalized positions
    /// * `p0` - initial generalized momenta
    /// * `input_fn` - optional control input u(t)
    /// * `config` - integration configuration
    pub fn integrate(
        &self,
        q0: &[f64],
        p0: &[f64],
        input_fn: Option<&dyn Fn(f64) -> Vec<f64>>,
        config: &PortHamiltonianConfig,
    ) -> IntegrateResult<PhResult> {
        if q0.len() != self.n_dof {
            return Err(IntegrateError::DimensionMismatch(format!(
                "q0 length {} != n_dof {}",
                q0.len(),
                self.n_dof
            )));
        }
        if p0.len() != self.n_dof {
            return Err(IntegrateError::DimensionMismatch(format!(
                "p0 length {} != n_dof {}",
                p0.len(),
                self.n_dof
            )));
        }

        let n = config.n_steps;
        let dt = config.dt;
        let mut times = Vec::with_capacity(n + 1);
        let mut states = Vec::with_capacity(n + 1);
        let mut energies = Vec::with_capacity(n + 1);
        let mut outputs = Vec::with_capacity(n + 1);

        let mut q = q0.to_vec();
        let mut p = p0.to_vec();
        let mut t = 0.0;

        // Initial state
        times.push(t);
        let mut state = q.clone();
        state.extend_from_slice(&p);
        states.push(state);
        let h0 = if config.energy_tracking {
            self.hamiltonian(&q, &p)
        } else {
            0.0
        };
        energies.push(h0);
        outputs.push(self.compute_output(&p));

        for _ in 0..n {
            let u = if let Some(f) = input_fn {
                f(t)
            } else {
                vec![0.0; self.n_inputs.max(1)]
            };

            let (q_new, p_new) = match config.integrator {
                PhIntegrator::StormerVerlet => self.stormer_verlet_step(&q, &p, &u, dt),
                PhIntegrator::MidpointRule => {
                    self.midpoint_rule_step(&q, &p, &u, dt, config.max_iter, config.tol)
                }
                PhIntegrator::DiscreteGradient => {
                    self.discrete_gradient_step(&q, &p, &u, dt, config.max_iter, config.tol)
                }
                PhIntegrator::CayleyTransform => self.cayley_transform_step(&q, &p, &u, dt),
            };

            q = q_new;
            p = p_new;
            t += dt;

            times.push(t);
            let mut state = q.clone();
            state.extend_from_slice(&p);
            states.push(state);

            let h = if config.energy_tracking {
                self.hamiltonian(&q, &p)
            } else {
                0.0
            };
            energies.push(h);
            outputs.push(self.compute_output(&p));
        }

        let h_final = *energies.last().unwrap_or(&0.0);
        let energy_drift = (h_final - h0).abs();

        Ok(PhResult {
            times,
            states,
            energies,
            outputs,
            energy_drift,
        })
    }

    /// Compute system output y = Bᵀ M⁻¹ p
    fn compute_output(&self, p: &[f64]) -> Vec<f64> {
        if self.n_inputs == 0 || self.input_matrix.is_empty() {
            return vec![];
        }
        let minv_p = self.mass_inv_vec(p);
        let n = self.n_dof;
        let m = self.n_inputs;
        let mut y = vec![0.0; m];
        // B is n×m, Bᵀ is m×n
        // y_j = sum_i B[i,j] * (M^{-1}p)[i]
        for j in 0..m {
            for i in 0..n {
                y[j] += self.input_matrix[i * m + j] * minv_p[i];
            }
        }
        y
    }

    /// Störmer-Verlet step (symplectic, 2nd order)
    ///
    /// p_{n+1/2} = p_n - h/2 * (∂V/∂q_n + D M⁻¹ p_n) + h/2 * B u
    /// q_{n+1}   = q_n + h * M⁻¹ p_{n+1/2}
    /// p_{n+1}   = p_{n+1/2} - h/2 * (∂V/∂q_{n+1} + D M⁻¹ p_{n+1/2}) + h/2 * B u
    fn stormer_verlet_step(
        &self,
        q: &[f64],
        p: &[f64],
        u: &[f64],
        dt: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = self.n_dof;
        let grad_v = (self.grad_potential)(q);
        let minv_p = self.mass_inv_vec(p);
        let d_minv_p = self.mat_vec(&self.damping, &minv_p);
        let bu = self.compute_bu(u);

        // Half-step momentum
        let mut p_half = vec![0.0; n];
        for i in 0..n {
            p_half[i] = p[i] - 0.5 * dt * (grad_v[i] + d_minv_p[i])
                + 0.5 * dt * bu.get(i).copied().unwrap_or(0.0);
        }

        // Full-step position
        let minv_p_half = self.mass_inv_vec(&p_half);
        let mut q_new = vec![0.0; n];
        for i in 0..n {
            q_new[i] = q[i] + dt * minv_p_half[i];
        }

        // Second half-step momentum
        let grad_v_new = (self.grad_potential)(&q_new);
        let d_minv_p_half = self.mat_vec(&self.damping, &minv_p_half);
        let mut p_new = vec![0.0; n];
        for i in 0..n {
            p_new[i] = p_half[i] - 0.5 * dt * (grad_v_new[i] + d_minv_p_half[i])
                + 0.5 * dt * bu.get(i).copied().unwrap_or(0.0);
        }

        (q_new, p_new)
    }

    /// Implicit midpoint rule step (3rd-order accurate for Hamiltonian systems)
    ///
    /// Uses fixed-point iteration to solve the implicit system:
    ///   q_new = q + h * M⁻¹ p_mid
    ///   p_new = p - h * (∂V/∂q_mid + D M⁻¹ p_mid) + h * B u
    /// where q_mid = (q + q_new)/2, p_mid = (p + p_new)/2
    fn midpoint_rule_step(
        &self,
        q: &[f64],
        p: &[f64],
        u: &[f64],
        dt: f64,
        max_iter: usize,
        tol: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = self.n_dof;
        let bu = self.compute_bu(u);

        // Initial guess: explicit Euler
        let grad_v = (self.grad_potential)(q);
        let minv_p = self.mass_inv_vec(p);
        let d_minv_p = self.mat_vec(&self.damping, &minv_p);

        let mut q_new: Vec<f64> = (0..n).map(|i| q[i] + dt * minv_p[i]).collect();
        let mut p_new: Vec<f64> = (0..n)
            .map(|i| p[i] - dt * (grad_v[i] + d_minv_p[i]) + dt * bu.get(i).copied().unwrap_or(0.0))
            .collect();

        // Fixed-point iteration
        for _ in 0..max_iter {
            let q_mid: Vec<f64> = (0..n).map(|i| 0.5 * (q[i] + q_new[i])).collect();
            let p_mid: Vec<f64> = (0..n).map(|i| 0.5 * (p[i] + p_new[i])).collect();

            let grad_v_mid = (self.grad_potential)(&q_mid);
            let minv_p_mid = self.mass_inv_vec(&p_mid);
            let d_minv_p_mid = self.mat_vec(&self.damping, &minv_p_mid);

            let q_new_next: Vec<f64> = (0..n).map(|i| q[i] + dt * minv_p_mid[i]).collect();
            let p_new_next: Vec<f64> = (0..n)
                .map(|i| {
                    p[i] - dt * (grad_v_mid[i] + d_minv_p_mid[i])
                        + dt * bu.get(i).copied().unwrap_or(0.0)
                })
                .collect();

            // Check convergence
            let err_q: f64 = q_new_next
                .iter()
                .zip(q_new.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let err_p: f64 = p_new_next
                .iter()
                .zip(p_new.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);

            q_new = q_new_next;
            p_new = p_new_next;

            if err_q < tol && err_p < tol {
                break;
            }
        }

        (q_new, p_new)
    }

    /// Discrete gradient step (exactly energy-preserving)
    ///
    /// Uses the Gonzalez discrete gradient:
    ///   ∇_d H ≈ (H(q_new,p_new) - H(q,p)) * (state_new - state) / ||state_new - state||²
    ///
    /// For mechanical systems, this simplifies to an averaged gradient approach:
    ///   ∂V/∂q|_discrete ≈ (∂V/∂q(q) + ∂V/∂q(q_new)) / 2
    ///   with an extra correction term for exact energy preservation
    fn discrete_gradient_step(
        &self,
        q: &[f64],
        p: &[f64],
        u: &[f64],
        dt: f64,
        max_iter: usize,
        tol: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = self.n_dof;
        let bu = self.compute_bu(u);

        // Predictor: Störmer-Verlet
        let (mut q_new, mut p_new) = self.stormer_verlet_step(q, p, u, dt);

        // Corrector iterations using Gonzalez discrete gradient
        for _ in 0..max_iter {
            let grad_v_old = (self.grad_potential)(q);
            let grad_v_new = (self.grad_potential)(&q_new);

            // Average gradient (midpoint discrete gradient for quadratic + correction)
            let mut grad_v_avg: Vec<f64> = (0..n)
                .map(|i| 0.5 * (grad_v_old[i] + grad_v_new[i]))
                .collect();

            // Gonzalez correction for non-quadratic potentials
            let v_old = (self.potential)(q);
            let v_new = (self.potential)(&q_new);
            let dq: Vec<f64> = (0..n).map(|i| q_new[i] - q[i]).collect();
            let dq_norm_sq: f64 = dq.iter().map(|x| x * x).sum();
            if dq_norm_sq > f64::EPSILON {
                let avg_dot: f64 = (0..n)
                    .map(|i| 0.5 * (grad_v_old[i] + grad_v_new[i]) * dq[i])
                    .sum();
                let correction = (v_new - v_old - avg_dot) / dq_norm_sq;
                for i in 0..n {
                    grad_v_avg[i] += correction * dq[i];
                }
            }

            // Average M⁻¹p
            let p_avg: Vec<f64> = (0..n).map(|i| 0.5 * (p[i] + p_new[i])).collect();
            let minv_p_avg = self.mass_inv_vec(&p_avg);
            let d_minv_p_avg = self.mat_vec(&self.damping, &minv_p_avg);

            let q_new_next: Vec<f64> = (0..n).map(|i| q[i] + dt * minv_p_avg[i]).collect();
            let p_new_next: Vec<f64> = (0..n)
                .map(|i| {
                    p[i] - dt * (grad_v_avg[i] + d_minv_p_avg[i])
                        + dt * bu.get(i).copied().unwrap_or(0.0)
                })
                .collect();

            let err_q: f64 = q_new_next
                .iter()
                .zip(q_new.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let err_p: f64 = p_new_next
                .iter()
                .zip(p_new.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);

            q_new = q_new_next;
            p_new = p_new_next;

            if err_q < tol && err_p < tol {
                break;
            }
        }

        (q_new, p_new)
    }

    /// Cayley transform step (for linear PH systems)
    ///
    /// Approximates the matrix exponential via Cayley transform:
    ///   x_{n+1} = (I + h/2 * A)⁻¹ (I - h/2 * A) x_n
    /// where A = (J - R) Q for linear PH systems
    fn cayley_transform_step(
        &self,
        q: &[f64],
        p: &[f64],
        u: &[f64],
        dt: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        // Use midpoint rule as a proxy for linear systems
        // (Cayley = midpoint for linear systems)
        self.midpoint_rule_step(q, p, u, dt, 20, 1e-12)
    }

    /// Compute B * u (input force)
    fn compute_bu(&self, u: &[f64]) -> Vec<f64> {
        if self.n_inputs == 0 || self.input_matrix.is_empty() || u.is_empty() {
            return vec![0.0; self.n_dof];
        }
        let n = self.n_dof;
        let m = self.n_inputs;
        let mut bu = vec![0.0; n];
        for i in 0..n {
            for j in 0..m.min(u.len()) {
                bu[i] += self.input_matrix[i * m + j] * u[j];
            }
        }
        bu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create harmonic oscillator system
    /// H = p²/(2m) + ω²q²/2
    fn harmonic_oscillator(mass: f64, omega: f64) -> PortHamiltonianSystem {
        let omega_sq = omega * omega;
        PortHamiltonianSystem::new_mechanical(
            1,
            vec![mass],
            Box::new(move |q: &[f64]| 0.5 * omega_sq * q[0] * q[0]),
            Box::new(move |q: &[f64]| vec![omega_sq * q[0]]),
            vec![0.0], // no damping
        )
    }

    #[test]
    fn test_ph_config_default() {
        let config = PortHamiltonianConfig::default();
        assert_eq!(config.integrator, PhIntegrator::StormerVerlet);
        assert!((config.dt - 0.01).abs() < 1e-15);
        assert_eq!(config.n_steps, 100);
        assert!(config.energy_tracking);
        assert_eq!(config.max_iter, 10);
    }

    #[test]
    fn test_ph_result_shapes() {
        let sys = harmonic_oscillator(1.0, 1.0);
        let config = PortHamiltonianConfig {
            n_steps: 50,
            ..Default::default()
        };
        let result = sys
            .integrate(&[1.0], &[0.0], None, &config)
            .expect("Integration should succeed");

        assert_eq!(result.times.len(), 51);
        assert_eq!(result.states.len(), 51);
        assert_eq!(result.energies.len(), 51);
        // Each state has 2 * n_dof components
        assert_eq!(result.states[0].len(), 2);
    }

    #[test]
    fn test_harmonic_oscillator_energy_conservation_stormer_verlet() {
        // H = p²/2 + q²/2, initial: q=1, p=0 -> H=0.5
        let sys = harmonic_oscillator(1.0, 1.0);
        let config = PortHamiltonianConfig {
            integrator: PhIntegrator::StormerVerlet,
            dt: 0.01,
            n_steps: 1000,
            energy_tracking: true,
            ..Default::default()
        };
        let result = sys
            .integrate(&[1.0], &[0.0], None, &config)
            .expect("Integration should succeed");

        let h0 = result.energies[0];
        // Check energy conservation within 0.1%
        for &h in &result.energies {
            let rel_err = (h - h0).abs() / h0.abs().max(1e-14);
            assert!(
                rel_err < 0.001,
                "Energy conservation violated: H0={h0:.6}, H={h:.6}, rel_err={rel_err:.6e}"
            );
        }
    }

    #[test]
    fn test_damped_oscillator_energy_decrease() {
        // H = p²/2 + q²/2 with damping D=0.5
        // Energy should decrease monotonically (with damping)
        let sys = PortHamiltonianSystem::new_mechanical(
            1,
            vec![1.0],
            Box::new(|q: &[f64]| 0.5 * q[0] * q[0]),
            Box::new(|q: &[f64]| vec![q[0]]),
            vec![0.5], // positive damping
        );

        let config = PortHamiltonianConfig {
            integrator: PhIntegrator::StormerVerlet,
            dt: 0.01,
            n_steps: 200,
            energy_tracking: true,
            ..Default::default()
        };
        let result = sys
            .integrate(&[1.0], &[0.0], None, &config)
            .expect("Integration should succeed");

        let h0 = result.energies[0];
        let h_final = *result.energies.last().expect("Should have energies");

        // With positive damping, total energy should decrease
        assert!(
            h_final < h0,
            "Damped system: energy should decrease. H0={h0:.6}, H_final={h_final:.6}"
        );
    }

    #[test]
    fn test_discrete_gradient_energy_preservation() {
        // Undamped harmonic oscillator with discrete gradient
        let sys = harmonic_oscillator(1.0, 1.0);
        let config = PortHamiltonianConfig {
            integrator: PhIntegrator::DiscreteGradient,
            dt: 0.05,
            n_steps: 200,
            energy_tracking: true,
            max_iter: 20,
            tol: 1e-12,
        };
        let result = sys
            .integrate(&[1.0], &[0.0], None, &config)
            .expect("Integration should succeed");

        // energy_drift should be very small for discrete gradient method
        assert!(
            result.energy_drift < 0.01,
            "Discrete gradient energy drift too large: {}",
            result.energy_drift
        );
    }

    #[test]
    fn test_midpoint_rule_step() {
        let sys = harmonic_oscillator(1.0, 1.0);
        let config = PortHamiltonianConfig {
            integrator: PhIntegrator::MidpointRule,
            dt: 0.01,
            n_steps: 100,
            energy_tracking: true,
            max_iter: 15,
            tol: 1e-10,
        };
        let result = sys
            .integrate(&[1.0], &[0.0], None, &config)
            .expect("Integration should succeed");

        // Midpoint rule conserves quadratic invariants (harmonic oscillator is quadratic)
        let h0 = result.energies[0];
        let h_final = *result.energies.last().expect("Should have energies");
        let rel_err = (h_final - h0).abs() / h0.abs().max(1e-14);
        assert!(
            rel_err < 1e-6,
            "Midpoint rule should preserve energy for quadratic H: rel_err={rel_err:.2e}"
        );
    }

    #[test]
    fn test_two_dof_system() {
        // 2-DOF uncoupled harmonic oscillator
        // H = (p1² + p2²)/2 + (q1² + q2²)/2
        let sys = PortHamiltonianSystem::new_mechanical(
            2,
            vec![1.0, 0.0, 0.0, 1.0], // identity mass matrix
            Box::new(|q: &[f64]| 0.5 * (q[0] * q[0] + q[1] * q[1])),
            Box::new(|q: &[f64]| vec![q[0], q[1]]),
            vec![0.0, 0.0, 0.0, 0.0], // no damping
        );

        let config = PortHamiltonianConfig {
            n_steps: 50,
            ..Default::default()
        };
        let result = sys
            .integrate(&[1.0, 0.5], &[0.0, 1.0], None, &config)
            .expect("Integration should succeed");

        assert_eq!(result.states[0].len(), 4); // q1, q2, p1, p2
        let h0 = result.energies[0];
        let h_final = *result.energies.last().expect("Should have energies");
        let rel_err = (h_final - h0).abs() / h0.abs().max(1e-14);
        assert!(
            rel_err < 0.001,
            "2-DOF energy conservation failed: {rel_err:.2e}"
        );
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let sys = harmonic_oscillator(1.0, 1.0);
        let config = PortHamiltonianConfig::default();
        // q0 has wrong length
        let result = sys.integrate(&[1.0, 2.0], &[0.0], None, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_hamiltonian_computation() {
        let sys = harmonic_oscillator(1.0, 2.0);
        // H = p²/2 + ω²q²/2 = 0 + 4*1/2 = 2.0 at q=1, p=0
        let h = sys.hamiltonian(&[1.0], &[0.0]);
        assert!((h - 2.0).abs() < 1e-14, "Hamiltonian at q=1,p=0: {h}");

        // H = p²/2m + ω²q²/2 = 1/2 + 0 = 0.5 at q=0, p=1
        let h2 = sys.hamiltonian(&[0.0], &[1.0]);
        assert!((h2 - 0.5).abs() < 1e-14, "Hamiltonian at q=0,p=1: {h2}");
    }

    #[test]
    fn test_cayley_transform_step() {
        let sys = harmonic_oscillator(1.0, 1.0);
        let config = PortHamiltonianConfig {
            integrator: PhIntegrator::CayleyTransform,
            dt: 0.01,
            n_steps: 100,
            energy_tracking: true,
            ..Default::default()
        };
        let result = sys
            .integrate(&[1.0], &[0.0], None, &config)
            .expect("Integration should succeed");
        assert_eq!(result.times.len(), 101);
        let h0 = result.energies[0];
        let h_final = *result.energies.last().expect("Should have energies");
        let rel_err = (h_final - h0).abs() / h0.abs().max(1e-14);
        assert!(
            rel_err < 1e-4,
            "Cayley transform energy drift: {rel_err:.2e}"
        );
    }
}
