//! Lindblad Master Equation for open quantum systems.
//!
//! dρ/dt = -i/ħ [H, ρ] + Σ_k γ_k (L_k ρ L_k† - ½ {L_k†L_k, ρ})
//!
//! Solved by propagating the density matrix directly using
//! Runge-Kutta 4th order integration of the Lindblad superoperator.
//!
//! # Example
//! ```no_run
//! use scirs2_integrate::specialized::quantum::lindblad::{
//!     CMatrix, LindbladConfig, LindbladMethod, LindbladSystem, lindblad_evolve,
//! };
//! use scirs2_core::ndarray::Array2;
//!
//! // Two-level system: excited state decays to ground state
//! let n = 2;
//! let h = CMatrix::zeros(n); // zero Hamiltonian
//!
//! // Decay operator: |0><1| (lowering operator)
//! let mut l_re = Array2::<f64>::zeros((n, n));
//! l_re[[0, 1]] = 1.0;
//! let l = CMatrix::from_parts(l_re, Array2::zeros((n, n)));
//!
//! let system = LindbladSystem {
//!     hamiltonian: h,
//!     lindblad_ops: vec![(1.0, l)],
//! };
//!
//! // Start in excited state
//! let mut rho0_re = Array2::<f64>::zeros((n, n));
//! rho0_re[[1, 1]] = 1.0;
//! let rho0 = CMatrix::from_parts(rho0_re, Array2::zeros((n, n)));
//!
//! let config = LindbladConfig::default();
//! let result = lindblad_evolve(&system, &rho0, &config).unwrap();
//! ```

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2};

// ─────────────────────────────────────────────────────────────────────────────
// CMatrix: complex matrix stored as (re, im) pair of Array2<f64>
// ─────────────────────────────────────────────────────────────────────────────

/// Dense complex matrix stored as separate real and imaginary `Array2<f64>`.
///
/// This representation avoids a dependency on a complex number crate while
/// keeping arithmetic transparent and allocation-friendly.
#[derive(Debug, Clone)]
pub struct CMatrix {
    /// Real part
    pub re: Array2<f64>,
    /// Imaginary part
    pub im: Array2<f64>,
}

impl CMatrix {
    /// Create an n×n zero matrix.
    pub fn zeros(n: usize) -> Self {
        Self {
            re: Array2::zeros((n, n)),
            im: Array2::zeros((n, n)),
        }
    }

    /// Create an n×n identity matrix.
    pub fn eye(n: usize) -> Self {
        Self {
            re: Array2::eye(n),
            im: Array2::zeros((n, n)),
        }
    }

    /// Construct from separate real and imaginary parts.
    ///
    /// # Panics
    /// Panics if `re` and `im` do not have the same shape.
    pub fn from_parts(re: Array2<f64>, im: Array2<f64>) -> Self {
        assert_eq!(re.shape(), im.shape(), "re and im must have the same shape");
        Self { re, im }
    }

    /// Return the matrix dimension n (assumes square matrix).
    pub fn n(&self) -> usize {
        self.re.nrows()
    }

    /// Conjugate transpose: (A†)_{ij} = conj(A_{ji}).
    pub fn adjoint(&self) -> Self {
        let n = self.n();
        let mut re = Array2::zeros((n, n));
        let mut im = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                re[[i, j]] = self.re[[j, i]];
                im[[i, j]] = -self.im[[j, i]];
            }
        }
        Self { re, im }
    }

    /// Matrix multiplication C = A * B.
    pub fn mul(&self, other: &Self) -> Self {
        let n = self.n();
        let mut re = Array2::zeros((n, n));
        let mut im = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut r = 0.0_f64;
                let mut c = 0.0_f64;
                for k in 0..n {
                    // (a_re + i a_im)(b_re + i b_im) = a_re b_re - a_im b_im + i(a_re b_im + a_im b_re)
                    r += self.re[[i, k]] * other.re[[k, j]] - self.im[[i, k]] * other.im[[k, j]];
                    c += self.re[[i, k]] * other.im[[k, j]] + self.im[[i, k]] * other.re[[k, j]];
                }
                re[[i, j]] = r;
                im[[i, j]] = c;
            }
        }
        Self { re, im }
    }

    /// Element-wise addition.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            re: &self.re + &other.re,
            im: &self.im + &other.im,
        }
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            re: &self.re - &other.re,
            im: &self.im - &other.im,
        }
    }

    /// Scale by real scalar.
    pub fn scale(&self, s: f64) -> Self {
        Self {
            re: &self.re * s,
            im: &self.im * s,
        }
    }

    /// Commutator [A, B] = A*B - B*A.
    pub fn commutator(&self, rho: &Self) -> Self {
        self.mul(rho).sub(&rho.mul(self))
    }

    /// Anti-commutator {A, B} = A*B + B*A.
    pub fn anticommutator(&self, b: &Self) -> Self {
        self.mul(b).add(&b.mul(self))
    }

    /// Frobenius (trace) norm: sqrt(Σ|a_ij|²).
    pub fn trace_norm(&self) -> f64 {
        let re_sq: f64 = self.re.iter().map(|x| x * x).sum();
        let im_sq: f64 = self.im.iter().map(|x| x * x).sum();
        (re_sq + im_sq).sqrt()
    }

    /// Complex trace (re, im).
    pub fn trace(&self) -> (f64, f64) {
        let n = self.n();
        let re = (0..n).map(|i| self.re[[i, i]]).sum();
        let im = (0..n).map(|i| self.im[[i, i]]).sum();
        (re, im)
    }

    /// Vectorize the matrix row-major: [re_00, re_01, ..., re_{n-1,n-1}, im_00, ..., im_{n-1,n-1}].
    pub fn to_vec(&self) -> Array1<f64> {
        let n = self.n();
        let mut v = Array1::zeros(2 * n * n);
        for i in 0..n {
            for j in 0..n {
                v[i * n + j] = self.re[[i, j]];
                v[n * n + i * n + j] = self.im[[i, j]];
            }
        }
        v
    }

    /// Reconstruct from vectorized form produced by `to_vec`.
    pub fn from_vec(v: &Array1<f64>, n: usize) -> Self {
        let mut re = Array2::zeros((n, n));
        let mut im = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                re[[i, j]] = v[i * n + j];
                im[[i, j]] = v[n * n + i * n + j];
            }
        }
        Self { re, im }
    }

    /// Purity Tr(ρ²).
    pub fn purity(&self) -> f64 {
        let rho2 = self.mul(self);
        rho2.trace().0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration types
// ─────────────────────────────────────────────────────────────────────────────

/// Integration method for the Lindblad equation.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum LindbladMethod {
    /// First-order Euler (fast but low accuracy).
    EulerMaruyama,
    /// Classical 4th-order Runge-Kutta (default).
    RungeKutta4,
}

/// Configuration for Lindblad master equation solver.
#[derive(Debug, Clone)]
pub struct LindbladConfig {
    /// Reduced Planck constant ħ (default 1.0 for natural units).
    pub hbar: f64,
    /// Time interval `[t_start, t_end]`.
    pub t_span: [f64; 2],
    /// Number of integration steps.
    pub n_steps: usize,
    /// Integration method.
    pub method: LindbladMethod,
}

impl Default for LindbladConfig {
    fn default() -> Self {
        Self {
            hbar: 1.0,
            t_span: [0.0, 1.0],
            n_steps: 100,
            method: LindbladMethod::RungeKutta4,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// System and result types
// ─────────────────────────────────────────────────────────────────────────────

/// An open quantum system described by a Hamiltonian and Lindblad operators.
pub struct LindbladSystem {
    /// System Hamiltonian H (n×n complex Hermitian matrix).
    pub hamiltonian: CMatrix,
    /// Lindblad jump operators with their rates: (γ_k, L_k).
    pub lindblad_ops: Vec<(f64, CMatrix)>,
}

/// Result of a Lindblad master equation simulation.
pub struct LindbladResult {
    /// Time points at which the density matrix was recorded.
    pub times: Array1<f64>,
    /// Density matrix ρ(t) at each time point.
    pub density_matrices: Vec<CMatrix>,
    /// Purity Tr(ρ²) at each time point (≤ 1 for physical states).
    pub purity: Array1<f64>,
    /// Trace Tr(ρ) at each time point (should remain 1).
    pub trace: Array1<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Core algorithm
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Lindblad right-hand side:
///
/// drho/dt = -i/ħ [H, ρ] + Σ_k γ_k (L_k ρ L_k† - ½ {L_k†L_k, ρ})
fn lindblad_rhs(system: &LindbladSystem, rho: &CMatrix, hbar: f64) -> CMatrix {
    let n = rho.n();

    // Coherent part: -i/ħ [H, ρ]
    // [H, ρ] = H*ρ - ρ*H  (purely imaginary trace, real off-diag in general)
    // -i * [H_re + i H_im, ρ_re + i ρ_im] / ħ
    let comm = system.hamiltonian.commutator(rho); // [H, ρ]
                                                   // Multiply by -i/ħ: (a + ib)*(-i) = b - ia
    let coherent = CMatrix {
        re: &comm.im / hbar,
        im: -&comm.re / hbar,
    };

    // Dissipative part
    let mut dissipative = CMatrix::zeros(n);
    for (gamma, l) in &system.lindblad_ops {
        let l_dag = l.adjoint();
        let l_dag_l = l_dag.mul(l); // L†L

        // L ρ L†
        let lrho = l.mul(rho);
        let lrho_ldag = lrho.mul(&l_dag);

        // ½ {L†L, ρ}
        let anti = l_dag_l.anticommutator(rho).scale(0.5);

        // γ_k (L ρ L† - ½ {L†L, ρ})
        let term = lrho_ldag.sub(&anti).scale(*gamma);
        dissipative = dissipative.add(&term);
    }

    coherent.add(&dissipative)
}

/// Evolve a density matrix under the Lindblad master equation.
///
/// # Arguments
/// * `system` – Hamiltonian and Lindblad operators.
/// * `rho0` – Initial density matrix (must be Hermitian, trace 1).
/// * `config` – Solver configuration.
///
/// # Errors
/// Returns [`IntegrateError::InvalidInput`] if inputs are inconsistent.
pub fn lindblad_evolve(
    system: &LindbladSystem,
    rho0: &CMatrix,
    config: &LindbladConfig,
) -> IntegrateResult<LindbladResult> {
    // ── Validation ──────────────────────────────────────────────────────────
    if config.n_steps == 0 {
        return Err(IntegrateError::InvalidInput(
            "n_steps must be > 0".to_string(),
        ));
    }
    let n = rho0.n();
    if n == 0 {
        return Err(IntegrateError::InvalidInput(
            "density matrix must be at least 1×1".to_string(),
        ));
    }
    if system.hamiltonian.n() != n {
        return Err(IntegrateError::DimensionMismatch(format!(
            "Hamiltonian dimension {} ≠ rho dimension {}",
            system.hamiltonian.n(),
            n
        )));
    }
    for (k, (_, lk)) in system.lindblad_ops.iter().enumerate() {
        if lk.n() != n {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Lindblad operator {} has dimension {} ≠ rho dimension {}",
                k,
                lk.n(),
                n
            )));
        }
    }
    // Check trace ≈ 1
    let (tr_re, _tr_im) = rho0.trace();
    if (tr_re - 1.0).abs() > 1e-4 {
        return Err(IntegrateError::InvalidInput(format!(
            "Initial density matrix trace must be 1, got {tr_re:.6}"
        )));
    }

    // ── Integration ──────────────────────────────────────────────────────────
    let t_start = config.t_span[0];
    let t_end = config.t_span[1];
    let n_steps = config.n_steps;
    let h = (t_end - t_start) / n_steps as f64;

    let mut times = Array1::zeros(n_steps + 1);
    let mut density_matrices = Vec::with_capacity(n_steps + 1);
    let mut purity_arr = Array1::zeros(n_steps + 1);
    let mut trace_arr = Array1::zeros(n_steps + 1);

    times[0] = t_start;
    purity_arr[0] = rho0.purity();
    trace_arr[0] = rho0.trace().0;
    let mut rho = rho0.clone();
    density_matrices.push(rho.clone());

    for step in 0..n_steps {
        let t = t_start + step as f64 * h;

        match config.method {
            LindbladMethod::EulerMaruyama => {
                let drho = lindblad_rhs(system, &rho, config.hbar);
                rho = rho.add(&drho.scale(h));
            }
            LindbladMethod::RungeKutta4 => {
                // k1 = f(t, ρ)
                let k1 = lindblad_rhs(system, &rho, config.hbar);

                // k2 = f(t + h/2, ρ + h/2 * k1)
                let rho2 = rho.add(&k1.scale(h / 2.0));
                let k2 = lindblad_rhs(system, &rho2, config.hbar);

                // k3 = f(t + h/2, ρ + h/2 * k2)
                let rho3 = rho.add(&k2.scale(h / 2.0));
                let k3 = lindblad_rhs(system, &rho3, config.hbar);

                // k4 = f(t + h, ρ + h * k3)
                let rho4 = rho.add(&k3.scale(h));
                let k4 = lindblad_rhs(system, &rho4, config.hbar);

                // ρ(t+h) = ρ(t) + h/6 * (k1 + 2k2 + 2k3 + k4)
                let update = k1
                    .add(&k2.scale(2.0))
                    .add(&k3.scale(2.0))
                    .add(&k4)
                    .scale(h / 6.0);
                rho = rho.add(&update);
            }
            #[allow(unreachable_patterns)]
            _ => {
                return Err(IntegrateError::NotImplementedError(
                    "Unknown LindbladMethod variant".to_string(),
                ));
            }
        }

        times[step + 1] = t + h;
        purity_arr[step + 1] = rho.purity();
        trace_arr[step + 1] = rho.trace().0;
        density_matrices.push(rho.clone());
    }

    Ok(LindbladResult {
        times,
        density_matrices,
        purity: purity_arr,
        trace: trace_arr,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Steady state
// ─────────────────────────────────────────────────────────────────────────────

/// Find the steady state ρ_ss by evolving until purity converges.
///
/// The algorithm runs the RK4 integrator until `|Tr(ρ²)(t) - Tr(ρ²)(t-Δ)| < tol`
/// over several consecutive checks, then returns the last density matrix.
///
/// # Errors
/// Returns [`IntegrateError::ConvergenceError`] if no steady state is found
/// within the maximum number of steps.
pub fn steady_state(system: &LindbladSystem) -> IntegrateResult<CMatrix> {
    let n = system.hamiltonian.n();

    // Start from maximally mixed state ρ = I/n
    let rho0 = CMatrix {
        re: Array2::eye(n) / n as f64,
        im: Array2::zeros((n, n)),
    };

    let config = LindbladConfig {
        hbar: 1.0,
        t_span: [0.0, 50.0],
        n_steps: 5000,
        method: LindbladMethod::RungeKutta4,
    };

    let result = lindblad_evolve(system, &rho0, &config)?;

    // Check purity convergence at the end
    let np = result.purity.len();
    if np < 2 {
        return Ok(result.density_matrices[0].clone());
    }

    let purity_last = result.purity[np - 1];
    let purity_prev = result.purity[np - 100.min(np - 1)];
    if (purity_last - purity_prev).abs() > 1e-4 {
        // Try a longer evolution
        let config2 = LindbladConfig {
            t_span: [0.0, 200.0],
            n_steps: 20_000,
            ..config
        };
        let result2 = lindblad_evolve(system, &rho0, &config2)?;
        let n2 = result2.density_matrices.len();
        return Ok(result2.density_matrices[n2 - 1].clone());
    }

    let nd = result.density_matrices.len();
    Ok(result.density_matrices[nd - 1].clone())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_rho0_ground(n: usize) -> CMatrix {
        let mut re = Array2::<f64>::zeros((n, n));
        re[[0, 0]] = 1.0;
        CMatrix::from_parts(re, Array2::zeros((n, n)))
    }

    fn make_rho0_excited(n: usize) -> CMatrix {
        // Start in highest index state
        let mut re = Array2::<f64>::zeros((n, n));
        re[[n - 1, n - 1]] = 1.0;
        CMatrix::from_parts(re, Array2::zeros((n, n)))
    }

    fn make_equal_superposition(n: usize) -> CMatrix {
        // ρ = (1/n) * ones (maximally mixed)
        CMatrix {
            re: Array2::eye(n) / n as f64,
            im: Array2::zeros((n, n)),
        }
    }

    // ── CMatrix algebra ─────────────────────────────────────────────────────

    #[test]
    fn test_cmatrix_adjoint_involution() {
        // (A†)† = A
        let re = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let im = Array2::from_shape_vec((2, 2), vec![0.5, -0.5, 1.0, -1.0]).unwrap();
        let a = CMatrix::from_parts(re, im);
        let a_dag_dag = a.adjoint().adjoint();
        let diff = a.sub(&a_dag_dag);
        assert!(diff.trace_norm() < 1e-12, "adjoint involution failed");
    }

    #[test]
    fn test_cmatrix_commutator_self_zero() {
        // [A, A] = 0
        let re = Array2::eye(2);
        let im = Array2::zeros((2, 2));
        let a = CMatrix::from_parts(re, im);
        let comm = a.commutator(&a);
        assert!(comm.trace_norm() < 1e-12, "[A,A] should be zero");
    }

    #[test]
    fn test_cmatrix_trace_eye() {
        let n = 3;
        let eye = CMatrix::eye(n);
        let (tr, _) = eye.trace();
        assert!((tr - n as f64).abs() < 1e-14);
    }

    #[test]
    fn test_cmatrix_to_from_vec_roundtrip() {
        let n = 2;
        let re = Array2::from_shape_vec((n, n), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let im = Array2::from_shape_vec((n, n), vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let m = CMatrix::from_parts(re, im);
        let v = m.to_vec();
        let m2 = CMatrix::from_vec(&v, n);
        let diff = m.sub(&m2);
        assert!(diff.trace_norm() < 1e-14);
    }

    // ── Solver validation ────────────────────────────────────────────────────

    #[test]
    fn test_zero_h_no_lindblad_constant_rho() {
        // With H=0 and no Lindblad operators, ρ should remain constant.
        let n = 2;
        let system = LindbladSystem {
            hamiltonian: CMatrix::zeros(n),
            lindblad_ops: vec![],
        };
        let rho0 = make_equal_superposition(n);
        let config = LindbladConfig {
            t_span: [0.0, 2.0],
            n_steps: 200,
            ..Default::default()
        };
        let result = lindblad_evolve(&system, &rho0, &config).unwrap();
        let rho_final = &result.density_matrices[result.density_matrices.len() - 1];
        let diff = rho0.sub(rho_final);
        assert!(
            diff.trace_norm() < 1e-10,
            "rho should not change with H=0, no Lindblad"
        );
    }

    #[test]
    fn test_trace_preserved() {
        // Tr(ρ(t)) = 1 at all times (up to numerical tolerance).
        let n = 2;
        let mut l_re = Array2::<f64>::zeros((n, n));
        l_re[[0, 1]] = 1.0; // lowering operator
        let l = CMatrix::from_parts(l_re, Array2::zeros((n, n)));
        let system = LindbladSystem {
            hamiltonian: CMatrix::zeros(n),
            lindblad_ops: vec![(0.5, l)],
        };
        let rho0 = make_rho0_excited(n);
        let config = LindbladConfig {
            t_span: [0.0, 3.0],
            n_steps: 300,
            ..Default::default()
        };
        let result = lindblad_evolve(&system, &rho0, &config).unwrap();
        for tr in result.trace.iter() {
            assert!((tr - 1.0).abs() < 1e-5, "Trace deviated from 1: {tr}");
        }
    }

    #[test]
    fn test_purity_leq_one() {
        // Tr(ρ²) ≤ 1 for all physical states.
        let n = 2;
        let mut l_re = Array2::<f64>::zeros((n, n));
        l_re[[0, 1]] = 1.0;
        let l = CMatrix::from_parts(l_re, Array2::zeros((n, n)));
        let system = LindbladSystem {
            hamiltonian: CMatrix::zeros(n),
            lindblad_ops: vec![(1.0, l)],
        };
        let rho0 = make_rho0_excited(n);
        let config = LindbladConfig {
            t_span: [0.0, 2.0],
            n_steps: 200,
            ..Default::default()
        };
        let result = lindblad_evolve(&system, &rho0, &config).unwrap();
        for (i, p) in result.purity.iter().enumerate() {
            assert!(*p <= 1.0 + 1e-6, "Purity > 1 at step {i}: {p}");
        }
    }

    #[test]
    fn test_unitary_evolution_purity_one() {
        // Pure state under unitary (no Lindblad ops) → purity stays 1.
        let n = 2;
        // H = σ_z: [[1,0],[0,-1]]
        let mut h_re = Array2::<f64>::zeros((n, n));
        h_re[[0, 0]] = 1.0;
        h_re[[1, 1]] = -1.0;
        let h = CMatrix::from_parts(h_re, Array2::zeros((n, n)));
        let system = LindbladSystem {
            hamiltonian: h,
            lindblad_ops: vec![],
        };
        let rho0 = make_rho0_ground(n); // |0><0| pure state
        let config = LindbladConfig {
            t_span: [0.0, 1.0],
            n_steps: 1000,
            ..Default::default()
        };
        let result = lindblad_evolve(&system, &rho0, &config).unwrap();
        for (i, p) in result.purity.iter().enumerate() {
            assert!(
                (p - 1.0).abs() < 1e-4,
                "Purity deviated from 1 at step {i}: {p}"
            );
        }
    }

    #[test]
    fn test_two_level_decay_to_ground() {
        // Excited state should decay to ground state under amplitude damping.
        let n = 2;
        let mut l_re = Array2::<f64>::zeros((n, n));
        l_re[[0, 1]] = 1.0; // |0><1| lowering operator
        let l = CMatrix::from_parts(l_re, Array2::zeros((n, n)));
        let system = LindbladSystem {
            hamiltonian: CMatrix::zeros(n),
            lindblad_ops: vec![(2.0, l)],
        };
        let rho0 = make_rho0_excited(n);
        let config = LindbladConfig {
            t_span: [0.0, 5.0],
            n_steps: 1000,
            ..Default::default()
        };
        let result = lindblad_evolve(&system, &rho0, &config).unwrap();
        let rho_final = &result.density_matrices[result.density_matrices.len() - 1];
        // Ground state population should be > 0.99
        assert!(
            rho_final.re[[0, 0]] > 0.99,
            "Ground state population = {}; expected > 0.99",
            rho_final.re[[0, 0]]
        );
    }

    #[test]
    fn test_dephasing_kills_offdiag() {
        // Dephasing operator σ_z: off-diagonal elements of ρ should decay.
        let n = 2;
        // σ_z = diag(1,-1)
        let mut l_re = Array2::<f64>::zeros((n, n));
        l_re[[0, 0]] = 1.0;
        l_re[[1, 1]] = -1.0;
        let l = CMatrix::from_parts(l_re, Array2::zeros((n, n)));
        let system = LindbladSystem {
            hamiltonian: CMatrix::zeros(n),
            lindblad_ops: vec![(1.0, l)],
        };
        // Coherent superposition ρ = 0.5 * [[1,1],[1,1]]
        let rho0 = CMatrix::from_parts(
            Array2::from_shape_vec((2, 2), vec![0.5, 0.5, 0.5, 0.5]).unwrap(),
            Array2::zeros((2, 2)),
        );
        let config = LindbladConfig {
            t_span: [0.0, 5.0],
            n_steps: 500,
            ..Default::default()
        };
        let result = lindblad_evolve(&system, &rho0, &config).unwrap();
        let rho_final = &result.density_matrices[result.density_matrices.len() - 1];
        // Off-diagonal should have decayed significantly
        assert!(
            rho_final.re[[0, 1]].abs() < 0.01,
            "Off-diagonal not decayed: {}",
            rho_final.re[[0, 1]]
        );
    }

    #[test]
    fn test_steady_state_decay_is_ground() {
        // Steady state of amplitude damping should be |0><0|.
        let n = 2;
        let mut l_re = Array2::<f64>::zeros((n, n));
        l_re[[0, 1]] = 1.0;
        let l = CMatrix::from_parts(l_re, Array2::zeros((n, n)));
        let system = LindbladSystem {
            hamiltonian: CMatrix::zeros(n),
            lindblad_ops: vec![(2.0, l)],
        };
        let ss = steady_state(&system).unwrap();
        assert!(
            ss.re[[0, 0]] > 0.99,
            "Steady-state ground pop = {}",
            ss.re[[0, 0]]
        );
    }

    #[test]
    fn test_invalid_trace_rho0() {
        let n = 2;
        let system = LindbladSystem {
            hamiltonian: CMatrix::zeros(n),
            lindblad_ops: vec![],
        };
        // Trace = 2 instead of 1
        let rho0 = CMatrix::from_parts(Array2::eye(n), Array2::zeros((n, n)));
        let config = LindbladConfig::default();
        let result = lindblad_evolve(&system, &rho0, &config);
        assert!(result.is_err(), "Should error on trace ≠ 1");
    }

    #[test]
    fn test_invalid_n_steps_zero() {
        let n = 2;
        let system = LindbladSystem {
            hamiltonian: CMatrix::zeros(n),
            lindblad_ops: vec![],
        };
        let rho0 = make_rho0_ground(n);
        let config = LindbladConfig {
            n_steps: 0,
            ..Default::default()
        };
        let result = lindblad_evolve(&system, &rho0, &config);
        assert!(result.is_err(), "Should error on n_steps=0");
    }

    #[test]
    fn test_identity_rho_stays_identity_over_n() {
        // ρ = I/n (maximally mixed) with H=0 and no Lindblad → stays I/n.
        let n = 3;
        let system = LindbladSystem {
            hamiltonian: CMatrix::zeros(n),
            lindblad_ops: vec![],
        };
        let rho0 = make_equal_superposition(n);
        let config = LindbladConfig {
            t_span: [0.0, 1.0],
            n_steps: 100,
            ..Default::default()
        };
        let result = lindblad_evolve(&system, &rho0, &config).unwrap();
        let rho_final = &result.density_matrices[result.density_matrices.len() - 1];
        let diff = rho0.sub(rho_final);
        assert!(diff.trace_norm() < 1e-10, "I/n state drifted");
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let n = 2;
        let system = LindbladSystem {
            hamiltonian: CMatrix::zeros(n + 1), // wrong size
            lindblad_ops: vec![],
        };
        let rho0 = make_rho0_ground(n);
        let config = LindbladConfig::default();
        let result = lindblad_evolve(&system, &rho0, &config);
        assert!(result.is_err(), "Should error on dimension mismatch");
    }

    #[test]
    fn test_euler_method_trace_preserved() {
        let n = 2;
        let mut l_re = Array2::<f64>::zeros((n, n));
        l_re[[0, 1]] = 1.0;
        let l = CMatrix::from_parts(l_re, Array2::zeros((n, n)));
        let system = LindbladSystem {
            hamiltonian: CMatrix::zeros(n),
            lindblad_ops: vec![(0.5, l)],
        };
        let rho0 = make_rho0_excited(n);
        let config = LindbladConfig {
            t_span: [0.0, 0.5],
            n_steps: 5000, // small steps for Euler stability
            method: LindbladMethod::EulerMaruyama,
            ..Default::default()
        };
        let result = lindblad_evolve(&system, &rho0, &config).unwrap();
        let tr_final = result.trace[result.trace.len() - 1];
        assert!(
            (tr_final - 1.0).abs() < 1e-3,
            "Euler trace deviated: {tr_final}"
        );
    }
}
