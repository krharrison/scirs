//! VQE (Variational Quantum Eigensolver) for ground-state energy estimation
//!
//! The VQE algorithm prepares a parametrized quantum state (ansatz) and minimizes
//! the expectation value ⟨ψ(θ)|H|ψ(θ)⟩ over the parameters θ.

use crate::error::OptimizeError;
use crate::quantum_classical::statevector::{cabs2, Statevector};
use crate::quantum_classical::QcResult;

// ─── Pauli operators ───────────────────────────────────────────────────────

/// Single-qubit Pauli operator.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PauliOp {
    /// Identity
    I,
    /// Pauli X
    X,
    /// Pauli Y
    Y,
    /// Pauli Z
    Z,
}

// ─── Pauli Hamiltonian ─────────────────────────────────────────────────────

/// A Hamiltonian expressed as a weighted sum of Pauli tensor products.
///
/// H = Σ_k c_k ⊗_{i} P_{k,i}
///
/// Each term is `(coefficient, [(qubit_index, PauliOp)])`.
#[derive(Debug, Clone)]
pub struct PauliHamiltonian {
    /// Hamiltonian terms: `(weight, list_of_(qubit, pauli)_pairs)`
    pub terms: Vec<(f64, Vec<(usize, PauliOp)>)>,
    /// Number of qubits in the system
    pub n_qubits: usize,
}

impl PauliHamiltonian {
    /// Create a new empty Hamiltonian for `n_qubits` qubits.
    pub fn new(n_qubits: usize) -> Self {
        Self {
            terms: Vec::new(),
            n_qubits,
        }
    }

    /// Add a term to the Hamiltonian.
    pub fn add_term(&mut self, coefficient: f64, ops: Vec<(usize, PauliOp)>) {
        self.terms.push((coefficient, ops));
    }

    /// Compute ⟨ψ|H|ψ⟩ using the statevector.
    ///
    /// Each term is evaluated via: `⟨ψ|P_k|ψ⟩ = ⟨ψ|P_k^† P_k|ψ⟩`
    /// For Hermitian Paulis we use diagonal or off-diagonal matrix elements.
    pub fn expectation(&self, sv: &Statevector) -> QcResult<f64> {
        let mut total = 0.0;
        for (coeff, ops) in &self.terms {
            // Build a copy of the statevector and apply the Pauli string
            let mut temp = sv.clone();
            for &(qubit, ref pauli) in ops {
                match pauli {
                    PauliOp::I => {} // identity: do nothing
                    PauliOp::X => temp.apply_x(qubit)?,
                    PauliOp::Y => temp.apply_y(qubit)?,
                    PauliOp::Z => temp.apply_z(qubit)?,
                    _ => {} // exhaustive match on non_exhaustive enum
                }
            }
            // Compute ⟨ψ|P_k|ψ⟩ = Re(ψ†·P_k ψ)
            let dot: f64 = sv
                .amplitudes
                .iter()
                .zip(temp.amplitudes.iter())
                .map(|(&(ar, ai), &(br, bi))| ar * br + ai * bi)
                .sum();
            total += coeff * dot;
        }
        Ok(total)
    }

    /// Create the transverse-field Ising model: H = -J Σ Z_i Z_{i+1} - h Σ X_i
    pub fn transverse_ising_1d(n_qubits: usize, j: f64, h: f64) -> Self {
        let mut ham = Self::new(n_qubits);
        // ZZ couplings
        for i in 0..(n_qubits - 1) {
            ham.add_term(-j, vec![(i, PauliOp::Z), (i + 1, PauliOp::Z)]);
        }
        // Transverse field (X)
        for i in 0..n_qubits {
            ham.add_term(-h, vec![(i, PauliOp::X)]);
        }
        ham
    }
}

// ─── Statevector extensions for Pauli gates ────────────────────────────────

impl Statevector {
    /// Apply Pauli X (bit flip) to `qubit`.
    pub fn apply_x(&mut self, qubit: usize) -> QcResult<()> {
        // Hadamard → Rz(π) → Hadamard is equivalent to X, but direct swap is simpler:
        let bit = 1usize << qubit;
        let dim = self.amplitudes.len();
        if qubit >= self.n_qubits {
            return Err(OptimizeError::ValueError(format!(
                "Qubit {qubit} out of range"
            )));
        }
        for i in 0..dim {
            if i & bit == 0 {
                self.amplitudes.swap(i, i | bit);
            }
        }
        Ok(())
    }

    /// Apply Pauli Y to `qubit`.
    ///
    /// Y = [[0, -i], [i, 0]]
    /// Y|0⟩ = i|1⟩,  Y|1⟩ = -i|0⟩
    pub fn apply_y(&mut self, qubit: usize) -> QcResult<()> {
        if qubit >= self.n_qubits {
            return Err(OptimizeError::ValueError(format!(
                "Qubit {qubit} out of range"
            )));
        }
        let bit = 1usize << qubit;
        let dim = self.amplitudes.len();
        for i in 0..dim {
            if i & bit == 0 {
                let j = i | bit;
                let (ar, ai) = self.amplitudes[i]; // |0⟩ component
                let (br, bi) = self.amplitudes[j]; // |1⟩ component
                                                   // Y|0⟩ = i|1⟩ → new amplitude at j: (i * a) = (-ai, ar)
                                                   // Y|1⟩ = -i|0⟩ → new amplitude at i: (-i * b) = (bi, -br)
                self.amplitudes[i] = (bi, -br);
                self.amplitudes[j] = (-ai, ar);
            }
        }
        Ok(())
    }

    /// Apply Pauli Z (phase flip) to `qubit`.
    ///
    /// Z|0⟩ = |0⟩,  Z|1⟩ = -|1⟩
    pub fn apply_z(&mut self, qubit: usize) -> QcResult<()> {
        if qubit >= self.n_qubits {
            return Err(OptimizeError::ValueError(format!(
                "Qubit {qubit} out of range"
            )));
        }
        let bit = 1usize << qubit;
        for (i, amp) in self.amplitudes.iter_mut().enumerate() {
            if i & bit != 0 {
                amp.0 = -amp.0;
                amp.1 = -amp.1;
            }
        }
        Ok(())
    }
}

// ─── Hardware-efficient ansatz ─────────────────────────────────────────────

/// Hardware-efficient variational ansatz.
///
/// Structure per layer:
/// - Ry(θ) on each qubit
/// - Rz(φ) on each qubit
/// - CNOT entangling gates: qubit 0 → 1, qubit 1 → 2, ...
///
/// Total parameters per layer: 2 * n_qubits (Ry angles + Rz angles)
/// Final Ry layer: n_qubits extra parameters
/// Total: (2 * n_layers + 1) * n_qubits
#[derive(Debug, Clone)]
pub struct HardwareEfficientAnsatz {
    /// Number of qubits
    pub n_qubits: usize,
    /// Number of entangling layers
    pub n_layers: usize,
}

impl HardwareEfficientAnsatz {
    /// Create a new hardware-efficient ansatz.
    pub fn new(n_qubits: usize, n_layers: usize) -> Self {
        Self { n_qubits, n_layers }
    }

    /// Total number of parameters: (2 * n_layers + 1) * n_qubits
    pub fn n_params(&self) -> usize {
        (2 * self.n_layers + 1) * self.n_qubits
    }

    /// Build the ansatz statevector from the parameter vector.
    pub fn run(&self, params: &[f64]) -> QcResult<Statevector> {
        let expected = self.n_params();
        if params.len() != expected {
            return Err(OptimizeError::ValueError(format!(
                "Expected {expected} parameters, got {}",
                params.len()
            )));
        }

        let mut state = Statevector::zero_state(self.n_qubits)?;
        let n = self.n_qubits;
        let mut offset = 0;

        for _layer in 0..self.n_layers {
            // Ry rotations
            for q in 0..n {
                state.apply_ry(q, params[offset + q])?;
            }
            offset += n;

            // Rz rotations
            for q in 0..n {
                state.apply_rz(q, params[offset + q])?;
            }
            offset += n;

            // CNOT entangling: chain pattern
            for q in 0..(n - 1) {
                state.apply_cnot(q, q + 1)?;
            }
        }

        // Final Ry layer (no entangling after)
        for q in 0..n {
            state.apply_ry(q, params[offset + q])?;
        }

        Ok(state)
    }
}

// ─── VQE optimizer ─────────────────────────────────────────────────────────

/// VQE optimizer: minimizes ⟨ψ(θ)|H|ψ(θ)⟩ over ansatz parameters θ.
#[derive(Debug, Clone)]
pub struct VqeOptimizer {
    /// The Hamiltonian to minimize
    pub hamiltonian: PauliHamiltonian,
    /// The variational ansatz
    pub ansatz: HardwareEfficientAnsatz,
    /// Initial parameters
    pub init_params: Vec<f64>,
    /// Maximum gradient-descent iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Learning rate for gradient descent
    pub learning_rate: f64,
}

impl VqeOptimizer {
    /// Create a new VQE optimizer.
    pub fn new(
        hamiltonian: PauliHamiltonian,
        ansatz: HardwareEfficientAnsatz,
        init_params: Vec<f64>,
    ) -> Self {
        Self {
            hamiltonian,
            ansatz,
            init_params,
            max_iter: 500,
            tol: 1e-7,
            learning_rate: 0.1,
        }
    }

    /// Evaluate the energy ⟨H⟩ at the given parameters.
    pub fn energy(&self, params: &[f64]) -> QcResult<f64> {
        let state = self.ansatz.run(params)?;
        self.hamiltonian.expectation(&state)
    }

    /// Compute gradient using the parameter-shift rule.
    ///
    /// ∂E/∂θ_k = 0.5 * [E(θ_k + π/2) - E(θ_k - π/2)]
    pub fn gradient(&self, params: &[f64]) -> QcResult<Vec<f64>> {
        let n = params.len();
        let shift = std::f64::consts::FRAC_PI_2;
        let mut grad = vec![0.0; n];

        for k in 0..n {
            let mut p_plus = params.to_vec();
            let mut p_minus = params.to_vec();
            p_plus[k] += shift;
            p_minus[k] -= shift;
            let e_plus = self.energy(&p_plus)?;
            let e_minus = self.energy(&p_minus)?;
            grad[k] = 0.5 * (e_plus - e_minus);
        }
        Ok(grad)
    }

    /// Run VQE optimization using gradient descent with momentum.
    pub fn run(&mut self) -> QcResult<VqeResult> {
        let n_params = self.ansatz.n_params();
        if self.init_params.len() != n_params {
            return Err(OptimizeError::ValueError(format!(
                "Expected {n_params} initial parameters, got {}",
                self.init_params.len()
            )));
        }

        let mut params = self.init_params.clone();
        let mut n_evals = 0usize;

        // Momentum gradient descent (Adam-like adaptive learning rate)
        let beta1: f64 = 0.9;
        let beta2: f64 = 0.999;
        let eps_adam: f64 = 1e-8;
        let mut m = vec![0.0; n_params]; // first moment
        let mut v = vec![0.0; n_params]; // second moment
        let mut t = 0u32;

        let mut prev_energy = self.energy(&params)?;
        n_evals += 1;

        for _iter in 0..self.max_iter {
            let grad = self.gradient(&params)?;
            n_evals += 2 * n_params; // 2 evals per parameter
            t += 1;

            let t_f = t as f64;
            let lr_t =
                self.learning_rate * (1.0 - beta2.powf(t_f)).sqrt() / (1.0 - beta1.powf(t_f));

            for k in 0..n_params {
                m[k] = beta1 * m[k] + (1.0 - beta1) * grad[k];
                v[k] = beta2 * v[k] + (1.0 - beta2) * grad[k] * grad[k];
                params[k] -= lr_t * m[k] / (v[k].sqrt() + eps_adam);
            }

            let energy = self.energy(&params)?;
            n_evals += 1;

            if (prev_energy - energy).abs() < self.tol {
                return Ok(VqeResult {
                    ground_state_energy: energy,
                    optimal_params: params,
                    n_evaluations: n_evals,
                    converged: true,
                });
            }
            prev_energy = energy;
        }

        Ok(VqeResult {
            ground_state_energy: prev_energy,
            optimal_params: params,
            n_evaluations: n_evals,
            converged: false,
        })
    }
}

// ─── Result type ───────────────────────────────────────────────────────────

/// Result of a VQE optimization run.
#[derive(Debug, Clone)]
pub struct VqeResult {
    /// Estimated ground-state energy
    pub ground_state_energy: f64,
    /// Optimal ansatz parameters
    pub optimal_params: Vec<f64>,
    /// Total circuit evaluations
    pub n_evaluations: usize,
    /// Whether the optimizer converged
    pub converged: bool,
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    #[test]
    fn test_pauli_z_expectation_zero_state() {
        // |0⟩: ⟨Z⟩ should be +1
        let sv = Statevector::zero_state(1).unwrap();
        let mut ham = PauliHamiltonian::new(1);
        ham.add_term(1.0, vec![(0, PauliOp::Z)]);
        let e = ham.expectation(&sv).unwrap();
        assert!((e - 1.0).abs() < EPS, "Expected +1, got {e}");
    }

    #[test]
    fn test_pauli_z_expectation_one_state() {
        // |1⟩: ⟨Z⟩ should be -1
        let mut sv = Statevector::zero_state(1).unwrap();
        sv.amplitudes[0] = (0.0, 0.0);
        sv.amplitudes[1] = (1.0, 0.0);
        let mut ham = PauliHamiltonian::new(1);
        ham.add_term(1.0, vec![(0, PauliOp::Z)]);
        let e = ham.expectation(&sv).unwrap();
        assert!((e + 1.0).abs() < EPS, "Expected -1, got {e}");
    }

    #[test]
    fn test_vqe_ising_2qubit_energy() {
        // H = -J * Z0Z1 - h * X0 - h * X1 with J=1, h=0.5
        // Ground state energy for 2-qubit Ising at h=0.5, J=1.0:
        // Eigenvalues of 2x2 Ising can be computed analytically.
        // E_min = -sqrt(J² + h²) ~ -sqrt(1.25) ≈ -1.118 per coupling
        // Full 4x4 ground state ~ -1.5 to -2
        let j = 1.0_f64;
        let h = 0.5_f64;
        let ham = PauliHamiltonian::transverse_ising_1d(2, j, h);

        let ansatz = HardwareEfficientAnsatz::new(2, 2);
        let n_p = ansatz.n_params();
        // Initialize with some non-trivial parameters
        let init_params: Vec<f64> = (0..n_p).map(|i| 0.1 * (i as f64 + 1.0)).collect();

        let mut vqe = VqeOptimizer::new(ham, ansatz, init_params);
        vqe.max_iter = 300;
        vqe.learning_rate = 0.05;

        let result = vqe.run().unwrap();
        // The exact ground state for 2-qubit transverse Ising H = -ZZ - 0.5*X0 - 0.5*X1
        // is approximately -sqrt(2) ≈ -1.414.
        // The pure classical ground state (ZZ only, no transverse field) is -J = -1.0.
        // VQE with HEA should find something better than the classical minimum (-1.0).
        // We require energy ≤ -1.2 to confirm quantum variational improvement.
        assert!(
            result.ground_state_energy <= -1.2 * j,
            "VQE energy {:.4} should be ≤ -1.2 (below classical minimum of -J=-1.0)",
            result.ground_state_energy
        );
    }

    #[test]
    fn test_vqe_gradient_vs_finite_differences() {
        let ham = PauliHamiltonian::transverse_ising_1d(2, 1.0, 1.0);
        let ansatz = HardwareEfficientAnsatz::new(2, 1);
        let n_p = ansatz.n_params();
        let params: Vec<f64> = (0..n_p).map(|i| 0.3 * (i as f64 + 1.0)).collect();
        let init = params.clone();

        let vqe = VqeOptimizer::new(ham, ansatz, init);
        let grad = vqe.gradient(&params).unwrap();

        let eps = 1e-5;
        for k in 0..n_p {
            let mut pp = params.clone();
            let mut pm = params.clone();
            pp[k] += eps;
            pm[k] -= eps;
            let fd = (vqe.energy(&pp).unwrap() - vqe.energy(&pm).unwrap()) / (2.0 * eps);
            assert!(
                (grad[k] - fd).abs() < 0.01,
                "Parameter-shift gradient {:.4} vs FD {:.4} at param {k}",
                grad[k],
                fd
            );
        }
    }

    #[test]
    fn test_hea_norm_preserved() {
        let ansatz = HardwareEfficientAnsatz::new(3, 2);
        let n_p = ansatz.n_params();
        let params: Vec<f64> = (0..n_p).map(|i| 0.5 + 0.1 * i as f64).collect();
        let state = ansatz.run(&params).unwrap();
        let norm = state.norm_squared();
        assert!((norm - 1.0).abs() < 1e-12, "HEA norm must be 1, got {norm}");
    }

    #[test]
    fn test_pauli_x_expectation() {
        // |+⟩ = H|0⟩ → ⟨X⟩ = +1
        let mut sv = Statevector::zero_state(1).unwrap();
        sv.apply_hadamard(0).unwrap();
        let mut ham = PauliHamiltonian::new(1);
        ham.add_term(1.0, vec![(0, PauliOp::X)]);
        let e = ham.expectation(&sv).unwrap();
        assert!((e - 1.0).abs() < 1e-10, "⟨X⟩ of |+⟩ should be 1, got {e}");
    }

    #[test]
    fn test_hea_n_params() {
        // (2*n_layers + 1) * n_qubits
        let ansatz = HardwareEfficientAnsatz::new(4, 3);
        assert_eq!(ansatz.n_params(), 7 * 4);
    }
}
