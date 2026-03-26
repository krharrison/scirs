//! Quantum Phase Estimation (QPE) circuit simulation.
//!
//! QPE estimates the eigenvalue e^{2πiφ} of a unitary operator U given an
//! eigenstate |ψ⟩, using an ancilla register of n_ancilla qubits as a phase
//! readout. The algorithm:
//!
//! 1. Initialise ancilla register in |0…0⟩, eigenstate register in |ψ⟩.
//! 2. Apply Hadamard to each ancilla qubit.
//! 3. For ancilla qubit j (0 = most significant): apply controlled-U^{2^j}
//!    acting on the eigenstate register.
//! 4. Apply inverse QFT to the ancilla register.
//! 5. Measure ancilla register — most probable outcome encodes φ.
//!
//! This module simulates the 1-qubit eigenvalue problem:
//!   U|1⟩ = e^{2πiφ}|1⟩,   U|0⟩ = |0⟩
//! which is sufficient to test and benchmark the full QPE protocol.

use std::f64::consts::PI;

use scirs2_core::numeric::Complex64;

use crate::error::{FFTError, FFTResult};
use crate::quantum::qft::{iqft, QftConfig, Statevector};

/// Result of a Quantum Phase Estimation simulation.
#[derive(Debug, Clone)]
pub struct PhaseEstimationResult {
    /// Estimated phase φ ∈ [0, 1) such that the eigenvalue is e^{2πiφ}.
    pub phase_estimate: f64,
    /// Probability of the dominant ancilla measurement outcome.
    pub confidence: f64,
    /// Full probability distribution over all 2^n_ancilla ancilla states,
    /// ordered from |0…0⟩ to |1…1⟩.
    pub ancilla_probs: Vec<f64>,
}

/// Configuration for Quantum Phase Estimation.
#[derive(Debug, Clone)]
pub struct PhaseEstimationConfig {
    /// Number of ancilla (readout) qubits. Precision ≈ 1 / 2^n_ancilla.
    /// Default: 8 (i.e. ≈ 1/256 precision).
    pub n_ancilla: usize,
    /// Number of independent trials used for iterative QPE (currently unused;
    /// included for API forward compatibility). Default: 1.
    pub n_trials: usize,
}

impl Default for PhaseEstimationConfig {
    fn default() -> Self {
        Self {
            n_ancilla: 8,
            n_trials: 1,
        }
    }
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Apply a single-qubit Hadamard to qubit `qubit` in the full statevector
/// of size 2^n_qubits. Re-exported from qft module logic but simplified here
/// so phase_estimation.rs has no circular dependency.
fn apply_hadamard_ancilla(state: &mut Statevector, qubit: usize, n_qubits: usize) {
    let n = state.len();
    let step = 1_usize << (n_qubits - 1 - qubit);
    let inv_sqrt2 = 1.0_f64 / 2.0_f64.sqrt();

    let mut i = 0;
    while i < n {
        for j in 0..step {
            let idx0 = i + j;
            let idx1 = i + j + step;
            let a0 = state[idx0];
            let a1 = state[idx1];
            state[idx0] = Complex64::new(inv_sqrt2 * (a0.re + a1.re), inv_sqrt2 * (a0.im + a1.im));
            state[idx1] = Complex64::new(inv_sqrt2 * (a0.re - a1.re), inv_sqrt2 * (a0.im - a1.im));
        }
        i += 2 * step;
    }
}

/// Simulate controlled-U^{2^power} for the 1-qubit diagonal unitary
/// U|1⟩ = e^{2πiφ}|1⟩, U|0⟩ = |0⟩.
///
/// The full statevector has n_qubits total qubits: `ancilla` qubits (indices
/// 0..n_ancilla−1) followed by the 1-qubit eigenstate register (index
/// n_ancilla).  When both the ancilla qubit `anc_bit` is |1⟩ and the
/// eigenstate qubit (= target) is |1⟩, we multiply the amplitude by
/// exp(2πi φ 2^power).
fn controlled_phase_power(
    state: &mut Statevector,
    ancilla_qubit: usize,
    target_qubit: usize,
    phase: f64,
    power: u64,
    n_qubits: usize,
) {
    let n = state.len();
    let total_phase = 2.0 * PI * phase * (power as f64);
    let rot = Complex64::new(total_phase.cos(), total_phase.sin());

    let anc_bit = 1_usize << (n_qubits - 1 - ancilla_qubit);
    let tgt_bit = 1_usize << (n_qubits - 1 - target_qubit);
    let both_mask = anc_bit | tgt_bit;

    for idx in 0..n {
        if (idx & both_mask) == both_mask {
            state[idx] *= rot;
        }
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Simulate Quantum Phase Estimation for the 1-qubit diagonal phase unitary
/// U|1⟩ = e^{2πiφ}|1⟩.
///
/// The function constructs a combined ancilla+eigenstate register
/// (n_ancilla + 1 qubits total), performs the full QPE circuit, and returns
/// the estimated phase together with the ancilla measurement distribution.
///
/// # Arguments
/// * `phase` – True phase φ ∈ [0, 1).
/// * `config` – QPE configuration.
///
/// # Returns
/// A [`PhaseEstimationResult`] containing the estimate, confidence, and full
/// ancilla probability distribution.
///
/// # Errors
/// Returns `FFTError` if n_ancilla is 0 or the QFT step fails.
///
/// # Examples
/// ```
/// use scirs2_fft::quantum::{PhaseEstimationConfig, quantum_phase_estimation};
///
/// // φ = 0.5 is representable exactly with 1 bit, so 4 ancilla qubits suffice.
/// let cfg = PhaseEstimationConfig { n_ancilla: 4, ..Default::default() };
/// let result = quantum_phase_estimation(0.5, &cfg).unwrap();
/// assert!((result.phase_estimate - 0.5).abs() < 1.0 / 16.0);
/// ```
pub fn quantum_phase_estimation(
    phase: f64,
    config: &PhaseEstimationConfig,
) -> FFTResult<PhaseEstimationResult> {
    let na = config.n_ancilla;
    if na == 0 {
        return Err(FFTError::ValueError(
            "n_ancilla must be at least 1".to_string(),
        ));
    }

    // Total qubits: na ancilla + 1 eigenstate qubit
    let n_total = na + 1;
    let dim = 1_usize << n_total;

    // Initialise: ancilla = |0…0⟩, eigenstate = |1⟩  →  |0…0 1⟩
    // Index in little-endian-like layout: ancilla is high-order, eigenstate is qubit n_total−1
    let eigenstate_bit = 1_usize; // qubit index n_total−1 maps to bit 0 of index
    let mut state = vec![Complex64::new(0.0, 0.0); dim];
    state[eigenstate_bit] = Complex64::new(1.0, 0.0);

    // Step 1: Hadamard on each ancilla qubit
    for anc in 0..na {
        apply_hadamard_ancilla(&mut state, anc, n_total);
    }

    // Step 2: Controlled-U^{2^k} for each ancilla qubit.
    // Following the standard QPE circuit: ancilla qubit j (0 = most significant)
    // controls U^{2^(na-1-j)}, so that after the IQFT the ancilla register
    // directly encodes the binary expansion of φ with the MSB at qubit 0.
    for anc in 0..na {
        let k = na - 1 - anc; // so qubit 0 controls U^{2^(na-1)}, qubit na-1 controls U^1
        let power = 1_u64 << k;
        controlled_phase_power(&mut state, anc, na, phase, power, n_total);
    }

    // Step 3: Inverse QFT on the ancilla register only.
    // We extract the ancilla register marginal state, apply IQFT, then rebuild
    // the full statevector (since eigenstate is |1⟩ throughout, we can factorise).
    //
    // Because the eigenstate register is |1⟩ (never changed, controlled ops only
    // add phase when it is |1⟩), the full state is:
    //   |ancilla_state⟩ ⊗ |1⟩
    // We can extract the ancilla amplitudes by looking at indices with bit0 = 1.
    let ancilla_dim = 1_usize << na;
    let mut ancilla_state: Statevector = (0..ancilla_dim)
        .map(|anc_idx| {
            // Full index = (anc_idx << 1) | 1  (eigenstate bit = 1)
            let full_idx = (anc_idx << 1) | 1;
            if full_idx < dim {
                state[full_idx]
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect();

    // Apply IQFT to ancilla state
    let iqft_cfg = QftConfig {
        n_qubits: na,
        apply_swap: true,
        approximate: false,
        approx_threshold: 1e-10,
    };
    ancilla_state = iqft(&ancilla_state, &iqft_cfg)?;

    // Measurement probabilities for ancilla
    let ancilla_probs: Vec<f64> = ancilla_state
        .iter()
        .map(|a| a.re * a.re + a.im * a.im)
        .collect();

    // Most likely outcome
    let (best_idx, &best_prob) = ancilla_probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| {
            FFTError::ComputationError("Empty ancilla probability vector".to_string())
        })?;

    let phase_estimate = best_idx as f64 / ancilla_dim as f64;

    Ok(PhaseEstimationResult {
        phase_estimate,
        confidence: best_prob,
        ancilla_probs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_phase_estimation_half() {
        // φ = 0.5 is exactly representable with 1 ancilla bit, so with
        // 4 ancilla qubits we expect probability ≈1 for the |1000⟩ state (index 8).
        let cfg = PhaseEstimationConfig {
            n_ancilla: 4,
            ..Default::default()
        };
        let result = quantum_phase_estimation(0.5, &cfg).expect("QPE failed");
        assert!(
            (result.phase_estimate - 0.5).abs() < 1.0 / 16.0,
            "Expected estimate ≈ 0.5 but got {}",
            result.phase_estimate
        );
        assert!(result.confidence > 0.9, "Expected high confidence");
    }

    #[test]
    fn test_phase_estimation_quarter() {
        // φ = 0.25 is representable exactly with 2 ancilla bits.
        let cfg = PhaseEstimationConfig {
            n_ancilla: 6,
            ..Default::default()
        };
        let result = quantum_phase_estimation(0.25, &cfg).expect("QPE failed");
        let tolerance = 1.0 / (1_u64 << cfg.n_ancilla) as f64;
        assert!(
            (result.phase_estimate - 0.25).abs() <= tolerance,
            "Expected estimate within {tolerance} of 0.25 but got {}",
            result.phase_estimate
        );
    }

    #[test]
    fn test_phase_estimation_zero() {
        // φ = 0.0 → U is identity on |1⟩; all ancilla phases are 0.
        // The most likely outcome is |0…0⟩.
        let cfg = PhaseEstimationConfig {
            n_ancilla: 4,
            ..Default::default()
        };
        let result = quantum_phase_estimation(0.0, &cfg).expect("QPE failed");
        assert_relative_eq!(result.phase_estimate, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_phase_estimation_probabilities_sum_to_one() {
        let cfg = PhaseEstimationConfig {
            n_ancilla: 5,
            ..Default::default()
        };
        let result = quantum_phase_estimation(0.375, &cfg).expect("QPE failed");
        let total: f64 = result.ancilla_probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Ancilla probabilities should sum to 1; got {total}"
        );
    }

    #[test]
    fn test_phase_estimation_error_zero_ancilla() {
        let cfg = PhaseEstimationConfig {
            n_ancilla: 0,
            n_trials: 1,
        };
        let result = quantum_phase_estimation(0.5, &cfg);
        assert!(result.is_err());
    }

    #[test]
    fn test_phase_estimation_increasing_precision() {
        // With more ancilla qubits the estimate should be more accurate.
        let phase = 0.3;
        let cfg4 = PhaseEstimationConfig {
            n_ancilla: 4,
            n_trials: 1,
        };
        let cfg8 = PhaseEstimationConfig {
            n_ancilla: 8,
            n_trials: 1,
        };

        let r4 = quantum_phase_estimation(phase, &cfg4).expect("QPE4 failed");
        let r8 = quantum_phase_estimation(phase, &cfg8).expect("QPE8 failed");

        let err4 = (r4.phase_estimate - phase).abs();
        let err8 = (r8.phase_estimate - phase).abs();

        // 8 ancilla bits can represent multiples of 1/256 ≈ 0.0039
        // 4 ancilla bits can represent multiples of 1/16 = 0.0625
        // err8 should be ≤ 1/256, err4 ≤ 1/16
        assert!(
            err8 <= 1.0 / 256.0 + 1e-10,
            "8-ancilla error {err8} exceeds 1/256"
        );
        assert!(
            err4 <= 1.0 / 16.0 + 1e-10,
            "4-ancilla error {err4} exceeds 1/16"
        );
    }
}
