//! N-qubit statevector simulator
//!
//! Represents quantum states as dense complex amplitude vectors of size 2^n,
//! and applies single- and two-qubit gates exactly.

use crate::error::OptimizeError;
use crate::quantum_classical::QcResult;

/// Complex multiply: (a+ib)(c+id) = (ac-bd) + i(ad+bc)
#[inline]
pub fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// Complex add: (a+ib) + (c+id) = (a+c) + i(b+d)
#[inline]
pub fn cadd(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 + b.0, a.1 + b.1)
}

/// Complex magnitude squared: |z|² = re² + im²
#[inline]
pub fn cabs2(z: (f64, f64)) -> f64 {
    z.0 * z.0 + z.1 * z.1
}

/// N-qubit statevector: stores 2^n complex amplitudes as (re, im) pairs.
///
/// Qubit indexing convention: qubit 0 is the *least* significant bit.
/// So the basis state |b_{n-1} ... b_1 b_0⟩ corresponds to index
/// `b_0 + 2*b_1 + ... + 2^(n-1)*b_{n-1}`.
#[derive(Debug, Clone)]
pub struct Statevector {
    /// Complex amplitudes: `amplitudes[k] = (re, im)` for basis state `|k⟩`
    pub amplitudes: Vec<(f64, f64)>,
    /// Number of qubits
    pub n_qubits: usize,
}

impl Statevector {
    /// Create the zero state |0...0⟩ for `n` qubits.
    pub fn zero_state(n: usize) -> QcResult<Self> {
        if n == 0 {
            return Err(OptimizeError::ValueError(
                "Number of qubits must be at least 1".to_string(),
            ));
        }
        if n > 30 {
            return Err(OptimizeError::ValueError(format!(
                "Too many qubits: {n}; maximum supported is 30"
            )));
        }
        let dim = 1usize << n;
        let mut amplitudes = vec![(0.0_f64, 0.0_f64); dim];
        amplitudes[0] = (1.0, 0.0);
        Ok(Self {
            amplitudes,
            n_qubits: n,
        })
    }

    /// Total norm squared: should remain 1.0 after any unitary operation.
    pub fn norm_squared(&self) -> f64 {
        self.amplitudes.iter().map(|&z| cabs2(z)).sum()
    }

    /// Total norm (Euclidean).
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Apply Hadamard gate to `qubit`.
    ///
    /// H = (1/√2) [[1, 1], [1, -1]]
    ///
    /// Pairs basis states that differ only in bit `qubit`.
    pub fn apply_hadamard(&mut self, qubit: usize) -> QcResult<()> {
        self.check_qubit(qubit)?;
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let dim = self.amplitudes.len();
        let bit = 1usize << qubit;

        for i in 0..dim {
            if i & bit == 0 {
                let j = i | bit; // partner with bit `qubit` set
                let (a, b) = (self.amplitudes[i], self.amplitudes[j]);
                self.amplitudes[i] = ((a.0 + b.0) * inv_sqrt2, (a.1 + b.1) * inv_sqrt2);
                self.amplitudes[j] = ((a.0 - b.0) * inv_sqrt2, (a.1 - b.1) * inv_sqrt2);
            }
        }
        Ok(())
    }

    /// Apply Rz(θ) gate to `qubit`.
    ///
    /// Rz(θ) = [[e^{-iθ/2}, 0], [0, e^{iθ/2}]]
    ///
    /// States with bit `qubit` = 0 get phase e^{-iθ/2}; states with bit = 1
    /// get phase e^{+iθ/2}.
    pub fn apply_rz(&mut self, qubit: usize, theta: f64) -> QcResult<()> {
        self.check_qubit(qubit)?;
        let half = theta / 2.0;
        let phase0 = (half.cos(), -half.sin()); // e^{-iθ/2}
        let phase1 = (half.cos(), half.sin()); // e^{+iθ/2}
        let bit = 1usize << qubit;

        for (i, amp) in self.amplitudes.iter_mut().enumerate() {
            if i & bit == 0 {
                *amp = cmul(*amp, phase0);
            } else {
                *amp = cmul(*amp, phase1);
            }
        }
        Ok(())
    }

    /// Apply Rx(θ) gate to `qubit`.
    ///
    /// Rx(θ) = [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]
    pub fn apply_rx(&mut self, qubit: usize, theta: f64) -> QcResult<()> {
        self.check_qubit(qubit)?;
        let half = theta / 2.0;
        let c = half.cos();
        let s = half.sin();
        let bit = 1usize << qubit;
        let dim = self.amplitudes.len();

        for i in 0..dim {
            if i & bit == 0 {
                let j = i | bit;
                let (a, b) = (self.amplitudes[i], self.amplitudes[j]);
                // |0⟩ → c|0⟩ - i·s|1⟩
                // |1⟩ → -i·s|0⟩ + c|1⟩
                self.amplitudes[i] = cadd(
                    (a.0 * c, a.1 * c),
                    (b.1 * s, -b.0 * s), // -i*s * b = (b.im*s, -b.re*s)
                );
                self.amplitudes[j] = cadd(
                    (a.1 * s, -a.0 * s), // -i*s * a
                    (b.0 * c, b.1 * c),
                );
            }
        }
        Ok(())
    }

    /// Apply Ry(θ) gate to `qubit`.
    ///
    /// Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
    pub fn apply_ry(&mut self, qubit: usize, theta: f64) -> QcResult<()> {
        self.check_qubit(qubit)?;
        let half = theta / 2.0;
        let c = half.cos();
        let s = half.sin();
        let bit = 1usize << qubit;
        let dim = self.amplitudes.len();

        for i in 0..dim {
            if i & bit == 0 {
                let j = i | bit;
                let (a, b) = (self.amplitudes[i], self.amplitudes[j]);
                // |0⟩ → c|0⟩ - s|1⟩,  |1⟩ → s|0⟩ + c|1⟩
                self.amplitudes[i] = (a.0 * c - b.0 * s, a.1 * c - b.1 * s);
                self.amplitudes[j] = (a.0 * s + b.0 * c, a.1 * s + b.1 * c);
            }
        }
        Ok(())
    }

    /// Apply CNOT gate with `control` qubit and `target` qubit.
    ///
    /// Flips the target qubit when the control qubit is |1⟩.
    pub fn apply_cnot(&mut self, control: usize, target: usize) -> QcResult<()> {
        self.check_qubit(control)?;
        self.check_qubit(target)?;
        if control == target {
            return Err(OptimizeError::ValueError(
                "CNOT control and target must be different qubits".to_string(),
            ));
        }
        let ctrl_bit = 1usize << control;
        let tgt_bit = 1usize << target;
        let dim = self.amplitudes.len();

        for i in 0..dim {
            // Only process states where control is |1⟩ and target is |0⟩
            if (i & ctrl_bit != 0) && (i & tgt_bit == 0) {
                let j = i | tgt_bit; // same state but with target flipped to |1⟩
                self.amplitudes.swap(i, j);
            }
        }
        Ok(())
    }

    /// Apply Rzz(θ) gate to qubits `q1` and `q2`.
    ///
    /// Rzz(θ) = e^{-iθ/2 · Z⊗Z}
    ///
    /// For basis state |b1, b2⟩:
    /// - If b1 XOR b2 = 0 (same bits): phase e^{-iθ/2}
    /// - If b1 XOR b2 = 1 (different bits): phase e^{+iθ/2}
    pub fn apply_rzz(&mut self, q1: usize, q2: usize, theta: f64) -> QcResult<()> {
        self.check_qubit(q1)?;
        self.check_qubit(q2)?;
        if q1 == q2 {
            return Err(OptimizeError::ValueError(
                "Rzz: q1 and q2 must be different qubits".to_string(),
            ));
        }
        let half = theta / 2.0;
        let phase_same = (half.cos(), -half.sin()); // e^{-iθ/2} when ZZ eigenvalue = +1
        let phase_diff = (half.cos(), half.sin()); // e^{+iθ/2} when ZZ eigenvalue = -1
        let bit1 = 1usize << q1;
        let bit2 = 1usize << q2;

        for (i, amp) in self.amplitudes.iter_mut().enumerate() {
            let b1 = (i & bit1) != 0;
            let b2 = (i & bit2) != 0;
            if b1 == b2 {
                *amp = cmul(*amp, phase_same);
            } else {
                *amp = cmul(*amp, phase_diff);
            }
        }
        Ok(())
    }

    /// Compute ⟨Z_i Z_j⟩ expectation value.
    ///
    /// Z_i Z_j has eigenvalue +1 when bits i and j are equal, -1 otherwise.
    pub fn expectation_zz(&self, q1: usize, q2: usize) -> QcResult<f64> {
        self.check_qubit(q1)?;
        self.check_qubit(q2)?;
        let bit1 = 1usize << q1;
        let bit2 = 1usize << q2;

        let value = self
            .amplitudes
            .iter()
            .enumerate()
            .map(|(i, &amp)| {
                let b1 = (i & bit1) != 0;
                let b2 = (i & bit2) != 0;
                let sign = if b1 == b2 { 1.0 } else { -1.0 };
                sign * cabs2(amp)
            })
            .sum();
        Ok(value)
    }

    /// Compute ⟨Z_k⟩ expectation value.
    ///
    /// Z_k has eigenvalue +1 when bit k is 0, and -1 when bit k is 1.
    pub fn expectation_z(&self, qubit: usize) -> QcResult<f64> {
        self.check_qubit(qubit)?;
        let bit = 1usize << qubit;

        let value = self
            .amplitudes
            .iter()
            .enumerate()
            .map(|(i, &amp)| {
                let sign = if i & bit == 0 { 1.0 } else { -1.0 };
                sign * cabs2(amp)
            })
            .sum();
        Ok(value)
    }

    /// Return the index of the basis state with the highest probability.
    pub fn most_probable_state(&self) -> usize {
        self.amplitudes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                cabs2(**a)
                    .partial_cmp(&cabs2(**b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Decode a basis state index into a bitstring (qubit 0 = LSB).
    pub fn index_to_bits(&self, idx: usize) -> Vec<bool> {
        (0..self.n_qubits).map(|k| (idx >> k) & 1 == 1).collect()
    }

    fn check_qubit(&self, qubit: usize) -> QcResult<()> {
        if qubit >= self.n_qubits {
            return Err(OptimizeError::ValueError(format!(
                "Qubit index {qubit} out of range for {}-qubit register",
                self.n_qubits
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    #[test]
    fn test_zero_state_amplitude() {
        let sv = Statevector::zero_state(3).unwrap();
        assert_eq!(sv.amplitudes.len(), 8);
        assert!((sv.amplitudes[0].0 - 1.0).abs() < EPS);
        assert!(sv.amplitudes[0].1.abs() < EPS);
        for &amp in &sv.amplitudes[1..] {
            assert!(cabs2(amp) < EPS);
        }
    }

    #[test]
    fn test_hadamard_creates_plus_state() {
        let mut sv = Statevector::zero_state(1).unwrap();
        sv.apply_hadamard(0).unwrap();
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!((sv.amplitudes[0].0 - inv_sqrt2).abs() < EPS);
        assert!((sv.amplitudes[1].0 - inv_sqrt2).abs() < EPS);
        assert!(sv.amplitudes[0].1.abs() < EPS);
        assert!(sv.amplitudes[1].1.abs() < EPS);
    }

    #[test]
    fn test_cnot_10_to_11() {
        // Prepare |10⟩: qubit 1 = 1, qubit 0 = 0 → index = 2 (binary 10)
        let mut sv = Statevector::zero_state(2).unwrap();
        // Apply X (= H Rz(π) H) to qubit 1 to set it to |1⟩
        // Simpler: directly set amplitude
        sv.amplitudes[0] = (0.0, 0.0);
        sv.amplitudes[2] = (1.0, 0.0); // index 2 = |10⟩
        sv.apply_cnot(1, 0).unwrap();
        // After CNOT: control=1(=bit 1), target=0 → |10⟩ → |11⟩ (index 3)
        assert!(cabs2(sv.amplitudes[3]) > 1.0 - EPS);
        assert!(cabs2(sv.amplitudes[2]) < EPS);
    }

    #[test]
    fn test_rz_phase_rotation() {
        // Rz(π)|0⟩ should give e^{-iπ/2}|0⟩ = -i|0⟩
        let mut sv = Statevector::zero_state(1).unwrap();
        sv.apply_rz(0, std::f64::consts::PI).unwrap();
        assert!(sv.amplitudes[0].0.abs() < EPS);
        assert!((sv.amplitudes[0].1 + 1.0).abs() < EPS); // -i
    }

    #[test]
    fn test_norm_preserved_after_gates() {
        let mut sv = Statevector::zero_state(3).unwrap();
        sv.apply_hadamard(0).unwrap();
        sv.apply_hadamard(1).unwrap();
        sv.apply_cnot(0, 1).unwrap();
        sv.apply_rz(2, 0.7).unwrap();
        sv.apply_rzz(0, 2, 1.2).unwrap();
        let norm = sv.norm_squared();
        assert!((norm - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_expectation_z_basis_states() {
        // |0⟩ → ⟨Z⟩ = +1
        let sv0 = Statevector::zero_state(1).unwrap();
        let ez0 = sv0.expectation_z(0).unwrap();
        assert!((ez0 - 1.0).abs() < EPS);

        // |1⟩ → ⟨Z⟩ = -1
        let mut sv1 = Statevector::zero_state(1).unwrap();
        sv1.amplitudes[0] = (0.0, 0.0);
        sv1.amplitudes[1] = (1.0, 0.0);
        let ez1 = sv1.expectation_z(0).unwrap();
        assert!((ez1 + 1.0).abs() < EPS);
    }

    #[test]
    fn test_expectation_zz() {
        // |00⟩ → ⟨Z0 Z1⟩ = +1
        let sv = Statevector::zero_state(2).unwrap();
        let ezz = sv.expectation_zz(0, 1).unwrap();
        assert!((ezz - 1.0).abs() < EPS);

        // |10⟩ → bit0=0, bit1=1 → different → ⟨ZZ⟩ = -1
        let mut sv2 = Statevector::zero_state(2).unwrap();
        sv2.amplitudes[0] = (0.0, 0.0);
        sv2.amplitudes[2] = (1.0, 0.0); // index 2 = bit1=1,bit0=0
        let ezz2 = sv2.expectation_zz(0, 1).unwrap();
        assert!((ezz2 + 1.0).abs() < EPS);
    }
}
