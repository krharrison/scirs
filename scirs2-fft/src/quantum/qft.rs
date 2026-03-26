//! Quantum Fourier Transform (QFT) circuit simulation.
//!
//! This module implements statevector simulation of the QFT circuit, which is
//! the quantum analog of the Discrete Fourier Transform (DFT). For an n-qubit
//! register of size N = 2^n:
//!
//! ```text
//! QFT |j⟩ = (1/√N) Σ_{k=0}^{N-1} exp(2πi j k / N) |k⟩
//! ```
//!
//! The circuit consists of Hadamard gates followed by controlled phase-rotation
//! gates R\_k = \[\[1,0\],\[0,exp(2*pi*i/2^k)\]\], then optionally bit-reversal SWAP gates.

use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

use crate::error::{FFTError, FFTResult};

/// An n-qubit statevector: 2^n complex amplitudes in computational basis order.
///
/// Index `j` represents basis state |j⟩. The amplitudes satisfy
/// Σ|a_j|² = 1 (normalisation is maintained by the QFT circuit).
pub type Statevector = Vec<Complex64>;

/// Configuration for the Quantum Fourier Transform.
#[derive(Debug, Clone)]
pub struct QftConfig {
    /// Number of qubits in the register. Register dimension = 2^n_qubits.
    pub n_qubits: usize,
    /// If `true` (default), apply bit-reversal SWAP gates at the end so the
    /// output ordering matches the standard DFT ordering.
    pub apply_swap: bool,
    /// If `true`, drop controlled-phase gates with |θ| < `approx_threshold`,
    /// enabling the approximate QFT used in many quantum algorithms.
    pub approximate: bool,
    /// Rotation angle threshold below which gates are dropped when `approximate`
    /// is `true`. Default: 1e-10.
    pub approx_threshold: f64,
}

impl Default for QftConfig {
    fn default() -> Self {
        Self {
            n_qubits: 3,
            apply_swap: true,
            approximate: false,
            approx_threshold: 1e-10,
        }
    }
}

// ── Internal gate helpers ────────────────────────────────────────────────────

/// Apply a single-qubit Hadamard to qubit `qubit` (0 = most significant).
///
/// Iterates over all 2^n amplitude pairs and applies H = (1/√2)[[1,1],[1,-1]].
fn apply_hadamard(state: &mut Statevector, qubit: usize, n_qubits: usize) {
    let n = state.len(); // = 2^n_qubits
    let step = 1_usize << (n_qubits - 1 - qubit);
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();

    let mut i = 0;
    while i < n {
        let block_start = i;
        // Each block of 2*step amplitudes contains one |0⟩ half and one |1⟩ half
        // for this qubit.
        for j in 0..step {
            let idx0 = block_start + j;
            let idx1 = block_start + j + step;
            let a0 = state[idx0];
            let a1 = state[idx1];
            state[idx0] = Complex64::new(inv_sqrt2 * (a0.re + a1.re), inv_sqrt2 * (a0.im + a1.im));
            state[idx1] = Complex64::new(inv_sqrt2 * (a0.re - a1.re), inv_sqrt2 * (a0.im - a1.im));
        }
        i += 2 * step;
    }
}

/// Apply controlled-R_k phase gate: if `control` qubit is |1⟩ AND `target`
/// qubit is |1⟩, multiply amplitude by exp(2πi / 2^k).
///
/// In the QFT circuit `k` is the distance (j+1) where `control` is qubit j
/// and `target` is the qubit being processed.
fn apply_controlled_phase(
    state: &mut Statevector,
    control: usize,
    target: usize,
    k: usize,
    n_qubits: usize,
) {
    let n = state.len();
    let theta = 2.0 * PI / (1_u64 << k) as f64;
    let phase = Complex64::new(theta.cos(), theta.sin());

    let control_bit = 1_usize << (n_qubits - 1 - control);
    let target_bit = 1_usize << (n_qubits - 1 - target);
    let both_mask = control_bit | target_bit;

    for idx in 0..n {
        // Gate fires when both control and target qubits are |1⟩
        if (idx & both_mask) == both_mask {
            state[idx] *= phase;
        }
    }
}

/// Apply a SWAP gate exchanging the amplitudes of qubits `q1` and `q2`.
fn apply_swap(state: &mut Statevector, q1: usize, q2: usize, n_qubits: usize) {
    let n = state.len();
    let bit1 = 1_usize << (n_qubits - 1 - q1);
    let bit2 = 1_usize << (n_qubits - 1 - q2);

    for idx in 0..n {
        // Only process pairs once: when bit1 is set but bit2 is not
        if (idx & bit1) != 0 && (idx & bit2) == 0 {
            let other = (idx & !bit1) | bit2;
            state.swap(idx, other);
        }
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Quantum Fourier Transform on an n-qubit statevector.
///
/// Implements the standard QFT circuit: for each qubit t from 0 to n-1 apply
/// a Hadamard gate followed by controlled-R_k gates from each subsequent qubit
/// j > t, then (optionally) bit-reversal SWAP gates at the end.
///
/// # Arguments
/// * `state` - Input statevector of length 2^n.
/// * `config` - QFT configuration.
///
/// # Returns
/// The transformed statevector (same length), or `FFTError` if the state size
/// is not a power of two consistent with `config.n_qubits`.
///
/// # Examples
/// ```
/// use scirs2_fft::quantum::{QftConfig, basis_state, qft, measure_probs};
/// use approx::assert_relative_eq;
///
/// // QFT|0⟩ on 2 qubits gives uniform superposition
/// let cfg = QftConfig { n_qubits: 2, ..Default::default() };
/// let state = basis_state(0, 2);
/// let out = qft(&state, &cfg).unwrap();
/// let probs = measure_probs(&out);
/// for p in &probs {
///     assert_relative_eq!(*p, 0.25, epsilon = 1e-12);
/// }
/// ```
pub fn qft(state: &Statevector, config: &QftConfig) -> FFTResult<Statevector> {
    let n = config.n_qubits;
    let expected_len = 1_usize << n;

    if state.len() != expected_len {
        return Err(FFTError::DimensionError(format!(
            "Statevector length {} does not match 2^{} = {}",
            state.len(),
            n,
            expected_len
        )));
    }
    if n == 0 {
        return Ok(state.clone());
    }

    let mut out = state.clone();

    // Apply QFT circuit qubit by qubit
    for t in 0..n {
        apply_hadamard(&mut out, t, n);
        for j in (t + 1)..n {
            let k = j - t + 1; // rotation order: R_2, R_3, ...
            let theta = 2.0 * PI / (1_u64 << k) as f64;
            if config.approximate && theta.abs() < config.approx_threshold {
                continue;
            }
            apply_controlled_phase(&mut out, j, t, k, n);
        }
    }

    // Bit-reversal permutation via SWAP gates
    if config.apply_swap {
        for i in 0..(n / 2) {
            apply_swap(&mut out, i, n - 1 - i, n);
        }
    }

    Ok(out)
}

/// Inverse Quantum Fourier Transform.
///
/// Applies the QFT† circuit: reverse qubit order, reverse gate order, and
/// negate all rotation angles.
///
/// # Examples
/// ```
/// use scirs2_fft::quantum::{QftConfig, basis_state, qft, iqft};
/// use approx::assert_relative_eq;
///
/// let n = 3;
/// let cfg = QftConfig { n_qubits: n, ..Default::default() };
/// let state = basis_state(5, n);
/// let transformed = qft(&state, &cfg).unwrap();
/// let recovered = iqft(&transformed, &cfg).unwrap();
/// for (a, b) in state.iter().zip(recovered.iter()) {
///     assert_relative_eq!(a.re, b.re, epsilon = 1e-10);
///     assert_relative_eq!(a.im, b.im, epsilon = 1e-10);
/// }
/// ```
pub fn iqft(state: &Statevector, config: &QftConfig) -> FFTResult<Statevector> {
    let n = config.n_qubits;
    let expected_len = 1_usize << n;

    if state.len() != expected_len {
        return Err(FFTError::DimensionError(format!(
            "Statevector length {} does not match 2^{} = {}",
            state.len(),
            n,
            expected_len
        )));
    }
    if n == 0 {
        return Ok(state.clone());
    }

    let mut out = state.clone();

    // Undo bit-reversal first (same SWAP circuit, self-inverse)
    if config.apply_swap {
        for i in 0..(n / 2) {
            apply_swap(&mut out, i, n - 1 - i, n);
        }
    }

    // Apply inverse QFT circuit (reverse qubit order, negate phases)
    for t in (0..n).rev() {
        for j in (t + 1..n).rev() {
            let k = j - t + 1;
            let theta = 2.0 * PI / (1_u64 << k) as f64;
            if config.approximate && theta.abs() < config.approx_threshold {
                continue;
            }
            // Inverse controlled-phase: negate θ
            apply_controlled_phase_neg(&mut out, j, t, k, n);
        }
        apply_hadamard(&mut out, t, n);
    }

    Ok(out)
}

/// Apply controlled-R_k† (conjugate-transpose, i.e. negative phase rotation).
fn apply_controlled_phase_neg(
    state: &mut Statevector,
    control: usize,
    target: usize,
    k: usize,
    n_qubits: usize,
) {
    let n = state.len();
    let theta = -2.0 * PI / (1_u64 << k) as f64;
    let phase = Complex64::new(theta.cos(), theta.sin());

    let control_bit = 1_usize << (n_qubits - 1 - control);
    let target_bit = 1_usize << (n_qubits - 1 - target);
    let both_mask = control_bit | target_bit;

    for idx in 0..n {
        if (idx & both_mask) == both_mask {
            state[idx] *= phase;
        }
    }
}

/// Construct the full N×N QFT unitary matrix for n_qubits qubits.
///
/// Entry (row k, col j) = exp(2πi j k / N) / √N, identical to the
/// normalised DFT matrix. Useful for validation.
///
/// # Examples
/// ```
/// use scirs2_fft::quantum::qft_matrix;
/// use approx::assert_relative_eq;
///
/// let m = qft_matrix(2); // 4×4 matrix
/// // First row: all 1/2
/// for j in 0..4 {
///     assert_relative_eq!(m[[0, j]].re, 0.5, epsilon = 1e-12);
///     assert_relative_eq!(m[[0, j]].im, 0.0, epsilon = 1e-12);
/// }
/// ```
pub fn qft_matrix(n_qubits: usize) -> Array2<Complex64> {
    let n = 1_usize << n_qubits;
    let inv_sqrt_n = 1.0 / (n as f64).sqrt();
    let mut mat = Array2::zeros((n, n));

    for row in 0..n {
        for col in 0..n {
            let angle = 2.0 * PI * (row * col) as f64 / n as f64;
            mat[[row, col]] = Complex64::new(inv_sqrt_n * angle.cos(), inv_sqrt_n * angle.sin());
        }
    }
    mat
}

/// Construct the computational basis state |j⟩ as a statevector of length 2^n.
///
/// The returned vector has amplitude 1+0i at index j and 0 elsewhere.
///
/// # Examples
/// ```
/// use scirs2_fft::quantum::basis_state;
///
/// let sv = basis_state(3, 3); // |011⟩ in 3-qubit space
/// assert_eq!(sv.len(), 8);
/// assert!((sv[3].re - 1.0).abs() < 1e-15);
/// ```
pub fn basis_state(j: usize, n_qubits: usize) -> Statevector {
    let n = 1_usize << n_qubits;
    let mut sv = vec![Complex64::new(0.0, 0.0); n];
    if j < n {
        sv[j] = Complex64::new(1.0, 0.0);
    }
    sv
}

/// Compute measurement probabilities |a_j|² for each basis state.
///
/// # Examples
/// ```
/// use scirs2_fft::quantum::{basis_state, measure_probs};
///
/// let sv = basis_state(2, 2);
/// let probs = measure_probs(&sv);
/// assert!((probs[2] - 1.0).abs() < 1e-15);
/// ```
pub fn measure_probs(state: &Statevector) -> Vec<f64> {
    state.iter().map(|a| a.re * a.re + a.im * a.im).collect()
}

/// Compute the DFT of a real-valued signal using QFT circuit simulation.
///
/// This demonstrates the equivalence between QFT and DFT: the signal length
/// must be a power of 2, and the result matches the standard DFT
/// `X[k] = Sum_{j} x[j] exp(-2*pi*i*j*k / N)` (unnormalised, using QFT convention).
///
/// Because the QFT uses the *positive* exponent convention, the output is the
/// complex conjugate of the standard DFT; we conjugate at the end to match
/// the conventional (negative-exponent) DFT.
///
/// # Errors
/// Returns `FFTError::DimensionError` if the signal length is not a power of 2.
///
/// # Examples
/// ```
/// use scirs2_fft::quantum::qft_dft;
/// use approx::assert_relative_eq;
/// use std::f64::consts::PI;
///
/// let n = 4;
/// let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).cos()).collect();
/// let dft = qft_dft(&signal).unwrap();
/// // cos(2π k/N) has DFT with peaks at k=1 and k=N-1
/// assert_relative_eq!(dft[1].re, 2.0, epsilon = 1e-9);
/// ```
pub fn qft_dft(signal: &[f64]) -> FFTResult<Vec<Complex64>> {
    let n = signal.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Check power-of-two
    if !n.is_power_of_two() {
        return Err(FFTError::DimensionError(format!(
            "Signal length {n} is not a power of two; QFT requires power-of-two sizes"
        )));
    }
    let n_qubits = n.trailing_zeros() as usize;

    // Build superposition state: |ψ⟩ = Σ_j x[j] |j⟩  (generally not normalised)
    let state: Statevector = signal.iter().map(|&v| Complex64::new(v, 0.0)).collect();

    let config = QftConfig {
        n_qubits,
        apply_swap: true,
        approximate: false,
        approx_threshold: 1e-10,
    };

    let qft_out = qft(&state, &config)?;

    // QFT uses +2πi convention; conjugate to get standard (−2πi) DFT
    // and scale by √N to match unnormalised DFT
    let sqrt_n = (n as f64).sqrt();
    Ok(qft_out
        .iter()
        .map(|c| Complex64::new(c.re * sqrt_n, -c.im * sqrt_n))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_qft_basis_state_0_uniform_superposition() {
        // QFT|0⟩ on n qubits should give uniform superposition with
        // amplitude 1/√N at every basis state.
        for n in 1..=4 {
            let cfg = QftConfig {
                n_qubits: n,
                ..Default::default()
            };
            let state = basis_state(0, n);
            let out = qft(&state, &cfg).expect("qft failed");
            let probs = measure_probs(&out);
            let expected = 1.0 / (1_usize << n) as f64;
            for p in &probs {
                assert!(
                    (*p - expected).abs() < 1e-12,
                    "n={n}: probability should be {expected} but got {p}"
                );
            }
        }
    }

    #[test]
    fn test_qft_matrix_equals_dft_matrix() {
        // The QFT matrix should equal the normalised DFT matrix.
        // Entry (k,j) = exp(2πi jk/N) / √N
        let n = 3;
        let mat = qft_matrix(n);
        let dim = 1_usize << n;

        for k in 0..dim {
            for j in 0..dim {
                let angle = 2.0 * PI * (j * k) as f64 / dim as f64;
                let expected_re = angle.cos() / (dim as f64).sqrt();
                let expected_im = angle.sin() / (dim as f64).sqrt();
                assert_relative_eq!(mat[[k, j]].re, expected_re, epsilon = 1e-12);
                assert_relative_eq!(mat[[k, j]].im, expected_im, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_qft_iqft_roundtrip() {
        // IQFT(QFT(|ψ⟩)) should recover |ψ⟩ for an arbitrary state.
        let n = 3;
        let cfg = QftConfig {
            n_qubits: n,
            ..Default::default()
        };
        let dim = 1_usize << n;

        // Arbitrary normalised state
        let norm_factor = 1.0 / (dim as f64).sqrt();
        let state: Statevector = (0..dim)
            .map(|j| {
                let angle = 2.0 * PI * j as f64 / dim as f64;
                Complex64::new(norm_factor * angle.cos(), norm_factor * angle.sin())
            })
            .collect();

        let transformed = qft(&state, &cfg).expect("qft failed");
        let recovered = iqft(&transformed, &cfg).expect("iqft failed");

        for (a, b) in state.iter().zip(recovered.iter()) {
            assert!(
                (a.re - b.re).abs() < 1e-10,
                "real part mismatch in IQFT∘QFT roundtrip"
            );
            assert!(
                (a.im - b.im).abs() < 1e-10,
                "imag part mismatch in IQFT∘QFT roundtrip"
            );
        }
    }

    #[test]
    fn test_qft_computational_basis_states() {
        // QFT|j⟩ = (1/√N) Σ_k exp(2πijk/N) |k⟩
        let n = 2;
        let dim = 1_usize << n;
        let cfg = QftConfig {
            n_qubits: n,
            ..Default::default()
        };

        for j in 0..dim {
            let state = basis_state(j, n);
            let out = qft(&state, &cfg).expect("qft failed");

            let inv_sqrt_n = 1.0 / (dim as f64).sqrt();
            for k in 0..dim {
                let angle = 2.0 * PI * (j * k) as f64 / dim as f64;
                let expected = Complex64::new(inv_sqrt_n * angle.cos(), inv_sqrt_n * angle.sin());
                assert!(
                    (out[k].re - expected.re).abs() < 1e-12,
                    "j={j} k={k}: real mismatch"
                );
                assert!(
                    (out[k].im - expected.im).abs() < 1e-12,
                    "j={j} k={k}: imag mismatch"
                );
            }
        }
    }

    #[test]
    fn test_qft_1qubit_is_hadamard() {
        // 1-qubit QFT is the Hadamard gate: maps |0⟩→(|0⟩+|1⟩)/√2.
        let cfg = QftConfig {
            n_qubits: 1,
            ..Default::default()
        };

        let s0 = basis_state(0, 1);
        let out0 = qft(&s0, &cfg).expect("qft failed");
        assert_relative_eq!(out0[0].re, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-12);
        assert_relative_eq!(out0[1].re, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-12);

        let s1 = basis_state(1, 1);
        let out1 = qft(&s1, &cfg).expect("qft failed");
        assert_relative_eq!(out1[0].re, 1.0 / 2.0_f64.sqrt(), epsilon = 1e-12);
        assert_relative_eq!(out1[1].re, -1.0 / 2.0_f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn test_qft_vs_classical_dft() {
        // qft_dft should match a naive DFT computation.
        let n = 8;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                (2.0 * PI * i as f64 / n as f64).cos()
                    + 0.5 * (4.0 * PI * i as f64 / n as f64).sin()
            })
            .collect();

        let qft_result = qft_dft(&signal).expect("qft_dft failed");

        // Reference naive DFT (negative exponent)
        let mut dft_ref = vec![Complex64::new(0.0, 0.0); n];
        for k in 0..n {
            for j in 0..n {
                let angle = -2.0 * PI * (j * k) as f64 / n as f64;
                dft_ref[k] += Complex64::new(signal[j] * angle.cos(), signal[j] * angle.sin());
            }
        }

        for k in 0..n {
            assert!(
                (qft_result[k].re - dft_ref[k].re).abs() < 1e-8,
                "k={k}: real mismatch QFT vs DFT"
            );
            assert!(
                (qft_result[k].im - dft_ref[k].im).abs() < 1e-8,
                "k={k}: imag mismatch QFT vs DFT"
            );
        }
    }

    #[test]
    fn test_qft_approximate_close_to_exact() {
        // Approximate QFT with very small threshold should nearly match exact QFT.
        let n = 4;
        let exact_cfg = QftConfig {
            n_qubits: n,
            ..Default::default()
        };
        let approx_cfg = QftConfig {
            n_qubits: n,
            approximate: true,
            approx_threshold: 1e-15, // effectively exact
            ..Default::default()
        };

        let state = basis_state(5, n);
        let exact = qft(&state, &exact_cfg).expect("exact qft failed");
        let approx = qft(&state, &approx_cfg).expect("approx qft failed");

        for (a, b) in exact.iter().zip(approx.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-12);
            assert_relative_eq!(a.im, b.im, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_qft_error_wrong_state_size() {
        // Should return an error if state size ≠ 2^n_qubits.
        let cfg = QftConfig {
            n_qubits: 3,
            ..Default::default()
        };
        let bad_state = vec![Complex64::new(1.0, 0.0); 5]; // not 8
        let result = qft(&bad_state, &cfg);
        assert!(result.is_err());
    }
}
