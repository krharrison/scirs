//! Property-based tests for FFT mathematical invariants in scirs2-fft.
//!
//! Tests cover:
//! * Reversibility: ifft(fft(x)) ≈ x
//! * Linearity: fft(α·x + β·y) ≈ α·fft(x) + β·fft(y)
//! * Parseval's theorem: Σ|x_n|² ≈ Σ|X_k|²/N
//! * Conjugate symmetry for real signals
//! * Zero-signal identity

use proptest::prelude::*;
use scirs2_core::numeric::Complex64;
use scirs2_fft::{fft, ifft};

// ─────────────────────────────────────────────────────────────────────────────
// Strategies
// ─────────────────────────────────────────────────────────────────────────────

/// A finite, bounded real signal value.
fn f64_signal() -> impl Strategy<Value = f64> {
    -100.0f64..100.0f64
}

/// A scaling coefficient (avoid extreme values that amplify numerical errors).
fn scale_strategy() -> impl Strategy<Value = f64> {
    -5.0f64..5.0f64
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Element-wise maximum absolute difference between two complex slices.
fn max_abs_diff_complex(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).norm())
        .fold(0.0_f64, f64::max)
}

/// Sum of squared magnitudes (power) of a complex slice.
fn power(v: &[Complex64]) -> f64 {
    v.iter().map(|c| c.norm_sqr()).sum()
}

/// Build a complex signal from a real one (imaginary parts = 0).
fn real_to_complex(v: &[f64]) -> Vec<Complex64> {
    v.iter().map(|&x| Complex64::new(x, 0.0)).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Reversibility: ifft(fft(x)) ≈ x
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    /// For any real input x of length 4 (power-of-two), ifft(fft(x)) reconstructs x.
    #[test]
    fn prop_fft_ifft_real_roundtrip_4(
        s0 in f64_signal(), s1 in f64_signal(), s2 in f64_signal(), s3 in f64_signal(),
    ) {
        let signal = vec![s0, s1, s2, s3];
        let len = signal.len();
        let spectrum = fft(&signal, None).expect("fft should succeed");
        let reconstructed = ifft(&spectrum, Some(len)).expect("ifft should succeed");
        let input_complex = real_to_complex(&signal);
        let err = max_abs_diff_complex(&reconstructed, &input_complex);
        let signal_max = signal.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        let tol = 1e-9 * (1.0 + signal_max) * (len as f64).sqrt();
        prop_assert!(err < tol, "ifft(fft(x)) roundtrip error = {} for len={}", err, len);
    }

    /// For any real input x of length 8, ifft(fft(x)) reconstructs x.
    #[test]
    fn prop_fft_ifft_real_roundtrip_8(
        s in proptest::collection::vec(f64_signal(), 8usize),
    ) {
        let len = s.len();
        let spectrum = fft(&s, None).expect("fft should succeed");
        let reconstructed = ifft(&spectrum, Some(len)).expect("ifft should succeed");
        let input_complex = real_to_complex(&s);
        let err = max_abs_diff_complex(&reconstructed, &input_complex);
        let signal_max = s.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        let tol = 1e-9 * (1.0 + signal_max) * (len as f64).sqrt();
        prop_assert!(err < tol, "ifft(fft(x)) roundtrip error = {} for len={}", err, len);
    }

    /// For any real input x of length 16, ifft(fft(x)) reconstructs x.
    #[test]
    fn prop_fft_ifft_real_roundtrip_16(
        s in proptest::collection::vec(f64_signal(), 16usize),
    ) {
        let len = s.len();
        let spectrum = fft(&s, None).expect("fft should succeed");
        let reconstructed = ifft(&spectrum, Some(len)).expect("ifft should succeed");
        let input_complex = real_to_complex(&s);
        let err = max_abs_diff_complex(&reconstructed, &input_complex);
        let signal_max = s.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        let tol = 1e-9 * (1.0 + signal_max) * (len as f64).sqrt();
        prop_assert!(err < tol, "ifft(fft(x)) roundtrip error = {} for len={}", err, len);
    }

    /// For any real input x of length 32, ifft(fft(x)) reconstructs x.
    #[test]
    fn prop_fft_ifft_real_roundtrip_32(
        s in proptest::collection::vec(f64_signal(), 32usize),
    ) {
        let len = s.len();
        let spectrum = fft(&s, None).expect("fft should succeed");
        let reconstructed = ifft(&spectrum, Some(len)).expect("ifft should succeed");
        let input_complex = real_to_complex(&s);
        let err = max_abs_diff_complex(&reconstructed, &input_complex);
        let signal_max = s.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        let tol = 1e-9 * (1.0 + signal_max) * (len as f64).sqrt();
        prop_assert!(err < tol, "ifft(fft(x)) roundtrip error = {} for len={}", err, len);
    }

    /// For complex input of length 16, ifft(fft(x)) reconstructs x.
    #[test]
    fn prop_fft_ifft_complex_roundtrip_16(
        re in proptest::collection::vec(f64_signal(), 16usize),
        im in proptest::collection::vec(f64_signal(), 16usize),
    ) {
        let len = re.len();
        let signal: Vec<Complex64> = re.iter().zip(im.iter()).map(|(&r, &i)| Complex64::new(r, i)).collect();
        let spectrum = fft(&signal, None).expect("fft should succeed");
        let reconstructed = ifft(&spectrum, Some(len)).expect("ifft should succeed");
        let err = max_abs_diff_complex(&reconstructed, &signal);
        let signal_max = signal.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
        let tol = 1e-9 * (1.0 + signal_max) * (len as f64).sqrt();
        prop_assert!(err < tol, "ifft(fft(x)) complex roundtrip error = {} for len={}", err, len);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Linearity: fft(α·x + β·y) ≈ α·fft(x) + β·fft(y)
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    /// FFT linearity over real signals of length 8.
    #[test]
    fn prop_fft_linearity_real_8(
        x in proptest::collection::vec(f64_signal(), 8usize),
        y in proptest::collection::vec(f64_signal(), 8usize),
        alpha in scale_strategy(),
        beta in scale_strategy(),
    ) {
        let len = x.len();
        let combination: Vec<f64> = x.iter().zip(y.iter())
            .map(|(xi, yi)| alpha * xi + beta * yi)
            .collect();

        let fft_combo = fft(&combination, None).expect("fft combo");
        let fft_x = fft(&x, None).expect("fft x");
        let fft_y = fft(&y, None).expect("fft y");

        let linear_combo: Vec<Complex64> = fft_x.iter().zip(fft_y.iter())
            .map(|(fx, fy)| Complex64::new(alpha, 0.0) * fx + Complex64::new(beta, 0.0) * fy)
            .collect();

        let err = max_abs_diff_complex(&fft_combo, &linear_combo);
        let max_input = x.iter().chain(y.iter()).map(|v| v.abs()).fold(0.0_f64, f64::max);
        let scale = (alpha.abs() + beta.abs() + 1.0) * (max_input + 1.0) * (len as f64).sqrt();
        let tol = 1e-9 * scale;

        prop_assert!(
            err < tol,
            "FFT linearity violation (len=8): error={}, tol={}", err, tol
        );
    }

    /// FFT linearity over real signals of length 16.
    #[test]
    fn prop_fft_linearity_real_16(
        x in proptest::collection::vec(f64_signal(), 16usize),
        y in proptest::collection::vec(f64_signal(), 16usize),
        alpha in scale_strategy(),
        beta in scale_strategy(),
    ) {
        let len = x.len();
        let combination: Vec<f64> = x.iter().zip(y.iter())
            .map(|(xi, yi)| alpha * xi + beta * yi)
            .collect();

        let fft_combo = fft(&combination, None).expect("fft combo");
        let fft_x = fft(&x, None).expect("fft x");
        let fft_y = fft(&y, None).expect("fft y");

        let linear_combo: Vec<Complex64> = fft_x.iter().zip(fft_y.iter())
            .map(|(fx, fy)| Complex64::new(alpha, 0.0) * fx + Complex64::new(beta, 0.0) * fy)
            .collect();

        let err = max_abs_diff_complex(&fft_combo, &linear_combo);
        let max_input = x.iter().chain(y.iter()).map(|v| v.abs()).fold(0.0_f64, f64::max);
        let scale = (alpha.abs() + beta.abs() + 1.0) * (max_input + 1.0) * (len as f64).sqrt();
        let tol = 1e-9 * scale;

        prop_assert!(
            err < tol,
            "FFT linearity violation (len=16): error={}, tol={}", err, tol
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parseval's theorem: Σ|x_n|² = Σ|X_k|²/N
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    /// Parseval's theorem for real signals of length 8.
    #[test]
    fn prop_parseval_real_8(s in proptest::collection::vec(f64_signal(), 8usize)) {
        let len = s.len();
        let signal_complex = real_to_complex(&s);
        let spectrum = fft(&s, None).expect("fft should succeed");
        let time_power = power(&signal_complex);
        let freq_power_normalized = power(&spectrum) / (len as f64);
        let tol = if time_power < 1e-30 { 1e-10 } else { 1e-9 * time_power };
        prop_assert!(
            (time_power - freq_power_normalized).abs() < tol,
            "Parseval violated (8): Σ|x|²={:.6e}, Σ|X|²/N={:.6e}, delta={:.3e}",
            time_power, freq_power_normalized, (time_power - freq_power_normalized).abs()
        );
    }

    /// Parseval's theorem for real signals of length 16.
    #[test]
    fn prop_parseval_real_16(s in proptest::collection::vec(f64_signal(), 16usize)) {
        let len = s.len();
        let signal_complex = real_to_complex(&s);
        let spectrum = fft(&s, None).expect("fft should succeed");
        let time_power = power(&signal_complex);
        let freq_power_normalized = power(&spectrum) / (len as f64);
        let tol = if time_power < 1e-30 { 1e-10 } else { 1e-9 * time_power };
        prop_assert!(
            (time_power - freq_power_normalized).abs() < tol,
            "Parseval violated (16): Σ|x|²={:.6e}, Σ|X|²/N={:.6e}, delta={:.3e}",
            time_power, freq_power_normalized, (time_power - freq_power_normalized).abs()
        );
    }

    /// Parseval's theorem for real signals of length 32.
    #[test]
    fn prop_parseval_real_32(s in proptest::collection::vec(f64_signal(), 32usize)) {
        let len = s.len();
        let signal_complex = real_to_complex(&s);
        let spectrum = fft(&s, None).expect("fft should succeed");
        let time_power = power(&signal_complex);
        let freq_power_normalized = power(&spectrum) / (len as f64);
        let tol = if time_power < 1e-30 { 1e-10 } else { 1e-9 * time_power };
        prop_assert!(
            (time_power - freq_power_normalized).abs() < tol,
            "Parseval violated (32): Σ|x|²={:.6e}, Σ|X|²/N={:.6e}, delta={:.3e}",
            time_power, freq_power_normalized, (time_power - freq_power_normalized).abs()
        );
    }

    /// Parseval's theorem for complex signals of length 16.
    #[test]
    fn prop_parseval_complex_16(
        re in proptest::collection::vec(f64_signal(), 16usize),
        im in proptest::collection::vec(f64_signal(), 16usize),
    ) {
        let len = re.len();
        let signal: Vec<Complex64> = re.iter().zip(im.iter()).map(|(&r, &i)| Complex64::new(r, i)).collect();
        let spectrum = fft(&signal, None).expect("fft should succeed");
        let time_power = power(&signal);
        let freq_power_normalized = power(&spectrum) / (len as f64);
        let tol = if time_power < 1e-30 { 1e-10 } else { 1e-9 * time_power };
        prop_assert!(
            (time_power - freq_power_normalized).abs() < tol,
            "Parseval complex violated (16): Σ|x|²={:.6e}, Σ|X|²/N={:.6e}, delta={:.3e}",
            time_power, freq_power_normalized, (time_power - freq_power_normalized).abs()
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Additional structural invariants
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    /// The FFT of a real signal satisfies conjugate symmetry: X[k] = conj(X[N-k]).
    #[test]
    fn prop_fft_conjugate_symmetry_8(s in proptest::collection::vec(f64_signal(), 8usize)) {
        let len = s.len();
        let spectrum = fft(&s, None).expect("fft should succeed");
        let signal_max = s.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let tol = 1e-9 * (1.0 + signal_max * (len as f64).sqrt());
        for k in 1..(len / 2) {
            let xk = spectrum[k];
            let xnk = spectrum[len - k];
            let err_re = (xk.re - xnk.re).abs();
            let err_im = (xk.im + xnk.im).abs();
            prop_assert!(
                err_re < tol && err_im < tol,
                "Conjugate symmetry violated at k={}: X[k]={:?}, expected conj(X[N-k])={:?}",
                k, xk, Complex64::new(xnk.re, -xnk.im)
            );
        }
    }

    /// The FFT of a real signal of length 16 satisfies conjugate symmetry.
    #[test]
    fn prop_fft_conjugate_symmetry_16(s in proptest::collection::vec(f64_signal(), 16usize)) {
        let len = s.len();
        let spectrum = fft(&s, None).expect("fft should succeed");
        let signal_max = s.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let tol = 1e-9 * (1.0 + signal_max * (len as f64).sqrt());
        for k in 1..(len / 2) {
            let xk = spectrum[k];
            let xnk = spectrum[len - k];
            let err_re = (xk.re - xnk.re).abs();
            let err_im = (xk.im + xnk.im).abs();
            prop_assert!(
                err_re < tol && err_im < tol,
                "Conjugate symmetry violated at k={}: X[k]={:?}, expected conj={:?}",
                k, xk, Complex64::new(xnk.re, -xnk.im)
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic structural tests (not strictly property-based but included here)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_fft_zero_signal_length_8() {
    let zeros: Vec<f64> = vec![0.0; 8];
    let spectrum = fft(&zeros, None).expect("fft of zeros should succeed");
    for (k, &c) in spectrum.iter().enumerate() {
        assert!(c.norm() < 1e-15, "fft(0)[{k}] = {c:?} should be 0");
    }
}

#[test]
fn test_fft_zero_signal_length_32() {
    let zeros: Vec<f64> = vec![0.0; 32];
    let spectrum = fft(&zeros, None).expect("fft of zeros should succeed");
    for (k, &c) in spectrum.iter().enumerate() {
        assert!(c.norm() < 1e-15, "fft(0)[{k}] = {c:?} should be 0");
    }
}

/// FFT of an impulse signal [1, 0, 0, ..., 0] should be all-ones.
#[test]
fn test_fft_impulse_is_flat() {
    let n = 16;
    let mut impulse = vec![0.0f64; n];
    impulse[0] = 1.0;
    let spectrum = fft(&impulse, None).expect("fft of impulse should succeed");
    assert_eq!(
        spectrum.len(),
        n,
        "Spectrum length should equal input length"
    );
    for (k, &c) in spectrum.iter().enumerate() {
        let err = (c.norm() - 1.0).abs();
        assert!(
            err < 1e-12,
            "Impulse FFT bin {} = {:?} magnitude should be 1.0, error = {}",
            k,
            c,
            err
        );
    }
}
