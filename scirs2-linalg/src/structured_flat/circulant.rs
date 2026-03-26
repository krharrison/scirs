//! Circulant matrix operations: O(N log N) matvec, eigenvalues, solve.
//!
//! All operations are implemented using a manual Cooley-Tukey FFT (radix-2 DIT)
//! and its inverse, staying entirely within this module without external FFT deps.

use scirs2_core::num_complex::Complex64;

use super::types::FlatCirculant;
use crate::error::{LinalgError, LinalgResult};

// ─────────────────────────────────────────────────────────────────────────────
// Internal FFT (radix-2 Cooley-Tukey DIT, power-of-two only; falls back to DFT)
// ─────────────────────────────────────────────────────────────────────────────

/// DFT of a real signal — O(N²) fallback for non-power-of-2 or small N.
fn dft(x: &[f64]) -> Vec<Complex64> {
    let n = x.len();
    use std::f64::consts::TAU;
    (0..n)
        .map(|k| {
            x.iter()
                .enumerate()
                .map(|(j, &xj)| {
                    let angle = -TAU * (k * j) as f64 / n as f64;
                    Complex64::new(xj, 0.0) * Complex64::from_polar(1.0, angle)
                })
                .sum()
        })
        .collect()
}

/// IDFT of a complex spectrum — O(N²) fallback.
fn idft(x: &[Complex64]) -> Vec<f64> {
    let n = x.len();
    use std::f64::consts::TAU;
    (0..n)
        .map(|k| {
            x.iter()
                .enumerate()
                .map(|(j, &xj)| {
                    let angle = TAU * (k * j) as f64 / n as f64;
                    (xj * Complex64::from_polar(1.0, angle)).re
                })
                .sum::<f64>()
                / n as f64
        })
        .collect()
}

/// Return the highest power of 2 ≤ n, or 0 if n == 0.
fn prev_pow2(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut p = 1;
    while p * 2 <= n {
        p *= 2;
    }
    p
}

fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// In-place Cooley-Tukey radix-2 DIT FFT (n must be a power of two).
fn fft_inplace(buf: &mut [Complex64]) {
    let n = buf.len();
    debug_assert!(
        is_power_of_two(n),
        "fft_inplace requires power-of-two length"
    );

    // Bit-reversal permutation.
    let bits = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = bit_reverse(i, bits);
        if i < j {
            buf.swap(i, j);
        }
    }

    use std::f64::consts::TAU;
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let w_base = Complex64::from_polar(1.0, -TAU / len as f64);
        for start in (0..n).step_by(len) {
            let mut w = Complex64::new(1.0, 0.0);
            for k in 0..half {
                let u = buf[start + k];
                let v = buf[start + k + half] * w;
                buf[start + k] = u + v;
                buf[start + k + half] = u - v;
                w *= w_base;
            }
        }
        len *= 2;
    }
}

/// In-place inverse FFT (n must be a power of two).
fn ifft_inplace(buf: &mut [Complex64]) {
    // Conjugate → FFT → conjugate → scale.
    for x in buf.iter_mut() {
        *x = x.conj();
    }
    fft_inplace(buf);
    let n = buf.len() as f64;
    for x in buf.iter_mut() {
        *x = x.conj() / n;
    }
}

fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut r = 0usize;
    for _ in 0..bits {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    r
}

/// Fast DFT: uses Cooley-Tukey if N is a power of two, else DFT fallback.
fn fast_dft(x: &[f64]) -> Vec<Complex64> {
    let n = x.len();
    if is_power_of_two(n) {
        let mut buf: Vec<Complex64> = x.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        fft_inplace(&mut buf);
        buf
    } else {
        dft(x)
    }
}

/// Fast IDFT: uses Cooley-Tukey if N is a power of two, else IDFT fallback.
fn fast_idft(x: &[Complex64]) -> Vec<f64> {
    let n = x.len();
    if is_power_of_two(n) {
        let mut buf = x.to_vec();
        ifft_inplace(&mut buf);
        buf.iter().map(|v| v.re).collect()
    } else {
        idft(x)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FlatCirculant operations
// ─────────────────────────────────────────────────────────────────────────────

impl FlatCirculant {
    /// Matrix-vector product y = C * x using FFT convolution.
    ///
    /// Complexity: O(N log N).
    pub fn matvec(&self, x: &[f64]) -> LinalgResult<Vec<f64>> {
        let n = self.n;
        if x.len() != n {
            return Err(LinalgError::ShapeError(format!(
                "x has length {} but matrix is {}×{}",
                x.len(),
                n,
                n
            )));
        }

        // y = IDFT( DFT(first_row) ⊙ DFT(x) )
        let c_hat = fast_dft(&self.first_row);
        let x_hat = fast_dft(x);
        let y_hat: Vec<Complex64> = c_hat
            .iter()
            .zip(x_hat.iter())
            .map(|(ci, xi)| ci * xi)
            .collect();
        Ok(fast_idft(&y_hat))
    }

    /// Eigenvalues of the circulant matrix: λ_k = DFT(first_row)_k.
    pub fn eigenvalues(&self) -> Vec<Complex64> {
        fast_dft(&self.first_row)
    }

    /// Solve C * x = b using eigendecomposition.
    ///
    /// Complexity: O(N log N).
    ///
    /// # Errors
    /// Returns [`LinalgError::SingularMatrixError`] if any eigenvalue is (near) zero.
    pub fn solve(&self, b: &[f64]) -> LinalgResult<Vec<f64>> {
        let n = self.n;
        if b.len() != n {
            return Err(LinalgError::ShapeError(format!(
                "b has length {} but matrix is {}×{}",
                b.len(),
                n,
                n
            )));
        }

        let c_hat = fast_dft(&self.first_row);
        let b_hat = fast_dft(b);

        let thresh = 1e-14 * c_hat.iter().map(|z| z.norm()).fold(0.0_f64, f64::max);

        let x_hat: Vec<Complex64> = c_hat
            .iter()
            .zip(b_hat.iter())
            .map(|(ci, bi)| {
                if ci.norm() < thresh {
                    Err(LinalgError::SingularMatrixError(
                        "circulant matrix is singular (zero eigenvalue)".into(),
                    ))
                } else {
                    Ok(bi / ci)
                }
            })
            .collect::<LinalgResult<Vec<_>>>()?;

        Ok(fast_idft(&x_hat))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn e1(n: usize) -> Vec<f64> {
        let mut v = vec![0.0_f64; n];
        v[0] = 1.0;
        v
    }

    #[test]
    fn test_circulant_matvec_first_basis() {
        // Convention: C[i,j] = c[(i-j+n)%n].
        // C * e_0 gives column 0: C[i,0] = c[(i-0+n)%n] = c[i].
        let c = vec![1.0_f64, 2.0, 3.0, 4.0];
        let mat = FlatCirculant::new(c.clone()).expect("failed");
        let y = mat.matvec(&e1(4)).expect("matvec failed");
        // Column 0: y[i] = c[i].
        for i in 0..4 {
            assert_relative_eq!(y[i], c[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_circulant_first_row_def() {
        // Row 0: C[0,j] = c[(0-j+n)%n] = c[(n-j)%n].
        // So dense[0][0] = c[0], dense[0][1] = c[n-1], dense[0][2] = c[n-2].
        let c = vec![5.0_f64, 3.0, 2.0];
        let n = 3;
        let mat = FlatCirculant::new(c.clone()).expect("failed");
        let dense = mat.to_dense();
        // Row 0: [c[0], c[2], c[1]] for n=3.
        assert_relative_eq!(dense[0], c[0], epsilon = 1e-12); // j=0: c[0]
        assert_relative_eq!(dense[1], c[n - 1], epsilon = 1e-12); // j=1: c[2]
        assert_relative_eq!(dense[2], c[n - 2], epsilon = 1e-12); // j=2: c[1]
    }

    #[test]
    fn test_circulant_solve_roundtrip() {
        let c = vec![4.0_f64, 1.0, 2.0, 1.0];
        let mat = FlatCirculant::new(c).expect("failed");
        let b = vec![1.0_f64, 2.0, 3.0, 4.0];
        let x = mat.solve(&b).expect("solve failed");
        let y = mat.matvec(&x).expect("matvec failed");
        for i in 0..4 {
            assert_relative_eq!(y[i], b[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_circulant_eigenvalues_length() {
        let c = vec![1.0_f64, 2.0, 3.0, 4.0];
        let mat = FlatCirculant::new(c).expect("failed");
        let eigs = mat.eigenvalues();
        assert_eq!(eigs.len(), 4);
    }
}
