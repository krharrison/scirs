//! ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques)
//!
//! ESPRIT exploits the shift-invariance structure of a ULA to estimate DOA
//! without grid search. For an M-element ULA, the two overlapping sub-arrays
//! of size M-1 are related by a rotational shift:
//!
//! 1. Eigendecompose R to get signal subspace E_s (d dominant eigenvectors)
//! 2. Partition: E1 = E_s[0:M-1, :], E2 = E_s[1:M, :]
//! 3. Rotation matrix: Phi = E1^dagger * E2
//! 4. DOAs from eigenvalues: theta_k = arcsin(angle(lambda_k) * lambda / (2*pi*d))
//!
//! This module provides:
//! - [`ESPRIT`]: standard LS-ESPRIT
//! - [`TlsESPRIT`]: Total Least Squares ESPRIT (better accuracy in noise)
//!
//! Pure Rust, no unwrap(), snake_case naming.

use crate::beamforming::array::estimate_covariance;
use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of ESPRIT DOA estimation
#[derive(Debug, Clone)]
pub struct ESPRITResult {
    /// Estimated DOA angles in radians
    pub doa_estimates: Vec<f64>,
    /// Rotational invariance eigenvalues (phase shifts)
    pub phase_shifts: Vec<Complex64>,
    /// Signal subspace eigenvalues (of the covariance matrix)
    pub signal_eigenvalues: Vec<f64>,
}

// ---------------------------------------------------------------------------
// ESPRIT
// ---------------------------------------------------------------------------

/// LS-ESPRIT DOA estimator
///
/// Uses least-squares solution for the shift-invariance equation.
#[derive(Debug, Clone)]
pub struct ESPRIT {
    /// Number of array elements
    n_elements: usize,
    /// Number of sources
    n_sources: usize,
    /// Inter-element spacing in wavelengths
    element_spacing: f64,
}

impl ESPRIT {
    /// Create a new ESPRIT estimator
    ///
    /// # Arguments
    ///
    /// * `n_elements` - Total number of ULA elements (>= 3)
    /// * `n_sources` - Number of signal sources
    /// * `element_spacing` - Element spacing in wavelengths (typically 0.5)
    pub fn new(n_elements: usize, n_sources: usize, element_spacing: f64) -> SignalResult<Self> {
        if n_elements < 3 {
            return Err(SignalError::ValueError(
                "ESPRIT requires at least 3 elements".to_string(),
            ));
        }
        if n_sources == 0 || n_sources >= n_elements {
            return Err(SignalError::ValueError(format!(
                "n_sources ({}) must be in [1, {})",
                n_sources, n_elements
            )));
        }
        if element_spacing <= 0.0 {
            return Err(SignalError::ValueError(
                "element_spacing must be positive".to_string(),
            ));
        }
        Ok(Self {
            n_elements,
            n_sources,
            element_spacing,
        })
    }

    /// Estimate DOA from snapshot data
    ///
    /// # Arguments
    ///
    /// * `data` - Snapshot matrix `[n_elements][n_snapshots]`
    pub fn estimate(&self, data: &[Vec<Complex64>]) -> SignalResult<ESPRITResult> {
        if data.len() != self.n_elements {
            return Err(SignalError::DimensionMismatch(format!(
                "Data has {} rows, expected {}",
                data.len(),
                self.n_elements
            )));
        }
        let cov_nested = estimate_covariance(data)?;
        self.estimate_from_covariance(&cov_nested)
    }

    /// Estimate DOA from covariance matrix
    pub fn estimate_from_covariance(
        &self,
        covariance: &[Vec<Complex64>],
    ) -> SignalResult<ESPRITResult> {
        let m = self.n_elements;
        let d = self.n_sources;

        if covariance.len() != m {
            return Err(SignalError::DimensionMismatch(format!(
                "Covariance size {} does not match n_elements {}",
                covariance.len(),
                m
            )));
        }

        // Convert to flat
        let mut cov_flat = vec![Complex64::new(0.0, 0.0); m * m];
        for i in 0..m {
            for j in 0..m {
                cov_flat[i * m + j] = covariance[i][j];
            }
        }

        let (eigenvalues, eigenvectors) = hermitian_eig_flat(&cov_flat, m)?;
        let signal_eigenvalues = eigenvalues[..d].to_vec();

        // Signal subspace: first d eigenvectors
        let es: Vec<&Vec<Complex64>> = eigenvectors[..d].iter().collect();

        // Build E1 and E2 sub-arrays (m-1 x d)
        let m1 = m - 1;
        let mut e1 = vec![Complex64::new(0.0, 0.0); m1 * d];
        let mut e2 = vec![Complex64::new(0.0, 0.0); m1 * d];
        for col in 0..d {
            for row in 0..m1 {
                e1[row * d + col] = es[col][row];
                e2[row * d + col] = es[col][row + 1];
            }
        }

        // LS solution: Phi = (E1^H E1)^{-1} E1^H E2
        let e1h_e1 = complex_gram(&e1, m1, d);
        let e1h_e2 = complex_cross_gram(&e1, &e2, m1, d);
        let psi = complex_solve_dd(&e1h_e1, &e1h_e2, d)?;

        // Eigenvalues of Phi
        let phase_shifts = eigenvalues_complex_matrix(&psi, d)?;

        // Convert to DOA
        let doa_estimates = phase_shifts_to_doa(&phase_shifts, self.element_spacing);

        Ok(ESPRITResult {
            doa_estimates,
            phase_shifts,
            signal_eigenvalues,
        })
    }

    /// Get number of elements
    pub fn n_elements(&self) -> usize {
        self.n_elements
    }

    /// Get number of sources
    pub fn n_sources(&self) -> usize {
        self.n_sources
    }
}

// ---------------------------------------------------------------------------
// TLS-ESPRIT
// ---------------------------------------------------------------------------

/// Total Least Squares ESPRIT
///
/// Treats both E1 and E2 as noisy observations, using SVD of [E1 | E2]
/// to define the TLS solution. Generally provides better accuracy than LS-ESPRIT.
#[derive(Debug, Clone)]
pub struct TlsESPRIT {
    /// Number of array elements
    n_elements: usize,
    /// Number of sources
    n_sources: usize,
    /// Element spacing in wavelengths
    element_spacing: f64,
}

impl TlsESPRIT {
    /// Create a new TLS-ESPRIT estimator
    pub fn new(n_elements: usize, n_sources: usize, element_spacing: f64) -> SignalResult<Self> {
        if n_elements < 3 {
            return Err(SignalError::ValueError(
                "TLS-ESPRIT requires at least 3 elements".to_string(),
            ));
        }
        if n_sources == 0 || n_sources >= n_elements {
            return Err(SignalError::ValueError(format!(
                "n_sources ({}) must be in [1, {})",
                n_sources, n_elements
            )));
        }
        if element_spacing <= 0.0 {
            return Err(SignalError::ValueError(
                "element_spacing must be positive".to_string(),
            ));
        }
        Ok(Self {
            n_elements,
            n_sources,
            element_spacing,
        })
    }

    /// Estimate DOA from snapshot data
    pub fn estimate(&self, data: &[Vec<Complex64>]) -> SignalResult<ESPRITResult> {
        if data.len() != self.n_elements {
            return Err(SignalError::DimensionMismatch(format!(
                "Data has {} rows, expected {}",
                data.len(),
                self.n_elements
            )));
        }
        let cov_nested = estimate_covariance(data)?;
        self.estimate_from_covariance(&cov_nested)
    }

    /// Estimate DOA from covariance matrix
    pub fn estimate_from_covariance(
        &self,
        covariance: &[Vec<Complex64>],
    ) -> SignalResult<ESPRITResult> {
        let m = self.n_elements;
        let d = self.n_sources;

        if covariance.len() != m {
            return Err(SignalError::DimensionMismatch(format!(
                "Covariance size {} does not match n_elements {}",
                covariance.len(),
                m
            )));
        }

        let mut cov_flat = vec![Complex64::new(0.0, 0.0); m * m];
        for i in 0..m {
            for j in 0..m {
                cov_flat[i * m + j] = covariance[i][j];
            }
        }

        let (eigenvalues, eigenvectors) = hermitian_eig_flat(&cov_flat, m)?;
        let signal_eigenvalues = eigenvalues[..d].to_vec();

        let es: Vec<&Vec<Complex64>> = eigenvectors[..d].iter().collect();
        let m1 = m - 1;

        // Build [E1 | E2] as (m-1) x 2d matrix
        let two_d = 2 * d;
        let mut e12 = vec![Complex64::new(0.0, 0.0); m1 * two_d];
        for col in 0..d {
            for row in 0..m1 {
                e12[row * two_d + col] = es[col][row]; // E1 left half
                e12[row * two_d + d + col] = es[col][row + 1]; // E2 right half
            }
        }

        // Compute right singular vectors via A^H A
        let v = compute_right_singular_vectors(&e12, m1, two_d)?;

        // Partition V into 4 blocks (each d x d)
        // V12: rows 0..d, columns d..2d (the "noise" right singular vectors)
        // V22: rows d..2d, columns d..2d
        let mut v12 = vec![Complex64::new(0.0, 0.0); d * d];
        let mut v22 = vec![Complex64::new(0.0, 0.0); d * d];
        for i in 0..d {
            for j in 0..d {
                let col_in_v = d + j;
                v12[i * d + j] = v[i * two_d + col_in_v];
                v22[i * d + j] = v[(d + i) * two_d + col_in_v];
            }
        }

        // Psi_TLS = -V12 * V22^{-1}
        let v22_inv = complex_matrix_inv(&v22, d)?;
        let psi_tls = complex_matmul_dd(&v12, &v22_inv, d);
        let psi_neg: Vec<Complex64> = psi_tls.iter().map(|x| -(*x)).collect();

        let phase_shifts = eigenvalues_complex_matrix(&psi_neg, d)?;
        let doa_estimates = phase_shifts_to_doa(&phase_shifts, self.element_spacing);

        Ok(ESPRITResult {
            doa_estimates,
            phase_shifts,
            signal_eigenvalues,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Convert phase shifts to DOA angles
fn phase_shifts_to_doa(phase_shifts: &[Complex64], element_spacing: f64) -> Vec<f64> {
    phase_shifts
        .iter()
        .filter_map(|&z| {
            let phi = z.arg();
            let sin_theta = -phi / (2.0 * PI * element_spacing);
            if sin_theta.abs() <= 1.0 {
                Some(sin_theta.asin())
            } else {
                None
            }
        })
        .collect()
}

/// Compute Gram matrix A^H A where A is (m x d) row-major
fn complex_gram(a: &[Complex64], m: usize, d: usize) -> Vec<Complex64> {
    let mut g = vec![Complex64::new(0.0, 0.0); d * d];
    for i in 0..d {
        for j in 0..d {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..m {
                sum += a[k * d + i].conj() * a[k * d + j];
            }
            g[i * d + j] = sum;
        }
    }
    g
}

/// Compute cross-Gram matrix A^H B
fn complex_cross_gram(a: &[Complex64], b: &[Complex64], m: usize, d: usize) -> Vec<Complex64> {
    let mut g = vec![Complex64::new(0.0, 0.0); d * d];
    for i in 0..d {
        for j in 0..d {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..m {
                sum += a[k * d + i].conj() * b[k * d + j];
            }
            g[i * d + j] = sum;
        }
    }
    g
}

/// Solve AX = B for X where A and B are d x d
fn complex_solve_dd(a: &[Complex64], b: &[Complex64], d: usize) -> SignalResult<Vec<Complex64>> {
    let inv = complex_matrix_inv(a, d)?;
    Ok(complex_matmul_dd(&inv, b, d))
}

/// d x d matrix multiplication (row-major)
fn complex_matmul_dd(a: &[Complex64], b: &[Complex64], d: usize) -> Vec<Complex64> {
    let mut c = vec![Complex64::new(0.0, 0.0); d * d];
    for i in 0..d {
        for j in 0..d {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..d {
                sum += a[i * d + k] * b[k * d + j];
            }
            c[i * d + j] = sum;
        }
    }
    c
}

/// Complex matrix inversion via LU with partial pivoting
fn complex_matrix_inv(a: &[Complex64], n: usize) -> SignalResult<Vec<Complex64>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut lu = a.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();

    for k in 0..n {
        let mut max_val = 0.0_f64;
        let mut max_row = k;
        for i in k..n {
            let v = lu[i * n + k].norm();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        if max_val < 1e-14 {
            return Err(SignalError::ComputationError(
                "Matrix is singular or near-singular".to_string(),
            ));
        }
        if max_row != k {
            for j in 0..n {
                lu.swap(k * n + j, max_row * n + j);
            }
            perm.swap(k, max_row);
        }
        let pivot = lu[k * n + k];
        for i in (k + 1)..n {
            lu[i * n + k] = lu[i * n + k] / pivot;
            for j in (k + 1)..n {
                let fac = lu[i * n + k];
                let sub = fac * lu[k * n + j];
                lu[i * n + j] -= sub;
            }
        }
    }

    let mut inv = vec![Complex64::new(0.0, 0.0); n * n];
    for col in 0..n {
        let mut rhs = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..n {
            if perm[i] == col {
                rhs[i] = Complex64::new(1.0, 0.0);
                break;
            }
        }
        // Forward substitution
        let mut y = rhs;
        for i in 0..n {
            for j in 0..i {
                let l = lu[i * n + j];
                let yj = y[j];
                y[i] -= l * yj;
            }
        }
        // Backward substitution
        let mut x = y;
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                let u = lu[i * n + j];
                let xj = x[j];
                x[i] -= u * xj;
            }
            x[i] = x[i] / lu[i * n + i];
        }
        for i in 0..n {
            inv[i * n + col] = x[i];
        }
    }
    Ok(inv)
}

/// Compute right singular vectors via A^H A
fn compute_right_singular_vectors(
    a: &[Complex64],
    m: usize,
    n_cols: usize,
) -> SignalResult<Vec<Complex64>> {
    let mut aha = vec![Complex64::new(0.0, 0.0); n_cols * n_cols];
    for i in 0..n_cols {
        for j in 0..n_cols {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..m {
                sum += a[k * n_cols + i].conj() * a[k * n_cols + j];
            }
            aha[i * n_cols + j] = sum;
        }
    }
    let (_, evecs) = hermitian_eig_flat(&aha, n_cols)?;
    let mut v = vec![Complex64::new(0.0, 0.0); n_cols * n_cols];
    for col in 0..n_cols {
        for row in 0..n_cols {
            v[row * n_cols + col] = evecs[col][row];
        }
    }
    Ok(v)
}

/// Eigenvalues of a general complex matrix via QR iteration
fn eigenvalues_complex_matrix(a: &[Complex64], d: usize) -> SignalResult<Vec<Complex64>> {
    if d == 0 {
        return Ok(Vec::new());
    }
    let mut h = a.to_vec();
    let max_iter = 300 * d;
    let eps = 1e-10;

    for _ in 0..max_iter {
        let mut max_off = 0.0_f64;
        for i in 1..d {
            max_off = max_off.max(h[i * d + (i - 1)].norm());
        }
        if max_off < eps {
            break;
        }
        // Simple shift
        let shift = if d >= 2 {
            h[(d - 1) * d + (d - 1)]
        } else {
            Complex64::new(0.0, 0.0)
        };
        let mut hs = h.clone();
        for i in 0..d {
            hs[i * d + i] -= shift;
        }
        let (q, r) = complex_qr_thin(&hs, d)?;
        h = complex_matmul_dd(&r, &q, d);
        for i in 0..d {
            h[i * d + i] += shift;
        }
    }
    Ok((0..d).map(|i| h[i * d + i]).collect())
}

/// Complex QR decomposition via Gram-Schmidt
fn complex_qr_thin(a: &[Complex64], d: usize) -> SignalResult<(Vec<Complex64>, Vec<Complex64>)> {
    let mut q = vec![Complex64::new(0.0, 0.0); d * d];
    let mut r = vec![Complex64::new(0.0, 0.0); d * d];

    for j in 0..d {
        let mut v: Vec<Complex64> = (0..d).map(|i| a[i * d + j]).collect();
        for k in 0..j {
            let q_col: Vec<Complex64> = (0..d).map(|i| q[i * d + k]).collect();
            let dot: Complex64 = q_col
                .iter()
                .zip(v.iter())
                .fold(Complex64::new(0.0, 0.0), |acc, (qi, vi)| {
                    acc + qi.conj() * vi
                });
            r[k * d + j] = dot;
            for i in 0..d {
                v[i] -= dot * q_col[i];
            }
        }
        let norm = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        r[j * d + j] = Complex64::new(norm, 0.0);
        if norm > 1e-14 {
            for i in 0..d {
                q[i * d + j] = v[i] / norm;
            }
        } else {
            q[j * d + j] = Complex64::new(1.0, 0.0);
        }
    }
    Ok((q, r))
}

/// Hermitian eigendecomposition (same as in music.rs)
fn hermitian_eig_flat(
    matrix: &[Complex64],
    n: usize,
) -> SignalResult<(Vec<f64>, Vec<Vec<Complex64>>)> {
    if matrix.len() != n * n {
        return Err(SignalError::DimensionMismatch(format!(
            "Matrix length {} does not match n={} (expected {})",
            matrix.len(),
            n,
            n * n
        )));
    }
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let n2 = 2 * n;
    let mut real_mat = vec![0.0_f64; n2 * n2];
    for i in 0..n {
        for j in 0..n {
            let c = matrix[i * n + j];
            real_mat[i * n2 + j] = c.re;
            real_mat[i * n2 + (n + j)] = -c.im;
            real_mat[(n + i) * n2 + j] = c.im;
            real_mat[(n + i) * n2 + (n + j)] = c.re;
        }
    }

    let (eigs_real, evecs_real) = jacobi_eig(&real_mat, n2)?;

    let mut indexed: Vec<(f64, usize)> = eigs_real
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut eigenvalues: Vec<f64> = Vec::with_capacity(n);
    let mut eigenvectors: Vec<Vec<Complex64>> = Vec::with_capacity(n);
    let mut used = 0;
    let mut i = 0;
    while used < n && i < indexed.len() {
        let (val, col_idx) = indexed[i];
        if used > 0 {
            let prev_val = eigenvalues[used - 1];
            if (val - prev_val).abs() < 1e-8 * (prev_val.abs() + 1.0) {
                i += 1;
                continue;
            }
        }
        eigenvalues.push(val);
        let real_ev = &evecs_real[col_idx];
        let cv: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new(real_ev[k], real_ev[n + k]))
            .collect();
        let norm = cv.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        let cv_norm = if norm > 1e-14 {
            cv.iter().map(|x| *x / norm).collect()
        } else {
            cv
        };
        eigenvectors.push(cv_norm);
        used += 1;
        i += 1;
    }

    while eigenvalues.len() < n {
        eigenvalues.push(0.0);
        let mut zero_ev = vec![Complex64::new(0.0, 0.0); n];
        let idx = eigenvalues.len() - 1;
        if idx < n {
            zero_ev[idx] = Complex64::new(1.0, 0.0);
        }
        eigenvectors.push(zero_ev);
    }

    Ok((eigenvalues, eigenvectors))
}

/// Jacobi eigenvalue algorithm
fn jacobi_eig(matrix: &[f64], n: usize) -> SignalResult<(Vec<f64>, Vec<Vec<f64>>)> {
    if matrix.len() != n * n {
        return Err(SignalError::DimensionMismatch(
            "Matrix size mismatch in Jacobi".to_string(),
        ));
    }
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let mut a = matrix.to_vec();
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let max_iter = 100 * n * n;
    let eps = 1e-12;

    for _ in 0..max_iter {
        let mut max_val = 0.0_f64;
        let mut p = 0_usize;
        let mut q = 1_usize;
        for ii in 0..n {
            for jj in (ii + 1)..n {
                let val = a[ii * n + jj].abs();
                if val > max_val {
                    max_val = val;
                    p = ii;
                    q = jj;
                }
            }
        }
        if max_val < eps {
            break;
        }

        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        let theta = if (aqq - app).abs() < 1e-30 {
            PI / 4.0
        } else {
            0.5 * (2.0 * apq / (aqq - app)).atan()
        };

        let cos_t = theta.cos();
        let sin_t = theta.sin();

        let mut new_a = a.clone();
        for r in 0..n {
            if r != p && r != q {
                let arp = a[r * n + p];
                let arq = a[r * n + q];
                new_a[r * n + p] = cos_t * arp - sin_t * arq;
                new_a[p * n + r] = new_a[r * n + p];
                new_a[r * n + q] = sin_t * arp + cos_t * arq;
                new_a[q * n + r] = new_a[r * n + q];
            }
        }
        new_a[p * n + p] = cos_t * cos_t * app - 2.0 * cos_t * sin_t * apq + sin_t * sin_t * aqq;
        new_a[q * n + q] = sin_t * sin_t * app + 2.0 * cos_t * sin_t * apq + cos_t * cos_t * aqq;
        new_a[p * n + q] = 0.0;
        new_a[q * n + p] = 0.0;
        a = new_a;

        for r in 0..n {
            let vrp = v[r * n + p];
            let vrq = v[r * n + q];
            v[r * n + p] = cos_t * vrp - sin_t * vrq;
            v[r * n + q] = sin_t * vrp + cos_t * vrq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    let eigenvectors: Vec<Vec<f64>> = (0..n)
        .map(|col| (0..n).map(|row| v[row * n + col]).collect())
        .collect();

    Ok((eigenvalues, eigenvectors))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// ULA steering vector (fast)
    fn ula_sv(m: usize, theta: f64, d: f64) -> Vec<Complex64> {
        let phase_inc = -2.0 * PI * d * theta.sin();
        (0..m)
            .map(|k| {
                let ph = phase_inc * k as f64;
                Complex64::new(ph.cos(), ph.sin())
            })
            .collect()
    }

    /// Generate snapshot data for a single source
    fn make_snapshot(
        n_elements: usize,
        d: f64,
        theta: f64,
        n_snapshots: usize,
        snr: f64,
    ) -> Vec<Vec<Complex64>> {
        let sv = ula_sv(n_elements, theta, d);
        let noise_std = (1.0 / snr).sqrt();
        let mut rng: u64 = 54321;

        (0..n_elements)
            .map(|m| {
                (0..n_snapshots)
                    .map(|k| {
                        let phase = 2.0 * PI * 0.1 * k as f64;
                        let signal = Complex64::new(phase.cos(), phase.sin()) * sv[m];
                        rng = rng
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let nr = ((rng >> 33) as f64 / u32::MAX as f64 - 0.5) * 2.0 * noise_std;
                        rng = rng
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let ni = ((rng >> 33) as f64 / u32::MAX as f64 - 0.5) * 2.0 * noise_std;
                        signal + Complex64::new(nr, ni)
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_esprit_single_source() {
        let n_el = 8;
        let theta = 20.0_f64.to_radians();
        let data = make_snapshot(n_el, 0.5, theta, 300, 30.0);
        let esprit = ESPRIT::new(n_el, 1, 0.5).expect("should create ESPRIT");
        let result = esprit.estimate(&data).expect("should estimate");
        assert!(
            !result.doa_estimates.is_empty(),
            "Should have a DOA estimate"
        );
        let est_deg = result.doa_estimates[0].to_degrees();
        let true_deg = theta.to_degrees();
        assert!(
            (est_deg - true_deg).abs() < 8.0,
            "ESPRIT DOA error: est={:.2} deg, true={:.2} deg",
            est_deg,
            true_deg
        );
    }

    #[test]
    fn test_esprit_negative_angle() {
        let n_el = 8;
        let theta = -15.0_f64.to_radians();
        let data = make_snapshot(n_el, 0.5, theta, 300, 30.0);
        let esprit = ESPRIT::new(n_el, 1, 0.5).expect("should create ESPRIT");
        let result = esprit.estimate(&data).expect("should estimate");
        assert!(!result.doa_estimates.is_empty());
        let est_deg = result.doa_estimates[0].to_degrees();
        let true_deg = theta.to_degrees();
        assert!(
            (est_deg - true_deg).abs() < 8.0,
            "ESPRIT DOA error: est={:.2} deg, true={:.2} deg",
            est_deg,
            true_deg
        );
    }

    #[test]
    fn test_tls_esprit_single_source() {
        let n_el = 8;
        let theta = -10.0_f64.to_radians();
        let data = make_snapshot(n_el, 0.5, theta, 300, 30.0);
        let tls = TlsESPRIT::new(n_el, 1, 0.5).expect("should create TLS-ESPRIT");
        let result = tls.estimate(&data).expect("should estimate");
        assert!(!result.doa_estimates.is_empty());
        let est_deg = result.doa_estimates[0].to_degrees();
        let true_deg = theta.to_degrees();
        assert!(
            (est_deg - true_deg).abs() < 8.0,
            "TLS-ESPRIT DOA error: est={:.2} deg, true={:.2} deg",
            est_deg,
            true_deg
        );
    }

    #[test]
    fn test_tls_esprit_handles_noise() {
        // Test with lower SNR -- TLS should still work
        let n_el = 12;
        let theta = 25.0_f64.to_radians();
        let data = make_snapshot(n_el, 0.5, theta, 500, 10.0);
        let tls = TlsESPRIT::new(n_el, 1, 0.5).expect("should create TLS-ESPRIT");
        let result = tls.estimate(&data).expect("should estimate");
        assert!(!result.doa_estimates.is_empty());
        let est_deg = result.doa_estimates[0].to_degrees();
        let true_deg = theta.to_degrees();
        assert!(
            (est_deg - true_deg).abs() < 15.0,
            "TLS-ESPRIT DOA error in noise: est={:.2} deg, true={:.2} deg",
            est_deg,
            true_deg
        );
    }

    #[test]
    fn test_esprit_validation() {
        assert!(ESPRIT::new(2, 1, 0.5).is_err());
        assert!(ESPRIT::new(4, 0, 0.5).is_err());
        assert!(ESPRIT::new(4, 4, 0.5).is_err());
        assert!(ESPRIT::new(4, 1, 0.0).is_err());
    }

    #[test]
    fn test_tls_esprit_validation() {
        assert!(TlsESPRIT::new(2, 1, 0.5).is_err());
        assert!(TlsESPRIT::new(4, 0, 0.5).is_err());
        assert!(TlsESPRIT::new(4, 1, -0.5).is_err());
    }

    #[test]
    fn test_esprit_signal_eigenvalues() {
        let n_el = 8;
        let theta = 0.3;
        let data = make_snapshot(n_el, 0.5, theta, 300, 30.0);
        let esprit = ESPRIT::new(n_el, 1, 0.5).expect("should create ESPRIT");
        let result = esprit.estimate(&data).expect("should estimate");
        // Signal eigenvalue should be significantly larger than zero
        assert!(
            result.signal_eigenvalues[0] > 0.1,
            "Signal eigenvalue should be significant: {}",
            result.signal_eigenvalues[0]
        );
    }
}
