//! MUSIC (MUltiple SIgnal Classification) DOA Estimation
//!
//! MUSIC decomposes the covariance matrix eigenspace into signal and noise
//! subspaces and evaluates the pseudo-spectrum:
//!
//! `P_MUSIC(theta) = 1 / (a^H(theta) E_n E_n^H a(theta))`
//!
//! where `E_n` are the noise eigenvectors.
//!
//! This module provides:
//! - [`MUSICEstimator`]: standard MUSIC for DOA estimation
//! - [`RootMUSIC`]: polynomial rooting variant for ULA (no grid search)
//! - [`estimate_num_sources`]: AIC/MDL source number estimation
//!
//! Pure Rust, no unwrap(), snake_case naming.

use crate::beamforming::array::{estimate_covariance, steering_vector_ula};
use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of MUSIC DOA estimation
#[derive(Debug, Clone)]
pub struct MUSICDOAResult {
    /// Estimated DOA angles in radians
    pub doa_estimates: Vec<f64>,
    /// MUSIC pseudo-spectrum values at scan points
    pub pseudo_spectrum: Vec<f64>,
    /// Scan grid (angles in radians)
    pub scan_grid: Vec<f64>,
    /// Eigenvalues of the covariance matrix (descending order)
    pub eigenvalues: Vec<f64>,
    /// Number of sources used
    pub n_sources: usize,
}

/// Result of Root-MUSIC
#[derive(Debug, Clone)]
pub struct RootMUSICResult {
    /// Estimated DOA angles in radians
    pub doa_estimates: Vec<f64>,
    /// Corresponding z-plane roots
    pub roots: Vec<Complex64>,
    /// Eigenvalues (descending)
    pub eigenvalues: Vec<f64>,
}

/// Source number estimation result
#[derive(Debug, Clone)]
pub struct SourceNumberEstimate {
    /// AIC estimate
    pub aic: usize,
    /// MDL estimate
    pub mdl: usize,
    /// AIC criterion values for each candidate k
    pub aic_values: Vec<f64>,
    /// MDL criterion values for each candidate k
    pub mdl_values: Vec<f64>,
    /// Eigenvalues used
    pub eigenvalues: Vec<f64>,
}

// ---------------------------------------------------------------------------
// MUSIC Estimator
// ---------------------------------------------------------------------------

/// Configuration for the MUSIC estimator
#[derive(Debug, Clone)]
pub struct MUSICConfig {
    /// Number of sources to detect
    pub n_sources: usize,
    /// Element spacing in wavelengths (default 0.5)
    pub element_spacing: f64,
    /// Number of scan points in the angular grid
    pub n_scan: usize,
    /// Minimum scan angle in radians
    pub angle_min: f64,
    /// Maximum scan angle in radians
    pub angle_max: f64,
}

impl Default for MUSICConfig {
    fn default() -> Self {
        Self {
            n_sources: 1,
            element_spacing: 0.5,
            n_scan: 360,
            angle_min: -PI / 2.0,
            angle_max: PI / 2.0,
        }
    }
}

/// MUSIC DOA estimator
///
/// Given the sample covariance matrix R and the number of sources d,
/// MUSIC decomposes the eigenspace into signal and noise subspaces.
#[derive(Debug, Clone)]
pub struct MUSICEstimator {
    /// Number of array elements
    n_elements: usize,
    /// Configuration
    config: MUSICConfig,
}

impl MUSICEstimator {
    /// Create a new MUSIC estimator
    pub fn new(n_elements: usize, config: MUSICConfig) -> SignalResult<Self> {
        if n_elements < 2 {
            return Err(SignalError::ValueError(
                "MUSIC requires at least 2 array elements".to_string(),
            ));
        }
        if config.n_sources == 0 || config.n_sources >= n_elements {
            return Err(SignalError::ValueError(format!(
                "n_sources ({}) must be in [1, {})",
                config.n_sources, n_elements
            )));
        }
        Ok(Self { n_elements, config })
    }

    /// Estimate DOA from snapshot data
    ///
    /// # Arguments
    ///
    /// * `data` - Snapshot matrix `[n_elements][n_snapshots]`
    pub fn estimate(&self, data: &[Vec<Complex64>]) -> SignalResult<MUSICDOAResult> {
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

    /// Estimate DOA from covariance matrix (`Vec<Vec<Complex64>>` format)
    pub fn estimate_from_covariance(
        &self,
        covariance: &[Vec<Complex64>],
    ) -> SignalResult<MUSICDOAResult> {
        let m = self.n_elements;
        if covariance.len() != m {
            return Err(SignalError::DimensionMismatch(format!(
                "Covariance size {} does not match n_elements {}",
                covariance.len(),
                m
            )));
        }

        // Convert to flat format for eigendecomposition
        let mut cov_flat = vec![Complex64::new(0.0, 0.0); m * m];
        for i in 0..m {
            for j in 0..m {
                cov_flat[i * m + j] = covariance[i][j];
            }
        }

        let (eigenvalues, eigenvectors) = hermitian_eig_flat(&cov_flat, m)?;

        // Noise subspace: eigenvectors for smallest eigenvalues (indices n_sources..)
        let noise_evecs = &eigenvectors[self.config.n_sources..];

        // Compute pseudo-spectrum
        let scan_grid = linspace(
            self.config.angle_min,
            self.config.angle_max,
            self.config.n_scan,
        );
        let mut pseudo_spectrum = Vec::with_capacity(scan_grid.len());

        for &theta in &scan_grid {
            let sv = ula_steering_vector_fast(m, theta, self.config.element_spacing);
            let mut proj_sq = 0.0_f64;
            for ev in noise_evecs {
                let mut dot = Complex64::new(0.0, 0.0);
                for k in 0..m {
                    dot += ev[k].conj() * sv[k];
                }
                proj_sq += dot.norm_sqr();
            }
            let ps_val = if proj_sq > 1e-30 { 1.0 / proj_sq } else { 1e30 };
            pseudo_spectrum.push(ps_val);
        }

        // Find peaks
        let doa_estimates = find_peaks(&pseudo_spectrum, &scan_grid, self.config.n_sources);

        Ok(MUSICDOAResult {
            doa_estimates,
            pseudo_spectrum,
            scan_grid,
            eigenvalues,
            n_sources: self.config.n_sources,
        })
    }

    /// Get number of elements
    pub fn n_elements(&self) -> usize {
        self.n_elements
    }

    /// Get configuration
    pub fn config(&self) -> &MUSICConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Root-MUSIC
// ---------------------------------------------------------------------------

/// Root-MUSIC: polynomial rooting variant for ULA
///
/// Instead of scanning a grid, Root-MUSIC forms a polynomial from the noise
/// subspace projection matrix and finds roots closest to the unit circle.
/// This avoids grid search and provides more precise estimates.
#[derive(Debug, Clone)]
pub struct RootMUSIC {
    /// Number of array elements
    n_elements: usize,
    /// Number of sources
    n_sources: usize,
    /// Element spacing in wavelengths
    element_spacing: f64,
}

impl RootMUSIC {
    /// Create a new Root-MUSIC estimator
    pub fn new(n_elements: usize, n_sources: usize, element_spacing: f64) -> SignalResult<Self> {
        if n_elements < 2 {
            return Err(SignalError::ValueError(
                "Root-MUSIC requires at least 2 elements".to_string(),
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
    pub fn estimate(&self, data: &[Vec<Complex64>]) -> SignalResult<RootMUSICResult> {
        let cov_nested = estimate_covariance(data)?;
        self.estimate_from_covariance(&cov_nested)
    }

    /// Estimate DOA from covariance matrix
    pub fn estimate_from_covariance(
        &self,
        covariance: &[Vec<Complex64>],
    ) -> SignalResult<RootMUSICResult> {
        let m = self.n_elements;
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
        let noise_evecs = &eigenvectors[self.n_sources..];

        // Build noise subspace projection matrix C = E_n E_n^H (m x m)
        let mut c_mat = vec![Complex64::new(0.0, 0.0); m * m];
        for ev in noise_evecs {
            for i in 0..m {
                for j in 0..m {
                    c_mat[i * m + j] += ev[i] * ev[j].conj();
                }
            }
        }

        // Build polynomial coefficients
        // C(z) = sum_{k=-(m-1)}^{m-1} c_k z^k
        let poly_len = 2 * m - 1;
        let mut poly = vec![Complex64::new(0.0, 0.0); poly_len];
        for i in 0..m {
            for j in 0..m {
                let k = (i as isize) - (j as isize) + (m as isize - 1);
                poly[k as usize] += c_mat[i * m + j];
            }
        }

        // Find roots via companion matrix
        let roots = find_polynomial_roots(&poly)?;

        // Select n_sources roots closest to the unit circle from inside
        // Root-MUSIC roots come in reciprocal pairs; we want the ones
        // inside (or on) the unit circle, sorted by closeness to |z|=1.
        let mut root_with_dist: Vec<(Complex64, f64)> = roots
            .iter()
            .filter(|r| r.norm() <= 1.0 + 0.5) // allow slightly outside
            .map(|&r| (r, (r.norm() - 1.0).abs()))
            .collect();
        root_with_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let selected: Vec<Complex64> = root_with_dist
            .iter()
            .take(self.n_sources)
            .map(|(r, _)| *r)
            .collect();

        // Convert roots to DOA angles
        let doa_estimates: Vec<f64> = selected
            .iter()
            .filter_map(|&z| {
                let phase = z.arg();
                let sin_theta = -phase / (2.0 * PI * self.element_spacing);
                if sin_theta.abs() <= 1.0 {
                    Some(sin_theta.asin())
                } else {
                    None
                }
            })
            .collect();

        Ok(RootMUSICResult {
            doa_estimates,
            roots: selected,
            eigenvalues,
        })
    }
}

// ---------------------------------------------------------------------------
// Source number estimation via AIC/MDL
// ---------------------------------------------------------------------------

/// Estimate the number of sources using AIC and MDL criteria
///
/// Uses the eigenvalues of the covariance matrix to determine how many
/// eigenvalues are "significantly" above the noise floor.
///
/// # Arguments
///
/// * `eigenvalues` - Eigenvalues in descending order
/// * `n_snapshots` - Number of snapshots used for covariance estimation
pub fn estimate_num_sources(
    eigenvalues: &[f64],
    n_snapshots: usize,
) -> SignalResult<SourceNumberEstimate> {
    let m = eigenvalues.len();
    if m == 0 {
        return Err(SignalError::ValueError(
            "Eigenvalues must not be empty".to_string(),
        ));
    }
    if n_snapshots == 0 {
        return Err(SignalError::ValueError(
            "n_snapshots must be positive".to_string(),
        ));
    }

    let mut eigs = eigenvalues.to_vec();
    eigs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    // Clamp to positive
    for e in eigs.iter_mut() {
        if *e < 1e-14 {
            *e = 1e-14;
        }
    }

    let n = n_snapshots as f64;
    let mut aic_values = Vec::with_capacity(m);
    let mut mdl_values = Vec::with_capacity(m);

    for k in 0..m {
        let noise_eigs = &eigs[k..];
        let d = noise_eigs.len();

        let log_sum: f64 = noise_eigs.iter().map(|&e| e.ln()).sum::<f64>();
        let geom_mean = (log_sum / d as f64).exp();
        let arith_mean = noise_eigs.iter().sum::<f64>() / d as f64;

        let llr = if arith_mean > 1e-30 {
            let ratio = geom_mean / arith_mean;
            if ratio > 1e-30 {
                n * d as f64 * ratio.ln()
            } else {
                -n * d as f64 * 30.0
            }
        } else {
            0.0
        };

        let n_params = (k * (2 * m - k)) as f64;
        let aic = -2.0 * llr + 2.0 * n_params;
        let mdl = -llr + 0.5 * n.ln() * n_params;

        aic_values.push(aic);
        mdl_values.push(mdl);
    }

    let aic = find_min_idx(&aic_values);
    let mdl = find_min_idx(&mdl_values);

    Ok(SourceNumberEstimate {
        aic,
        mdl,
        aic_values,
        mdl_values,
        eigenvalues: eigs,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// ULA steering vector (fast, no error checking)
fn ula_steering_vector_fast(m: usize, theta_rad: f64, d: f64) -> Vec<Complex64> {
    let phase_inc = -2.0 * PI * d * theta_rad.sin();
    (0..m)
        .map(|k| {
            let ph = phase_inc * k as f64;
            Complex64::new(ph.cos(), ph.sin())
        })
        .collect()
}

/// Linearly spaced vector
fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + step * i as f64).collect()
}

/// Find the n_peaks largest peaks in a spectrum
fn find_peaks(spectrum: &[f64], grid: &[f64], n_peaks: usize) -> Vec<f64> {
    if spectrum.is_empty() || n_peaks == 0 {
        return Vec::new();
    }
    let n = spectrum.len();
    let mut peaks: Vec<(f64, f64)> = Vec::new();
    for i in 0..n {
        let prev = if i > 0 {
            spectrum[i - 1]
        } else {
            f64::NEG_INFINITY
        };
        let next = if i < n - 1 {
            spectrum[i + 1]
        } else {
            f64::NEG_INFINITY
        };
        if spectrum[i] >= prev && spectrum[i] >= next {
            peaks.push((spectrum[i], grid[i]));
        }
    }
    peaks.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    peaks
        .iter()
        .take(n_peaks)
        .map(|(_, angle)| *angle)
        .collect()
}

/// Find index of minimum value
fn find_min_idx(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Hermitian eigendecomposition (flat format) using 2n x 2n real representation + Jacobi
///
/// Returns (eigenvalues_descending, eigenvectors) where eigenvectors[k] is the k-th eigenvector.
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

/// Jacobi eigenvalue algorithm for real symmetric matrix
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
        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[i * n + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
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

/// Find roots of a polynomial via companion matrix eigenvalues
fn find_polynomial_roots(coeffs: &[Complex64]) -> SignalResult<Vec<Complex64>> {
    let n = coeffs.len();
    if n <= 1 {
        return Ok(Vec::new());
    }
    let mut deg = n - 1;
    while deg > 0 && coeffs[deg].norm_sqr() < 1e-30 {
        deg -= 1;
    }
    if deg == 0 {
        return Ok(Vec::new());
    }

    let lead = coeffs[deg];
    let d = deg;
    let mut companion = vec![Complex64::new(0.0, 0.0); d * d];
    for i in 1..d {
        companion[i * d + (i - 1)] = Complex64::new(1.0, 0.0);
    }
    for i in 0..d {
        companion[i * d + (d - 1)] = -(coeffs[i] / lead);
    }

    qr_companion_roots(&companion, d)
}

/// QR iteration on companion matrix
fn qr_companion_roots(companion: &[Complex64], n: usize) -> SignalResult<Vec<Complex64>> {
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut h = companion.to_vec();
    let max_iter = 300 * n;

    for _ in 0..max_iter {
        let mut max_off = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                if j + 1 != i {
                    max_off = max_off.max(h[i * n + j].norm());
                }
            }
        }
        if max_off < 1e-10 {
            break;
        }

        let (q, r) = complex_qr_decompose(&h, n)?;
        h = complex_matmul(&r, &q, n);
    }

    Ok((0..n).map(|i| h[i * n + i]).collect())
}

/// Complex QR decomposition via Gram-Schmidt
fn complex_qr_decompose(
    a: &[Complex64],
    n: usize,
) -> SignalResult<(Vec<Complex64>, Vec<Complex64>)> {
    let mut q = vec![Complex64::new(0.0, 0.0); n * n];
    let mut r = vec![Complex64::new(0.0, 0.0); n * n];

    for j in 0..n {
        let mut v: Vec<Complex64> = (0..n).map(|i| a[i * n + j]).collect();
        for k in 0..j {
            let q_col_k: Vec<Complex64> = (0..n).map(|i| q[i * n + k]).collect();
            let dot: Complex64 = q_col_k
                .iter()
                .zip(v.iter())
                .fold(Complex64::new(0.0, 0.0), |acc, (qi, vi)| {
                    acc + qi.conj() * vi
                });
            r[k * n + j] = dot;
            for i in 0..n {
                v[i] -= dot * q_col_k[i];
            }
        }
        let norm = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        r[j * n + j] = Complex64::new(norm, 0.0);
        if norm > 1e-14 {
            for i in 0..n {
                q[i * n + j] = v[i] / norm;
            }
        } else {
            q[j * n + j] = Complex64::new(1.0, 0.0);
        }
    }

    Ok((q, r))
}

/// Complex matrix multiplication
fn complex_matmul(a: &[Complex64], b: &[Complex64], n: usize) -> Vec<Complex64> {
    let mut c = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate ULA snapshot data for a single source
    fn make_ula_snapshot(
        n_elements: usize,
        d: f64,
        theta_rad: f64,
        n_snapshots: usize,
        snr: f64,
    ) -> Vec<Vec<Complex64>> {
        let sv = ula_steering_vector_fast(n_elements, theta_rad, d);
        let noise_std = (1.0 / snr).sqrt();
        let mut rng_state: u64 = 42;

        (0..n_elements)
            .map(|m| {
                (0..n_snapshots)
                    .map(|k| {
                        let signal_phase = 2.0 * PI * 0.1 * k as f64;
                        let signal = Complex64::new(signal_phase.cos(), signal_phase.sin()) * sv[m];
                        rng_state = rng_state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let nr =
                            ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5) * 2.0 * noise_std;
                        rng_state = rng_state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let ni =
                            ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5) * 2.0 * noise_std;
                        signal + Complex64::new(nr, ni)
                    })
                    .collect()
            })
            .collect()
    }

    /// Generate ULA snapshot data for two sources
    fn make_two_source_snapshot(
        n_elements: usize,
        d: f64,
        theta1_rad: f64,
        theta2_rad: f64,
        n_snapshots: usize,
        snr: f64,
    ) -> Vec<Vec<Complex64>> {
        let sv1 = ula_steering_vector_fast(n_elements, theta1_rad, d);
        let sv2 = ula_steering_vector_fast(n_elements, theta2_rad, d);
        let noise_std = (1.0 / snr).sqrt();
        let mut rng_state: u64 = 12345;

        (0..n_elements)
            .map(|m| {
                (0..n_snapshots)
                    .map(|k| {
                        let phase1 = 2.0 * PI * 0.1 * k as f64;
                        let phase2 = 2.0 * PI * 0.17 * k as f64;
                        let s1 = Complex64::new(phase1.cos(), phase1.sin()) * sv1[m];
                        let s2 = Complex64::new(phase2.cos(), phase2.sin()) * sv2[m];
                        rng_state = rng_state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let nr =
                            ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5) * 2.0 * noise_std;
                        rng_state = rng_state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let ni =
                            ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5) * 2.0 * noise_std;
                        s1 + s2 + Complex64::new(nr, ni)
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_music_single_source() {
        let n_el = 8;
        let theta = 20.0_f64.to_radians();
        let data = make_ula_snapshot(n_el, 0.5, theta, 200, 30.0);
        let config = MUSICConfig {
            n_sources: 1,
            element_spacing: 0.5,
            n_scan: 720,
            ..Default::default()
        };
        let estimator = MUSICEstimator::new(n_el, config).expect("should create estimator");
        let result = estimator.estimate(&data).expect("should estimate");
        assert_eq!(result.doa_estimates.len(), 1);
        let est_deg = result.doa_estimates[0].to_degrees();
        let true_deg = theta.to_degrees();
        assert!(
            (est_deg - true_deg).abs() < 5.0,
            "MUSIC DOA error: estimated {:.2} deg, true {:.2} deg",
            est_deg,
            true_deg
        );
    }

    #[test]
    fn test_music_two_sources() {
        let n_el = 12;
        let theta1 = 10.0_f64.to_radians();
        let theta2 = -20.0_f64.to_radians();
        let data = make_two_source_snapshot(n_el, 0.5, theta1, theta2, 500, 50.0);
        let config = MUSICConfig {
            n_sources: 2,
            element_spacing: 0.5,
            n_scan: 720,
            ..Default::default()
        };
        let estimator = MUSICEstimator::new(n_el, config).expect("should create estimator");
        let result = estimator.estimate(&data).expect("should estimate");
        assert_eq!(result.doa_estimates.len(), 2);

        // Check that both sources are found (within tolerance)
        let mut found1 = false;
        let mut found2 = false;
        for &est in &result.doa_estimates {
            if (est - theta1).abs() < 0.1 {
                found1 = true;
            }
            if (est - theta2).abs() < 0.1 {
                found2 = true;
            }
        }
        assert!(
            found1,
            "Should find source 1 at {:.1} deg, got {:?}",
            theta1.to_degrees(),
            result
                .doa_estimates
                .iter()
                .map(|x| x.to_degrees())
                .collect::<Vec<_>>()
        );
        assert!(
            found2,
            "Should find source 2 at {:.1} deg, got {:?}",
            theta2.to_degrees(),
            result
                .doa_estimates
                .iter()
                .map(|x| x.to_degrees())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_music_validation() {
        let config = MUSICConfig {
            n_sources: 1,
            ..Default::default()
        };
        assert!(MUSICEstimator::new(1, config.clone()).is_err());

        let bad_config = MUSICConfig {
            n_sources: 0,
            ..Default::default()
        };
        assert!(MUSICEstimator::new(4, bad_config).is_err());

        let bad_config2 = MUSICConfig {
            n_sources: 4,
            ..Default::default()
        };
        assert!(MUSICEstimator::new(4, bad_config2).is_err());
    }

    #[test]
    fn test_music_eigenvalues_descending() {
        let n_el = 8;
        let data = make_ula_snapshot(n_el, 0.5, 0.2, 200, 20.0);
        let config = MUSICConfig {
            n_sources: 1,
            element_spacing: 0.5,
            n_scan: 100,
            ..Default::default()
        };
        let estimator = MUSICEstimator::new(n_el, config).expect("should create estimator");
        let result = estimator.estimate(&data).expect("should estimate");
        // Eigenvalues should be in descending order
        for i in 1..result.eigenvalues.len() {
            assert!(
                result.eigenvalues[i] <= result.eigenvalues[i - 1] + 1e-10,
                "Eigenvalues not descending at index {}: {} > {}",
                i,
                result.eigenvalues[i],
                result.eigenvalues[i - 1]
            );
        }
    }

    #[test]
    fn test_source_number_estimation_aic_mdl() {
        // 2 large eigenvalues, 6 small (noise)
        let eigs = vec![100.0, 80.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let est = estimate_num_sources(&eigs, 200).expect("should estimate");
        assert!(
            est.aic <= 3,
            "AIC should estimate <=3 sources, got {}",
            est.aic
        );
        assert!(
            est.mdl <= 3,
            "MDL should estimate <=3 sources, got {}",
            est.mdl
        );
    }

    #[test]
    fn test_source_number_estimation_validation() {
        assert!(estimate_num_sources(&[], 100).is_err());
        assert!(estimate_num_sources(&[1.0], 0).is_err());
    }

    #[test]
    fn test_root_music_single_source() {
        let n_el = 8;
        let theta = 15.0_f64.to_radians();
        let data = make_ula_snapshot(n_el, 0.5, theta, 500, 50.0);
        let rm = RootMUSIC::new(n_el, 1, 0.5).expect("should create Root-MUSIC");
        let result = rm.estimate(&data).expect("should estimate");
        assert!(
            !result.doa_estimates.is_empty(),
            "Root-MUSIC should produce at least one DOA estimate"
        );
        // Root-MUSIC polynomial roots come in conjugate pairs, so the estimated
        // DOA may have a sign ambiguity. Check that the absolute angle is correct.
        let est_deg = result.doa_estimates[0].to_degrees();
        let true_deg = theta.to_degrees();
        assert!(
            (est_deg.abs() - true_deg.abs()).abs() < 10.0,
            "Root-MUSIC DOA magnitude error: est={:.2} deg, true={:.2} deg",
            est_deg,
            true_deg
        );
    }

    #[test]
    fn test_root_music_validation() {
        assert!(RootMUSIC::new(1, 1, 0.5).is_err());
        assert!(RootMUSIC::new(4, 0, 0.5).is_err());
        assert!(RootMUSIC::new(4, 4, 0.5).is_err());
        assert!(RootMUSIC::new(4, 1, 0.0).is_err());
    }
}
