//! ESPRIT (Estimation of Signal Parameters via Rotational Invariance Techniques)
//!
//! Provides:
//! - [`ESPRIT1D`]: 1D ESPRIT for ULA — exploits shift-invariance
//! - [`TlsEsprit`]: Total Least Squares ESPRIT (improved bias/variance)
//! - [`ESPRIT2D`]: 2D ESPRIT for planar arrays (URA)
//!
//! References:
//! - Roy, R. & Kailath, T. (1989). "ESPRIT — estimation of signal parameters via rotational
//!   invariance techniques." IEEE Trans. ASSP, 37(7), 984–995.
//! - Zoltowski, M.D. et al. (1993). "Closed-form 2-D angle estimation with rectangular arrays in
//!   element space or beamspace." IEEE Trans. SP, 41(1), 316–328.
//! - Haardt, M. & Nossek, J.A. (1995). "Unitary ESPRIT: How to obtain increased estimation
//!   accuracy with a reduced computational burden." IEEE Trans. SP, 43(5), 1232–1242.

use crate::error::{SignalError, SignalResult};
use crate::subspace::array_processing::{hermitian_eig, SpatialCovariance};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of 1D ESPRIT
#[derive(Debug, Clone)]
pub struct ESPRITResult {
    /// Estimated DOA angles in radians
    pub doa_estimates: Vec<f64>,
    /// Rotational invariance eigenvalues (z = exp(jω), one per source)
    pub phase_shifts: Vec<Complex64>,
    /// Signal subspace eigenvalues (of the covariance matrix)
    pub signal_eigenvalues: Vec<f64>,
}

/// Result of 2D ESPRIT
#[derive(Debug, Clone)]
pub struct ESPRIT2DResult {
    /// Estimated elevation angles in radians
    pub elevation: Vec<f64>,
    /// Estimated azimuth angles in radians
    pub azimuth: Vec<f64>,
    /// Phase shifts in x-direction
    pub phase_shifts_x: Vec<Complex64>,
    /// Phase shifts in y-direction
    pub phase_shifts_y: Vec<Complex64>,
}

// ---------------------------------------------------------------------------
// 1D ESPRIT for ULA
// ---------------------------------------------------------------------------

/// 1D ESPRIT for Uniform Linear Arrays
///
/// Exploits the shift-invariance of the ULA steering matrix. For an M-element ULA,
/// the two overlapping sub-arrays of size `M-1` are related by a rotational shift.
///
/// The algorithm:
/// 1. Compute the sample covariance matrix `R` from the snapshot data.
/// 2. Eigendecompose `R`; take the `d` dominant eigenvectors as `E_s`.
/// 3. Partition `E_s` into two sub-arrays `E1 = E_s[0:M-1,:]` and `E2 = E_s[1:M,:]`.
/// 4. Solve the shift invariance equation `Ψ = (E1^H E1)^{-1} E1^H E2` (LS)
///    or via SVD/TLS (see [`TlsEsprit`]).
/// 5. Eigenvalues of `Ψ` give phase shifts → DOA angles.
#[derive(Debug, Clone)]
pub struct ESPRIT1D {
    /// Number of array elements
    pub n_elements: usize,
    /// Number of sources
    pub n_sources: usize,
    /// Inter-element spacing in wavelengths
    pub element_spacing: f64,
}

impl ESPRIT1D {
    /// Create a new ESPRIT1D estimator
    ///
    /// # Arguments
    ///
    /// * `n_elements`      - Total number of ULA elements
    /// * `n_sources`       - Number of signal sources
    /// * `element_spacing` - Element spacing in wavelengths (typically 0.5)
    pub fn new(n_elements: usize, n_sources: usize, element_spacing: f64) -> SignalResult<Self> {
        if n_elements < 3 {
            return Err(SignalError::ValueError(
                "ESPRIT requires at least 3 elements".to_string(),
            ));
        }
        if n_sources == 0 || n_sources >= n_elements {
            return Err(SignalError::ValueError(format!(
                "n_sources ({n_sources}) must be in [1, {n_elements})"
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

    /// Estimate DOA from snapshot matrix `data[element][snapshot]`
    pub fn estimate(&self, data: &[Vec<Complex64>]) -> SignalResult<ESPRITResult> {
        let cov = SpatialCovariance::estimate(data)?;
        self.estimate_from_covariance(&cov)
    }

    /// Estimate DOA from precomputed covariance
    pub fn estimate_from_covariance(
        &self,
        covariance: &SpatialCovariance,
    ) -> SignalResult<ESPRITResult> {
        if covariance.size != self.n_elements {
            return Err(SignalError::DimensionMismatch(format!(
                "Covariance size {} ≠ n_elements {}",
                covariance.size, self.n_elements
            )));
        }
        let m = self.n_elements;
        let d = self.n_sources;

        let (eigenvalues, eigenvectors) = hermitian_eig(&covariance.matrix, m)?;
        let signal_eigenvalues: Vec<f64> = eigenvalues[..d].to_vec();

        // Signal subspace: first d eigenvectors (largest eigenvalues)
        // Each eigenvector is a column; we need E_s as m×d matrix
        // es[k] = eigenvectors[k] (k-th eigenvector, length m)
        let es: Vec<&Vec<Complex64>> = eigenvectors[..d].iter().collect();

        // Build two sub-arrays of size (m-1) × d
        // E1 = E_s[0..m-1, :], E2 = E_s[1..m, :]
        let m1 = m - 1;
        // E1 as flat column-major: e1[row*d + col] = es[col][row]
        let mut e1 = vec![Complex64::new(0.0, 0.0); m1 * d];
        let mut e2 = vec![Complex64::new(0.0, 0.0); m1 * d];
        for col in 0..d {
            for row in 0..m1 {
                e1[row * d + col] = es[col][row];
                e2[row * d + col] = es[col][row + 1];
            }
        }

        // Solve Ψ (d×d) via LS: Ψ = (E1^H E1)^{-1} E1^H E2
        // E1^H E1 is d×d, E1^H E2 is d×d
        let e1h_e1 = complex_gram(&e1, m1, d);
        let e1h_e2 = complex_cross_gram(&e1, &e2, m1, d);

        // Solve E1^H E1 * Ψ = E1^H E2 via inversion
        let psi = complex_solve_dd(&e1h_e1, &e1h_e2, d)?;

        // Eigenvalues of Ψ → phase shifts
        let phase_shifts = eigenvalues_complex_matrix(&psi, d)?;

        // Convert phase shifts to DOA angles
        let doa_estimates: Vec<f64> = phase_shifts
            .iter()
            .filter_map(|&z| {
                let phi = z.arg();
                // phi = -2*pi*d*sin(theta)  → sin(theta) = -phi/(2*pi*d)
                let sin_theta = -phi / (2.0 * PI * self.element_spacing);
                if sin_theta.abs() <= 1.0 {
                    Some(sin_theta.asin())
                } else {
                    None
                }
            })
            .collect();

        Ok(ESPRITResult {
            doa_estimates,
            phase_shifts,
            signal_eigenvalues,
        })
    }
}

// ---------------------------------------------------------------------------
// TLS-ESPRIT
// ---------------------------------------------------------------------------

/// Total Least Squares ESPRIT
///
/// TLS-ESPRIT improves over LS-ESPRIT by treating both `E1` and `E2` as noisy
/// observations. It is based on the SVD of `[E1 | E2]` and exploits the
/// minimum singular value to define the TLS solution.
#[derive(Debug, Clone)]
pub struct TlsEsprit {
    /// Number of array elements
    pub n_elements: usize,
    /// Number of sources
    pub n_sources: usize,
    /// Element spacing in wavelengths
    pub element_spacing: f64,
}

impl TlsEsprit {
    /// Create a new TLS-ESPRIT estimator
    pub fn new(n_elements: usize, n_sources: usize, element_spacing: f64) -> SignalResult<Self> {
        if n_elements < 3 {
            return Err(SignalError::ValueError(
                "TLS-ESPRIT requires at least 3 elements".to_string(),
            ));
        }
        if n_sources == 0 || n_sources >= n_elements {
            return Err(SignalError::ValueError(format!(
                "n_sources ({n_sources}) must be in [1, {n_elements})"
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
        let cov = SpatialCovariance::estimate(data)?;
        self.estimate_from_covariance(&cov)
    }

    /// Estimate from precomputed covariance
    pub fn estimate_from_covariance(
        &self,
        covariance: &SpatialCovariance,
    ) -> SignalResult<ESPRITResult> {
        if covariance.size != self.n_elements {
            return Err(SignalError::DimensionMismatch(format!(
                "Covariance size {} ≠ n_elements {}",
                covariance.size, self.n_elements
            )));
        }
        let m = self.n_elements;
        let d = self.n_sources;

        let (eigenvalues, eigenvectors) = hermitian_eig(&covariance.matrix, m)?;
        let signal_eigenvalues: Vec<f64> = eigenvalues[..d].to_vec();

        let es: Vec<&Vec<Complex64>> = eigenvectors[..d].iter().collect();
        let m1 = m - 1;

        // Build [E1 | E2] as (m-1) × 2d matrix
        let mut e12 = vec![Complex64::new(0.0, 0.0); m1 * (2 * d)];
        for col in 0..d {
            for row in 0..m1 {
                e12[row * (2 * d) + col] = es[col][row];         // E1 left half
                e12[row * (2 * d) + d + col] = es[col][row + 1]; // E2 right half
            }
        }

        // SVD of [E1 | E2]: we need the right singular vectors
        // V^H of [E1 | E2] gives us V₁₂ (2d × 2d) — we need the last d singular vectors
        // For TLS: partition V = [[V11, V12], [V21, V22]] (d×d blocks)
        // and Ψ_TLS = -V12 V22^{-1}
        let v = compute_right_singular_vectors(&e12, m1, 2 * d)?;

        // Partition V into 4 blocks (each d×d)
        let mut v12 = vec![Complex64::new(0.0, 0.0); d * d];
        let mut v22 = vec![Complex64::new(0.0, 0.0); d * d];
        for i in 0..d {
            for j in 0..d {
                // v is (2d)×(2d); columns correspond to sorted singular vectors
                // We want the *last* 2d - d = d right singular vectors (smallest)
                // They are columns d..2d-1 of V (0-indexed)
                // v[row*(2d)+col] for row in 0..2d, col in d..2d
                let col_in_v = d + j; // col index in V (0-indexed among the 2d cols)
                v12[i * d + j] = v[i * (2 * d) + col_in_v];
                v22[i * d + j] = v[(d + i) * (2 * d) + col_in_v];
            }
        }

        // Ψ_TLS = -V12 * V22^{-1}
        let v22_inv = complex_matrix_inv(&v22, d)?;
        let psi_tls = complex_matmul_dd(&v12, &v22_inv, d);
        // Negate
        let psi_neg: Vec<Complex64> = psi_tls.iter().map(|x| -(*x)).collect();

        let phase_shifts = eigenvalues_complex_matrix(&psi_neg, d)?;

        let doa_estimates: Vec<f64> = phase_shifts
            .iter()
            .filter_map(|&z| {
                let phi = z.arg();
                let sin_theta = -phi / (2.0 * PI * self.element_spacing);
                if sin_theta.abs() <= 1.0 {
                    Some(sin_theta.asin())
                } else {
                    None
                }
            })
            .collect();

        Ok(ESPRITResult {
            doa_estimates,
            phase_shifts,
            signal_eigenvalues,
        })
    }
}

// ---------------------------------------------------------------------------
// 2D ESPRIT for planar arrays (URA)
// ---------------------------------------------------------------------------

/// 2D ESPRIT for Uniform Rectangular Arrays (URA)
///
/// Extends ESPRIT to two dimensions for joint azimuth/elevation estimation.
/// The array has `nx` elements in the x-direction and `ny` elements in the y-direction.
///
/// Two shift invariance equations are exploited simultaneously:
/// - `E1_x`, `E2_x` for the x-direction shift
/// - `E1_y`, `E2_y` for the y-direction shift
///
/// The 2D phase-shift matrices `Φ_x` and `Φ_y` are then jointly diagonalised to
/// pair the azimuth/elevation estimates.
#[derive(Debug, Clone)]
pub struct ESPRIT2D {
    /// Number of elements in x direction
    pub nx: usize,
    /// Number of elements in y direction
    pub ny: usize,
    /// Number of sources
    pub n_sources: usize,
    /// Element spacing in x-direction (wavelengths)
    pub dx: f64,
    /// Element spacing in y-direction (wavelengths)
    pub dy: f64,
}

impl ESPRIT2D {
    /// Create a new 2D ESPRIT estimator
    ///
    /// # Arguments
    ///
    /// * `nx`, `ny`      - Elements in x and y directions
    /// * `n_sources`     - Number of sources
    /// * `dx`, `dy`      - Element spacings in wavelengths
    pub fn new(
        nx: usize,
        ny: usize,
        n_sources: usize,
        dx: f64,
        dy: f64,
    ) -> SignalResult<Self> {
        let total = nx * ny;
        if nx < 2 || ny < 2 {
            return Err(SignalError::ValueError(
                "2D ESPRIT requires at least 2 elements in each direction".to_string(),
            ));
        }
        if n_sources == 0 || n_sources >= total {
            return Err(SignalError::ValueError(format!(
                "n_sources ({n_sources}) must be in [1, {total})"
            )));
        }
        if dx <= 0.0 || dy <= 0.0 {
            return Err(SignalError::ValueError(
                "Element spacings must be positive".to_string(),
            ));
        }
        Ok(Self {
            nx,
            ny,
            n_sources,
            dx,
            dy,
        })
    }

    /// Estimate 2D DOA from snapshot data
    ///
    /// `data` is `[nx*ny][n_snapshots]`; rows are ordered as element (xi, yi)
    /// with row index `i = xi * ny + yi`.
    pub fn estimate(&self, data: &[Vec<Complex64>]) -> SignalResult<ESPRIT2DResult> {
        let m = self.nx * self.ny;
        if data.len() != m {
            return Err(SignalError::DimensionMismatch(format!(
                "data has {} rows, expected {}",
                data.len(),
                m
            )));
        }
        let cov = SpatialCovariance::estimate(data)?;
        self.estimate_from_covariance(&cov)
    }

    /// Estimate 2D DOA from precomputed covariance
    pub fn estimate_from_covariance(
        &self,
        covariance: &SpatialCovariance,
    ) -> SignalResult<ESPRIT2DResult> {
        let m = self.nx * self.ny;
        if covariance.size != m {
            return Err(SignalError::DimensionMismatch(format!(
                "Covariance size {} ≠ nx*ny={}",
                covariance.size, m
            )));
        }
        let d = self.n_sources;
        let nx = self.nx;
        let ny = self.ny;

        let (_, eigenvectors) = hermitian_eig(&covariance.matrix, m)?;
        let es: Vec<&Vec<Complex64>> = eigenvectors[..d].iter().collect();

        // Build x-shift sub-arrays: indices (xi, yi) vs (xi+1, yi) for xi in 0..nx-1
        // Row index for element (xi, yi) = xi * ny + yi
        let mx = (nx - 1) * ny; // number of pairs in x
        let my = nx * (ny - 1); // number of pairs in y

        let mut e1x = vec![Complex64::new(0.0, 0.0); mx * d];
        let mut e2x = vec![Complex64::new(0.0, 0.0); mx * d];
        let mut e1y = vec![Complex64::new(0.0, 0.0); my * d];
        let mut e2y = vec![Complex64::new(0.0, 0.0); my * d];

        for col in 0..d {
            let mut rx = 0usize;
            for xi in 0..(nx - 1) {
                for yi in 0..ny {
                    let idx0 = xi * ny + yi;
                    let idx1 = (xi + 1) * ny + yi;
                    e1x[rx * d + col] = es[col][idx0];
                    e2x[rx * d + col] = es[col][idx1];
                    rx += 1;
                }
            }
            let mut ry = 0usize;
            for xi in 0..nx {
                for yi in 0..(ny - 1) {
                    let idx0 = xi * ny + yi;
                    let idx1 = xi * ny + yi + 1;
                    e1y[ry * d + col] = es[col][idx0];
                    e2y[ry * d + col] = es[col][idx1];
                    ry += 1;
                }
            }
        }

        // LS ESPRIT for x-direction
        let e1x_h_e1x = complex_gram(&e1x, mx, d);
        let e1x_h_e2x = complex_cross_gram(&e1x, &e2x, mx, d);
        let psi_x = complex_solve_dd(&e1x_h_e1x, &e1x_h_e2x, d)?;

        // LS ESPRIT for y-direction
        let e1y_h_e1y = complex_gram(&e1y, my, d);
        let e1y_h_e2y = complex_cross_gram(&e1y, &e2y, my, d);
        let psi_y = complex_solve_dd(&e1y_h_e1y, &e1y_h_e2y, d)?;

        // Eigenvalues of Ψx and Ψy give phase shifts
        let phase_x = eigenvalues_complex_matrix(&psi_x, d)?;
        let phase_y = eigenvalues_complex_matrix(&psi_y, d)?;

        // Pair phase shifts via automatic pairing using a joint diagonalisation approach
        // Simple approach: jointly diagonalise Ψx and Ψy sharing same eigenvector matrix
        // For well-separated sources, we can use the pairing of the shared eigenvectors
        let (paired_x, paired_y) = pair_2d_estimates(&psi_x, &psi_y, d)?;

        // Convert to elevation and azimuth
        let elevation: Vec<f64> = paired_x
            .iter()
            .filter_map(|&z| {
                let phi_x = -z.arg() / (2.0 * PI * self.dx);
                if phi_x.abs() <= 1.0 {
                    Some(phi_x.asin())
                } else {
                    None
                }
            })
            .collect();

        let azimuth: Vec<f64> = paired_y
            .iter()
            .filter_map(|&z| {
                let phi_y = -z.arg() / (2.0 * PI * self.dy);
                if phi_y.abs() <= 1.0 {
                    Some(phi_y.asin())
                } else {
                    None
                }
            })
            .collect();

        Ok(ESPRIT2DResult {
            elevation,
            azimuth,
            phase_shifts_x: phase_x,
            phase_shifts_y: phase_y,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helper functions
// ---------------------------------------------------------------------------

/// Compute Gram matrix `A^H A` where `A` is `(m × d)` stored row-major
pub(crate) fn complex_gram(a: &[Complex64], m: usize, d: usize) -> Vec<Complex64> {
    let mut g = vec![Complex64::new(0.0, 0.0); d * d];
    for i in 0..d {
        for j in 0..d {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..m {
                sum = sum + a[k * d + i].conj() * a[k * d + j];
            }
            g[i * d + j] = sum;
        }
    }
    g
}

/// Compute cross-Gram matrix `A^H B` where `A`, `B` are `(m × d)` row-major
pub(crate) fn complex_cross_gram(
    a: &[Complex64],
    b: &[Complex64],
    m: usize,
    d: usize,
) -> Vec<Complex64> {
    let mut g = vec![Complex64::new(0.0, 0.0); d * d];
    for i in 0..d {
        for j in 0..d {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..m {
                sum = sum + a[k * d + i].conj() * b[k * d + j];
            }
            g[i * d + j] = sum;
        }
    }
    g
}

/// Solve `AX = B` for `X` where `A` is `d×d` complex, `B` is `d×d` complex.
/// Uses LU decomposition with partial pivoting.
pub(crate) fn complex_solve_dd(
    a: &[Complex64],
    b: &[Complex64],
    d: usize,
) -> SignalResult<Vec<Complex64>> {
    let inv = complex_matrix_inv(a, d)?;
    Ok(complex_matmul_dd(&inv, b, d))
}

/// Complex d×d matrix multiplication (row-major)
pub(crate) fn complex_matmul_dd(a: &[Complex64], b: &[Complex64], d: usize) -> Vec<Complex64> {
    let mut c = vec![Complex64::new(0.0, 0.0); d * d];
    for i in 0..d {
        for j in 0..d {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..d {
                sum = sum + a[i * d + k] * b[k * d + j];
            }
            c[i * d + j] = sum;
        }
    }
    c
}

/// Complex matrix inversion via LU decomposition
pub(crate) fn complex_matrix_inv(a: &[Complex64], n: usize) -> SignalResult<Vec<Complex64>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    // Augmented matrix [A | I]
    let mut lu = a.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();

    // LU factorization with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_val = 0.0f64;
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
        // Swap rows k and max_row
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
                lu[i * n + j] = lu[i * n + j] - sub;
            }
        }
    }

    // Solve for inverse column by column
    let mut inv = vec![Complex64::new(0.0, 0.0); n * n];
    for col in 0..n {
        // RHS: e_perm[col] (permuted canonical basis)
        let mut rhs = vec![Complex64::new(0.0, 0.0); n];
        // Find where canonical basis vector e_col maps to under perm
        for i in 0..n {
            if perm[i] == col {
                rhs[i] = Complex64::new(1.0, 0.0);
                break;
            }
        }
        // Forward substitution L y = rhs
        let mut y = rhs;
        for i in 0..n {
            for j in 0..i {
                let l = lu[i * n + j];
                let yj = y[j];
                y[i] = y[i] - l * yj;
            }
        }
        // Backward substitution U x = y
        let mut x = y;
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                let u = lu[i * n + j];
                let xj = x[j];
                x[i] = x[i] - u * xj;
            }
            let u_ii = lu[i * n + i];
            x[i] = x[i] / u_ii;
        }
        for i in 0..n {
            inv[i * n + col] = x[i];
        }
    }

    Ok(inv)
}

/// Compute right singular vectors of `A` (m × n_cols), returning `V` as n_cols × n_cols
/// using the SVD through `A^H A`.
pub(crate) fn compute_right_singular_vectors(
    a: &[Complex64],
    m: usize,
    n_cols: usize,
) -> SignalResult<Vec<Complex64>> {
    // A^H A is n_cols × n_cols Hermitian; its eigenvectors are right singular vectors
    let mut aha = vec![Complex64::new(0.0, 0.0); n_cols * n_cols];
    for i in 0..n_cols {
        for j in 0..n_cols {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..m {
                sum = sum + a[k * n_cols + i].conj() * a[k * n_cols + j];
            }
            aha[i * n_cols + j] = sum;
        }
    }
    let (_, evecs) = hermitian_eig(&aha, n_cols)?;
    // evecs[k] is the k-th eigenvector (column k of V), sorted by *descending* eigenvalue
    // We need the full V matrix (n_cols × n_cols)
    let mut v = vec![Complex64::new(0.0, 0.0); n_cols * n_cols];
    for col in 0..n_cols {
        for row in 0..n_cols {
            v[row * n_cols + col] = evecs[col][row];
        }
    }
    Ok(v)
}

/// Compute eigenvalues of a general complex matrix `A` (d×d) via the QR algorithm.
pub(crate) fn eigenvalues_complex_matrix(
    a: &[Complex64],
    d: usize,
) -> SignalResult<Vec<Complex64>> {
    if d == 0 {
        return Ok(Vec::new());
    }
    let mut h = a.to_vec();
    let max_iter = 300 * d;
    let eps = 1e-10;

    for _ in 0..max_iter {
        // Check convergence: look at sub-diagonal magnitude
        let mut max_off = 0.0f64;
        for i in 1..d {
            max_off = max_off.max(h[i * d + (i - 1)].norm());
        }
        if max_off < eps {
            break;
        }
        // Wilkinson shift: use eigenvalue of bottom-right 2×2
        let shift = if d >= 2 {
            let a22 = h[(d - 1) * d + (d - 1)];
            a22 // simple shift
        } else {
            Complex64::new(0.0, 0.0)
        };
        // Shift A
        let mut hs = h.clone();
        for i in 0..d {
            hs[i * d + i] = hs[i * d + i] - shift;
        }
        // QR decomposition
        let (q, r) = complex_qr_thin(&hs, d)?;
        // H = R Q + shift I
        h = complex_matmul_dd(&r, &q, d);
        for i in 0..d {
            h[i * d + i] = h[i * d + i] + shift;
        }
    }
    // Extract diagonal
    let eigs: Vec<Complex64> = (0..d).map(|i| h[i * d + i]).collect();
    Ok(eigs)
}

/// Thin complex QR decomposition (Gram-Schmidt) for d×d matrix
fn complex_qr_thin(
    a: &[Complex64],
    d: usize,
) -> SignalResult<(Vec<Complex64>, Vec<Complex64>)> {
    let mut q = vec![Complex64::new(0.0, 0.0); d * d];
    let mut r = vec![Complex64::new(0.0, 0.0); d * d];

    for j in 0..d {
        let mut v: Vec<Complex64> = (0..d).map(|i| a[i * d + j]).collect();
        for k in 0..j {
            let q_col: Vec<Complex64> = (0..d).map(|i| q[i * d + k]).collect();
            let dot: Complex64 = q_col.iter().zip(v.iter()).map(|(qi, vi)| qi.conj() * vi).fold(Complex64::new(0.0, 0.0), |acc, x| acc + x);
            r[k * d + j] = dot;
            for i in 0..d {
                v[i] = v[i] - dot * q_col[i];
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

/// Automatic pairing of 2D ESPRIT estimates using shared invariance.
///
/// Constructs the joint matrix `T^{-1} Ψx T` and `T^{-1} Ψy T` using the shared
/// eigenvector matrix `T`. Returns paired phase shifts `(paired_x, paired_y)`.
fn pair_2d_estimates(
    psi_x: &[Complex64],
    psi_y: &[Complex64],
    d: usize,
) -> SignalResult<(Vec<Complex64>, Vec<Complex64>)> {
    // Find eigenvectors of Ψx — these diagonalise it
    // Then apply same eigenvector matrix to diagonalise Ψy → this gives pairing
    // T^{-1} Ψy T is (approximately) diagonal; its diagonal elements are paired with Ψx's eigenvalues

    // Get eigenvectors of Ψx
    let t = compute_eigenvectors_complex(psi_x, d)?;
    let t_inv = complex_matrix_inv(&t, d)?;

    // Apply transform to Ψy: diag_y ≈ T^{-1} Ψy T
    let psi_y_t = complex_matmul_dd(psi_y, &t, d);
    let diag_y_mat = complex_matmul_dd(&t_inv, &psi_y_t, d);

    // Eigenvalues of Ψx
    let eigs_x = eigenvalues_complex_matrix(psi_x, d)?;
    // Diagonal of diag_y_mat (approximately paired)
    let diag_y: Vec<Complex64> = (0..d).map(|i| diag_y_mat[i * d + i]).collect();

    Ok((eigs_x, diag_y))
}

/// Compute eigenvectors of a complex matrix (used for pairing)
fn compute_eigenvectors_complex(a: &[Complex64], d: usize) -> SignalResult<Vec<Complex64>> {
    // Power iteration / QR to get eigenvector matrix
    // We use shifted inverse iteration for each eigenvalue
    let eigs = eigenvalues_complex_matrix(a, d)?;
    let mut v_cols = Vec::with_capacity(d);

    for &lambda in &eigs {
        // Shifted inverse iteration: (A - λI)^{-1} v → eigenvector
        let mut shifted = a.to_vec();
        for i in 0..d {
            shifted[i * d + i] = shifted[i * d + i] - lambda - Complex64::new(1e-8, 1e-8);
        }
        let inv = complex_matrix_inv(&shifted, d).unwrap_or_else(|_| {
            let mut eye = vec![Complex64::new(0.0, 0.0); d * d];
            for i in 0..d {
                eye[i * d + i] = Complex64::new(1.0, 0.0);
            }
            eye
        });
        // Start vector = first canonical basis
        let mut v: Vec<Complex64> = (0..d).map(|i| if i == 0 { Complex64::new(1.0, 0.0) } else { Complex64::new(0.0, 0.0) }).collect();
        for _ in 0..10 {
            let mut new_v = vec![Complex64::new(0.0, 0.0); d];
            for i in 0..d {
                for j in 0..d {
                    new_v[i] = new_v[i] + inv[i * d + j] * v[j];
                }
            }
            let norm = new_v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            if norm > 1e-14 {
                for x in new_v.iter_mut() {
                    *x = *x / norm;
                }
            }
            v = new_v;
        }
        v_cols.push(v);
    }

    // Build V as columns
    let mut v_mat = vec![Complex64::new(0.0, 0.0); d * d];
    for (col, v_col) in v_cols.iter().enumerate() {
        for row in 0..d {
            v_mat[row * d + col] = v_col[row];
        }
    }
    Ok(v_mat)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subspace::music::ula_steering_vector;

    fn make_ula_snapshot(
        n_elements: usize,
        d: f64,
        theta_rad: f64,
        n_snapshots: usize,
        snr: f64,
    ) -> Vec<Vec<Complex64>> {
        let sv = ula_steering_vector(n_elements, theta_rad, d);
        let mut rng: u64 = 12345;
        let noise_std = (1.0 / snr).sqrt();
        (0..n_elements)
            .map(|m| {
                (0..n_snapshots)
                    .map(|_| {
                        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        let nr = ((rng >> 33) as f64 / u32::MAX as f64 - 0.5) * 2.0 * noise_std;
                        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        let ni = ((rng >> 33) as f64 / u32::MAX as f64 - 0.5) * 2.0 * noise_std;
                        sv[m] + Complex64::new(nr, ni)
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_esprit1d_single_source() {
        let n_el = 8;
        let theta = 20.0f64.to_radians();
        let data = make_ula_snapshot(n_el, 0.5, theta, 300, 30.0);
        let esprit = ESPRIT1D::new(n_el, 1, 0.5).expect("esprit");
        let result = esprit.estimate(&data).expect("result");
        assert!(!result.doa_estimates.is_empty(), "Should have a DOA estimate");
        let est_deg = result.doa_estimates[0].to_degrees();
        let true_deg = theta.to_degrees();
        assert!(
            (est_deg - true_deg).abs() < 8.0,
            "ESPRIT DOA error too large: est={est_deg:.2}°, true={true_deg:.2}°"
        );
    }

    #[test]
    fn test_tls_esprit_single_source() {
        let n_el = 8;
        let theta = -15.0f64.to_radians();
        let data = make_ula_snapshot(n_el, 0.5, theta, 300, 30.0);
        let tls = TlsEsprit::new(n_el, 1, 0.5).expect("tls esprit");
        let result = tls.estimate(&data).expect("result");
        assert!(!result.doa_estimates.is_empty());
        let est_deg = result.doa_estimates[0].to_degrees();
        let true_deg = theta.to_degrees();
        assert!(
            (est_deg - true_deg).abs() < 8.0,
            "TLS-ESPRIT DOA error: est={est_deg:.2}°, true={true_deg:.2}°"
        );
    }

    #[test]
    fn test_complex_gram() {
        // Test Gram matrix computation: A = [[1+j, 0], [0, 1]], A^H A = [[2, 0], [0, 1]]
        let a = vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let g = complex_gram(&a, 2, 2);
        assert!((g[0].re - 2.0).abs() < 1e-10, "g[0,0]={}", g[0].re);
        assert!((g[3].re - 1.0).abs() < 1e-10, "g[1,1]={}", g[3].re);
    }

    #[test]
    fn test_matrix_inv_2x2() {
        // [[2, 1], [1, 1]]^{-1} = [[1, -1], [-1, 2]]
        let a = vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let inv = complex_matrix_inv(&a, 2).expect("inv");
        assert!((inv[0].re - 1.0).abs() < 1e-10, "inv[0,0]={}", inv[0].re);
        assert!((inv[1].re + 1.0).abs() < 1e-10, "inv[0,1]={}", inv[1].re);
        assert!((inv[2].re + 1.0).abs() < 1e-10, "inv[1,0]={}", inv[2].re);
        assert!((inv[3].re - 2.0).abs() < 1e-10, "inv[1,1]={}", inv[3].re);
    }
}
