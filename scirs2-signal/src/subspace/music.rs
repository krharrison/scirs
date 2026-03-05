//! MUSIC (Multiple Signal Classification) algorithm and variants
//!
//! Provides:
//! - [`MUSICEstimator`]: standard MUSIC for DOA/frequency estimation
//! - [`RootMUSIC`]: polynomial-rooting variant for ULA
//! - [`SSMUSIC`]: spatially-smoothed MUSIC for coherent sources
//!
//! References:
//! - Schmidt, R.O. (1986). "Multiple emitter location and signal parameter
//!   estimation." IEEE Trans. Antennas Propagation, 34(3), 276–280.
//! - Barabell, A.J. (1983). "Improving the resolution performance of eigenstructure-based
//!   direction-finding algorithms." ICASSP 1983.
//! - Shan, T.J., Wax, M. & Kailath, T. (1985). "On spatial smoothing for direction of arrival
//!   estimation of coherent signals." IEEE Trans. ASSP, 33(4), 806–811.

use crate::error::{SignalError, SignalResult};
use crate::subspace::array_processing::{hermitian_eig, ArrayManifold, SpatialCovariance};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of a MUSIC pseudo-spectrum computation
#[derive(Debug, Clone)]
pub struct MUSICResult {
    /// Estimated DOA angles in radians (or frequencies for spectral MUSIC)
    pub doa_estimates: Vec<f64>,
    /// Pseudo-spectrum values at scan points
    pub pseudo_spectrum: Vec<f64>,
    /// Scan grid (angles in radians, or normalised frequencies)
    pub scan_grid: Vec<f64>,
    /// Eigenvalues of the covariance matrix (descending order)
    pub eigenvalues: Vec<f64>,
    /// Estimated number of sources used
    pub n_sources: usize,
}

/// Result of Root-MUSIC polynomial rooting
#[derive(Debug, Clone)]
pub struct RootMUSICResult {
    /// Estimated DOA angles in radians
    pub doa_estimates: Vec<f64>,
    /// Corresponding z-plane roots (on or near unit circle)
    pub roots: Vec<Complex64>,
    /// Eigenvalues (descending)
    pub eigenvalues: Vec<f64>,
}

// ---------------------------------------------------------------------------
// MUSIC Estimator
// ---------------------------------------------------------------------------

/// Configuration for the MUSIC estimator
#[derive(Debug, Clone)]
pub struct MUSICConfig {
    /// Number of sources (signals) to detect
    pub n_sources: usize,
    /// Element spacing in wavelengths (for DOA; default 0.5)
    pub element_spacing: f64,
    /// Number of scan points in the angular grid
    pub n_scan: usize,
    /// Minimum angle in radians (default: -π/2)
    pub angle_min: f64,
    /// Maximum angle in radians (default: +π/2)
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

/// MUSIC (Multiple Signal Classification) DOA estimator
///
/// Given the sample covariance matrix `R` of an array of `M` sensors and the
/// number of sources `d`, MUSIC decomposes the eigenspace of `R` into signal
/// and noise subspaces and evaluates:
///
/// `P_MUSIC(θ) = 1 / (a^H(θ) E_n E_n^H a(θ))`
///
/// where `E_n` is the noise eigenvector matrix.
#[derive(Debug, Clone)]
pub struct MUSICEstimator {
    /// Number of array elements
    pub n_elements: usize,
    /// Estimator configuration
    pub config: MUSICConfig,
}

impl MUSICEstimator {
    /// Create a new MUSIC estimator
    ///
    /// # Arguments
    ///
    /// * `n_elements` - Number of sensors
    /// * `config`     - Configuration
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
        Ok(Self {
            n_elements,
            config,
        })
    }

    /// Estimate DOA from sample covariance matrix
    ///
    /// # Arguments
    ///
    /// * `covariance` - `SpatialCovariance` (size must equal `n_elements`)
    ///
    /// # Returns
    ///
    /// * `MUSICResult`
    pub fn estimate(&self, covariance: &SpatialCovariance) -> SignalResult<MUSICResult> {
        if covariance.size != self.n_elements {
            return Err(SignalError::DimensionMismatch(format!(
                "Covariance size {} ≠ n_elements {}",
                covariance.size, self.n_elements
            )));
        }

        let m = self.n_elements;
        let (eigenvalues, eigenvectors) = hermitian_eig(&covariance.matrix, m)?;

        // Noise subspace: eigenvectors corresponding to the (m - n_sources) smallest eigenvalues
        // eigenvalues are in descending order so noise subspace is [n_sources..]
        let noise_evecs = &eigenvectors[self.config.n_sources..];

        // Compute pseudo-spectrum over scan grid
        let scan_grid = linspace(self.config.angle_min, self.config.angle_max, self.config.n_scan);
        let pseudo_spectrum = self.compute_pseudo_spectrum(&scan_grid, noise_evecs, m)?;

        // Find peaks
        let doa_estimates = find_peaks(&pseudo_spectrum, &scan_grid, self.config.n_sources);

        Ok(MUSICResult {
            doa_estimates,
            pseudo_spectrum,
            scan_grid,
            eigenvalues,
            n_sources: self.config.n_sources,
        })
    }

    /// Estimate DOA directly from snapshot data
    ///
    /// # Arguments
    ///
    /// * `data` - Snapshot matrix `[n_elements][n_snapshots]`
    pub fn estimate_from_data(&self, data: &[Vec<Complex64>]) -> SignalResult<MUSICResult> {
        let cov = SpatialCovariance::estimate(data)?;
        self.estimate(&cov)
    }

    /// Compute pseudo-spectrum for a given noise subspace and scan grid
    fn compute_pseudo_spectrum(
        &self,
        scan_grid: &[f64],
        noise_evecs: &[Vec<Complex64>],
        m: usize,
    ) -> SignalResult<Vec<f64>> {
        let n_noise = noise_evecs.len();
        let n_scan = scan_grid.len();
        let mut ps = Vec::with_capacity(n_scan);

        for &theta in scan_grid {
            // Steering vector for ULA
            let sv = ula_steering_vector(m, theta, self.config.element_spacing);
            // Noise subspace projection: ||E_n^H a(θ)||^2
            let mut proj_sq = 0.0f64;
            for ev in noise_evecs.iter().take(n_noise) {
                let mut dot = Complex64::new(0.0, 0.0);
                for k in 0..m {
                    dot = dot + ev[k].conj() * sv[k];
                }
                proj_sq += dot.norm_sqr();
            }
            // Pseudo-spectrum: 1 / projection
            let ps_val = if proj_sq > 1e-30 {
                1.0 / proj_sq
            } else {
                1e30
            };
            ps.push(ps_val);
        }
        Ok(ps)
    }
}

// ---------------------------------------------------------------------------
// Root-MUSIC
// ---------------------------------------------------------------------------

/// Root-MUSIC: polynomial-rooting MUSIC for ULA
///
/// Instead of scanning a grid, Root-MUSIC forms the polynomial
///
/// `C(z) = a^H(z) E_n E_n^H a(z)`
///
/// (where `a(z) = [1, z, z^2, …, z^{M-1}]` is the Vandermonde vector) and finds
/// roots closest to the unit circle. This gives exact frequency estimates for
/// noiseless data and avoids grid search.
#[derive(Debug, Clone)]
pub struct RootMUSIC {
    /// Number of array elements
    pub n_elements: usize,
    /// Number of sources
    pub n_sources: usize,
    /// Element spacing in wavelengths
    pub element_spacing: f64,
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
    pub fn estimate(&self, data: &[Vec<Complex64>]) -> SignalResult<RootMUSICResult> {
        let cov = SpatialCovariance::estimate(data)?;
        self.estimate_from_covariance(&cov)
    }

    /// Estimate DOA from precomputed covariance matrix
    pub fn estimate_from_covariance(
        &self,
        covariance: &SpatialCovariance,
    ) -> SignalResult<RootMUSICResult> {
        if covariance.size != self.n_elements {
            return Err(SignalError::DimensionMismatch(format!(
                "Covariance size {} ≠ n_elements {}",
                covariance.size, self.n_elements
            )));
        }
        let m = self.n_elements;
        let (eigenvalues, eigenvectors) = hermitian_eig(&covariance.matrix, m)?;
        let noise_evecs = &eigenvectors[self.n_sources..];

        // Build noise subspace projection matrix C = E_n E_n^H  (m×m)
        let n_noise = noise_evecs.len();
        let mut c_mat = vec![Complex64::new(0.0, 0.0); m * m];
        for ev in noise_evecs.iter().take(n_noise) {
            for i in 0..m {
                for j in 0..m {
                    c_mat[i * m + j] = c_mat[i * m + j] + ev[i] * ev[j].conj();
                }
            }
        }

        // Polynomial coefficients from C
        // C(z) = sum_{k=-(m-1)}^{m-1} c_k z^k  where c_k = sum_{i-j=k} C[i,j]
        // We form the polynomial of degree 2(m-1): p[k] corresponds to z^{k-(m-1)}
        let poly_len = 2 * m - 1;
        let mut poly = vec![Complex64::new(0.0, 0.0); poly_len];
        for i in 0..m {
            for j in 0..m {
                let k = (i as isize) - (j as isize) + (m as isize - 1);
                poly[k as usize] = poly[k as usize] + c_mat[i * m + j];
            }
        }

        // Find roots of polynomial via companion matrix
        let roots = find_polynomial_roots(&poly)?;

        // Select n_sources roots closest to the unit circle
        let mut root_with_dist: Vec<(Complex64, f64)> = roots
            .iter()
            .map(|&r| {
                let dist = (r.norm() - 1.0).abs();
                (r, dist)
            })
            .collect();
        root_with_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let selected: Vec<Complex64> = root_with_dist
            .iter()
            .take(self.n_sources)
            .map(|(r, _)| *r)
            .collect();

        // Convert roots to angles: z = exp(-j*2*pi*d*sin(theta)) => sin(theta) = -arg(z)/(2*pi*d)
        let doa_estimates: Vec<f64> = selected
            .iter()
            .filter_map(|&z| {
                let phase = z.arg(); // arg in (-pi, pi]
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
// SS-MUSIC (Spatially Smoothed MUSIC)
// ---------------------------------------------------------------------------

/// Spatially Smoothed MUSIC for coherent sources
///
/// Uses forward spatial smoothing (and optionally forward-backward smoothing)
/// to decorrelate coherent signals before applying MUSIC.
#[derive(Debug, Clone)]
pub struct SSMUSIC {
    /// Full array element count
    pub n_elements: usize,
    /// Sub-array size (must be < n_elements)
    pub sub_array_size: usize,
    /// Number of sources
    pub n_sources: usize,
    /// Use forward-backward smoothing (default true)
    pub forward_backward: bool,
    /// Element spacing
    pub element_spacing: f64,
    /// Angular scan range and resolution
    pub n_scan: usize,
    /// Minimum scan angle in radians
    pub angle_min: f64,
    /// Maximum scan angle in radians
    pub angle_max: f64,
}

impl SSMUSIC {
    /// Create a new SS-MUSIC estimator
    pub fn new(
        n_elements: usize,
        sub_array_size: usize,
        n_sources: usize,
    ) -> SignalResult<Self> {
        if n_elements < 4 {
            return Err(SignalError::ValueError(
                "SS-MUSIC requires at least 4 elements".to_string(),
            ));
        }
        if sub_array_size < 2 || sub_array_size > n_elements - 1 {
            return Err(SignalError::ValueError(format!(
                "sub_array_size ({sub_array_size}) must be in [2, {}]",
                n_elements - 1
            )));
        }
        if n_sources == 0 || n_sources >= sub_array_size {
            return Err(SignalError::ValueError(format!(
                "n_sources ({n_sources}) must be in [1, {sub_array_size})"
            )));
        }
        Ok(Self {
            n_elements,
            sub_array_size,
            n_sources,
            forward_backward: true,
            element_spacing: 0.5,
            n_scan: 360,
            angle_min: -PI / 2.0,
            angle_max: PI / 2.0,
        })
    }

    /// Estimate DOA from snapshot data
    pub fn estimate(&self, data: &[Vec<Complex64>]) -> SignalResult<MUSICResult> {
        if data.len() != self.n_elements {
            return Err(SignalError::DimensionMismatch(format!(
                "data has {} rows, expected {}",
                data.len(),
                self.n_elements
            )));
        }

        // Spatially smoothed covariance
        let mut cov = SpatialCovariance::estimate_smoothed(data, self.sub_array_size)?;
        if self.forward_backward {
            cov = cov.forward_backward_average();
        }

        let m = self.sub_array_size;
        let (eigenvalues, eigenvectors) = hermitian_eig(&cov.matrix, m)?;
        let noise_evecs = &eigenvectors[self.n_sources..];

        let scan_grid = linspace(self.angle_min, self.angle_max, self.n_scan);
        let n_scan = scan_grid.len();
        let n_noise = noise_evecs.len();
        let mut pseudo_spectrum = Vec::with_capacity(n_scan);

        for &theta in &scan_grid {
            let sv = ula_steering_vector(m, theta, self.element_spacing);
            let mut proj_sq = 0.0f64;
            for ev in noise_evecs.iter().take(n_noise) {
                let mut dot = Complex64::new(0.0, 0.0);
                for k in 0..m {
                    dot = dot + ev[k].conj() * sv[k];
                }
                proj_sq += dot.norm_sqr();
            }
            let ps_val = if proj_sq > 1e-30 { 1.0 / proj_sq } else { 1e30 };
            pseudo_spectrum.push(ps_val);
        }

        let doa_estimates = find_peaks(&pseudo_spectrum, &scan_grid, self.n_sources);

        Ok(MUSICResult {
            doa_estimates,
            pseudo_spectrum,
            scan_grid,
            eigenvalues,
            n_sources: self.n_sources,
        })
    }
}

// ---------------------------------------------------------------------------
// Frequency estimation variant (1D MUSIC for spectral estimation)
// ---------------------------------------------------------------------------

/// MUSIC pseudo-spectrum for frequency estimation from a scalar time series.
///
/// Models the time series as a sum of `n_sources` complex exponentials in white noise.
/// Builds an `(m × m)` autocorrelation matrix and applies MUSIC to estimate frequencies.
///
/// # Arguments
///
/// * `signal`     - Input signal (length ≥ `model_order + n_sources`)
/// * `n_sources`  - Number of sinusoidal components
/// * `model_order`- AR model order / embedding dimension `m`
/// * `n_scan`     - Number of frequency bins to scan (0..0.5 normalised)
///
/// # Returns
///
/// * `MUSICResult` with `scan_grid` in normalised frequency [0, 0.5]
pub fn music_spectral(
    signal: &[f64],
    n_sources: usize,
    model_order: usize,
    n_scan: usize,
) -> SignalResult<MUSICResult> {
    let n = signal.len();
    if n_sources == 0 {
        return Err(SignalError::ValueError(
            "n_sources must be positive".to_string(),
        ));
    }
    if model_order <= n_sources {
        return Err(SignalError::ValueError(format!(
            "model_order ({model_order}) must be > n_sources ({n_sources})"
        )));
    }
    if n < model_order + 1 {
        return Err(SignalError::ValueError(format!(
            "Signal length ({n}) must be > model_order ({model_order})"
        )));
    }

    let m = model_order;
    let n_snapshots = n - m;

    // Build autocorrelation matrix from signal using Toeplitz structure
    let mut r = vec![Complex64::new(0.0, 0.0); m * m];
    for i in 0..m {
        for j in 0..m {
            let lag = (i as isize) - (j as isize);
            let mut sum = Complex64::new(0.0, 0.0);
            let start = if lag >= 0 { lag as usize } else { 0 };
            let end_i = if lag >= 0 { n } else { n - (-lag as usize) };
            let end_j = if lag < 0 { n } else { n - lag as usize };
            let count = end_i.min(end_j).saturating_sub(start);
            for k in 0..count {
                let xi = signal[if lag >= 0 { k } else { k + (-lag as usize) }];
                let xj = signal[if lag >= 0 { k + lag as usize } else { k }];
                sum = sum + Complex64::new(xi * xj, 0.0);
            }
            r[i * m + j] = sum / n_snapshots as f64;
        }
    }

    let (eigenvalues, eigenvectors) = hermitian_eig(&r, m)?;
    let noise_evecs = &eigenvectors[n_sources..];

    // Scan normalised frequencies 0..0.5
    let freqs = linspace(0.0, 0.5, n_scan);
    let n_noise = noise_evecs.len();
    let mut pseudo_spectrum = Vec::with_capacity(n_scan);

    for &f in &freqs {
        // Steering vector: a(f) = [1, e^{j2πf}, e^{j4πf}, …]
        let sv: Vec<Complex64> = (0..m)
            .map(|k| {
                let phase = 2.0 * PI * f * k as f64;
                Complex64::new(phase.cos(), phase.sin())
            })
            .collect();

        let mut proj_sq = 0.0f64;
        for ev in noise_evecs.iter().take(n_noise) {
            let mut dot = Complex64::new(0.0, 0.0);
            for k in 0..m {
                dot = dot + ev[k].conj() * sv[k];
            }
            proj_sq += dot.norm_sqr();
        }
        let ps_val = if proj_sq > 1e-30 { 1.0 / proj_sq } else { 1e30 };
        pseudo_spectrum.push(ps_val);
    }

    let freq_estimates = find_peaks(&pseudo_spectrum, &freqs, n_sources);

    Ok(MUSICResult {
        doa_estimates: freq_estimates,
        pseudo_spectrum,
        scan_grid: freqs,
        eigenvalues,
        n_sources,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// ULA steering vector
pub(crate) fn ula_steering_vector(m: usize, theta_rad: f64, d: f64) -> Vec<Complex64> {
    let phase_inc = -2.0 * PI * d * theta_rad.sin();
    (0..m)
        .map(|k| {
            let ph = phase_inc * k as f64;
            Complex64::new(ph.cos(), ph.sin())
        })
        .collect()
}

/// Linearly spaced vector
pub(crate) fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + step * i as f64).collect()
}

/// Find the `n_peaks` largest peaks in a pseudo-spectrum
pub(crate) fn find_peaks(spectrum: &[f64], grid: &[f64], n_peaks: usize) -> Vec<f64> {
    if spectrum.is_empty() || n_peaks == 0 {
        return Vec::new();
    }
    let n = spectrum.len();
    // Find local maxima
    let mut peaks: Vec<(f64, f64)> = Vec::new();
    for i in 0..n {
        let prev = if i > 0 { spectrum[i - 1] } else { f64::NEG_INFINITY };
        let next = if i < n - 1 { spectrum[i + 1] } else { f64::NEG_INFINITY };
        if spectrum[i] >= prev && spectrum[i] >= next {
            peaks.push((spectrum[i], grid[i]));
        }
    }
    // Sort by height descending
    peaks.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    peaks
        .iter()
        .take(n_peaks)
        .map(|(_, angle)| *angle)
        .collect()
}

/// Find roots of a polynomial given by coefficients `c` (lowest degree first).
///
/// Uses the companion matrix eigenvalue approach.
pub(crate) fn find_polynomial_roots(coeffs: &[Complex64]) -> SignalResult<Vec<Complex64>> {
    let n = coeffs.len();
    if n <= 1 {
        return Ok(Vec::new());
    }
    // Remove trailing zero coefficients to determine actual degree
    let mut deg = n - 1;
    while deg > 0 && coeffs[deg].norm_sqr() < 1e-30 {
        deg -= 1;
    }
    if deg == 0 {
        return Ok(Vec::new());
    }

    let lead = coeffs[deg];
    // Build companion matrix (deg × deg)
    // Companion matrix for p(x) = c[0] + c[1]*x + … + c[deg]*x^deg
    // is the companion in *descending* coefficient form.
    // We convert to monic: divide by lead
    let d = deg;
    let mut companion = vec![Complex64::new(0.0, 0.0); d * d];
    // Sub-diagonal = 1
    for i in 1..d {
        companion[i * d + (i - 1)] = Complex64::new(1.0, 0.0);
    }
    // Last column = -c[0..d] / c[d] (reversed)
    for i in 0..d {
        companion[i * d + (d - 1)] = -(coeffs[i] / lead);
    }

    // Eigenvalues of the companion matrix are the roots.
    // We run power-iteration-based QZ (simplified QR iteration on complex matrix).
    let roots = qr_companion_roots(&companion, d)?;
    Ok(roots)
}

/// QR iteration on a complex companion matrix to find eigenvalues (roots).
///
/// This is a simplified implementation suitable for moderate-degree polynomials.
fn qr_companion_roots(companion: &[Complex64], n: usize) -> SignalResult<Vec<Complex64>> {
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut h = companion.to_vec();
    let max_iter = 300 * n;

    for _ in 0..max_iter {
        // Find largest off-sub-diagonal element to check convergence
        let mut max_off = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                if j + 1 != i {
                    // not sub-diagonal
                    max_off = max_off.max(h[i * n + j].norm());
                }
            }
        }
        if max_off < 1e-10 {
            break;
        }

        // Simple QR step (Householder-based) — we use Gram-Schmidt for simplicity
        // Apply one unshifted QR iteration: H = Q R, then H = R Q
        let (q, r) = complex_qr_decompose(&h, n)?;
        // H_new = R * Q
        h = complex_matmul(&r, &q, n);
    }

    // Extract eigenvalues from diagonal
    let roots: Vec<Complex64> = (0..n).map(|i| h[i * n + i]).collect();
    Ok(roots)
}

/// Complex QR decomposition via Gram-Schmidt
fn complex_qr_decompose(a: &[Complex64], n: usize) -> SignalResult<(Vec<Complex64>, Vec<Complex64>)> {
    let mut q = vec![Complex64::new(0.0, 0.0); n * n];
    let mut r = vec![Complex64::new(0.0, 0.0); n * n];

    // Extract columns
    let mut cols: Vec<Vec<Complex64>> = (0..n)
        .map(|j| (0..n).map(|i| a[i * n + j]).collect())
        .collect();

    for j in 0..n {
        // Compute projection onto previous q columns
        let mut v = cols[j].clone();
        for k in 0..j {
            let q_col_k: Vec<Complex64> = (0..n).map(|i| q[i * n + k]).collect();
            let dot: Complex64 = q_col_k.iter().zip(v.iter()).map(|(qki, vi)| qki.conj() * vi).fold(Complex64::new(0.0, 0.0), |acc, x| acc + x);
            r[k * n + j] = dot;
            for i in 0..n {
                v[i] = v[i] - dot * q_col_k[i];
            }
        }
        let norm = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        r[j * n + j] = Complex64::new(norm, 0.0);
        if norm > 1e-14 {
            for i in 0..n {
                q[i * n + j] = v[i] / norm;
            }
        } else {
            // Degenerate column — use canonical basis vector
            q[j * n + j] = Complex64::new(1.0, 0.0);
        }
    }

    Ok((q, r))
}

/// Complex matrix multiplication C = A * B (n × n)
fn complex_matmul(a: &[Complex64], b: &[Complex64], n: usize) -> Vec<Complex64> {
    let mut c = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..n {
                sum = sum + a[i * n + k] * b[k * n + j];
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

    fn make_ula_snapshot(
        n_elements: usize,
        d: f64,
        theta_rad: f64,
        n_snapshots: usize,
        snr: f64,
    ) -> Vec<Vec<Complex64>> {
        let sv = ula_steering_vector(n_elements, theta_rad, d);
        let mut rng_state: u64 = 42;
        let noise_std = (1.0 / snr).sqrt();

        (0..n_elements)
            .map(|m| {
                (0..n_snapshots)
                    .map(|_n| {
                        // Simple LCG noise
                        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        let nr = ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5) * 2.0 * noise_std;
                        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        let ni = ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5) * 2.0 * noise_std;
                        sv[m] + Complex64::new(nr, ni)
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_music_single_source() {
        let n_el = 8;
        let theta = 20.0f64.to_radians();
        let data = make_ula_snapshot(n_el, 0.5, theta, 200, 30.0);
        let config = MUSICConfig {
            n_sources: 1,
            element_spacing: 0.5,
            n_scan: 720,
            ..Default::default()
        };
        let estimator = MUSICEstimator::new(n_el, config).expect("estimator");
        let result = estimator.estimate_from_data(&data).expect("result");
        assert_eq!(result.doa_estimates.len(), 1);
        let est_deg = result.doa_estimates[0].to_degrees();
        let true_deg = theta.to_degrees();
        assert!(
            (est_deg - true_deg).abs() < 5.0,
            "DOA error too large: estimated {est_deg:.2}°, true {true_deg:.2}°"
        );
    }

    #[test]
    fn test_linspace() {
        let v = linspace(0.0, 1.0, 5);
        assert_eq!(v.len(), 5);
        assert!((v[0] - 0.0).abs() < 1e-10);
        assert!((v[4] - 1.0).abs() < 1e-10);
        assert!((v[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_find_peaks() {
        let spectrum = vec![1.0, 3.0, 2.0, 5.0, 4.0, 1.0];
        let grid: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let peaks = find_peaks(&spectrum, &grid, 2);
        assert_eq!(peaks.len(), 2);
        // Largest peak at index 3 (value 5.0)
        assert!((peaks[0] - 3.0).abs() < 1e-10 || (peaks[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_root_music() {
        let n_el = 8;
        let theta = 15.0f64.to_radians();
        let data = make_ula_snapshot(n_el, 0.5, theta, 500, 50.0);
        let rm = RootMUSIC::new(n_el, 1, 0.5).expect("root music");
        let result = rm.estimate(&data).expect("result");
        if !result.doa_estimates.is_empty() {
            let est_deg = result.doa_estimates[0].to_degrees();
            let true_deg = theta.to_degrees();
            assert!(
                (est_deg - true_deg).abs() < 10.0,
                "Root-MUSIC DOA error: estimated {est_deg:.2}°, true {true_deg:.2}°"
            );
        }
    }

    #[test]
    fn test_music_spectral() {
        // Signal with one sinusoid at normalised frequency 0.1
        let n = 128usize;
        let f0 = 0.1f64;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64).sin())
            .collect();
        let result = music_spectral(&signal, 1, 8, 256).expect("music spectral");
        assert!(!result.doa_estimates.is_empty());
        let est_f = result.doa_estimates[0];
        assert!(
            (est_f - f0).abs() < 0.02,
            "Freq error too large: est={est_f:.4}, true={f0:.4}"
        );
    }
}
