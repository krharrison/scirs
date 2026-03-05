//! Classical subspace spectral estimation methods
//!
//! Provides:
//! - [`Pisarenko`]: Pisarenko's single sinusoid frequency estimator
//! - [`MinNorm`]: Minimum-Norm method (extended Pisarenko to multiple sources)
//! - [`MVDR`]: Minimum Variance Distortionless Response beamformer/spectral estimator
//! - [`Capon`]: Capon's maximum-likelihood method (same as MVDR)
//! - [`ForwardBackward`]: Forward-Backward averaging for improved covariance estimate
//!
//! References:
//! - Pisarenko, V.F. (1973). "The retrieval of harmonics from a covariance function."
//!   Geophys. J. Royal Astron. Soc., 33, 347–366.
//! - Reddi, S.S. (1979). "Multiple source location — a digital approach."
//!   IEEE Trans. Aerosp. Electron. Syst., 15(1), 95–105.
//! - Capon, J. (1969). "High-resolution frequency-wavenumber spectrum analysis."
//!   Proc. IEEE, 57(8), 1408–1418.
//! - Johnson, D.H. & Dudgeon, D.E. (1993). "Array Signal Processing." Prentice Hall.

use crate::error::{SignalError, SignalResult};
use crate::subspace::array_processing::{hermitian_eig, SpatialCovariance};
use crate::subspace::esprit::complex_matrix_inv;
use crate::subspace::music::linspace;
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of Pisarenko frequency estimation
#[derive(Debug, Clone)]
pub struct PisarenkoResult {
    /// Estimated frequency(ies) in normalised units [0, 0.5)
    pub frequencies: Vec<f64>,
    /// Noise variance estimate
    pub noise_variance: f64,
    /// Minimum eigenvalue (equals noise variance for exact model)
    pub min_eigenvalue: f64,
}

/// Result of MVDR/Capon spectral estimation
#[derive(Debug, Clone)]
pub struct MVDRResult {
    /// Pseudo-spectrum values
    pub spectrum: Vec<f64>,
    /// Frequency/angle grid
    pub grid: Vec<f64>,
    /// Peak estimates
    pub peak_estimates: Vec<f64>,
}

/// Result of Min-Norm method
#[derive(Debug, Clone)]
pub struct MinNormResult {
    /// Pseudo-spectrum values
    pub pseudo_spectrum: Vec<f64>,
    /// Frequency/angle grid
    pub grid: Vec<f64>,
    /// Peak estimates (frequencies or DOA angles)
    pub peak_estimates: Vec<f64>,
    /// Eigenvalues of covariance matrix
    pub eigenvalues: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Pisarenko
// ---------------------------------------------------------------------------

/// Pisarenko's Harmonic Decomposition method
///
/// Estimates a single sinusoidal frequency from a time series by finding the
/// minimum eigenvector of the autocorrelation matrix. Only valid for exactly
/// one sinusoid in white noise.
///
/// The autocorrelation matrix of order `p+1` is formed and its minimum
/// eigenvector `v = [v₀, v₁, …, vₚ]` gives the frequency:
///
/// `ω = arccos(-Re(v₁) / (2 v₀))`   (for p=2 case)
#[derive(Debug, Clone)]
pub struct Pisarenko {
    /// Model order (autocorrelation matrix size = order+1)
    pub order: usize,
}

impl Pisarenko {
    /// Create a Pisarenko estimator
    ///
    /// # Arguments
    ///
    /// * `order` - Autocorrelation matrix order (minimum 2)
    pub fn new(order: usize) -> SignalResult<Self> {
        if order < 2 {
            return Err(SignalError::ValueError(
                "Pisarenko order must be at least 2".to_string(),
            ));
        }
        Ok(Self { order })
    }

    /// Estimate frequency from time series
    ///
    /// # Arguments
    ///
    /// * `signal` - Real-valued time series
    ///
    /// # Returns
    ///
    /// * `PisarenkoResult`
    pub fn estimate(&self, signal: &[f64]) -> SignalResult<PisarenkoResult> {
        let n = signal.len();
        let m = self.order + 1;
        if n < m + 1 {
            return Err(SignalError::ValueError(format!(
                "Signal length ({n}) must be > order+1 ({m})"
            )));
        }

        // Build (m × m) Toeplitz autocorrelation matrix
        let r = autocorrelation_matrix(signal, m)?;

        // Eigendecompose
        let (eigenvalues, eigenvectors) = hermitian_eig(&r, m)?;

        // Minimum eigenvalue → noise variance
        let min_eig_idx = m - 1; // last (smallest) in descending order
        let min_eigenvalue = eigenvalues[min_eig_idx];
        let noise_variance = min_eigenvalue;

        // Minimum eigenvector
        let min_evec = &eigenvectors[min_eig_idx];

        // For p=2: two-element min eigenvector gives one frequency
        // For general p: use the polynomial rooting approach
        // Polynomial: P(z) = sum_{k=0}^{m-1} v[k] z^k  (roots on unit circle)
        let poly: Vec<Complex64> = min_evec.clone();
        let roots = crate::subspace::music::find_polynomial_roots(&poly)?;

        // Keep roots closest to unit circle
        let mut root_dist: Vec<(Complex64, f64)> = roots
            .iter()
            .map(|&r_val| {
                let dist = (r_val.norm() - 1.0).abs();
                (r_val, dist)
            })
            .collect();
        root_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // The closest root to unit circle gives the sinusoid
        let n_freqs = 1; // Pisarenko handles one sinusoid per model
        let frequencies: Vec<f64> = root_dist
            .iter()
            .take(n_freqs)
            .filter_map(|(z, _)| {
                let phi = z.arg(); // φ = 2π f
                // Normalised frequency: f = φ / (2π), keep in [0, 0.5]
                let f = phi / (2.0 * PI);
                let f_norm = if f < 0.0 { f + 1.0 } else { f };
                if f_norm <= 0.5 {
                    Some(f_norm)
                } else {
                    Some(1.0 - f_norm) // fold back to [0, 0.5]
                }
            })
            .collect();

        Ok(PisarenkoResult {
            frequencies,
            noise_variance,
            min_eigenvalue,
        })
    }
}

// ---------------------------------------------------------------------------
// Min-Norm
// ---------------------------------------------------------------------------

/// Minimum-Norm pseudo-spectrum estimator
///
/// Extends the noise subspace projection approach: instead of projecting onto
/// the entire noise subspace (MUSIC), Min-Norm finds the minimum-norm vector
/// in the noise subspace that has a unity first component:
///
/// `d = e₁ - E_s (E_s^H e₁) / (e₁^H E_n E_n^H e₁)`
///
/// and then evaluates `P_MN(ω) = 1 / |d^H a(ω)|²`.
///
/// Min-Norm reduces spurious peaks compared to MUSIC and has improved resolution.
#[derive(Debug, Clone)]
pub struct MinNorm {
    /// Number of array elements / model order
    pub n_elements: usize,
    /// Number of sources
    pub n_sources: usize,
    /// Element spacing in wavelengths (for DOA) or 1 (for frequency estimation)
    pub element_spacing: f64,
    /// Number of scan points
    pub n_scan: usize,
    /// Grid minimum (angle or frequency)
    pub grid_min: f64,
    /// Grid maximum (angle or frequency)
    pub grid_max: f64,
    /// Mode: "doa" or "freq"
    pub mode: MinNormMode,
}

/// Operating mode for Min-Norm
#[derive(Debug, Clone, PartialEq)]
pub enum MinNormMode {
    /// DOA estimation for ULA (grid in radians)
    DOA,
    /// Spectral/frequency estimation (grid in normalised frequencies [0, 0.5])
    Frequency,
}

impl MinNorm {
    /// Create a new Min-Norm estimator for DOA
    pub fn new_doa(n_elements: usize, n_sources: usize, element_spacing: f64) -> SignalResult<Self> {
        if n_elements < 2 {
            return Err(SignalError::ValueError(
                "Min-Norm requires at least 2 elements".to_string(),
            ));
        }
        if n_sources == 0 || n_sources >= n_elements {
            return Err(SignalError::ValueError(format!(
                "n_sources ({n_sources}) must be in [1, {n_elements})"
            )));
        }
        Ok(Self {
            n_elements,
            n_sources,
            element_spacing,
            n_scan: 360,
            grid_min: -PI / 2.0,
            grid_max: PI / 2.0,
            mode: MinNormMode::DOA,
        })
    }

    /// Create a new Min-Norm estimator for frequency estimation
    pub fn new_freq(model_order: usize, n_sources: usize) -> SignalResult<Self> {
        if model_order < 2 {
            return Err(SignalError::ValueError(
                "Min-Norm model order must be at least 2".to_string(),
            ));
        }
        if n_sources == 0 || n_sources >= model_order {
            return Err(SignalError::ValueError(format!(
                "n_sources ({n_sources}) must be in [1, {model_order})"
            )));
        }
        Ok(Self {
            n_elements: model_order,
            n_sources,
            element_spacing: 1.0,
            n_scan: 512,
            grid_min: 0.0,
            grid_max: 0.5,
            mode: MinNormMode::Frequency,
        })
    }

    /// Estimate from precomputed covariance
    pub fn estimate(&self, covariance: &SpatialCovariance) -> SignalResult<MinNormResult> {
        if covariance.size != self.n_elements {
            return Err(SignalError::DimensionMismatch(format!(
                "Covariance size {} ≠ n_elements {}",
                covariance.size, self.n_elements
            )));
        }
        let m = self.n_elements;
        let d = self.n_sources;

        let (eigenvalues, eigenvectors) = hermitian_eig(&covariance.matrix, m)?;

        // Noise subspace matrix: columns [d..m]
        let n_noise = m - d;
        let noise_evecs: Vec<&Vec<Complex64>> = eigenvectors[d..].iter().collect();

        // Build noise subspace projection matrix E_n E_n^H (m × m)
        let mut en_ent = vec![Complex64::new(0.0, 0.0); m * m];
        for ev in noise_evecs.iter().take(n_noise) {
            for i in 0..m {
                for j in 0..m {
                    en_ent[i * m + j] = en_ent[i * m + j] + ev[i] * ev[j].conj();
                }
            }
        }

        // Min-Norm weight vector: d = e₁ - E_s (E_s^H e₁) / (e₁^H E_n E_n^H e₁)
        // e₁ = [1, 0, 0, …] (canonical basis)
        // E_s^H e₁ = first row of E_s^H = conjugate of first elements of signal eigenvectors
        let signal_evecs: Vec<&Vec<Complex64>> = eigenvectors[..d].iter().collect();

        // e₁^H E_n E_n^H e₁ = (E_n E_n^H)[0,0]
        let denom_val = en_ent[0];
        let denom = if denom_val.re.abs() > 1e-14 {
            denom_val.re
        } else {
            1e-14
        };

        // Compute E_s (E_s^H e₁): d-vector E_s^H e₁ = [es[k][0].conj() for k in 0..d]
        // Then E_s * that = sum_k es[k][0].conj() * es[k]
        let mut correction = vec![Complex64::new(0.0, 0.0); m];
        for k in 0..d {
            let weight = signal_evecs[k][0].conj();
            for i in 0..m {
                correction[i] = correction[i] + weight * signal_evecs[k][i];
            }
        }

        // d_vec = e₁ - correction / denom
        let mut d_vec = vec![Complex64::new(0.0, 0.0); m];
        d_vec[0] = Complex64::new(1.0, 0.0);
        for i in 0..m {
            d_vec[i] = d_vec[i] - correction[i] / denom;
        }

        // Scan grid
        let grid = linspace(self.grid_min, self.grid_max, self.n_scan);
        let mut pseudo_spectrum = Vec::with_capacity(self.n_scan);

        for &param in &grid {
            let sv = match self.mode {
                MinNormMode::DOA => {
                    let phase_inc = -2.0 * PI * self.element_spacing * param.sin();
                    (0..m)
                        .map(|k| {
                            let ph = phase_inc * k as f64;
                            Complex64::new(ph.cos(), ph.sin())
                        })
                        .collect::<Vec<_>>()
                }
                MinNormMode::Frequency => {
                    (0..m)
                        .map(|k| {
                            let ph = 2.0 * PI * param * k as f64;
                            Complex64::new(ph.cos(), ph.sin())
                        })
                        .collect::<Vec<_>>()
                }
            };

            let mut dot = Complex64::new(0.0, 0.0);
            for k in 0..m {
                dot = dot + d_vec[k].conj() * sv[k];
            }
            let ps = if dot.norm_sqr() > 1e-30 {
                1.0 / dot.norm_sqr()
            } else {
                1e30
            };
            pseudo_spectrum.push(ps);
        }

        let peak_estimates =
            crate::subspace::music::find_peaks(&pseudo_spectrum, &grid, self.n_sources);

        Ok(MinNormResult {
            pseudo_spectrum,
            grid,
            peak_estimates,
            eigenvalues,
        })
    }

    /// Estimate from snapshot data
    pub fn estimate_from_data(&self, data: &[Vec<Complex64>]) -> SignalResult<MinNormResult> {
        let cov = SpatialCovariance::estimate(data)?;
        self.estimate(&cov)
    }
}

// ---------------------------------------------------------------------------
// MVDR / Capon
// ---------------------------------------------------------------------------

/// MVDR (Minimum Variance Distortionless Response) beamformer / spectral estimator
///
/// Also known as Capon's method. The MVDR spectrum is:
///
/// `P_MVDR(θ) = 1 / (a^H(θ) R^{-1} a(θ))`
///
/// Provides better spatial resolution than conventional beamforming but requires
/// matrix inversion at each scan point (or equivalently once for the full grid).
///
/// Unlike MUSIC/ESPRIT, MVDR does **not** require knowledge of the number of sources.
#[derive(Debug, Clone)]
pub struct MVDR {
    /// Number of array elements
    pub n_elements: usize,
    /// Element spacing in wavelengths
    pub element_spacing: f64,
    /// Number of scan points
    pub n_scan: usize,
    /// Scan grid min (radians for DOA, normalised freq for spectral)
    pub grid_min: f64,
    /// Scan grid max
    pub grid_max: f64,
    /// Diagonal loading factor (regularisation, default 0)
    pub diagonal_loading: f64,
    /// Number of peak estimates to return
    pub n_peaks: usize,
    /// Mode
    pub mode: MVDRMode,
}

/// Operating mode for MVDR
#[derive(Debug, Clone, PartialEq)]
pub enum MVDRMode {
    /// DOA estimation (grid in radians)
    DOA,
    /// Frequency estimation (grid in normalised frequencies)
    Frequency,
}

impl MVDR {
    /// Create MVDR estimator for DOA
    pub fn new_doa(n_elements: usize, element_spacing: f64, n_peaks: usize) -> SignalResult<Self> {
        if n_elements < 2 {
            return Err(SignalError::ValueError(
                "MVDR requires at least 2 elements".to_string(),
            ));
        }
        if n_peaks == 0 {
            return Err(SignalError::ValueError(
                "n_peaks must be positive".to_string(),
            ));
        }
        Ok(Self {
            n_elements,
            element_spacing,
            n_scan: 360,
            grid_min: -PI / 2.0,
            grid_max: PI / 2.0,
            diagonal_loading: 0.0,
            n_peaks,
            mode: MVDRMode::DOA,
        })
    }

    /// Create MVDR estimator for frequency estimation
    pub fn new_freq(model_order: usize, n_peaks: usize) -> SignalResult<Self> {
        if model_order < 2 {
            return Err(SignalError::ValueError(
                "MVDR model order must be at least 2".to_string(),
            ));
        }
        Ok(Self {
            n_elements: model_order,
            element_spacing: 1.0,
            n_scan: 512,
            grid_min: 0.0,
            grid_max: 0.5,
            diagonal_loading: 0.0,
            n_peaks,
            mode: MVDRMode::Frequency,
        })
    }

    /// Estimate from precomputed covariance
    pub fn estimate(&self, covariance: &SpatialCovariance) -> SignalResult<MVDRResult> {
        if covariance.size != self.n_elements {
            return Err(SignalError::DimensionMismatch(format!(
                "Covariance size {} ≠ n_elements {}",
                covariance.size, self.n_elements
            )));
        }
        let m = self.n_elements;

        // Apply diagonal loading if requested
        let mut r = covariance.matrix.clone();
        if self.diagonal_loading > 0.0 {
            for i in 0..m {
                r[i * m + i] = r[i * m + i] + Complex64::new(self.diagonal_loading, 0.0);
            }
        }

        // Invert R
        let r_inv = complex_matrix_inv(&r, m)?;

        let grid = linspace(self.grid_min, self.grid_max, self.n_scan);
        let mut spectrum = Vec::with_capacity(self.n_scan);

        for &param in &grid {
            let sv = self.steering_vector(m, param);
            // a^H R^{-1} a
            let mut r_inv_a = vec![Complex64::new(0.0, 0.0); m];
            for i in 0..m {
                for j in 0..m {
                    r_inv_a[i] = r_inv_a[i] + r_inv[i * m + j] * sv[j];
                }
            }
            let mut ahria = Complex64::new(0.0, 0.0);
            for i in 0..m {
                ahria = ahria + sv[i].conj() * r_inv_a[i];
            }
            let s = if ahria.re > 1e-30 {
                1.0 / ahria.re
            } else {
                1e30
            };
            spectrum.push(s);
        }

        let peak_estimates =
            crate::subspace::music::find_peaks(&spectrum, &grid, self.n_peaks);

        Ok(MVDRResult {
            spectrum,
            grid,
            peak_estimates,
        })
    }

    /// Estimate from snapshot data
    pub fn estimate_from_data(&self, data: &[Vec<Complex64>]) -> SignalResult<MVDRResult> {
        let cov = SpatialCovariance::estimate(data)?;
        self.estimate(&cov)
    }

    fn steering_vector(&self, m: usize, param: f64) -> Vec<Complex64> {
        match self.mode {
            MVDRMode::DOA => {
                let phase_inc = -2.0 * PI * self.element_spacing * param.sin();
                (0..m)
                    .map(|k| {
                        let ph = phase_inc * k as f64;
                        Complex64::new(ph.cos(), ph.sin())
                    })
                    .collect()
            }
            MVDRMode::Frequency => (0..m)
                .map(|k| {
                    let ph = 2.0 * PI * param * k as f64;
                    Complex64::new(ph.cos(), ph.sin())
                })
                .collect(),
        }
    }
}

/// Capon's method — alias for MVDR (mathematically identical)
pub type Capon = MVDR;

// ---------------------------------------------------------------------------
// ForwardBackward averaging
// ---------------------------------------------------------------------------

/// Forward-Backward (FB) covariance averaging
///
/// The FB-averaged covariance is:
///
/// `R_FB = (R_f + J R_f* J) / 2`
///
/// where `J` is the exchange matrix (anti-diagonal ones). This operation
/// decorrelates coherent signals and can double the effective snapshot count
/// (reducing variance of the covariance estimate).
///
/// Use this as a preprocessing step before MUSIC, ESPRIT, or MVDR.
#[derive(Debug, Clone)]
pub struct ForwardBackward {
    /// Number of array elements
    pub n_elements: usize,
}

impl ForwardBackward {
    /// Create a ForwardBackward averager
    pub fn new(n_elements: usize) -> SignalResult<Self> {
        if n_elements < 2 {
            return Err(SignalError::ValueError(
                "ForwardBackward requires at least 2 elements".to_string(),
            ));
        }
        Ok(Self { n_elements })
    }

    /// Compute FB-averaged covariance from snapshot data
    pub fn covariance(&self, data: &[Vec<Complex64>]) -> SignalResult<SpatialCovariance> {
        if data.len() != self.n_elements {
            return Err(SignalError::DimensionMismatch(format!(
                "data has {} rows, expected {}",
                data.len(),
                self.n_elements
            )));
        }
        let forward_cov = SpatialCovariance::estimate(data)?;
        Ok(forward_cov.forward_backward_average())
    }

    /// Compute FB-averaged covariance from precomputed forward covariance
    pub fn average(&self, forward_cov: SpatialCovariance) -> SignalResult<SpatialCovariance> {
        if forward_cov.size != self.n_elements {
            return Err(SignalError::DimensionMismatch(format!(
                "Covariance size {} ≠ n_elements {}",
                forward_cov.size, self.n_elements
            )));
        }
        Ok(forward_cov.forward_backward_average())
    }
}

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

/// Build Toeplitz autocorrelation matrix of size `m×m` from signal
fn autocorrelation_matrix(signal: &[f64], m: usize) -> SignalResult<Vec<Complex64>> {
    let n = signal.len();
    if n < m {
        return Err(SignalError::ValueError(format!(
            "Signal length ({n}) must be >= m ({m})"
        )));
    }
    let mut r = vec![Complex64::new(0.0, 0.0); m * m];
    // r[i,j] = R(i-j) = (1/N) sum x[k]*x[k-(i-j)] for valid indices
    for i in 0..m {
        for j in 0..m {
            let lag = (i as isize) - (j as isize);
            let mut sum = 0.0f64;
            let mut count = 0usize;
            for k in 0..n {
                let k2 = k as isize - lag;
                if k2 >= 0 && k2 < n as isize {
                    sum += signal[k] * signal[k2 as usize];
                    count += 1;
                }
            }
            let val = if count > 0 { sum / count as f64 } else { 0.0 };
            r[i * m + j] = Complex64::new(val, 0.0);
        }
    }
    Ok(r)
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
        let mut rng: u64 = 99999;
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
    fn test_pisarenko_frequency() {
        let n = 256usize;
        let f0 = 0.1f64;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64).sin())
            .collect();
        let pis = Pisarenko::new(4).expect("pis");
        let result = pis.estimate(&signal).expect("result");
        if !result.frequencies.is_empty() {
            let est_f = result.frequencies[0];
            assert!(
                (est_f - f0).abs() < 0.05,
                "Pisarenko freq error: est={est_f:.4}, true={f0:.4}"
            );
        }
    }

    #[test]
    fn test_mvdr_doa() {
        let n_el = 8;
        let theta = 20.0f64.to_radians();
        let data = make_ula_snapshot(n_el, 0.5, theta, 200, 20.0);
        let mvdr = MVDR::new_doa(n_el, 0.5, 1).expect("mvdr");
        let result = mvdr.estimate_from_data(&data).expect("result");
        assert!(!result.peak_estimates.is_empty());
        let est_deg = result.peak_estimates[0].to_degrees();
        let true_deg = theta.to_degrees();
        assert!(
            (est_deg - true_deg).abs() < 10.0,
            "MVDR DOA error: est={est_deg:.2}°, true={true_deg:.2}°"
        );
    }

    #[test]
    fn test_minnorm_freq() {
        let n = 256usize;
        let f0 = 0.15f64;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64).sin())
            .collect();
        // Build autocorrelation matrix manually
        let m = 10usize;
        let r = autocorrelation_matrix(&signal, m).expect("r");
        let cov = SpatialCovariance {
            matrix: r,
            size: m,
            n_snapshots: n,
        };
        let mn = MinNorm::new_freq(m, 1).expect("mn");
        let result = mn.estimate(&cov).expect("result");
        assert!(!result.peak_estimates.is_empty());
        let est_f = result.peak_estimates[0];
        assert!(
            (est_f - f0).abs() < 0.05,
            "MinNorm freq error: est={est_f:.4}, true={f0:.4}"
        );
    }

    #[test]
    fn test_forward_backward() {
        let n_el = 6;
        let theta = 10.0f64.to_radians();
        let data = make_ula_snapshot(n_el, 0.5, theta, 100, 10.0);
        let fb = ForwardBackward::new(n_el).expect("fb");
        let cov = fb.covariance(&data).expect("cov");
        assert_eq!(cov.size, n_el);
        // Check Hermitian symmetry
        for i in 0..n_el {
            for j in 0..n_el {
                let diff = (cov.get(i, j) - cov.get(j, i).conj()).norm();
                assert!(diff < 1e-10, "FB cov not Hermitian at ({i},{j}): {diff}");
            }
        }
    }

    #[test]
    fn test_capon_alias() {
        // Capon = MVDR, just verify we can construct it
        let capon = Capon::new_doa(8, 0.5, 1).expect("capon");
        assert_eq!(capon.n_elements, 8);
    }
}
