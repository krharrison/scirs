//! Morlet wavelet filter bank for scattering transforms
//!
//! Constructs a dyadic filter bank of Morlet wavelets in the frequency domain,
//! along with a low-pass scaling function. The filter bank is parameterized by:
//! - J: number of octaves (scales)
//! - Q: quality factor (wavelets per octave)
//! - signal_length: length of the input signal (determines FFT size)

use std::f64::consts::PI;

use scirs2_core::numeric::Complex64;

use crate::error::{FFTError, FFTResult};

/// Configuration for a Morlet wavelet filter bank.
#[derive(Debug, Clone)]
pub struct FilterBankConfig {
    /// Number of octaves (logarithmic scale range)
    pub j_max: usize,
    /// Quality factors per order (wavelets per octave for each scattering order)
    pub quality_factors: Vec<usize>,
    /// Length of the input signal
    pub signal_length: usize,
    /// Center frequency of the mother wavelet (default: PI)
    pub xi0: f64,
    /// Bandwidth parameter sigma (default: computed from Q)
    pub sigma: Option<f64>,
}

impl FilterBankConfig {
    /// Create a new filter bank configuration with default parameters.
    ///
    /// # Arguments
    /// * `j_max` - Number of octaves
    /// * `quality_factors` - Quality factors per order (e.g., `[8, 1]` for Q1=8, Q2=1)
    /// * `signal_length` - Length of the input signal
    pub fn new(j_max: usize, quality_factors: Vec<usize>, signal_length: usize) -> Self {
        Self {
            j_max,
            quality_factors,
            signal_length,
            xi0: PI,
            sigma: None,
        }
    }

    /// Set the center frequency of the mother wavelet.
    #[must_use]
    pub fn with_xi0(mut self, xi0: f64) -> Self {
        self.xi0 = xi0;
        self
    }

    /// Set a custom bandwidth parameter.
    #[must_use]
    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.sigma = Some(sigma);
        self
    }
}

/// A single Morlet wavelet: psi(t) = C * exp(-t^2 / (2*sigma^2)) * exp(j*xi*t)
///
/// Stored in the frequency domain for efficient convolution.
#[derive(Debug, Clone)]
pub struct MorletWavelet {
    /// Center frequency (radians per sample)
    pub xi: f64,
    /// Bandwidth parameter
    pub sigma: f64,
    /// Scale index j
    pub j: usize,
    /// Sub-octave index within octave
    pub q_index: usize,
    /// Linear index (j * Q + q_index)
    pub linear_index: usize,
    /// Frequency-domain representation (complex-valued, length = fft_size)
    pub freq_response: Vec<Complex64>,
}

/// A wavelet filter bank containing all wavelets and the scaling function.
#[derive(Debug, Clone)]
pub struct FilterBank {
    /// Configuration used to build this filter bank
    pub config: FilterBankConfig,
    /// FFT size (next power of 2 >= signal_length)
    pub fft_size: usize,
    /// Wavelets for each scattering order (outer: order, inner: wavelet index)
    pub wavelets: Vec<Vec<MorletWavelet>>,
    /// Low-pass scaling function in frequency domain
    pub phi: Vec<Complex64>,
}

impl FilterBank {
    /// Construct a new filter bank from configuration.
    ///
    /// Builds Morlet wavelets at dyadic scales 2^(j/Q) for j = 0..J*Q-1,
    /// plus a low-pass scaling function phi that captures content below 2^J.
    pub fn new(config: FilterBankConfig) -> FFTResult<Self> {
        if config.j_max == 0 {
            return Err(FFTError::ValueError("j_max must be at least 1".to_string()));
        }
        if config.quality_factors.is_empty() {
            return Err(FFTError::ValueError(
                "quality_factors must have at least one entry".to_string(),
            ));
        }
        for (i, &q) in config.quality_factors.iter().enumerate() {
            if q == 0 {
                return Err(FFTError::ValueError(format!(
                    "quality_factors[{i}] must be at least 1"
                )));
            }
        }
        if config.signal_length == 0 {
            return Err(FFTError::ValueError(
                "signal_length must be positive".to_string(),
            ));
        }

        let fft_size = config.signal_length.next_power_of_two();

        // Build wavelets for each order
        let mut all_wavelets = Vec::new();
        for (order, &q) in config.quality_factors.iter().enumerate() {
            let sigma_base = compute_sigma_from_q(q, config.xi0, config.sigma);
            let wavelets =
                build_morlet_wavelets(config.j_max, q, config.xi0, sigma_base, fft_size, order)?;
            all_wavelets.push(wavelets);
        }

        // Build low-pass scaling function
        let sigma_phi = compute_sigma_from_q(config.quality_factors[0], config.xi0, config.sigma);
        let phi = build_scaling_function(config.j_max, sigma_phi, fft_size)?;

        Ok(Self {
            config,
            fft_size,
            wavelets: all_wavelets,
            phi,
        })
    }

    /// Number of first-order wavelets (J * Q1).
    pub fn num_first_order(&self) -> usize {
        self.wavelets.first().map_or(0, |w| w.len())
    }

    /// Number of second-order wavelets (J * Q2), if a second order exists.
    pub fn num_second_order(&self) -> usize {
        self.wavelets.get(1).map_or(0, |w| w.len())
    }

    /// Total number of wavelets across all orders.
    pub fn total_wavelets(&self) -> usize {
        self.wavelets.iter().map(|w| w.len()).sum()
    }
}

/// Compute sigma (bandwidth) from quality factor Q and center frequency xi0.
///
/// sigma = xi0 / (2 * ln(2)^(1/2) * Q) ensures that the half-power bandwidth
/// spans one octave divided by Q.
fn compute_sigma_from_q(q: usize, xi0: f64, custom_sigma: Option<f64>) -> f64 {
    if let Some(s) = custom_sigma {
        return s;
    }
    // sigma such that the wavelet has bandwidth ~ xi0 / Q
    // Using the relation: bandwidth = 2 * sigma * sqrt(2 * ln(2))
    // Q = xi0 / bandwidth => sigma = xi0 / (Q * 2 * sqrt(2 * ln(2)))
    let ln2_sqrt = (2.0_f64 * 2.0_f64.ln()).sqrt();
    xi0 / (q as f64 * ln2_sqrt)
}

/// Build Morlet wavelets at dyadic scales for a given quality factor.
fn build_morlet_wavelets(
    j_max: usize,
    q: usize,
    xi0: f64,
    sigma_base: f64,
    fft_size: usize,
    _order: usize,
) -> FFTResult<Vec<MorletWavelet>> {
    let total = j_max * q;
    let mut wavelets = Vec::with_capacity(total);
    let n = fft_size;

    for idx in 0..total {
        let j = idx / q;
        let q_index = idx % q;

        // Scale factor: 2^(idx / Q)
        let scale = 2.0_f64.powf(idx as f64 / q as f64);

        // Center frequency at this scale
        let xi = xi0 / scale;

        // Bandwidth at this scale
        let sigma = sigma_base * scale;

        // Build frequency-domain Morlet wavelet
        // Psi_hat(omega) = C * exp(-(omega - xi)^2 * sigma^2 / 2)
        // with correction term to ensure zero mean
        let mut freq_response = vec![Complex64::new(0.0, 0.0); n];
        let n_f64 = n as f64;

        for k in 0..n {
            // Normalized frequency: omega = 2*pi*k/N
            let omega = 2.0 * PI * k as f64 / n_f64;

            // Gaussian centered at xi
            let diff_pos = omega - xi;
            let gauss_pos = (-0.5 * diff_pos * diff_pos * sigma * sigma).exp();

            // Correction term for zero mean (subtract Gaussian at omega=0)
            let gauss_correction = (-0.5 * xi * xi * sigma * sigma).exp();
            let gauss_zero = (-0.5 * omega * omega * sigma * sigma).exp();

            let value = gauss_pos - gauss_correction * gauss_zero;
            freq_response[k] = Complex64::new(value, 0.0);
        }

        // Normalize: L2 norm in frequency domain = 1
        let energy: f64 = freq_response.iter().map(|c| c.norm_sqr()).sum();
        if energy > 1e-15 {
            let norm_factor = 1.0 / energy.sqrt();
            for c in &mut freq_response {
                *c = Complex64::new(c.re * norm_factor, c.im * norm_factor);
            }
        }

        wavelets.push(MorletWavelet {
            xi,
            sigma,
            j,
            q_index,
            linear_index: idx,
            freq_response,
        });
    }

    Ok(wavelets)
}

/// Build the low-pass scaling function phi in the frequency domain.
///
/// phi_hat(omega) = exp(-omega^2 * sigma_J^2 / 2) where sigma_J = sigma_base * 2^J
fn build_scaling_function(
    j_max: usize,
    sigma_base: f64,
    fft_size: usize,
) -> FFTResult<Vec<Complex64>> {
    let n = fft_size;
    let n_f64 = n as f64;
    let sigma_j = sigma_base * 2.0_f64.powi(j_max as i32);

    let mut phi = vec![Complex64::new(0.0, 0.0); n];

    for k in 0..n {
        let omega = 2.0 * PI * k as f64 / n_f64;
        // Wrap frequency to [-pi, pi]
        let omega_wrapped = if omega > PI { omega - 2.0 * PI } else { omega };
        let value = (-0.5 * omega_wrapped * omega_wrapped * sigma_j * sigma_j).exp();
        phi[k] = Complex64::new(value, 0.0);
    }

    // Normalize
    let energy: f64 = phi.iter().map(|c| c.norm_sqr()).sum();
    if energy > 1e-15 {
        let norm_factor = 1.0 / energy.sqrt();
        for c in &mut phi {
            *c = Complex64::new(c.re * norm_factor, c.im * norm_factor);
        }
    }

    Ok(phi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_bank_creation() {
        let config = FilterBankConfig::new(4, vec![8, 1], 1024);
        let fb = FilterBank::new(config).expect("filter bank creation should succeed");

        assert_eq!(fb.num_first_order(), 32); // J=4, Q=8 => 32 wavelets
        assert_eq!(fb.num_second_order(), 4); // J=4, Q=1 => 4 wavelets
        assert_eq!(fb.fft_size, 1024);
        assert_eq!(fb.phi.len(), 1024);
    }

    #[test]
    fn test_wavelet_frequency_peaks() {
        // Each wavelet should peak near its center frequency
        let config = FilterBankConfig::new(3, vec![4], 512);
        let fb = FilterBank::new(config).expect("filter bank creation should succeed");

        let first_order = &fb.wavelets[0];
        for w in first_order {
            // Find peak frequency bin
            let peak_bin = w
                .freq_response
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.norm_sqr()
                        .partial_cmp(&b.norm_sqr())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, _)| idx)
                .expect("should find peak");

            let peak_omega = 2.0 * PI * peak_bin as f64 / fb.fft_size as f64;

            // Peak should be near the wavelet's center frequency xi
            // Allow generous tolerance due to discretization
            let rel_error = if w.xi > 1e-6 {
                (peak_omega - w.xi).abs() / w.xi
            } else {
                peak_omega.abs()
            };
            assert!(
                rel_error < 0.5,
                "wavelet j={} q={}: peak_omega={:.4} vs xi={:.4}, rel_error={:.4}",
                w.j,
                w.q_index,
                peak_omega,
                w.xi,
                rel_error
            );
        }
    }

    #[test]
    fn test_dyadic_scaling() {
        // Wavelets should be spaced at octave intervals when Q=1
        let config = FilterBankConfig::new(4, vec![1], 1024);
        let fb = FilterBank::new(config).expect("filter bank creation should succeed");

        let wavelets = &fb.wavelets[0];
        // Check that center frequencies decrease by factor ~2 each octave
        for i in 0..wavelets.len() - 1 {
            let ratio = wavelets[i].xi / wavelets[i + 1].xi;
            // Should be approximately 2.0
            assert!(
                (ratio - 2.0).abs() < 0.1,
                "octave {i} to {}: ratio={:.4}, expected ~2.0",
                i + 1,
                ratio
            );
        }
    }

    #[test]
    fn test_filter_bank_invalid_config() {
        // j_max = 0
        let config = FilterBankConfig::new(0, vec![8], 1024);
        assert!(FilterBank::new(config).is_err());

        // empty quality factors
        let config = FilterBankConfig::new(4, vec![], 1024);
        assert!(FilterBank::new(config).is_err());

        // quality factor = 0
        let config = FilterBankConfig::new(4, vec![0], 1024);
        assert!(FilterBank::new(config).is_err());

        // signal_length = 0
        let config = FilterBankConfig::new(4, vec![8], 0);
        assert!(FilterBank::new(config).is_err());
    }

    #[test]
    fn test_wavelet_l2_normalization() {
        let config = FilterBankConfig::new(3, vec![4], 256);
        let fb = FilterBank::new(config).expect("filter bank creation should succeed");

        for w in &fb.wavelets[0] {
            let energy: f64 = w.freq_response.iter().map(|c| c.norm_sqr()).sum();
            assert!(
                (energy - 1.0).abs() < 1e-10,
                "wavelet j={} q={} has energy {:.6}, expected 1.0",
                w.j,
                w.q_index,
                energy
            );
        }
    }

    #[test]
    fn test_scaling_function_is_lowpass() {
        let config = FilterBankConfig::new(3, vec![4], 512);
        let fb = FilterBank::new(config).expect("filter bank creation should succeed");

        // phi should peak at DC (bin 0)
        let dc_mag = fb.phi[0].norm_sqr();
        let nyquist_bin = fb.fft_size / 2;
        let nyquist_mag = fb.phi[nyquist_bin].norm_sqr();

        assert!(
            dc_mag > nyquist_mag,
            "scaling function should peak at DC: dc={:.6} vs nyquist={:.6}",
            dc_mag,
            nyquist_mag
        );
    }
}
