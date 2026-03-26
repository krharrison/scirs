//! Adaptive Sparse FFT recovery algorithm.
//!
//! Implements a hash-based frequency localisation approach with multi-iteration
//! progressive sparsity refinement.  The algorithm:
//!
//! 1. Estimates the signal sparsity `k` from the power spectrum (energy-based
//!    elbow / knee-point method).
//! 2. Computes the full FFT of the signal.
//! 3. In each iteration:
//!    a. Applies a random permutation to the signal (redistributes aliasing).
//!    b. Sub-samples the permuted signal and computes a short FFT.
//!    c. Identifies the largest `k` peaks in the sub-sampled spectrum.
//!    d. Maps sub-spectrum bin candidates back to the original frequency grid
//!    using the inverse permutation.
//!    e. Validates candidates against the full spectrum energy threshold.
//! 4. After all iterations, collects the union of validated frequency bins and
//!    reads their coefficients directly from the full FFT.
//! 5. Returns the top-`max_sparsity` bins by energy.
//!
//! # References
//!
//! * Hassanieh, H., Indyk, P., Katabi, D., Price, E. "Simple and Practical
//!   Algorithm for Sparse Fourier Transform." SODA 2012.
//! * Pawar, S., Ramchandran, K. "FFAST: An Algorithm for Computing an Exactly
//!   k-Sparse DFT in O(k log k) Time." ISIT 2013.

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use scirs2_core::numeric::Complex64;
use std::collections::HashMap;

use super::estimation::estimate_sparsity;
use super::types::{AdaptiveSfftConfig, AdaptiveSfftResult};

/// Adaptive Sparse FFT algorithm.
///
/// Automatically estimates signal sparsity and recovers the sparse frequency
/// representation using hash-based frequency localisation.
///
/// # Examples
///
/// ```
/// use scirs2_fft::adaptive_sparse_fft::recovery::AdaptiveSparseFft;
/// use scirs2_fft::adaptive_sparse_fft::types::AdaptiveSfftConfig;
/// use std::f64::consts::PI;
///
/// let n = 256;
/// let freq_bin = 10_usize;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * freq_bin as f64 * i as f64 / n as f64).sin())
///     .collect();
///
/// let config = AdaptiveSfftConfig::default();
/// let solver = AdaptiveSparseFft::new();
/// let result = solver.compute(&signal, &config).expect("recovery should succeed");
///
/// // The single tone should be detected
/// assert!(!result.frequencies.is_empty());
/// ```
pub struct AdaptiveSparseFft {
    /// Random seed for permutation generation.
    seed: u64,
}

impl AdaptiveSparseFft {
    /// Create a new solver with a default seed.
    pub fn new() -> Self {
        Self {
            seed: 0x517c1e6f3a5b9d2e,
        }
    }

    /// Create a new solver with a specific seed for reproducibility.
    pub fn with_seed(seed: u64) -> Self {
        Self { seed }
    }

    /// Run the adaptive sparse FFT on `signal`.
    ///
    /// # Errors
    ///
    /// Returns an error if the signal is empty or has length < 4, or if any
    /// internal FFT computation fails.
    pub fn compute(
        &self,
        signal: &[f64],
        config: &AdaptiveSfftConfig,
    ) -> FFTResult<AdaptiveSfftResult> {
        let n = signal.len();
        if n < 4 {
            return Err(FFTError::ValueError(
                "Signal must have at least 4 samples".to_string(),
            ));
        }

        // Estimate initial sparsity
        let initial_k = estimate_sparsity(signal)?.max(1).min(config.max_sparsity);

        // Compute full FFT for reference: this gives us exact coefficients.
        let full_spectrum = fft(signal, None)?;
        let n_spectrum = full_spectrum.len();

        // Compute per-bin energy for thresholding
        let bin_energies: Vec<f64> = full_spectrum
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .collect();

        let total_energy: f64 = bin_energies.iter().sum();
        if total_energy < f64::EPSILON {
            return Ok(AdaptiveSfftResult::empty());
        }

        // Maximum energy in any single bin — used for relative threshold
        let max_bin_energy = bin_energies.iter().cloned().fold(0.0_f64, f64::max);

        // Absolute threshold per bin
        let abs_threshold = max_bin_energy * config.energy_threshold;

        let mut candidate_votes: HashMap<usize, usize> = HashMap::new();
        let mut rng_state = self.seed;
        let mut current_k = initial_k;
        let mut actual_iterations = 0;

        for _iter in 0..config.max_iterations {
            actual_iterations += 1;

            // Generate a permutation for this iteration
            let perm = generate_permutation(n, &mut rng_state);
            let inv_perm = invert_permutation(&perm);

            // Apply permutation to signal
            let permuted: Vec<f64> = (0..n).map(|i| signal[perm[i]]).collect();

            // Sub-sample to find candidate frequency locations
            let b = compute_bucket_count(n, current_k);
            let sub_signal = subsample_avg(&permuted, b);

            // FFT of sub-sampled signal
            let sub_spectrum = fft(&sub_signal, None)?;
            let b_actual = sub_spectrum.len();

            // Find top-`current_k` peaks in sub spectrum
            let peaks = find_top_peaks(&sub_spectrum, current_k);

            // Map back to original frequency grid
            for (sub_bin, _) in &peaks {
                // sub_bin `j` in the sub-sampled spectrum aliases to all
                // original permuted bins { j + m*B : m=0,1,... }
                let num_aliases = n.div_ceil(b_actual);
                for alias in 0..num_aliases {
                    let perm_bin = (sub_bin + alias * b_actual) % n_spectrum;
                    let orig_bin = inv_perm[perm_bin];
                    if orig_bin < n_spectrum && bin_energies[orig_bin] >= abs_threshold {
                        *candidate_votes.entry(orig_bin).or_insert(0) += 1;
                    }
                }
            }

            // Check captured fraction
            let captured: f64 = candidate_votes
                .keys()
                .map(|&b| bin_energies[b])
                .sum::<f64>();
            if captured / total_energy >= config.confidence {
                break;
            }

            // Refine sparsity: subtract found components from signal and re-estimate
            if actual_iterations < config.max_iterations {
                let found: HashMap<usize, Complex64> = candidate_votes
                    .keys()
                    .filter(|&&b| b < n_spectrum)
                    .map(|&b| (b, full_spectrum[b]))
                    .collect();
                let residual = subtract_components(signal, &found, n)?;
                let residual_k = estimate_sparsity(&residual).unwrap_or(1).max(1);
                current_k = residual_k.min(config.max_sparsity);
                if current_k == 0 {
                    break;
                }
            }
        }

        // If no candidates found via hashing, fall back to direct top-k from full spectrum
        if candidate_votes.is_empty() {
            let peaks = find_top_peaks(&full_spectrum, initial_k.min(config.max_sparsity));
            for (bin, _) in peaks {
                if bin_energies[bin] >= abs_threshold {
                    candidate_votes.insert(bin, 1);
                }
            }
            // If still empty, take the single maximum bin
            if candidate_votes.is_empty() {
                if let Some((max_bin, _)) = bin_energies
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                {
                    candidate_votes.insert(max_bin, 1);
                }
            }
        }

        // Collect final candidates, sorting by descending energy
        let mut candidate_list: Vec<(usize, Complex64)> = candidate_votes
            .keys()
            .filter(|&&b| b < n_spectrum)
            .map(|&b| (b, full_spectrum[b]))
            .collect();

        candidate_list.sort_by(|a, b| {
            let ea = a.1.re * a.1.re + a.1.im * a.1.im;
            let eb = b.1.re * b.1.re + b.1.im * b.1.im;
            eb.partial_cmp(&ea).unwrap_or(std::cmp::Ordering::Equal)
        });
        candidate_list.truncate(config.max_sparsity);

        // Sort by frequency index for deterministic output
        candidate_list.sort_by_key(|(idx, _)| *idx);

        let frequencies: Vec<usize> = candidate_list.iter().map(|(i, _)| *i).collect();
        let coefficients: Vec<Complex64> = candidate_list.iter().map(|(_, c)| *c).collect();

        // Compute captured energy fraction
        let captured_energy: f64 = coefficients.iter().map(|c| c.re * c.re + c.im * c.im).sum();
        let captured_fraction = (captured_energy / total_energy).min(1.0);

        Ok(AdaptiveSfftResult {
            estimated_sparsity: frequencies.len(),
            iterations: actual_iterations,
            total_energy,
            captured_energy_fraction: captured_fraction,
            frequencies,
            coefficients,
        })
    }
}

impl Default for AdaptiveSparseFft {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Generate a pseudo-random permutation of `{0, ..., n-1}` using Fisher-Yates
/// shuffle seeded by the given state (mutated in place via xorshift64).
fn generate_permutation(n: usize, state: &mut u64) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        let j = (*state as usize) % (i + 1);
        perm.swap(i, j);
    }
    perm
}

/// Compute the inverse of a permutation.
fn invert_permutation(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

/// Compute the bucket count for sub-sampling.
///
/// Chooses `B` such that the sub-spectrum has approximately `4k` bins,
/// rounded up to the nearest power of two for efficient FFT.
fn compute_bucket_count(n: usize, k: usize) -> usize {
    let b = (n / (4 * k.max(1))).max(1);
    b.next_power_of_two().min(n)
}

/// Sub-sample `signal` to `target_len` samples by averaging blocks.
///
/// This is a simple decimation with averaging (no anti-aliasing filter),
/// which is sufficient for the coarse frequency localisation step.
fn subsample_avg(signal: &[f64], target_len: usize) -> Vec<f64> {
    let n = signal.len();
    if target_len >= n {
        return signal.to_vec();
    }
    let block = n / target_len;
    (0..target_len)
        .map(|i| {
            let start = i * block;
            let end = (start + block).min(n);
            let count = (end - start) as f64;
            signal[start..end].iter().sum::<f64>() / count
        })
        .collect()
}

/// Find the top `k` frequency bins by magnitude in a complex spectrum.
///
/// Returns `(bin_index, magnitude)` pairs sorted by descending magnitude.
fn find_top_peaks(spectrum: &[Complex64], k: usize) -> Vec<(usize, f64)> {
    let mut magnitudes: Vec<(usize, f64)> = spectrum
        .iter()
        .enumerate()
        .map(|(i, c)| (i, (c.re * c.re + c.im * c.im).sqrt()))
        .collect();

    magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    magnitudes.truncate(k);
    magnitudes
}

/// Subtract candidate frequency components from the time-domain signal.
fn subtract_components(
    signal: &[f64],
    candidates: &HashMap<usize, Complex64>,
    n: usize,
) -> FFTResult<Vec<f64>> {
    if candidates.is_empty() {
        return Ok(signal.to_vec());
    }

    let mut candidate_spectrum: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n];
    for (&bin, &coeff) in candidates {
        if bin < n {
            candidate_spectrum[bin] = coeff;
        }
    }

    let approx = ifft(&candidate_spectrum, None)?;
    let residual: Vec<f64> = signal
        .iter()
        .zip(approx.iter())
        .map(|(&s, a)| s - a.re)
        .collect();

    Ok(residual)
}

/// Convenience function: run the adaptive sparse FFT with default config.
///
/// # Errors
///
/// Returns an error if the signal is too short or FFT computation fails.
///
/// # Examples
///
/// ```
/// use scirs2_fft::adaptive_sparse_fft::recovery::adaptive_sparse_fft_auto;
/// use std::f64::consts::PI;
///
/// let n = 128;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 5.0 * i as f64 / n as f64).sin())
///     .collect();
/// let result = adaptive_sparse_fft_auto(&signal).expect("should succeed");
/// assert!(!result.frequencies.is_empty());
/// ```
pub fn adaptive_sparse_fft_auto(signal: &[f64]) -> FFTResult<AdaptiveSfftResult> {
    let config = AdaptiveSfftConfig::default();
    let solver = AdaptiveSparseFft::new();
    solver.compute(signal, &config)
}
