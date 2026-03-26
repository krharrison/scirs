//! MUSIC (Multiple Signal Classification) frequency/phase estimator.
//!
//! Reference: Schmidt (1986). "Multiple emitter location and signal parameter estimation."
//! IEEE Trans. AP 34(3):276-280.

use std::f64::consts::PI;

use crate::error::{SignalError, SignalResult};

use super::esprit::{build_hankel, estimate_amplitude_phase};
use super::types::{FrequencyComponent, PhaseEstResult, PhaseMethod};

// ─── Jacobi EVD ──────────────────────────────────────────────────────────────

/// Symmetric Jacobi eigenvalue decomposition for a real symmetric n×n matrix.
///
/// On entry `a` is the upper triangle (stored in full row-major format).
/// On exit `a` contains eigenvalues on the diagonal, `v` contains eigenvectors as columns.
fn jacobi_evd(a: &mut [f64], n: usize, max_sweep: usize) -> Vec<Vec<f64>> {
    // Initialise V = I.
    let mut v: Vec<f64> = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    for _sweep in 0..max_sweep {
        let mut max_off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                let val = a[p * n + q].abs();
                if val > max_off {
                    max_off = val;
                }
            }
        }
        if max_off < 1e-13 {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-14 * max_off {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = 0.5 * (aqq - app) / apq;
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update A.
                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = a[r * n + p];
                    let arq = a[r * n + q];
                    a[r * n + p] = c * arp - s * arq;
                    a[p * n + r] = a[r * n + p];
                    a[r * n + q] = s * arp + c * arq;
                    a[q * n + r] = a[r * n + q];
                }

                // Update eigenvector matrix V.
                for r in 0..n {
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = c * vrp - s * vrq;
                    v[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }

    // Extract eigenvectors as column vectors.
    (0..n)
        .map(|col| (0..n).map(|row| v[row * n + col]).collect())
        .collect()
}

/// Return indices of the `k` smallest eigenvalues (on diagonal of `a` after EVD).
fn k_smallest_indices(a: &[f64], n: usize, k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f64)> = (0..n).map(|i| (i, a[i * n + i])).collect();
    indexed.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(k).map(|&(i, _)| i).collect()
}

// ─── Correlation matrix builder ───────────────────────────────────────────────

/// Build the sample correlation matrix R = X^T X / N from signal windows.
/// Uses the Hankel data matrix X (n_windows × win_len).
fn correlation_matrix(x: &[f64], n_windows: usize, win_len: usize) -> Vec<f64> {
    let mut r = vec![0.0f64; win_len * win_len];
    let scale = 1.0 / n_windows as f64;
    for i in 0..win_len {
        for j in 0..win_len {
            let mut s = 0.0;
            for k in 0..n_windows {
                s += x[k * win_len + i] * x[k * win_len + j];
            }
            r[i * win_len + j] = s * scale;
        }
    }
    r
}

// ─── Steering vector helpers ──────────────────────────────────────────────────

/// Compute a(f) · u_n · u_n^T · a(f) (real part, since signal is real).
/// a(f) = [1, e^{j2πf}, …, e^{j2πf(m-1)}] evaluated against real noise subspace.
///
/// For a real signal, the imaginary part of a(f)^H U_n U_n^H a(f) is zero
/// when U_n is real (which it is for a real correlation matrix).
/// We keep both real and imaginary parts for correctness.
fn music_denominator(freq_norm: f64, noise_vecs: &[Vec<f64>], m: usize) -> f64 {
    // a(f) = cos(2πf·k) + j·sin(2πf·k) for k=0..m-1
    let mut denom = 0.0f64;
    for u in noise_vecs {
        // P_u = |a^H u|^2 = (Σ_k cos(2πfk)·u[k])^2 + (Σ_k sin(2πfk)·u[k])^2
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for k in 0..m {
            let theta = 2.0 * PI * freq_norm * k as f64;
            re += theta.cos() * u[k];
            im += theta.sin() * u[k];
        }
        denom += re * re + im * im;
    }
    denom
}

// ─── Peak picker ─────────────────────────────────────────────────────────────

/// Find the indices of the `k` largest local maxima in `spectrum`.
fn find_peaks(spectrum: &[f64], k: usize) -> Vec<usize> {
    let n = spectrum.len();
    if n < 3 {
        return (0..n.min(k)).collect();
    }

    let mut peak_indices: Vec<usize> = (1..n - 1)
        .filter(|&i| spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1])
        .collect();

    // Sort peaks by descending value.
    peak_indices.sort_by(|&a, &b| {
        spectrum[b]
            .partial_cmp(&spectrum[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    peak_indices.truncate(k);
    peak_indices
}

// ─── MusicEstimator ───────────────────────────────────────────────────────────

/// MUSIC frequency and phase estimator.
#[derive(Debug, Clone)]
pub struct MusicEstimator {
    /// Number of signal components.
    pub num_components: usize,
    /// Signal subspace dimension (≥ num_components).
    pub subspace_dim: usize,
    /// Number of frequency points in the pseudospectrum sweep.
    pub freq_resolution: usize,
    /// Sample rate (Hz).
    pub fs: f64,
    /// Maximum Jacobi sweeps.
    pub max_iter: usize,
}

impl Default for MusicEstimator {
    fn default() -> Self {
        Self {
            num_components: 1,
            subspace_dim: 4,
            freq_resolution: 1024,
            fs: 1.0,
            max_iter: 100,
        }
    }
}

impl MusicEstimator {
    /// Create a new MUSIC estimator.
    pub fn new(num_components: usize, fs: f64) -> Self {
        let subspace_dim = (2 * num_components).max(4);
        Self {
            num_components,
            subspace_dim,
            fs,
            ..Default::default()
        }
    }

    /// Hankel window length.
    fn hankel_l(&self, n: usize) -> usize {
        let l_min = self.subspace_dim.saturating_sub(1);
        let l_default = n / 2;
        l_default.max(l_min).min(n.saturating_sub(1))
    }

    /// Compute the MUSIC pseudospectrum for `signal` at the given normalised
    /// frequencies (cycles/sample).
    pub fn pseudospectrum(&self, signal: &[f64], freqs_norm: &[f64]) -> Vec<f64> {
        // Build correlation matrix.
        let n = signal.len();
        let l = self.hankel_l(n);
        let (x, n_windows, win_len) = match build_hankel(signal, l) {
            Ok(v) => v,
            Err(_) => return vec![0.0; freqs_norm.len()],
        };

        let mut r = correlation_matrix(&x, n_windows, win_len);

        // EVD via Jacobi.
        let eigvecs = jacobi_evd(&mut r, win_len, self.max_iter);
        let noise_count = win_len.saturating_sub(self.num_components);

        // Noise subspace: eigenvectors with smallest eigenvalues.
        let noise_indices = k_smallest_indices(&r, win_len, noise_count);
        let noise_vecs: Vec<Vec<f64>> = noise_indices.iter().map(|&i| eigvecs[i].clone()).collect();

        // Evaluate pseudospectrum.
        freqs_norm
            .iter()
            .map(|&f| {
                let d = music_denominator(f, &noise_vecs, win_len);
                if d < 1e-20 {
                    1e20
                } else {
                    1.0 / d
                }
            })
            .collect()
    }

    /// Estimate frequency components from `signal`.
    pub fn estimate(&self, signal: &[f64]) -> SignalResult<PhaseEstResult> {
        let n = signal.len();
        if n < 4 {
            return Err(SignalError::InvalidArgument(format!(
                "Signal too short for MUSIC: need ≥ 4 samples, got {n}"
            )));
        }
        if self.num_components == 0 {
            return Err(SignalError::InvalidArgument(
                "num_components must be ≥ 1".into(),
            ));
        }

        // Sweep frequencies 0..0.5 (normalised, one-sided).
        let m = self.freq_resolution;
        let freqs: Vec<f64> = (0..m).map(|i| i as f64 / (2.0 * m as f64)).collect();
        let spectrum = self.pseudospectrum(signal, &freqs);

        // Find top-num_components peaks.
        let peak_idx = find_peaks(&spectrum, self.num_components);

        if peak_idx.is_empty() {
            // Fallback: pick highest-value bins.
            let mut indexed: Vec<(usize, f64)> = spectrum.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let fallback: Vec<usize> = indexed
                .iter()
                .take(self.num_components)
                .map(|&(i, _)| i)
                .collect();

            let components = fallback
                .iter()
                .map(|&i| {
                    let f_norm = freqs[i];
                    let f_hz = f_norm * self.fs;
                    let (amp, phase) = estimate_amplitude_phase(signal, f_norm);
                    FrequencyComponent {
                        frequency: f_hz,
                        amplitude: amp,
                        phase,
                    }
                })
                .collect();

            return Ok(PhaseEstResult::new(components, PhaseMethod::Music));
        }

        let components: Vec<FrequencyComponent> = peak_idx
            .iter()
            .map(|&i| {
                let f_norm = freqs[i];
                let f_hz = f_norm * self.fs;
                let (amp, phase) = estimate_amplitude_phase(signal, f_norm);
                FrequencyComponent {
                    frequency: f_hz,
                    amplitude: amp,
                    phase,
                }
            })
            .collect();

        Ok(PhaseEstResult::new(components, PhaseMethod::Music))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine(freq_hz: f64, phase: f64, amp: f64, fs: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| amp * (2.0 * PI * freq_hz / fs * i as f64 + phase).sin())
            .collect()
    }

    #[test]
    fn test_music_single_tone() {
        let fs = 1000.0;
        let n = 256;
        let sig = make_sine(150.0, 0.0, 1.0, fs, n);
        let est = MusicEstimator::new(1, fs);
        let result = est.estimate(&sig).expect("music single ok");
        assert_eq!(result.components.len(), 1);
        let f = result.components[0].frequency;
        assert!((f - 150.0).abs() < 5.0, "Expected ~150 Hz, got {f:.2} Hz");
    }

    #[test]
    fn test_music_pseudospectrum_peaks() {
        let fs = 1000.0;
        let n = 512;
        let mut sig = make_sine(100.0, 0.0, 1.0, fs, n);
        let s2 = make_sine(300.0, 0.0, 1.0, fs, n);
        for (a, b) in sig.iter_mut().zip(s2.iter()) {
            *a += b;
        }

        let mut est = MusicEstimator::new(2, fs);
        est.freq_resolution = 2000;
        est.subspace_dim = 6;

        // Pseudospectrum at 100 Hz and 300 Hz should be high.
        let test_freqs = vec![100.0 / fs, 300.0 / fs, 200.0 / fs];
        let ps = est.pseudospectrum(&sig, &test_freqs);
        assert!(
            ps[0] > ps[2],
            "100 Hz should dominate over 200 Hz: {:.2} vs {:.2}",
            ps[0],
            ps[2]
        );
        assert!(
            ps[1] > ps[2],
            "300 Hz should dominate over 200 Hz: {:.2} vs {:.2}",
            ps[1],
            ps[2]
        );
    }

    #[test]
    fn test_music_noise_subspace_orthogonal() {
        // Verify that EVD on the correlation matrix of a pure tone gives
        // exactly 2 dominant eigenvalues (real + imaginary parts) out of win_len.
        // A pure sinusoid of length N in a Hankel matrix of window L has rank 2
        // when N is large; the top 2 eigenvalues should dominate.
        let fs = 1000.0;
        let n = 256;
        let sig = make_sine(100.0, 0.0, 1.0, fs, n);
        // Use a larger Hankel window so the rank-2 structure is clear.
        let l = 30usize;
        let (x, n_windows, win_len) = build_hankel(&sig, l).expect("hankel ok");
        let mut r = correlation_matrix(&x, n_windows, win_len);
        // EVD.
        let _eigvecs = jacobi_evd(&mut r, win_len, 200);
        // Eigenvalues are now on the diagonal of r.
        let mut evals: Vec<f64> = (0..win_len).map(|i| r[i * win_len + i]).collect();
        evals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let top2_sum: f64 = evals[0] + evals[1];
        let total_sum: f64 = evals.iter().sum::<f64>().max(1e-30);
        // The top 2 eigenvalues should capture most of the energy (pure tone → rank 2).
        assert!(
            top2_sum / total_sum > 0.90,
            "Top-2 eigenvalue fraction {:.3} should be > 0.90 for pure tone (win_len={win_len})",
            top2_sum / total_sum
        );
    }

    #[test]
    fn test_music_too_few_samples() {
        let est = MusicEstimator::new(1, 1000.0);
        let result = est.estimate(&[1.0, 2.0]);
        assert!(result.is_err(), "Too-few-samples should return error");
    }

    #[test]
    fn test_music_freq_resolution() {
        let fs = 1000.0;
        let n = 128;
        let sig = make_sine(100.0, 0.0, 1.0, fs, n);

        let mut est = MusicEstimator::new(1, fs);
        est.freq_resolution = 512;
        let ps512 = est.pseudospectrum(
            &sig,
            &(0..512).map(|i| i as f64 / 1024.0).collect::<Vec<_>>(),
        );

        est.freq_resolution = 128;
        let ps128 = est.pseudospectrum(
            &sig,
            &(0..128).map(|i| i as f64 / 256.0).collect::<Vec<_>>(),
        );

        assert_eq!(ps512.len(), 512);
        assert_eq!(ps128.len(), 128);
    }

    #[test]
    fn test_jacobi_evd_symmetric() {
        // Build a known symmetric matrix and verify reconstruction.
        // A = [[5,2,0],[2,3,1],[0,1,4]]
        let mut a = vec![5.0f64, 2.0, 0.0, 2.0, 3.0, 1.0, 0.0, 1.0, 4.0];
        let a_orig = a.clone();
        let n = 3usize;
        let eigvecs = jacobi_evd(&mut a, n, 100);

        // Reconstruct A = V D V^T and compare with original.
        let evals: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();

        // Compute V D V^T.
        let mut recon = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    recon[i * n + j] += eigvecs[k][i] * evals[k] * eigvecs[k][j];
                }
            }
        }

        for i in 0..n {
            for j in 0..n {
                let diff = (recon[i * n + j] - a_orig[i * n + j]).abs();
                assert!(diff < 1e-8, "Reconstruction error at ({i},{j}): {diff:.2e}");
            }
        }
    }
}
