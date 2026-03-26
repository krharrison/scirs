//! Audio quality metrics: PESQ estimate, STOI, SI-SDR, SNR improvement,
//! and spectral convergence.
//!
//! These are simplified implementations suitable for benchmarking and
//! comparing speech enhancement methods.

use crate::error::{SignalError, SignalResult};

// ── Helpers ─────────────────────────────────────────────────────────────────

fn hann_window(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos()))
        .collect()
}

/// Compute power spectrum magnitude² for a windowed frame.
fn power_spectrum(frame: &[f64], n_fft: usize) -> Vec<f64> {
    let n_freq = n_fft / 2 + 1;
    let mut power = vec![0.0; n_freq];
    for k in 0..n_freq {
        let mut re = 0.0;
        let mut im = 0.0;
        for (t, &s) in frame.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * k as f64 * t as f64 / n_fft as f64;
            re += s * angle.cos();
            im += s * angle.sin();
        }
        power[k] = re * re + im * im;
    }
    power
}

/// Dot product.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// L2 norm squared.
fn norm_sq(a: &[f64]) -> f64 {
    a.iter().map(|&x| x * x).sum()
}

/// Convert Hz to Bark scale: bark = 13 * arctan(0.00076 * f) + 3.5 * arctan((f/7500)^2)
fn hz_to_bark(f: f64) -> f64 {
    13.0 * (0.00076 * f).atan() + 3.5 * (f / 7500.0).powi(2).atan()
}

// ── SI-SDR ──────────────────────────────────────────────────────────────────

/// Scale-invariant signal-to-distortion ratio (SI-SDR) in dB.
///
/// Defined as:
///   s_target = (<s, s_hat> / <s, s>) * s
///   e_noise  = s_hat - s_target
///   SI-SDR   = 10 * log10(||s_target||² / ||e_noise||²)
///
/// Returns `f64::INFINITY` if reference equals estimate (zero distortion).
/// Returns `f64::NEG_INFINITY` if reference is silent.
pub fn si_sdr(reference: &[f64], estimate: &[f64]) -> f64 {
    let n = reference.len().min(estimate.len());
    if n == 0 {
        return f64::NEG_INFINITY;
    }

    let ref_slice = &reference[..n];
    let est_slice = &estimate[..n];

    let ref_energy = norm_sq(ref_slice);
    if ref_energy < 1e-30 {
        return f64::NEG_INFINITY;
    }

    let proj = dot(ref_slice, est_slice) / ref_energy;

    // s_target = proj * reference
    let target_energy = proj * proj * ref_energy;

    // e_noise = estimate - s_target
    let noise_energy: f64 = ref_slice
        .iter()
        .zip(est_slice.iter())
        .map(|(&r, &e)| {
            let t = proj * r;
            (e - t) * (e - t)
        })
        .sum();

    if noise_energy < 1e-30 {
        return f64::INFINITY;
    }

    10.0 * (target_energy / noise_energy).log10()
}

// ── SNR improvement ─────────────────────────────────────────────────────────

/// Compute input SNR: 10 * log10(||clean||² / ||noisy - clean||²).
fn compute_snr(clean: &[f64], signal: &[f64]) -> f64 {
    let n = clean.len().min(signal.len());
    if n == 0 {
        return f64::NEG_INFINITY;
    }

    let clean_energy = norm_sq(&clean[..n]);
    if clean_energy < 1e-30 {
        return f64::NEG_INFINITY;
    }

    let noise_energy: f64 = clean[..n]
        .iter()
        .zip(signal[..n].iter())
        .map(|(&c, &s)| (s - c) * (s - c))
        .sum();

    if noise_energy < 1e-30 {
        return f64::INFINITY;
    }

    10.0 * (clean_energy / noise_energy).log10()
}

/// SNR improvement: SNR_out - SNR_in (in dB).
///
/// Positive values indicate the enhancement improved the SNR.
pub fn snr_improvement(clean: &[f64], noisy: &[f64], enhanced: &[f64]) -> f64 {
    let snr_in = compute_snr(clean, noisy);
    let snr_out = compute_snr(clean, enhanced);
    snr_out - snr_in
}

// ── STOI (Short-Time Objective Intelligibility) ─────────────────────────────

/// Short-time objective intelligibility (STOI) estimate.
///
/// Returns a value in [0, 1] where 1 indicates perfect intelligibility.
/// This is a simplified implementation using 1/3-octave band analysis.
pub fn stoi(reference: &[f64], degraded: &[f64], sample_rate: u32) -> f64 {
    let n = reference.len().min(degraded.len());
    if n == 0 {
        return 0.0;
    }

    let sr = sample_rate as f64;
    // STOI parameters
    let frame_len = (0.025 * sr) as usize; // 25 ms frames
    let hop = (0.010 * sr) as usize; // 10 ms hop
    let frame_len = frame_len.max(16);
    let hop = hop.max(1);
    let n_fft = frame_len.next_power_of_two();

    let window = hann_window(frame_len);

    // 1/3-octave center frequencies from 150 Hz to min(sr/2, 4500 Hz)
    let f_max = (sr / 2.0).min(4500.0);
    let mut center_freqs = Vec::new();
    let mut f = 150.0;
    while f <= f_max {
        center_freqs.push(f);
        f *= 2.0_f64.powf(1.0 / 3.0);
    }

    if center_freqs.is_empty() {
        return 0.0;
    }

    let n_bands = center_freqs.len();

    // Compute band energies for each frame
    let n_frames = if n >= frame_len {
        (n - frame_len) / hop + 1
    } else {
        return 0.0;
    };

    if n_frames == 0 {
        return 0.0;
    }

    let n_freq = n_fft / 2 + 1;
    let freq_res = sr / n_fft as f64;

    // 1/3-octave filter bank: for each band, sum bins in the range
    let band_indices: Vec<(usize, usize)> = center_freqs
        .iter()
        .map(|&fc| {
            let f_low = fc / 2.0_f64.powf(1.0 / 6.0);
            let f_high = fc * 2.0_f64.powf(1.0 / 6.0);
            let k_low = (f_low / freq_res).ceil() as usize;
            let k_high = ((f_high / freq_res).floor() as usize).min(n_freq - 1);
            (k_low.min(n_freq - 1), k_high)
        })
        .collect();

    let mut ref_bands = vec![vec![0.0; n_frames]; n_bands];
    let mut deg_bands = vec![vec![0.0; n_frames]; n_bands];

    for frame in 0..n_frames {
        let start = frame * hop;

        // Window and compute power spectrum
        let ref_frame: Vec<f64> = (0..frame_len)
            .map(|t| {
                let idx = start + t;
                if idx < n {
                    reference[idx] * window[t]
                } else {
                    0.0
                }
            })
            .collect();
        let deg_frame: Vec<f64> = (0..frame_len)
            .map(|t| {
                let idx = start + t;
                if idx < n {
                    degraded[idx] * window[t]
                } else {
                    0.0
                }
            })
            .collect();

        let ref_pow = power_spectrum(&ref_frame, n_fft);
        let deg_pow = power_spectrum(&deg_frame, n_fft);

        for (band, &(k_low, k_high)) in band_indices.iter().enumerate() {
            let mut ref_e = 0.0;
            let mut deg_e = 0.0;
            for k in k_low..=k_high {
                if k < n_freq {
                    ref_e += ref_pow[k];
                    deg_e += deg_pow[k];
                }
            }
            ref_bands[band][frame] = ref_e.sqrt();
            deg_bands[band][frame] = deg_e.sqrt();
        }
    }

    // STOI: average normalized correlation across bands and frame segments
    let seg_len = 15.min(n_frames); // ~150 ms segments
    if seg_len == 0 {
        return 0.0;
    }
    let n_segments = if n_frames >= seg_len {
        n_frames - seg_len + 1
    } else {
        return 0.0;
    };

    let mut total_corr = 0.0;
    let mut count = 0.0;

    for band in 0..n_bands {
        for seg in 0..n_segments {
            let ref_seg = &ref_bands[band][seg..seg + seg_len];
            let deg_seg = &deg_bands[band][seg..seg + seg_len];

            // Normalize and clip
            let ref_mean: f64 = ref_seg.iter().sum::<f64>() / seg_len as f64;
            let deg_mean: f64 = deg_seg.iter().sum::<f64>() / seg_len as f64;

            let ref_norm: Vec<f64> = ref_seg.iter().map(|&v| v - ref_mean).collect();
            let deg_norm: Vec<f64> = deg_seg.iter().map(|&v| v - deg_mean).collect();

            let ref_std = norm_sq(&ref_norm).sqrt();
            let deg_std = norm_sq(&deg_norm).sqrt();

            if ref_std > 1e-10 && deg_std > 1e-10 {
                let corr = dot(&ref_norm, &deg_norm) / (ref_std * deg_std);
                // Clip to [-1, 1]
                let corr = corr.max(-1.0).min(1.0);
                total_corr += corr;
                count += 1.0;
            }
        }
    }

    if count < 1.0 {
        return 0.0;
    }

    // Map average correlation to [0, 1]
    let avg = total_corr / count;
    // STOI uses the intermediate intelligibility metric d = average correlation
    // then maps through a logistic function, but for simplicity we clip to [0,1]
    ((avg + 1.0) / 2.0).max(0.0).min(1.0)
}

// ── PESQ estimate ───────────────────────────────────────────────────────────

/// Simplified PESQ (Perceptual Evaluation of Speech Quality) estimate.
///
/// Based on bark-scale disturbance measurement. Returns a value
/// approximately in [1.0, 4.5] (MOS-LQO scale).
///
/// This is a simplified approximation; for ITU-T P.862 compliance,
/// use a certified implementation.
pub fn pesq_estimate(reference: &[f64], degraded: &[f64], sample_rate: u32) -> f64 {
    let n = reference.len().min(degraded.len());
    if n == 0 {
        return 1.0; // Worst score
    }

    let sr = sample_rate as f64;
    let frame_len = (0.032 * sr) as usize; // 32 ms frames
    let hop = (0.016 * sr) as usize; // 16 ms hop
    let frame_len = frame_len.max(16);
    let hop = hop.max(1);
    let n_fft = frame_len.next_power_of_two();
    let n_freq = n_fft / 2 + 1;

    let window = hann_window(frame_len);

    let n_frames = if n >= frame_len {
        (n - frame_len) / hop + 1
    } else {
        return 1.0;
    };

    if n_frames == 0 {
        return 1.0;
    }

    // Bark bands: 0-24 Bark
    let n_bark = 24;
    let freq_res = sr / n_fft as f64;

    // Build bark filter bank
    let mut bark_bands: Vec<(usize, usize)> = Vec::with_capacity(n_bark);
    for b in 0..n_bark {
        let bark_low = b as f64;
        let bark_high = (b + 1) as f64;
        // Find frequency range for this bark band
        // Inverse bark: approximate
        let f_low = bark_to_hz(bark_low);
        let f_high = bark_to_hz(bark_high);
        let k_low = (f_low / freq_res).ceil() as usize;
        let k_high = ((f_high / freq_res).floor() as usize).min(n_freq - 1);
        bark_bands.push((k_low.min(n_freq - 1), k_high));
    }

    let mut total_disturbance = 0.0;

    for frame in 0..n_frames {
        let start = frame * hop;

        let ref_frame: Vec<f64> = (0..frame_len)
            .map(|t| {
                let idx = start + t;
                if idx < n {
                    reference[idx] * window[t]
                } else {
                    0.0
                }
            })
            .collect();
        let deg_frame: Vec<f64> = (0..frame_len)
            .map(|t| {
                let idx = start + t;
                if idx < n {
                    degraded[idx] * window[t]
                } else {
                    0.0
                }
            })
            .collect();

        let ref_pow = power_spectrum(&ref_frame, n_fft);
        let deg_pow = power_spectrum(&deg_frame, n_fft);

        // Bark-domain disturbance
        let mut frame_dist = 0.0;
        for &(k_low, k_high) in &bark_bands {
            let mut ref_e = 0.0;
            let mut deg_e = 0.0;
            for k in k_low..=k_high {
                if k < n_freq {
                    ref_e += ref_pow[k];
                    deg_e += deg_pow[k];
                }
            }
            // Loudness-domain disturbance (Zwicker loudness approximation)
            let ref_loud = ref_e.powf(0.3);
            let deg_loud = deg_e.powf(0.3);
            let dist = (ref_loud - deg_loud).abs();
            frame_dist += dist;
        }
        total_disturbance += frame_dist / n_bark as f64;
    }

    let avg_disturbance = total_disturbance / n_frames as f64;

    // Map disturbance to MOS-LQO scale [1, 4.5]
    // Lower disturbance → higher quality
    let mos = 4.5 - 3.5 * (1.0 - (-avg_disturbance * 2.0).exp());
    mos.max(1.0).min(4.5)
}

/// Approximate inverse Bark → Hz.
fn bark_to_hz(bark: f64) -> f64 {
    // Traunmüller's formula (approximation)
    let z = bark;
    if z < 2.0 {
        z * 100.0
    } else if z > 20.1 {
        // High bark values
        let hz = (z + 0.53) / (26.28 - z) * 1960.0;
        hz.max(0.0)
    } else {
        (z + 0.53) / (26.28 - z) * 1960.0
    }
}

// ── Spectral convergence ────────────────────────────────────────────────────

/// Spectral convergence: ||S_ref| - |S_est||_F / ||S_ref||_F
///
/// Returns a value in [0, 1] (approximately) where 0 means perfect match.
/// Uses STFT magnitude for comparison.
pub fn spectral_convergence(reference: &[f64], estimate: &[f64]) -> f64 {
    let n = reference.len().min(estimate.len());
    if n == 0 {
        return 1.0;
    }

    let n_fft = 512.min(n.next_power_of_two());
    let hop = n_fft / 4;
    let window = hann_window(n_fft);
    let n_freq = n_fft / 2 + 1;

    let n_frames = if n >= n_fft { (n - n_fft) / hop + 1 } else { 1 };

    let mut diff_sum = 0.0;
    let mut ref_sum = 0.0;

    for frame in 0..n_frames {
        let start = frame * hop;
        for k in 0..n_freq {
            let mut ref_re = 0.0;
            let mut ref_im = 0.0;
            let mut est_re = 0.0;
            let mut est_im = 0.0;

            for t in 0..n_fft {
                let idx = start + t;
                let ref_s = if idx < n { reference[idx] } else { 0.0 };
                let est_s = if idx < n { estimate[idx] } else { 0.0 };
                let w = if t < window.len() { window[t] } else { 0.0 };
                let angle = -2.0 * std::f64::consts::PI * k as f64 * t as f64 / n_fft as f64;
                let cos_a = angle.cos();
                let sin_a = angle.sin();

                ref_re += ref_s * w * cos_a;
                ref_im += ref_s * w * sin_a;
                est_re += est_s * w * cos_a;
                est_im += est_s * w * sin_a;
            }

            let ref_mag = (ref_re * ref_re + ref_im * ref_im).sqrt();
            let est_mag = (est_re * est_re + est_im * est_im).sqrt();
            let diff = ref_mag - est_mag;

            diff_sum += diff * diff;
            ref_sum += ref_mag * ref_mag;
        }
    }

    if ref_sum < 1e-30 {
        return if diff_sum < 1e-30 { 0.0 } else { 1.0 };
    }

    (diff_sum / ref_sum).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_signal(freq: f64, sr: u32, dur: f64) -> Vec<f64> {
        let n = (sr as f64 * dur) as usize;
        (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr as f64).sin())
            .collect()
    }

    fn add_noise(signal: &[f64], noise_level: f64, seed: u64) -> Vec<f64> {
        let mut lcg = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        signal
            .iter()
            .map(|&s| {
                lcg = lcg
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = ((lcg >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 2.0 * noise_level;
                s + noise
            })
            .collect()
    }

    #[test]
    fn test_si_sdr_identical() {
        let x = sine_signal(440.0, 8000, 0.1);
        let sdr = si_sdr(&x, &x);
        assert!(
            sdr > 100.0,
            "SI-SDR of identical signals should be very large, got {sdr}"
        );
    }

    #[test]
    fn test_si_sdr_with_noise() {
        let x = sine_signal(440.0, 8000, 0.1);
        let noise: Vec<f64> = add_noise(&vec![0.0; x.len()], 1.0, 42);
        let sdr = si_sdr(&x, &noise);
        assert!(
            sdr < 0.0,
            "SI-SDR with independent noise should be negative, got {sdr}"
        );
    }

    #[test]
    fn test_stoi_range() {
        let x = sine_signal(440.0, 16000, 0.5);
        let noisy = add_noise(&x, 0.3, 99);
        let s = stoi(&x, &noisy, 16000);
        assert!(s >= 0.0 && s <= 1.0, "STOI must be in [0,1], got {s}");
    }

    #[test]
    fn test_stoi_identical() {
        let x = sine_signal(440.0, 16000, 0.5);
        let s = stoi(&x, &x, 16000);
        assert!(
            s > 0.9,
            "STOI of identical signals should be close to 1, got {s}"
        );
    }

    #[test]
    fn test_snr_improvement_positive() {
        let clean = sine_signal(440.0, 8000, 0.2);
        let noisy = add_noise(&clean, 0.5, 123);
        // "Enhanced" is just clean + tiny noise (much better than noisy)
        let enhanced = add_noise(&clean, 0.01, 456);
        let imp = snr_improvement(&clean, &noisy, &enhanced);
        assert!(imp > 0.0, "SNR improvement should be positive, got {imp}");
    }

    #[test]
    fn test_spectral_convergence_identical() {
        let x = sine_signal(440.0, 8000, 0.1);
        let sc = spectral_convergence(&x, &x);
        assert!(
            sc < 0.01,
            "Spectral convergence of identical signals should be ~0, got {sc}"
        );
    }

    #[test]
    fn test_spectral_convergence_range() {
        let x = sine_signal(440.0, 8000, 0.1);
        let y = sine_signal(880.0, 8000, 0.1);
        let sc = spectral_convergence(&x, &y);
        assert!(
            sc >= 0.0,
            "Spectral convergence should be non-negative, got {sc}"
        );
        // For very different signals it should be closer to 1
        assert!(
            sc <= 2.0,
            "Spectral convergence should be bounded, got {sc}"
        );
    }

    #[test]
    fn test_pesq_estimate_basic() {
        let x = sine_signal(440.0, 16000, 0.5);
        let noisy = add_noise(&x, 0.3, 77);
        let pesq = pesq_estimate(&x, &noisy, 16000);
        assert!(
            pesq >= 1.0 && pesq <= 4.5,
            "PESQ should be in [1, 4.5], got {pesq}"
        );
    }

    #[test]
    fn test_pesq_identical() {
        let x = sine_signal(440.0, 16000, 0.5);
        let pesq = pesq_estimate(&x, &x, 16000);
        assert!(
            pesq > 3.0,
            "PESQ of identical signals should be high, got {pesq}"
        );
    }

    #[test]
    fn test_si_sdr_empty() {
        assert_eq!(si_sdr(&[], &[]), f64::NEG_INFINITY);
    }
}
