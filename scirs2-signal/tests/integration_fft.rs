//! Integration tests: scirs2-signal + scirs2-fft
//!
//! Covers:
//! - Signal filtering via FFT-based convolution
//! - Spectrogram computation and frequency localization
//! - Overlap-add convolution gives same result as direct convolution
//! - FFT filtering: low-pass filter in frequency domain
//! - Round-trip FFT: signal → spectrum → signal

use approx::assert_abs_diff_eq;
use scirs2_fft::{fftfreq, irfft, rfft};
use scirs2_signal::{
    filter::{butter, filtfilt, lfilter, FilterType},
    spectral::{spectrogram, stft},
};
use std::f64::consts::PI;

/// FFT-based convolution (full mode) - local implementation for testing
fn fft_convolve(a: &[f64], b: &[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let full_len = a.len() + b.len() - 1;
    let fft_len = full_len.next_power_of_two();

    let mut a_pad = vec![0.0; fft_len];
    a_pad[..a.len()].copy_from_slice(a);
    let mut b_pad = vec![0.0; fft_len];
    b_pad[..b.len()].copy_from_slice(b);

    let a_freq = scirs2_fft::fft(&a_pad, Some(fft_len))?;
    let b_freq = scirs2_fft::fft(&b_pad, Some(fft_len))?;

    let product: Vec<scirs2_core::numeric::Complex64> = a_freq
        .iter()
        .zip(b_freq.iter())
        .map(|(&af, &bf)| af * bf)
        .collect();

    let result = scirs2_fft::ifft(&product, Some(fft_len))?;
    Ok(result.iter().take(full_len).map(|c| c.re).collect())
}

/// Overlap-add convolution - local implementation for testing
fn overlap_add_convolve(
    signal: &[f64],
    kernel: &[f64],
    block_size: usize,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let n = signal.len();
    let m = kernel.len();
    let out_len = n + m - 1;
    let mut output = vec![0.0; out_len];

    let fft_size = (block_size + m - 1).next_power_of_two();

    // Pre-compute kernel FFT
    let mut k_pad = vec![0.0; fft_size];
    k_pad[..m].copy_from_slice(kernel);
    let k_freq = scirs2_fft::fft(&k_pad, Some(fft_size))?;

    let mut pos = 0;
    while pos < n {
        let end = (pos + block_size).min(n);
        let mut block = vec![0.0; fft_size];
        block[..end - pos].copy_from_slice(&signal[pos..end]);

        let block_freq = scirs2_fft::fft(&block, Some(fft_size))?;
        let product: Vec<scirs2_core::numeric::Complex64> = block_freq
            .iter()
            .zip(k_freq.iter())
            .map(|(&bf, &kf)| bf * kf)
            .collect();

        let result = scirs2_fft::ifft(&product, Some(fft_size))?;
        let seg_len = (end - pos + m - 1).min(fft_size);
        for i in 0..seg_len {
            if pos + i < out_len {
                output[pos + i] += result[i].re;
            }
        }
        pos += block_size;
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Helper: generate a sinusoidal test signal
// ---------------------------------------------------------------------------

fn generate_sine(freq: f64, fs: f64, n_samples: usize) -> Vec<f64> {
    (0..n_samples)
        .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
        .collect()
}

fn generate_cosine(freq: f64, fs: f64, n_samples: usize) -> Vec<f64> {
    (0..n_samples)
        .map(|i| (2.0 * PI * freq * i as f64 / fs).cos())
        .collect()
}

// ---------------------------------------------------------------------------
// 1. Round-trip rfft → irfft recovers original signal
// ---------------------------------------------------------------------------

#[test]
fn test_rfft_irfft_round_trip() {
    let fs = 1000.0_f64;
    let n = 128_usize;
    let signal = generate_sine(50.0, fs, n);

    let spectrum = rfft(&signal, None).expect("rfft failed");
    let recovered = irfft(&spectrum, Some(n)).expect("irfft failed");

    assert_eq!(
        recovered.len(),
        n,
        "Recovered signal length mismatch after round-trip"
    );
    for (i, (&orig, &rec)) in signal.iter().zip(recovered.iter()).enumerate() {
        assert_abs_diff_eq!(orig, rec, epsilon = 1e-8);
    }
}

// ---------------------------------------------------------------------------
// 2. FFT spectrum peak matches signal frequency
// ---------------------------------------------------------------------------

#[test]
fn test_fft_spectrum_peak_at_signal_frequency() {
    let fs = 1000.0_f64;
    let freq = 100.0_f64;
    let n = 1024_usize;
    let signal = generate_sine(freq, fs, n);

    let spectrum = rfft(&signal, None).expect("rfft failed");
    let freqs = fftfreq(n, 1.0 / fs).expect("fftfreq failed");

    // Compute magnitudes of the one-sided spectrum
    let n_freqs = n / 2 + 1;
    let magnitudes: Vec<f64> = spectrum[..n_freqs]
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
        .collect();

    // Find peak bin
    let peak_bin = magnitudes
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN in magnitudes"))
        .map(|(i, _)| i)
        .expect("Empty magnitudes");

    // Peak should be at or very near the target frequency
    let peak_freq = freqs[peak_bin].abs();
    assert_abs_diff_eq!(peak_freq, freq, epsilon = fs / n as f64 + 1.0);
}

// ---------------------------------------------------------------------------
// 3. Butterworth low-pass filter attenuates high frequency
// ---------------------------------------------------------------------------

#[test]
fn test_butterworth_lowpass_attenuates_high_frequency() {
    let fs = 1000.0_f64;
    let cutoff = 0.1_f64; // normalized: 0.1 * (fs/2) = 50 Hz
    let n = 512_usize;

    // Low-frequency component (10 Hz < 50 Hz): should pass
    let low_sig = generate_sine(10.0, fs, n);
    // High-frequency component (200 Hz >> 50 Hz): should be attenuated
    let high_sig = generate_sine(200.0, fs, n);

    let (b, a) = butter(4, cutoff, FilterType::Lowpass).expect("butter filter design failed");

    let filtered_low = lfilter(&b, &a, &low_sig).expect("lfilter (low) failed");
    let filtered_high = lfilter(&b, &a, &high_sig).expect("lfilter (high) failed");

    // Compute RMS of each filtered signal (skip first 50 samples to avoid transient)
    let skip = 50_usize;
    let rms = |v: &[f64]| -> f64 {
        let s: f64 = v[skip..].iter().map(|&x| x * x).sum();
        (s / (v.len() - skip) as f64).sqrt()
    };

    let rms_low = rms(&filtered_low);
    let rms_high = rms(&filtered_high);

    // Low frequency passes (RMS near 1/sqrt(2) ≈ 0.707 for sine)
    assert!(rms_low > 0.5, "Low frequency should pass but RMS={rms_low}");
    // High frequency is attenuated (RMS should be much less)
    assert!(
        rms_high < 0.1,
        "High frequency should be attenuated but RMS={rms_high}"
    );
}

// ---------------------------------------------------------------------------
// 4. filtfilt zero-phase filtering is symmetric
// ---------------------------------------------------------------------------

#[test]
fn test_filtfilt_zero_phase() {
    let fs = 1000.0_f64;
    let n = 256_usize;
    // Mixed signal: 20 Hz (pass) + 300 Hz (stop)
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            (2.0 * PI * 20.0 * t).sin() + 0.5 * (2.0 * PI * 300.0 * t).sin()
        })
        .collect();

    let (b, a) = butter(3, 0.1_f64, FilterType::Lowpass).expect("butter failed");
    let filtered = filtfilt(&b, &a, &signal).expect("filtfilt failed");

    assert_eq!(filtered.len(), n, "filtfilt output length mismatch");

    // The filtered signal RMS should be dominated by low frequency
    let skip = 20_usize;
    let rms_filtered: f64 = {
        let s: f64 = filtered[skip..].iter().map(|&x| x * x).sum();
        (s / (n - skip) as f64).sqrt()
    };
    assert!(
        rms_filtered > 0.3,
        "filtfilt output RMS too low: {rms_filtered}"
    );
    assert!(
        rms_filtered < 1.0,
        "filtfilt output RMS unexpectedly high: {rms_filtered}"
    );
}

// ---------------------------------------------------------------------------
// 5. FFT convolution equals direct convolution
// ---------------------------------------------------------------------------

#[test]
fn test_fft_convolve_equals_direct_convolve() {
    // Simple FIR kernel and signal
    let signal: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let kernel: Vec<f64> = vec![0.25, 0.5, 0.25]; // 3-tap smoothing filter

    // Direct (naive) linear convolution
    let n_out = signal.len() + kernel.len() - 1;
    let mut direct = vec![0.0_f64; n_out];
    for (i, &si) in signal.iter().enumerate() {
        for (j, &kj) in kernel.iter().enumerate() {
            direct[i + j] += si * kj;
        }
    }

    // FFT convolution
    let fft_result = fft_convolve(&signal, &kernel).expect("fft_convolve failed");

    assert_eq!(
        fft_result.len(),
        n_out,
        "fft_convolve output length mismatch"
    );
    for (i, (&fft_val, &direct_val)) in fft_result.iter().zip(direct.iter()).enumerate() {
        assert_abs_diff_eq!(fft_val, direct_val, epsilon = 1e-10);
    }
}

// ---------------------------------------------------------------------------
// 6. Overlap-add convolution matches direct convolution
// ---------------------------------------------------------------------------

#[test]
fn test_overlap_add_convolution_correctness() {
    let signal: Vec<f64> = (0..64_usize)
        .map(|i| (2.0 * PI * 0.1 * i as f64).sin())
        .collect();
    let kernel: Vec<f64> = vec![1.0 / 5.0; 5]; // 5-tap moving average

    // Direct convolution (full)
    let n_out = signal.len() + kernel.len() - 1;
    let mut direct = vec![0.0_f64; n_out];
    for (i, &si) in signal.iter().enumerate() {
        for (j, &kj) in kernel.iter().enumerate() {
            direct[i + j] += si * kj;
        }
    }

    // Overlap-add convolution
    let oa_result =
        overlap_add_convolve(&signal, &kernel, 16).expect("overlap_add_convolve failed");

    // Both should have same length
    assert_eq!(
        oa_result.len(),
        direct.len(),
        "overlap_add output length mismatch"
    );

    for (i, (&oa, &dc)) in oa_result.iter().zip(direct.iter()).enumerate() {
        assert_abs_diff_eq!(oa, dc, epsilon = 1e-10);
    }
}

// ---------------------------------------------------------------------------
// 7. STFT: frequency resolution and time steps
// ---------------------------------------------------------------------------

#[test]
fn test_stft_shape_and_frequency_resolution() {
    let fs = 1000.0_f64;
    let n_samples = 512_usize;
    let nperseg = 64_usize;
    let noverlap = 32_usize;

    let signal = generate_sine(100.0, fs, n_samples);

    let (freqs, times, stft_mat) = stft(
        &signal,
        Some(fs),
        Some("hann"),
        Some(nperseg),
        Some(noverlap),
        None,
        None,
        None,
        None,
    )
    .expect("STFT computation failed");

    // Number of frequency bins: nperseg/2 + 1
    let expected_freq_bins = nperseg / 2 + 1;
    assert_eq!(
        freqs.len(),
        expected_freq_bins,
        "Unexpected number of frequency bins"
    );

    // Number of time frames: roughly (n_samples - noverlap) / (nperseg - noverlap)
    assert!(!times.is_empty(), "STFT time axis is empty");

    // STFT matrix should be freqs x times
    assert_eq!(
        stft_mat.len(),
        freqs.len(),
        "STFT matrix row count mismatch"
    );
    for row in &stft_mat {
        assert_eq!(row.len(), times.len(), "STFT matrix column count mismatch");
    }

    // Frequency resolution: fs / nperseg
    let freq_resolution = fs / nperseg as f64;
    assert_abs_diff_eq!(freqs[1] - freqs[0], freq_resolution, epsilon = 1e-6);
}

// ---------------------------------------------------------------------------
// 8. Spectrogram: energy concentrated at signal frequency
// ---------------------------------------------------------------------------

#[test]
fn test_spectrogram_energy_at_signal_frequency() {
    let fs = 1000.0_f64;
    let signal_freq = 100.0_f64;
    let n_samples = 1024_usize;
    let nperseg = 128_usize;

    let signal = generate_sine(signal_freq, fs, n_samples);

    let (freqs, _times, psd_mat) = spectrogram(
        &signal,
        Some(fs),
        Some("hann"),
        Some(nperseg),
        None,
        None,
        None,
        Some("density"),
        Some("psd"),
    )
    .expect("Spectrogram computation failed");

    // Average power across all time frames
    let n_frames = psd_mat[0].len();
    let avg_power: Vec<f64> = psd_mat
        .iter()
        .map(|row| row.iter().sum::<f64>() / n_frames as f64)
        .collect();

    // Find frequency bin nearest to signal frequency
    let target_bin = freqs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            ((*a - signal_freq).abs())
                .partial_cmp(&(*b - signal_freq).abs())
                .expect("NaN in freq comparison")
        })
        .map(|(i, _)| i)
        .expect("Empty freq axis");

    // Peak power should be at or near signal frequency
    let peak_bin = avg_power
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN in power"))
        .map(|(i, _)| i)
        .expect("Empty power spectrum");

    // The peak bin should be within ±2 bins of the target frequency bin
    assert!(
        (peak_bin as isize - target_bin as isize).abs() <= 2,
        "Spectrogram peak at bin {peak_bin} (freq={:.1}Hz) vs expected bin {target_bin} (freq={signal_freq}Hz)",
        freqs[peak_bin]
    );
}

// ---------------------------------------------------------------------------
// 9. FFT: linearity property (FFT(a + b) = FFT(a) + FFT(b))
// ---------------------------------------------------------------------------

#[test]
fn test_fft_linearity() {
    let n = 64_usize;
    let a: Vec<f64> = generate_sine(10.0, 1000.0, n);
    let b: Vec<f64> = generate_cosine(30.0, 1000.0, n);
    let a_plus_b: Vec<f64> = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai + bi).collect();

    let fft_a = rfft(&a, None).expect("rfft(a) failed");
    let fft_b = rfft(&b, None).expect("rfft(b) failed");
    let fft_sum = rfft(&a_plus_b, None).expect("rfft(a+b) failed");

    // FFT(a+b) should equal FFT(a) + FFT(b) (complex sum)
    for (i, (fa, fb, fs)) in itertools_zip(fft_a.iter(), fft_b.iter(), fft_sum.iter()).enumerate() {
        let expected_re = fa.re + fb.re;
        let expected_im = fa.im + fb.im;
        assert_abs_diff_eq!(fs.re, expected_re, epsilon = 1e-10);
        assert_abs_diff_eq!(fs.im, expected_im, epsilon = 1e-10);
    }
}

// Helper to zip 3 iterators
fn itertools_zip<A, B, C>(
    a: impl Iterator<Item = A>,
    b: impl Iterator<Item = B>,
    c: impl Iterator<Item = C>,
) -> impl Iterator<Item = (A, B, C)> {
    a.zip(b).zip(c).map(|((a, b), c)| (a, b, c))
}

// ---------------------------------------------------------------------------
// 10. FFT convolution with delta function is identity
// ---------------------------------------------------------------------------

#[test]
fn test_fft_convolve_with_delta_is_identity() {
    let signal: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let delta: Vec<f64> = vec![1.0]; // single-sample impulse

    let result = fft_convolve(&signal, &delta).expect("fft_convolve with delta failed");

    // Convolution with delta should be identity
    assert_eq!(
        result.len(),
        signal.len(),
        "Length changed after delta convolution"
    );
    for (orig, res) in signal.iter().zip(result.iter()) {
        assert_abs_diff_eq!(*orig, *res, epsilon = 1e-10);
    }
}
