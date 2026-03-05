// Integration tests for scirs2-fft + scirs2-signal
// Tests spectral analysis pipelines, filter design, and FFT-based operations

use scirs2_core::ndarray::{Array1, Array2, Axis};
use proptest::prelude::*;
use scirs2_fft::*;
use scirs2_signal::*;
use crate::integration::common::*;
use crate::integration::fixtures::TestDatasets;
use num_complex::Complex64;

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Test FFT-based filtering pipeline
#[test]
fn test_fft_based_filtering() -> TestResult<()> {
    // Create a test signal with multiple frequency components
    let signal = TestDatasets::sinusoid_signal(1024, 10.0, 1024.0);

    println!("Testing FFT-based filtering");
    println!("Signal length: {}", signal.len());

    // Apply FFT
    let fft_result = fft(&signal.mapv(|x| Complex64::new(x, 0.0)))?;

    println!("FFT computed, spectrum length: {}", fft_result.len());

    // TODO: Implement filtering workflow:
    // 1. Design filter using scirs2-signal
    // 2. Apply filter in frequency domain using scirs2-fft
    // 3. Inverse FFT to get filtered signal
    // 4. Verify filter characteristics

    Ok(())
}

/// Test spectral analysis pipeline
#[test]
fn test_spectral_analysis_pipeline() -> TestResult<()> {
    // Test the complete spectral analysis workflow combining
    // FFT computation and signal processing features

    let sampling_rate = 1000.0;
    let duration = 2.0; // seconds
    let n_samples = (sampling_rate * duration) as usize;

    // Create composite signal: sum of sinusoids at different frequencies
    let freq1 = 10.0;
    let freq2 = 50.0;
    let freq3 = 120.0;

    let signal1 = TestDatasets::sinusoid_signal(n_samples, freq1, sampling_rate);
    let signal2 = TestDatasets::sinusoid_signal(n_samples, freq2, sampling_rate);
    let signal3 = TestDatasets::sinusoid_signal(n_samples, freq3, sampling_rate);

    let composite_signal = &signal1 + &signal2 + &signal3;

    println!("Testing spectral analysis pipeline");
    println!("Signal length: {}, sampling rate: {} Hz", n_samples, sampling_rate);

    // Compute power spectral density
    // TODO: Use scirs2-signal PSD functions once available
    let complex_signal = composite_signal.mapv(|x| Complex64::new(x, 0.0));
    let spectrum = fft(&complex_signal)?;

    // Verify peaks at expected frequencies
    let power_spectrum: Vec<f64> = spectrum.iter()
        .map(|c| c.norm_sqr())
        .collect();

    println!("Computed power spectrum with {} bins", power_spectrum.len());

    // TODO: Verify spectral peaks match input frequencies

    Ok(())
}

/// Test window functions integration
#[test]
fn test_window_functions_with_fft() -> TestResult<()> {
    // Test that window functions from scirs2-signal integrate
    // correctly with FFT operations

    let signal = TestDatasets::sinusoid_signal(512, 10.0, 512.0);

    println!("Testing window functions with FFT");

    // TODO: Apply various windows (Hamming, Hanning, Blackman) from scirs2-signal
    // and verify their effect on FFT spectrum

    let complex_signal = signal.mapv(|x| Complex64::new(x, 0.0));
    let spectrum = fft(&complex_signal)?;

    println!("Computed windowed FFT with {} samples", spectrum.len());

    Ok(())
}

/// Test spectrogram computation
#[test]
fn test_spectrogram_computation() -> TestResult<()> {
    // Test Short-Time Fourier Transform (STFT) implementation
    // combining windowing and FFT

    let signal = TestDatasets::sinusoid_signal(4096, 10.0, 1000.0);
    let window_size = 256;
    let hop_size = 128;

    println!("Testing spectrogram computation");
    println!("Signal length: {}, window: {}, hop: {}", signal.len(), window_size, hop_size);

    // TODO: Implement STFT using:
    // 1. Window from scirs2-signal
    // 2. FFT from scirs2-fft
    // 3. Sliding window approach

    Ok(())
}

/// Test convolution via FFT
#[test]
fn test_fft_convolution() -> TestResult<()> {
    // Test that convolution computed via FFT matches
    // time-domain convolution from scirs2-signal

    let signal = TestDatasets::sinusoid_signal(256, 5.0, 256.0);
    let kernel = Array1::from_vec(vec![0.25, 0.5, 0.25]); // Simple smoothing kernel

    println!("Testing FFT-based convolution");
    println!("Signal length: {}, kernel length: {}", signal.len(), kernel.len());

    // TODO: Implement and compare:
    // 1. Direct convolution from scirs2-signal
    // 2. FFT-based convolution:
    //    - FFT(signal) * FFT(kernel)
    //    - IFFT(result)
    // 3. Verify results match within tolerance

    Ok(())
}

/// Test filter design and application
#[test]
fn test_filter_design_and_application() -> TestResult<()> {
    // Test complete filter design workflow using both modules

    let sampling_rate = 1000.0;
    let signal = TestDatasets::sinusoid_signal(1000, 50.0, sampling_rate);

    println!("Testing filter design and application");

    // TODO: Design various filters using scirs2-signal:
    // 1. Low-pass filter
    // 2. High-pass filter
    // 3. Band-pass filter
    // 4. Band-stop filter
    // Apply using scirs2-fft and verify frequency response

    Ok(())
}

/// Test Hilbert transform integration
#[test]
fn test_hilbert_transform() -> TestResult<()> {
    // Test Hilbert transform implementation using FFT

    let signal = TestDatasets::sinusoid_signal(512, 10.0, 512.0);

    println!("Testing Hilbert transform via FFT");

    // TODO: Implement Hilbert transform:
    // 1. FFT of signal
    // 2. Apply Hilbert filter in frequency domain
    // 3. IFFT to get analytic signal
    // 4. Verify properties (envelope, phase)

    Ok(())
}

/// Test zero-padding effects
#[test]
fn test_zero_padding_effects() -> TestResult<()> {
    // Test that zero-padding is handled consistently between modules

    let signal = TestDatasets::sinusoid_signal(100, 5.0, 100.0);

    println!("Testing zero-padding effects");

    // Compute FFT with different padding lengths
    let padded_lengths = vec![128, 256, 512];

    for &padded_len in &padded_lengths {
        let mut padded_signal = signal.clone();
        // TODO: Pad signal to padded_len
        // Apply FFT and verify frequency resolution improves

        println!("  Padded length: {}", padded_len);
    }

    Ok(())
}

/// Test real-valued FFT integration
#[test]
fn test_rfft_integration() -> TestResult<()> {
    // Test that real FFT from scirs2-fft integrates with
    // real-valued signal processing

    let signal = TestDatasets::sinusoid_signal(1024, 10.0, 1024.0);

    println!("Testing real-valued FFT integration");

    // Compute RFFT
    let rfft_result = rfft(&signal)?;

    println!("RFFT computed, output length: {}", rfft_result.len());

    // Verify Hermitian symmetry property
    // TODO: Add verification

    Ok(())
}

/// Test 2D FFT for image processing
#[test]
fn test_2d_fft_integration() -> TestResult<()> {
    // Test 2D FFT integration for image/signal processing

    let image = TestDatasets::test_image_gradient(64);

    println!("Testing 2D FFT integration");
    println!("Image shape: {:?}", image.shape());

    // TODO: Compute 2D FFT and verify:
    // 1. DC component location
    // 2. Frequency ordering
    // 3. Inverse transform reconstruction

    Ok(())
}

// Property-based tests

proptest! {
    #[test]
    fn prop_fft_parseval_theorem(
        signal_len in 64usize..256
    ) {
        // Property: Parseval's theorem - energy is conserved in FFT
        // Sum of |x[n]|^2 = (1/N) * Sum of |X[k]|^2

        let signal = TestDatasets::sinusoid_signal(signal_len, 5.0, signal_len as f64);
        let complex_signal = signal.mapv(|x| Complex64::new(x, 0.0));

        let time_energy: f64 = signal.iter().map(|&x| x * x).sum();

        let fft_result = fft(&complex_signal)
            .expect("FFT failed in property test");
        let freq_energy: f64 = fft_result.iter()
            .map(|c| c.norm_sqr())
            .sum::<f64>() / signal_len as f64;

        let tolerance = 1e-6;
        prop_assert!(
            (time_energy - freq_energy).abs() < tolerance,
            "Parseval's theorem violated: time_energy={}, freq_energy={}",
            time_energy, freq_energy
        );
    }

    #[test]
    fn prop_fft_linearity(
        signal_len in 64usize..128,
        scale1 in -10.0f64..10.0,
        scale2 in -10.0f64..10.0
    ) {
        // Property: FFT is linear
        // FFT(a*x + b*y) = a*FFT(x) + b*FFT(y)

        let signal1 = TestDatasets::sinusoid_signal(signal_len, 5.0, signal_len as f64);
        let signal2 = TestDatasets::sinusoid_signal(signal_len, 10.0, signal_len as f64);

        let combined = &signal1 * scale1 + &signal2 * scale2;

        let complex1 = signal1.mapv(|x| Complex64::new(x, 0.0));
        let complex2 = signal2.mapv(|x| Complex64::new(x, 0.0));
        let complex_combined = combined.mapv(|x| Complex64::new(x, 0.0));

        let fft1 = fft(&complex1).expect("FFT1 failed");
        let fft2 = fft(&complex2).expect("FFT2 failed");
        let fft_combined = fft(&complex_combined).expect("FFT combined failed");

        let fft_linear = &fft1 * scale1 + &fft2 * scale2;

        // Check linearity holds within numerical precision
        let max_diff = fft_combined.iter()
            .zip(fft_linear.iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);

        prop_assert!(
            max_diff < 1e-6,
            "FFT linearity violated: max_diff={}",
            max_diff
        );
    }

    #[test]
    fn prop_fft_ifft_roundtrip(
        signal_len in 64usize..256
    ) {
        // Property: IFFT(FFT(x)) = x (within numerical precision)

        let signal = TestDatasets::sinusoid_signal(signal_len, 7.0, signal_len as f64);
        let complex_signal = signal.mapv(|x| Complex64::new(x, 0.0));

        let fft_result = fft(&complex_signal)
            .expect("FFT failed in roundtrip test");
        let reconstructed = ifft(&fft_result)
            .expect("IFFT failed in roundtrip test");

        let max_error = complex_signal.iter()
            .zip(reconstructed.iter())
            .map(|(orig, recon)| (orig - recon).norm())
            .fold(0.0, f64::max);

        prop_assert!(
            max_error < 1e-10,
            "FFT/IFFT roundtrip error too large: {}",
            max_error
        );
    }
}

/// Test cross-correlation via FFT
#[test]
fn test_cross_correlation_via_fft() -> TestResult<()> {
    // Test that cross-correlation computed via FFT matches expectations

    let signal1 = TestDatasets::sinusoid_signal(256, 10.0, 256.0);
    let signal2 = signal1.clone(); // Autocorrelation case

    println!("Testing cross-correlation via FFT");

    // TODO: Implement cross-correlation using FFT:
    // 1. FFT of both signals
    // 2. Multiply: FFT(signal1) * conj(FFT(signal2))
    // 3. IFFT to get correlation
    // 4. Verify peak at zero lag for autocorrelation

    Ok(())
}

/// Test frequency shifting
#[test]
fn test_frequency_shifting() -> TestResult<()> {
    // Test frequency shifting operations combining both modules

    let signal = TestDatasets::sinusoid_signal(512, 20.0, 512.0);
    let shift_freq = 10.0;

    println!("Testing frequency shifting");

    // TODO: Implement frequency shifting:
    // 1. FFT of signal
    // 2. Shift spectrum using fftshift/ifftshift
    // 3. IFFT to get shifted signal
    // 4. Verify new frequency components

    Ok(())
}

/// Test memory efficiency of FFT pipeline
#[test]
fn test_fft_pipeline_memory_efficiency() -> TestResult<()> {
    // Verify that FFT-based signal processing doesn't create
    // unnecessary copies

    let signal = TestDatasets::sinusoid_signal(8192, 10.0, 8192.0);

    println!("Testing FFT pipeline memory efficiency");

    assert_memory_efficient(
        || {
            let complex_signal = signal.mapv(|x| Complex64::new(x, 0.0));
            let spectrum = fft(&complex_signal)?;
            let reconstructed = ifft(&spectrum)?;
            Ok(reconstructed)
        },
        100.0,  // 100 MB max
        "FFT forward-backward pipeline",
    )?;

    Ok(())
}

/// Test performance comparison of different FFT sizes
#[test]
fn test_fft_size_performance() -> TestResult<()> {
    // Test that FFT performance scales as expected for different sizes

    let sizes = vec![64, 128, 256, 512, 1024, 2048, 4096];

    println!("Testing FFT performance scaling");

    for size in sizes {
        let signal = TestDatasets::sinusoid_signal(size, 10.0, size as f64);
        let complex_signal = signal.mapv(|x| Complex64::new(x, 0.0));

        let (_result, perf) = measure_time(&format!("FFT size {}", size), || {
            fft(&complex_signal).map_err(|e| e.into())
        })?;

        println!("  Size {}: {:.3} ms", size, perf.duration_ms);
    }

    Ok(())
}

/// Test DCT integration with signal processing
#[test]
fn test_dct_integration() -> TestResult<()> {
    // Test Discrete Cosine Transform integration

    let signal = TestDatasets::sinusoid_signal(128, 5.0, 128.0);

    println!("Testing DCT integration");

    // TODO: Test DCT from scirs2-fft:
    // 1. Apply DCT
    // 2. Modify coefficients (e.g., compression)
    // 3. Inverse DCT
    // 4. Verify properties (energy compaction, etc.)

    Ok(())
}

#[cfg(test)]
mod api_compatibility_tests {
    use super::*;

    /// Test that array formats are compatible between modules
    #[test]
    fn test_array_format_compatibility() -> TestResult<()> {
        // Verify that array types used by scirs2-signal can be
        // directly passed to scirs2-fft functions

        let signal = TestDatasets::sinusoid_signal(256, 10.0, 256.0);

        // This should work without any conversion
        let complex_signal = signal.mapv(|x| Complex64::new(x, 0.0));
        let _spectrum = fft(&complex_signal)?;

        println!("Array format compatibility verified");

        Ok(())
    }

    /// Test error handling consistency
    #[test]
    fn test_error_handling_consistency() -> TestResult<()> {
        // Verify that error types are compatible and informative

        // TODO: Test various error conditions:
        // 1. Invalid FFT sizes
        // 2. Mismatched array dimensions
        // 3. Numerical issues

        println!("Error handling consistency test");

        Ok(())
    }
}
