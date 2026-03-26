//! Tutorial: Fast Fourier Transform with SciRS2
//!
//! This tutorial covers computing FFTs, understanding frequency bins,
//! inverse FFTs, real-valued FFTs, and spectral analysis.
//!
//! Run with: cargo run -p scirs2-fft --example tutorial_fft

use scirs2_core::numeric::Complex64;
use scirs2_fft::{fft, fftfreq, ifft, rfft, FFTResult};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SciRS2 FFT Tutorial ===\n");

    section_basic_fft()?;
    section_frequency_bins()?;
    section_inverse_fft()?;
    section_real_fft()?;
    section_spectral_analysis()?;

    println!("\n=== Tutorial Complete ===");
    Ok(())
}

/// Section 1: Computing a basic FFT
fn section_basic_fft() -> FFTResult<()> {
    println!("--- 1. Basic FFT ---\n");

    // Create a simple signal: 8 samples of a cosine wave
    let n = 8;
    let signal: Vec<Complex64> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            Complex64::new((2.0 * PI * t).cos(), 0.0)
        })
        .collect();

    println!("Input signal (real parts):");
    for (i, s) in signal.iter().enumerate() {
        println!("  x[{}] = {:.4}", i, s.re);
    }

    // Compute the FFT
    let spectrum = fft(&signal, None)?;

    println!("\nFFT output (magnitude):");
    for (i, s) in spectrum.iter().enumerate() {
        let mag = (s.re * s.re + s.im * s.im).sqrt();
        println!("  X[{}] = {:.4} (mag = {:.4})", i, s, mag);
    }
    // A cosine at frequency bin 1 produces peaks at bins 1 and N-1
    println!();

    Ok(())
}

/// Section 2: Understanding frequency bins
fn section_frequency_bins() -> FFTResult<()> {
    println!("--- 2. Frequency Bins ---\n");

    let n = 8;
    let sample_rate = 100.0; // 100 Hz sampling rate

    // fftfreq returns the frequencies corresponding to each FFT bin
    let freqs = fftfreq(n, 1.0 / sample_rate)?;
    println!("Frequencies for N={}, fs={} Hz:", n, sample_rate);
    for (i, f) in freqs.iter().enumerate() {
        println!("  bin[{}] = {:.1} Hz", i, f);
    }
    // Bins 0..N/2 are positive frequencies, N/2+1..N-1 are negative
    println!();

    // fftshift rearranges so that zero-frequency is in the center
    // Convert to ndarray for fftshift
    let freqs_arr = scirs2_core::ndarray::Array1::from_vec(freqs.clone());
    let shifted = scirs2_fft::fftshift(&freqs_arr)?;
    println!("After fftshift (zero-centered):");
    for (i, f) in shifted.iter().enumerate() {
        println!("  bin[{}] = {:.1} Hz", i, f);
    }
    println!();

    Ok(())
}

/// Section 3: Inverse FFT (recovering the signal)
fn section_inverse_fft() -> FFTResult<()> {
    println!("--- 3. Inverse FFT ---\n");

    // Create a signal, transform, then invert
    let n = 16;
    let signal: Vec<Complex64> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            // Sum of two cosines: 2 Hz and 5 Hz components
            let val = (2.0 * PI * 2.0 * t).cos() + 0.5 * (2.0 * PI * 5.0 * t).cos();
            Complex64::new(val, 0.0)
        })
        .collect();

    // Forward FFT
    let spectrum = fft(&signal, None)?;

    // Inverse FFT
    let recovered = ifft(&spectrum, None)?;

    // Compare original and recovered
    println!("Original vs Recovered (first 8 samples):");
    for i in 0..8 {
        let diff = (signal[i].re - recovered[i].re).abs();
        println!(
            "  x[{}]: original={:.6}, recovered={:.6}, error={:.2e}",
            i, signal[i].re, recovered[i].re, diff
        );
    }
    println!("  (Round-trip error is near machine epsilon)\n");

    Ok(())
}

/// Section 4: Real-to-complex FFT (rfft) for real-valued signals
fn section_real_fft() -> FFTResult<()> {
    println!("--- 4. Real FFT (rfft) ---\n");

    // For real-valued input, rfft is more efficient because the output
    // is conjugate-symmetric, so only N/2+1 complex values are needed.
    let n = 16;
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (2.0 * PI * 3.0 * t).cos() // 3-cycle cosine
        })
        .collect();

    let spectrum = rfft(&signal, None)?;

    println!("Real signal with N={} samples:", n);
    println!(
        "  rfft output length: {} (N/2+1 = {})",
        spectrum.len(),
        n / 2 + 1
    );
    println!("\n  Magnitudes:");
    for (i, s) in spectrum.iter().enumerate() {
        let mag = (s.re * s.re + s.im * s.im).sqrt();
        if mag > 0.01 {
            println!("  bin[{}] magnitude = {:.4} (significant)", i, mag);
        } else {
            println!("  bin[{}] magnitude = {:.4}", i, mag);
        }
    }
    // Peak at bin 3, corresponding to the 3-cycle cosine
    println!();

    Ok(())
}

/// Section 5: Spectral analysis of a multi-tone signal
fn section_spectral_analysis() -> FFTResult<()> {
    println!("--- 5. Spectral Analysis ---\n");

    let sample_rate = 1000.0; // 1000 Hz
    let n = 1024;
    let dt = 1.0 / sample_rate;

    // Create a signal with 3 frequency components:
    //   50 Hz (amplitude 1.0), 120 Hz (amplitude 0.5), 300 Hz (amplitude 0.3)
    let signal: Vec<Complex64> = (0..n)
        .map(|i| {
            let t = i as f64 * dt;
            let val = 1.0 * (2.0 * PI * 50.0 * t).cos()
                + 0.5 * (2.0 * PI * 120.0 * t).cos()
                + 0.3 * (2.0 * PI * 300.0 * t).cos();
            Complex64::new(val, 0.0)
        })
        .collect();

    let spectrum = fft(&signal, None)?;
    let freqs = fftfreq(n, dt)?;

    // Find the dominant frequencies (only positive frequencies)
    println!("Signal: 50 Hz (A=1.0) + 120 Hz (A=0.5) + 300 Hz (A=0.3)");
    println!("Sample rate: {} Hz, N = {}\n", sample_rate, n);

    // Collect frequency-magnitude pairs for positive frequencies
    let mut peaks: Vec<(f64, f64)> = Vec::new();
    for i in 0..n / 2 {
        let mag = (spectrum[i].re * spectrum[i].re + spectrum[i].im * spectrum[i].im).sqrt()
            / (n as f64 / 2.0); // Normalize
        if mag > 0.1 {
            peaks.push((freqs[i], mag));
        }
    }

    println!("Detected frequency peaks:");
    for (freq, mag) in &peaks {
        println!("  {:.1} Hz, amplitude = {:.3}", freq, mag);
    }
    println!();

    Ok(())
}
