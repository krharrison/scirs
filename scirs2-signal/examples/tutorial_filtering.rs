//! Tutorial: Signal Processing and Filtering with SciRS2
//!
//! This tutorial covers designing and applying digital filters,
//! FIR filter design, and spectral estimation.
//!
//! Run with: cargo run -p scirs2-signal --example tutorial_filtering

use scirs2_signal::{butter, filtfilt, firwin, FilterType, SignalResult};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SciRS2 Signal Processing Tutorial ===\n");

    section_butterworth_filter()?;
    section_fir_filter()?;
    section_spectral_estimation()?;
    section_waveforms()?;

    println!("\n=== Tutorial Complete ===");
    Ok(())
}

/// Section 1: Designing and applying a Butterworth lowpass filter
fn section_butterworth_filter() -> SignalResult<()> {
    println!("--- 1. Butterworth Lowpass Filter ---\n");

    // Design a 4th-order Butterworth lowpass filter
    // Cutoff at 0.2 (normalized: fraction of Nyquist frequency)
    let (b, a) = butter(4, 0.2, FilterType::Lowpass)?;

    println!("Filter coefficients:");
    println!("  b (numerator):   {:?}", b);
    println!("  a (denominator): {:?}\n", a);

    // Create a test signal: 5 Hz sine + 50 Hz sine, sampled at 500 Hz
    let fs = 500.0;
    let n = 500;
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 50.0 * t).sin()
        })
        .collect();

    // Apply the filter using filtfilt (zero-phase filtering)
    // filtfilt applies the filter forward and backward, eliminating phase distortion
    let filtered = filtfilt(&b, &a, &signal)?;

    println!("Signal: 5 Hz + 50 Hz noise (N={})", n);
    println!("After lowpass filtering (cutoff ~50 Hz at fs=500):");
    println!(
        "  First 10 original values:  {:?}",
        &signal[..10]
            .iter()
            .map(|x| format!("{:.3}", x))
            .collect::<Vec<_>>()
    );
    println!(
        "  First 10 filtered values:  {:?}",
        &filtered[..10]
            .iter()
            .map(|x| format!("{:.3}", x))
            .collect::<Vec<_>>()
    );

    // The high-frequency component should be attenuated
    let original_range = signal.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - signal.iter().cloned().fold(f64::INFINITY, f64::min);
    let filtered_range = filtered.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - filtered.iter().cloned().fold(f64::INFINITY, f64::min);
    println!("  Original range:  {:.4}", original_range);
    println!("  Filtered range:  {:.4}", filtered_range);
    println!("  (Filtered range should be smaller -- high freq removed)\n");

    Ok(())
}

/// Section 2: FIR filter design using firwin
fn section_fir_filter() -> SignalResult<()> {
    println!("--- 2. FIR Filter Design ---\n");

    // Design a 31-tap FIR lowpass filter using a windowed sinc method
    // Cutoff at 0.3 (normalized frequency, fraction of Nyquist)
    // pass_zero=true means lowpass (pass DC)
    let taps = firwin(31, 0.3, "hamming", true)?;

    println!("FIR lowpass filter (31 taps, cutoff=0.3, hamming window):");
    println!("  Number of taps: {}", taps.len());
    println!(
        "  First 5 taps: {:?}",
        &taps[..5]
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
    );
    println!("  Center tap:   {:.6}", taps[15]);
    println!(
        "  Last 5 taps:  {:?}\n",
        &taps[26..]
            .iter()
            .map(|x| format!("{:.6}", x))
            .collect::<Vec<_>>()
    );

    // The filter is symmetric (linear phase)
    let is_symmetric = taps
        .iter()
        .zip(taps.iter().rev())
        .all(|(a, b)| (a - b).abs() < 1e-12);
    println!("  Filter is symmetric: {}", is_symmetric);
    println!("  (Linear phase FIR filters are always symmetric)\n");

    Ok(())
}

/// Section 3: Spectral estimation using Welch's method
fn section_spectral_estimation() -> SignalResult<()> {
    println!("--- 3. Spectral Estimation (Welch) ---\n");

    let fs = 1000.0;
    let n = 4096;

    // Create a signal with known frequency content
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            2.0 * (2.0 * PI * 100.0 * t).sin() + (2.0 * PI * 250.0 * t).sin()
        })
        .collect();

    // Welch's method: estimates power spectral density
    // It divides the signal into overlapping segments, windows each segment,
    // computes the periodogram, and averages them.
    let (freqs, psd) = scirs2_signal::welch(
        &signal,
        Some(fs),  // sample rate
        None,      // window type (default: Hann)
        Some(256), // nperseg
        Some(128), // noverlap
        None,      // nfft
        None,      // detrend
        None,      // scaling
    )?;

    println!(
        "Welch PSD estimate (fs={} Hz, segment=256, overlap=128):",
        fs
    );
    println!("  Number of frequency bins: {}", freqs.len());

    // Find peaks in the PSD
    let max_psd = psd.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("  Peak frequencies:");
    for i in 1..psd.len() - 1 {
        if psd[i] > psd[i - 1] && psd[i] > psd[i + 1] && psd[i] > max_psd * 0.1 {
            println!("    {:.1} Hz (PSD = {:.2})", freqs[i], psd[i]);
        }
    }
    println!();

    Ok(())
}

/// Section 4: Creating standard waveforms
fn section_waveforms() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- 4. Waveforms ---\n");

    // Generate time array for waveforms
    let n_samples = 1000;
    let t1 = 1.0_f64;
    let t_arr: Vec<f64> = (0..n_samples)
        .map(|i| i as f64 * t1 / n_samples as f64)
        .collect();

    // Chirp signal (frequency sweep from 10 Hz to 100 Hz)
    let chirp_signal = scirs2_signal::chirp(&t_arr, 10.0, t1, 100.0, "linear", 0.0)?;
    println!("Chirp signal (10-100 Hz linear sweep, 1s):");
    println!("  Length: {} samples", chirp_signal.len());
    println!(
        "  First 5 values: {:?}",
        &chirp_signal[..5]
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );
    println!();

    // Square wave: duty cycle determines the fraction of the period that is "high"
    let t_square: Vec<f64> = (0..20).map(|i| 2.0 * PI * i as f64 / 20.0).collect();
    let square_wave = scirs2_signal::square(&t_square, 0.5)?;
    println!("Square wave (1 cycle, 20 samples, 50% duty):");
    println!(
        "  {:?}\n",
        square_wave
            .iter()
            .map(|x| format!("{:.0}", x))
            .collect::<Vec<_>>()
    );

    // Sawtooth wave: width=1.0 gives a rising sawtooth
    let t_saw: Vec<f64> = (0..20).map(|i| 2.0 * PI * i as f64 / 20.0).collect();
    let saw_wave = scirs2_signal::sawtooth(&t_saw, 1.0)?;
    println!("Sawtooth wave (1 cycle, 20 samples):");
    println!(
        "  {:?}\n",
        saw_wave
            .iter()
            .map(|x| format!("{:.2}", x))
            .collect::<Vec<_>>()
    );

    Ok(())
}
