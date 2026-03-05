//! Audio Analysis Example — SciRS2 Signal
//!
//! Demonstrates:
//!   1. Synthetic audio signal (440 Hz + 880 Hz sine + white noise)
//!   2. FFT power spectrum via `periodogram`
//!   3. Top dominant frequencies
//!   4. MFCC feature extraction using `mfcc_extract` + `MfccConfig`
//!
//! Run with: cargo run -p scirs2-signal --example audio_analysis

use scirs2_signal::cepstral::{mfcc_extract, MfccConfig};
use scirs2_signal::spectral::periodogram;
use std::f64::consts::PI;

// ------------------------------------------------------------------ //
//  Signal generation                                                   //
// ------------------------------------------------------------------ //

/// Generate a synthetic audio snippet:
///   signal(t) = A1·sin(2π f1 t) + A2·sin(2π f2 t) + noise
fn generate_signal(
    duration_s: f64,
    sample_rate: f64,
    freqs: &[(f64, f64)], // (frequency_hz, amplitude)
    noise_amplitude: f64,
) -> Vec<f64> {
    let n = (duration_s * sample_rate) as usize;
    // Simple deterministic pseudo-noise (LCG)
    let mut lcg: u64 = 0x1234_5678_9ABC_DEF0;
    (0..n)
        .map(|i| {
            let t = i as f64 / sample_rate;
            let signal_part: f64 = freqs
                .iter()
                .map(|(f, a)| a * (2.0 * PI * f * t).sin())
                .sum();
            lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let noise = ((lcg >> 33) as f64 / u32::MAX as f64 - 0.5) * 2.0 * noise_amplitude;
            signal_part + noise
        })
        .collect()
}

// ------------------------------------------------------------------ //
//  Spectral peak finding                                              //
// ------------------------------------------------------------------ //

/// Return the top-k (frequency_hz, power) pairs from a periodogram result.
fn top_frequencies(freqs: &[f64], power: &[f64], k: usize) -> Vec<(f64, f64)> {
    let mut indexed: Vec<(f64, f64)> = freqs.iter().copied().zip(power.iter().copied()).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("no NaN"));
    indexed.truncate(k);
    indexed
}

// ------------------------------------------------------------------ //
//  MFCC summary statistics                                            //
// ------------------------------------------------------------------ //

/// Compute mean MFCC across all frames (one vector per coefficient).
fn mean_mfcc(mfcc_frames: &[Vec<f64>]) -> Vec<f64> {
    if mfcc_frames.is_empty() {
        return Vec::new();
    }
    let n_coeff = mfcc_frames[0].len();
    let n_frames = mfcc_frames.len() as f64;
    (0..n_coeff)
        .map(|c| mfcc_frames.iter().map(|f| f[c]).sum::<f64>() / n_frames)
        .collect()
}

/// Compute std-dev of MFCC across frames.
fn std_mfcc(mfcc_frames: &[Vec<f64>], means: &[f64]) -> Vec<f64> {
    if mfcc_frames.is_empty() {
        return Vec::new();
    }
    let n_coeff = means.len();
    let n_frames = mfcc_frames.len() as f64;
    (0..n_coeff)
        .map(|c| {
            let variance = mfcc_frames
                .iter()
                .map(|f| (f[c] - means[c]).powi(2))
                .sum::<f64>()
                / n_frames;
            variance.sqrt()
        })
        .collect()
}

// ------------------------------------------------------------------ //
//  main                                                               //
// ------------------------------------------------------------------ //

fn main() {
    const SAMPLE_RATE: f64 = 8_000.0; // 8 kHz (telephony band)
    const DURATION_S: f64 = 0.5; // 500 ms
    const NOISE_AMP: f64 = 0.05;

    let freqs = [(440.0_f64, 0.8_f64), (880.0, 0.4), (1320.0, 0.2)];

    println!("=== SciRS2 Audio Analysis Example ===\n");
    println!("Signal parameters:");
    println!("  Sample rate  : {SAMPLE_RATE:.0} Hz");
    println!("  Duration     : {DURATION_S:.3} s");
    for (f, a) in &freqs {
        println!("  Component    : {f:.0} Hz, amplitude {a:.2}");
    }
    println!("  Noise amp    : {NOISE_AMP:.3}\n");

    let signal = generate_signal(DURATION_S, SAMPLE_RATE, &freqs, NOISE_AMP);
    println!("Generated {} samples", signal.len());

    // RMS and peak
    let rms = (signal.iter().map(|x| x * x).sum::<f64>() / signal.len() as f64).sqrt();
    let peak = signal.iter().cloned().fold(0.0_f64, f64::max);
    println!("Signal RMS   : {rms:.4}");
    println!("Signal Peak  : {peak:.4}\n");

    // ------------------------------------------------------------------ //
    //  FFT Spectral Analysis                                               //
    // ------------------------------------------------------------------ //
    let (freqs_hz, power) = periodogram(&signal, Some(SAMPLE_RATE), Some("hann"), None, None, None)
        .expect("periodogram failed");

    println!("--- FFT Spectral Analysis ---");
    println!(
        "  FFT length   : {} points ({:.1} Hz resolution)",
        freqs_hz.len() * 2,
        freqs_hz[1] - freqs_hz[0]
    );

    let top5 = top_frequencies(&freqs_hz, &power, 5);
    println!("\n  Top 5 dominant frequencies:");
    println!("  {:<12} {:>14}", "Frequency", "Power (dB)");
    println!("  {}", "-".repeat(30));
    for (f, p) in &top5 {
        let db = 10.0 * (p + 1e-20).log10();
        println!("  {:<12.1} {:>14.2}", f, db);
    }

    // Verify detected peaks match input frequencies
    println!("\n  Frequency detection check:");
    for &(input_f, input_a) in &freqs {
        // Find nearest detected peak within ±20 Hz
        let closest = top5
            .iter()
            .min_by_key(|(f, _)| ((f - input_f).abs() * 100.0) as u64);
        if let Some((detected_f, _)) = closest {
            let err = (detected_f - input_f).abs();
            let status = if err < 20.0 { "OK" } else { "MISS" };
            println!(
                "    Input {:.0} Hz -> detected {:.1} Hz (err {:.1} Hz) [{}]",
                input_f, detected_f, err, status
            );
        }
    }

    // ------------------------------------------------------------------ //
    //  MFCC Feature Extraction                                            //
    // ------------------------------------------------------------------ //
    let frame_length = 256usize;
    let hop_length = 128usize;
    let n_mfcc = 13usize;

    let mfcc_config = MfccConfig {
        n_mfcc,
        sample_rate: SAMPLE_RATE,
        ..MfccConfig::new(SAMPLE_RATE)
    };

    let mfcc_frames = mfcc_extract(&signal, &mfcc_config, frame_length, hop_length)
        .expect("MFCC extraction failed");

    println!("\n--- MFCC Features ---");
    println!(
        "  Frame length : {} samples ({:.1} ms)",
        frame_length,
        1000.0 * frame_length as f64 / SAMPLE_RATE
    );
    println!(
        "  Hop length   : {} samples ({:.1} ms)",
        hop_length,
        1000.0 * hop_length as f64 / SAMPLE_RATE
    );
    println!("  N_MFCC       : {n_mfcc}");
    println!("  Frames       : {}", mfcc_frames.len());

    let means = mean_mfcc(&mfcc_frames);
    let stds = std_mfcc(&mfcc_frames, &means);

    println!("\n  MFCC statistics (mean ± std across frames):");
    println!("  {:<12} {:>10} {:>10}", "Coefficient", "Mean", "Std");
    println!("  {}", "-".repeat(36));
    for (c, (&m, &s)) in means.iter().zip(stds.iter()).enumerate() {
        println!("  MFCC[{:<4}]  {:>10.4} {:>10.4}", c, m, s);
    }

    // Brief first-frame display
    println!("\n  First-frame MFCC coefficients:");
    print!("  [");
    for (i, v) in mfcc_frames[0].iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{:.3}", v);
    }
    println!("]");

    println!("\nDone.");
}
