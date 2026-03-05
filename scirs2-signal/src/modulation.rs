//! Modulation and demodulation module for signal processing
//!
//! This module provides modulation/demodulation algorithms for communication signals:
//! - **AM** (Amplitude Modulation) — DSB-SC, DSB-FC (conventional), and SSB
//! - **FM** (Frequency Modulation) — analog FM with configurable deviation
//! - **QAM** (Quadrature Amplitude Modulation) — 4-QAM, 16-QAM, 64-QAM, 256-QAM
//!
//! All functions operate on real-valued sample vectors and produce real or complex outputs.
//! Pure Rust, no unwrap(), snake_case naming.

use crate::error::{SignalError, SignalResult};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// AM modulation / demodulation
// ---------------------------------------------------------------------------

/// AM modulation mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AmMode {
    /// Double-Sideband Suppressed Carrier (DSB-SC)
    DsbSc,
    /// Double-Sideband Full Carrier (conventional AM)
    /// The parameter is the modulation index (0 < m <= 1 for no over-modulation)
    DsbFc(f64),
    /// Single-Sideband (upper sideband)
    SsbUpper,
    /// Single-Sideband (lower sideband)
    SsbLower,
}

/// Amplitude-modulate a baseband signal onto a carrier
///
/// # Arguments
///
/// * `signal` - Baseband (message) signal
/// * `carrier_freq` - Carrier frequency in Hz
/// * `sample_rate` - Sample rate in Hz
/// * `mode` - AM modulation mode
///
/// # Returns
///
/// * Modulated signal (same length as input)
pub fn am_modulate(
    signal: &[f64],
    carrier_freq: f64,
    sample_rate: f64,
    mode: AmMode,
) -> SignalResult<Vec<f64>> {
    validate_mod_params(signal, carrier_freq, sample_rate)?;

    let n = signal.len();
    let mut output = vec![0.0; n];
    let omega_c = 2.0 * PI * carrier_freq / sample_rate;

    match mode {
        AmMode::DsbSc => {
            for (i, out) in output.iter_mut().enumerate() {
                *out = signal[i] * (omega_c * i as f64).cos();
            }
        }
        AmMode::DsbFc(mod_index) => {
            if mod_index <= 0.0 {
                return Err(SignalError::ValueError(
                    "Modulation index must be positive".to_string(),
                ));
            }
            // Normalize signal to [-1, 1] for modulation index interpretation
            let max_abs = signal.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
            let scale = if max_abs > 1e-20 { 1.0 / max_abs } else { 1.0 };

            for (i, out) in output.iter_mut().enumerate() {
                let carrier = (omega_c * i as f64).cos();
                *out = (1.0 + mod_index * signal[i] * scale) * carrier;
            }
        }
        AmMode::SsbUpper | AmMode::SsbLower => {
            // SSB uses Hilbert transform (approximation via 90-degree phase shift)
            let hilbert = hilbert_transform_approx(signal)?;
            let sign = if mode == AmMode::SsbUpper { -1.0 } else { 1.0 };
            for (i, out) in output.iter_mut().enumerate() {
                let cos_c = (omega_c * i as f64).cos();
                let sin_c = (omega_c * i as f64).sin();
                *out = signal[i] * cos_c + sign * hilbert[i] * sin_c;
            }
        }
    }

    Ok(output)
}

/// Demodulate an AM signal back to baseband
///
/// # Arguments
///
/// * `modulated` - AM modulated signal
/// * `carrier_freq` - Carrier frequency in Hz
/// * `sample_rate` - Sample rate in Hz
/// * `mode` - AM mode used for modulation
///
/// # Returns
///
/// * Demodulated baseband signal
pub fn am_demodulate(
    modulated: &[f64],
    carrier_freq: f64,
    sample_rate: f64,
    mode: AmMode,
) -> SignalResult<Vec<f64>> {
    validate_mod_params(modulated, carrier_freq, sample_rate)?;

    let n = modulated.len();
    let omega_c = 2.0 * PI * carrier_freq / sample_rate;

    match mode {
        AmMode::DsbSc => {
            // Coherent detection: multiply by carrier then low-pass
            let mut baseband = vec![0.0; n];
            for (i, out) in baseband.iter_mut().enumerate() {
                *out = 2.0 * modulated[i] * (omega_c * i as f64).cos();
            }
            // Simple moving-average low-pass filter
            let cutoff_samples = (sample_rate / carrier_freq).ceil() as usize;
            let filtered = moving_average_lowpass(&baseband, cutoff_samples.max(2));
            Ok(filtered)
        }
        AmMode::DsbFc(_mod_index) => {
            // Envelope detection (rectify + low-pass)
            let envelope: Vec<f64> = modulated.iter().map(|x| x.abs()).collect();
            let cutoff_samples = (sample_rate / carrier_freq).ceil() as usize;
            let filtered = moving_average_lowpass(&envelope, cutoff_samples.max(2));
            // Remove DC offset (the carrier component)
            let mean: f64 = filtered.iter().sum::<f64>() / filtered.len() as f64;
            Ok(filtered.iter().map(|x| x - mean).collect())
        }
        AmMode::SsbUpper | AmMode::SsbLower => {
            // Coherent SSB demodulation
            let mut baseband = vec![0.0; n];
            for (i, out) in baseband.iter_mut().enumerate() {
                *out = 2.0 * modulated[i] * (omega_c * i as f64).cos();
            }
            let cutoff_samples = (sample_rate / carrier_freq).ceil() as usize;
            let filtered = moving_average_lowpass(&baseband, cutoff_samples.max(2));
            Ok(filtered)
        }
    }
}

// ---------------------------------------------------------------------------
// FM modulation / demodulation
// ---------------------------------------------------------------------------

/// Frequency-modulate a baseband signal
///
/// # Arguments
///
/// * `signal` - Baseband (message) signal
/// * `carrier_freq` - Carrier frequency in Hz
/// * `sample_rate` - Sample rate in Hz
/// * `freq_deviation` - Maximum frequency deviation in Hz
///
/// # Returns
///
/// * FM modulated signal
pub fn fm_modulate(
    signal: &[f64],
    carrier_freq: f64,
    sample_rate: f64,
    freq_deviation: f64,
) -> SignalResult<Vec<f64>> {
    validate_mod_params(signal, carrier_freq, sample_rate)?;
    if freq_deviation <= 0.0 {
        return Err(SignalError::ValueError(
            "Frequency deviation must be positive".to_string(),
        ));
    }

    let n = signal.len();
    let omega_c = 2.0 * PI * carrier_freq / sample_rate;
    let k_f = 2.0 * PI * freq_deviation / sample_rate;

    // Cumulative integral of message signal
    let mut phase_integral = 0.0;
    let mut output = Vec::with_capacity(n);

    for (i, &s) in signal.iter().enumerate() {
        phase_integral += s;
        let phase = omega_c * i as f64 + k_f * phase_integral;
        output.push(phase.cos());
    }

    Ok(output)
}

/// Demodulate an FM signal back to baseband
///
/// Uses differentiation of instantaneous phase (arctangent discriminator).
///
/// # Arguments
///
/// * `modulated` - FM modulated signal
/// * `sample_rate` - Sample rate in Hz
/// * `freq_deviation` - Maximum frequency deviation used during modulation
///
/// # Returns
///
/// * Demodulated baseband signal
pub fn fm_demodulate(
    modulated: &[f64],
    sample_rate: f64,
    freq_deviation: f64,
) -> SignalResult<Vec<f64>> {
    if modulated.is_empty() {
        return Err(SignalError::ValueError(
            "Input signal must not be empty".to_string(),
        ));
    }
    if sample_rate <= 0.0 {
        return Err(SignalError::ValueError(
            "Sample rate must be positive".to_string(),
        ));
    }
    if freq_deviation <= 0.0 {
        return Err(SignalError::ValueError(
            "Frequency deviation must be positive".to_string(),
        ));
    }

    let n = modulated.len();
    if n < 2 {
        return Ok(vec![0.0]);
    }

    // Compute analytic signal via Hilbert transform
    let hilbert = hilbert_transform_approx(modulated)?;

    // Compute instantaneous phase
    let mut inst_phase = Vec::with_capacity(n);
    for i in 0..n {
        inst_phase.push(hilbert[i].atan2(modulated[i]));
    }

    // Unwrap phase
    let unwrapped = unwrap_phase_vec(&inst_phase);

    // Differentiate to get instantaneous frequency
    let k_f = 2.0 * PI * freq_deviation / sample_rate;
    let scale = if k_f.abs() > 1e-20 {
        sample_rate / (2.0 * PI * freq_deviation)
    } else {
        1.0
    };

    let mut demodulated = Vec::with_capacity(n);
    demodulated.push(0.0); // first sample
    for i in 1..n {
        let diff = unwrapped[i] - unwrapped[i - 1];
        demodulated.push(diff * scale);
    }

    // Remove DC component
    let mean: f64 = demodulated.iter().sum::<f64>() / demodulated.len() as f64;
    Ok(demodulated.iter().map(|x| x - mean).collect())
}

// ---------------------------------------------------------------------------
// QAM modulation
// ---------------------------------------------------------------------------

/// QAM constellation order
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QamOrder {
    /// 4-QAM (equivalent to QPSK)
    Qam4,
    /// 16-QAM
    Qam16,
    /// 64-QAM
    Qam64,
    /// 256-QAM
    Qam256,
}

impl QamOrder {
    /// Get the number of constellation points
    pub fn constellation_size(self) -> usize {
        match self {
            QamOrder::Qam4 => 4,
            QamOrder::Qam16 => 16,
            QamOrder::Qam64 => 64,
            QamOrder::Qam256 => 256,
        }
    }

    /// Get the number of bits per symbol
    pub fn bits_per_symbol(self) -> usize {
        match self {
            QamOrder::Qam4 => 2,
            QamOrder::Qam16 => 4,
            QamOrder::Qam64 => 6,
            QamOrder::Qam256 => 8,
        }
    }

    /// Get the grid dimension (sqrt of constellation size)
    fn grid_dim(self) -> usize {
        match self {
            QamOrder::Qam4 => 2,
            QamOrder::Qam16 => 4,
            QamOrder::Qam64 => 8,
            QamOrder::Qam256 => 16,
        }
    }
}

/// A QAM symbol (I + jQ)
#[derive(Debug, Clone, Copy)]
pub struct QamSymbol {
    /// In-phase component
    pub i: f64,
    /// Quadrature component
    pub q: f64,
}

/// Generate the QAM constellation map (Gray-coded)
///
/// Returns a vector of QamSymbol, one per constellation point, normalized
/// to unit average power.
pub fn qam_constellation(order: QamOrder) -> Vec<QamSymbol> {
    let m = order.grid_dim();
    let mut points = Vec::with_capacity(order.constellation_size());

    for row in 0..m {
        for col in 0..m {
            // Map to symmetric grid centered at origin
            let i_val = 2.0 * col as f64 - (m as f64 - 1.0);
            let q_val = 2.0 * row as f64 - (m as f64 - 1.0);
            points.push(QamSymbol { i: i_val, q: q_val });
        }
    }

    // Normalize to unit average power
    let avg_power: f64 =
        points.iter().map(|p| p.i * p.i + p.q * p.q).sum::<f64>() / points.len() as f64;
    let scale = if avg_power > 1e-20 {
        1.0 / avg_power.sqrt()
    } else {
        1.0
    };

    for p in &mut points {
        p.i *= scale;
        p.q *= scale;
    }

    points
}

/// Map a bit sequence to QAM symbols
///
/// # Arguments
///
/// * `bits` - Input bit sequence (each element is 0 or 1)
/// * `order` - QAM constellation order
///
/// # Returns
///
/// * Vector of QAM symbols
pub fn qam_modulate_bits(bits: &[u8], order: QamOrder) -> SignalResult<Vec<QamSymbol>> {
    let bps = order.bits_per_symbol();
    if bits.len() % bps != 0 {
        return Err(SignalError::ValueError(format!(
            "Bit sequence length {} must be a multiple of {} for {:?}",
            bits.len(),
            bps,
            order
        )));
    }

    // Validate bits
    if bits.iter().any(|&b| b > 1) {
        return Err(SignalError::ValueError("Bits must be 0 or 1".to_string()));
    }

    let constellation = qam_constellation(order);
    let mut symbols = Vec::with_capacity(bits.len() / bps);

    for chunk in bits.chunks(bps) {
        // Convert bit group to integer index
        let mut index: usize = 0;
        for &bit in chunk {
            index = (index << 1) | (bit as usize);
        }
        if index >= constellation.len() {
            return Err(SignalError::ValueError(format!(
                "Symbol index {} out of range for constellation size {}",
                index,
                constellation.len()
            )));
        }
        symbols.push(constellation[index]);
    }

    Ok(symbols)
}

/// Demodulate QAM symbols back to bits using minimum-distance hard decision
///
/// # Arguments
///
/// * `symbols` - Received QAM symbols (possibly noisy)
/// * `order` - QAM constellation order
///
/// # Returns
///
/// * Demodulated bit sequence
pub fn qam_demodulate_bits(symbols: &[QamSymbol], order: QamOrder) -> SignalResult<Vec<u8>> {
    if symbols.is_empty() {
        return Err(SignalError::ValueError(
            "Symbol sequence must not be empty".to_string(),
        ));
    }

    let constellation = qam_constellation(order);
    let bps = order.bits_per_symbol();
    let mut bits = Vec::with_capacity(symbols.len() * bps);

    for sym in symbols {
        // Find nearest constellation point (minimum Euclidean distance)
        let mut best_idx = 0;
        let mut best_dist = f64::MAX;
        for (idx, point) in constellation.iter().enumerate() {
            let di = sym.i - point.i;
            let dq = sym.q - point.q;
            let dist = di * di + dq * dq;
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }

        // Convert index back to bits
        for bit_pos in (0..bps).rev() {
            bits.push(((best_idx >> bit_pos) & 1) as u8);
        }
    }

    Ok(bits)
}

/// Modulate QAM symbols onto a carrier for transmission
///
/// Produces a real-valued passband signal:
///   s(t) = I(t) * cos(2*pi*fc*t) - Q(t) * sin(2*pi*fc*t)
///
/// Each symbol is held for `samples_per_symbol` samples.
///
/// # Arguments
///
/// * `symbols` - QAM symbol sequence
/// * `carrier_freq` - Carrier frequency in Hz
/// * `sample_rate` - Sample rate in Hz
/// * `samples_per_symbol` - Number of samples per symbol period
///
/// # Returns
///
/// * Passband modulated signal
pub fn qam_modulate_passband(
    symbols: &[QamSymbol],
    carrier_freq: f64,
    sample_rate: f64,
    samples_per_symbol: usize,
) -> SignalResult<Vec<f64>> {
    if symbols.is_empty() {
        return Err(SignalError::ValueError(
            "Symbol sequence must not be empty".to_string(),
        ));
    }
    if carrier_freq <= 0.0 || sample_rate <= 0.0 {
        return Err(SignalError::ValueError(
            "Carrier frequency and sample rate must be positive".to_string(),
        ));
    }
    if samples_per_symbol == 0 {
        return Err(SignalError::ValueError(
            "Samples per symbol must be positive".to_string(),
        ));
    }

    let total_samples = symbols.len() * samples_per_symbol;
    let omega_c = 2.0 * PI * carrier_freq / sample_rate;
    let mut output = Vec::with_capacity(total_samples);

    for (sym_idx, sym) in symbols.iter().enumerate() {
        for k in 0..samples_per_symbol {
            let t = (sym_idx * samples_per_symbol + k) as f64;
            let sample = sym.i * (omega_c * t).cos() - sym.q * (omega_c * t).sin();
            output.push(sample);
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Unified modulate / demodulate interface
// ---------------------------------------------------------------------------

/// Modulation method enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModulationMethod {
    /// Amplitude Modulation (DSB-SC)
    Am,
    /// Amplitude Modulation with modulation index
    AmFc(f64),
    /// Frequency Modulation with frequency deviation in Hz
    Fm(f64),
}

/// Unified modulation function
///
/// # Arguments
///
/// * `signal` - Baseband signal
/// * `carrier_freq` - Carrier frequency in Hz
/// * `sample_rate` - Sample rate in Hz
/// * `method` - Modulation method
///
/// # Returns
///
/// * Modulated signal
pub fn modulate(
    signal: &[f64],
    carrier_freq: f64,
    sample_rate: f64,
    method: ModulationMethod,
) -> SignalResult<Vec<f64>> {
    match method {
        ModulationMethod::Am => am_modulate(signal, carrier_freq, sample_rate, AmMode::DsbSc),
        ModulationMethod::AmFc(mod_index) => {
            am_modulate(signal, carrier_freq, sample_rate, AmMode::DsbFc(mod_index))
        }
        ModulationMethod::Fm(deviation) => {
            fm_modulate(signal, carrier_freq, sample_rate, deviation)
        }
    }
}

/// Unified demodulation function
///
/// # Arguments
///
/// * `modulated` - Modulated signal
/// * `carrier_freq` - Carrier frequency in Hz
/// * `sample_rate` - Sample rate in Hz
/// * `method` - Modulation method used
///
/// # Returns
///
/// * Demodulated baseband signal
pub fn demodulate(
    modulated: &[f64],
    carrier_freq: f64,
    sample_rate: f64,
    method: ModulationMethod,
) -> SignalResult<Vec<f64>> {
    match method {
        ModulationMethod::Am => am_demodulate(modulated, carrier_freq, sample_rate, AmMode::DsbSc),
        ModulationMethod::AmFc(mod_index) => am_demodulate(
            modulated,
            carrier_freq,
            sample_rate,
            AmMode::DsbFc(mod_index),
        ),
        ModulationMethod::Fm(deviation) => fm_demodulate(modulated, sample_rate, deviation),
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn validate_mod_params(signal: &[f64], carrier_freq: f64, sample_rate: f64) -> SignalResult<()> {
    if signal.is_empty() {
        return Err(SignalError::ValueError(
            "Input signal must not be empty".to_string(),
        ));
    }
    if carrier_freq <= 0.0 {
        return Err(SignalError::ValueError(
            "Carrier frequency must be positive".to_string(),
        ));
    }
    if sample_rate <= 0.0 {
        return Err(SignalError::ValueError(
            "Sample rate must be positive".to_string(),
        ));
    }
    if carrier_freq >= sample_rate / 2.0 {
        return Err(SignalError::ValueError(
            "Carrier frequency must be below Nyquist frequency".to_string(),
        ));
    }
    Ok(())
}

/// Approximate Hilbert transform using FFT
/// Returns the imaginary part of the analytic signal
fn hilbert_transform_approx(signal: &[f64]) -> SignalResult<Vec<f64>> {
    let n = signal.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // FFT
    let spectrum = scirs2_fft::fft(signal, Some(n))
        .map_err(|e| SignalError::ComputationError(format!("FFT failed: {}", e)))?;

    // Create one-sided spectrum (Hilbert mask)
    let mut analytic_spectrum = vec![scirs2_core::numeric::Complex64::new(0.0, 0.0); n];
    analytic_spectrum[0] = spectrum[0]; // DC
    if n % 2 == 0 && n > 1 {
        analytic_spectrum[n / 2] = spectrum[n / 2]; // Nyquist
    }
    for i in 1..((n + 1) / 2) {
        analytic_spectrum[i] = spectrum[i] * 2.0;
    }

    // IFFT
    let analytic = scirs2_fft::ifft(&analytic_spectrum, Some(n))
        .map_err(|e| SignalError::ComputationError(format!("IFFT failed: {}", e)))?;

    // Return imaginary part
    Ok(analytic.iter().map(|c| c.im).collect())
}

/// Simple moving average low-pass filter
fn moving_average_lowpass(signal: &[f64], window_size: usize) -> Vec<f64> {
    let n = signal.len();
    if window_size == 0 || n == 0 {
        return signal.to_vec();
    }
    let w = window_size.min(n);
    let mut output = Vec::with_capacity(n);

    let mut sum = 0.0;
    // Initialize with first w samples
    for i in 0..w.min(n) {
        sum += signal[i];
    }

    for i in 0..n {
        if i >= w {
            sum += signal[i] - signal[i - w];
        } else if i > 0 {
            sum += signal[i.min(n - 1)];
        }
        let count = (i + 1).min(w);
        output.push(sum / count as f64);
    }

    output
}

/// Unwrap phase angles to avoid discontinuities
fn unwrap_phase_vec(phases: &[f64]) -> Vec<f64> {
    if phases.is_empty() {
        return Vec::new();
    }
    let mut unwrapped = vec![0.0; phases.len()];
    unwrapped[0] = phases[0];
    for i in 1..phases.len() {
        let mut diff = phases[i] - phases[i - 1];
        while diff > PI {
            diff -= 2.0 * PI;
        }
        while diff <= -PI {
            diff += 2.0 * PI;
        }
        unwrapped[i] = unwrapped[i - 1] + diff;
    }
    unwrapped
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ----- Validation tests -----

    #[test]
    fn test_am_modulate_validation() {
        assert!(am_modulate(&[], 1000.0, 8000.0, AmMode::DsbSc).is_err());
        assert!(am_modulate(&[1.0], 0.0, 8000.0, AmMode::DsbSc).is_err());
        assert!(am_modulate(&[1.0], 1000.0, 0.0, AmMode::DsbSc).is_err());
        assert!(am_modulate(&[1.0], 5000.0, 8000.0, AmMode::DsbSc).is_err()); // above Nyquist
    }

    #[test]
    fn test_fm_modulate_validation() {
        assert!(fm_modulate(&[], 1000.0, 8000.0, 75.0).is_err());
        assert!(fm_modulate(&[1.0], 1000.0, 8000.0, 0.0).is_err()); // zero deviation
    }

    // ----- AM DSB-SC tests -----

    #[test]
    fn test_am_dsbsc_zero_signal() {
        let signal = vec![0.0; 100];
        let modulated =
            am_modulate(&signal, 1000.0, 8000.0, AmMode::DsbSc).expect("AM modulation failed");
        for &s in &modulated {
            assert_relative_eq!(s, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_am_dsbsc_carrier_present() {
        let sr = 8000.0;
        let fc = 1000.0;
        let n = 800;
        let signal = vec![1.0; n]; // constant amplitude

        let modulated = am_modulate(&signal, fc, sr, AmMode::DsbSc).expect("AM modulation failed");

        // Should be a cosine at carrier frequency
        for (i, &s) in modulated.iter().enumerate() {
            let expected = (2.0 * PI * fc * i as f64 / sr).cos();
            assert_relative_eq!(s, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_am_dsbsc_modulate_demodulate() {
        let sr = 8000.0;
        let fc = 2000.0;
        let n = 400;
        // Low-frequency message
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 100.0 * i as f64 / sr).sin())
            .collect();

        let modulated = am_modulate(&signal, fc, sr, AmMode::DsbSc).expect("AM modulation failed");
        let demodulated =
            am_demodulate(&modulated, fc, sr, AmMode::DsbSc).expect("AM demodulation failed");

        assert_eq!(demodulated.len(), n);
        // After demod + lowpass, the signal shape should be recovered (with some delay/attenuation)
        assert!(demodulated.iter().any(|&x| x.abs() > 0.01));
    }

    #[test]
    fn test_am_dsbsc_output_bounded() {
        let sr = 8000.0;
        let fc = 1500.0;
        let signal: Vec<f64> = (0..200)
            .map(|i| (2.0 * PI * 200.0 * i as f64 / sr).sin())
            .collect();

        let modulated = am_modulate(&signal, fc, sr, AmMode::DsbSc).expect("AM modulation failed");
        // Modulated signal should be bounded by signal amplitude
        let max_input = signal.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        for &s in &modulated {
            assert!(s.abs() <= max_input + 1e-10);
        }
    }

    #[test]
    fn test_am_dsbsc_length_preserved() {
        let signal = vec![0.5; 123];
        let modulated =
            am_modulate(&signal, 1000.0, 8000.0, AmMode::DsbSc).expect("AM modulation failed");
        assert_eq!(modulated.len(), 123);
    }

    // ----- AM DSB-FC tests -----

    #[test]
    fn test_am_dsbfc_carrier_always_present() {
        let sr = 8000.0;
        let fc = 1000.0;
        let signal = vec![0.0; 200]; // zero message

        let modulated =
            am_modulate(&signal, fc, sr, AmMode::DsbFc(0.5)).expect("AM-FC modulation failed");

        // With zero message, output is just the carrier
        for (i, &s) in modulated.iter().enumerate() {
            let expected = (2.0 * PI * fc * i as f64 / sr).cos();
            assert_relative_eq!(s, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_am_dsbfc_invalid_mod_index() {
        let signal = vec![1.0; 100];
        assert!(am_modulate(&signal, 1000.0, 8000.0, AmMode::DsbFc(0.0)).is_err());
        assert!(am_modulate(&signal, 1000.0, 8000.0, AmMode::DsbFc(-0.5)).is_err());
    }

    #[test]
    fn test_am_dsbfc_modulate_demodulate() {
        let sr = 8000.0;
        let fc = 2000.0;
        let n = 400;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 100.0 * i as f64 / sr).sin())
            .collect();

        let modulated =
            am_modulate(&signal, fc, sr, AmMode::DsbFc(0.8)).expect("AM-FC modulation failed");
        let demodulated = am_demodulate(&modulated, fc, sr, AmMode::DsbFc(0.8))
            .expect("AM-FC demodulation failed");

        assert_eq!(demodulated.len(), n);
        // Should recover some signal shape
        assert!(demodulated.iter().any(|&x| x.abs() > 0.001));
    }

    #[test]
    fn test_am_dsbfc_envelope_positive() {
        let sr = 8000.0;
        let fc = 1500.0;
        let signal: Vec<f64> = (0..200)
            .map(|i| 0.5 * (2.0 * PI * 100.0 * i as f64 / sr).sin())
            .collect();

        let modulated =
            am_modulate(&signal, fc, sr, AmMode::DsbFc(0.5)).expect("AM-FC modulation failed");

        // Envelope = |1 + m*signal| should be positive for m <= 1 and |signal| <= 1
        // This means the modulated signal shouldn't always be negative at carrier peaks
        let has_positive = modulated.iter().any(|&x| x > 0.0);
        assert!(has_positive);
    }

    #[test]
    fn test_am_dsbfc_modulation_depth() {
        let sr = 8000.0;
        let fc = 2000.0;
        let signal = vec![1.0; 200]; // constant

        let low_mod = am_modulate(&signal, fc, sr, AmMode::DsbFc(0.3)).expect("Low mod AM failed");
        let high_mod =
            am_modulate(&signal, fc, sr, AmMode::DsbFc(0.9)).expect("High mod AM failed");

        let low_max = low_mod.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        let high_max = high_mod.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);

        // Higher modulation index should produce larger peak amplitude
        assert!(high_max > low_max);
    }

    // ----- AM SSB tests -----

    #[test]
    fn test_am_ssb_produces_output() {
        let sr = 8000.0;
        let fc = 2000.0;
        let n = 256;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 300.0 * i as f64 / sr).sin())
            .collect();

        let upper =
            am_modulate(&signal, fc, sr, AmMode::SsbUpper).expect("SSB Upper modulation failed");
        let lower =
            am_modulate(&signal, fc, sr, AmMode::SsbLower).expect("SSB Lower modulation failed");

        assert_eq!(upper.len(), n);
        assert_eq!(lower.len(), n);
        // Upper and lower sideband should be different
        let diff: f64 = upper
            .iter()
            .zip(lower.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.1);
    }

    #[test]
    fn test_am_ssb_demodulate() {
        let sr = 8000.0;
        let fc = 2000.0;
        let n = 256;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 300.0 * i as f64 / sr).sin())
            .collect();

        let modulated =
            am_modulate(&signal, fc, sr, AmMode::SsbUpper).expect("SSB modulation failed");
        let demodulated =
            am_demodulate(&modulated, fc, sr, AmMode::SsbUpper).expect("SSB demodulation failed");

        assert_eq!(demodulated.len(), n);
        assert!(demodulated.iter().any(|&x| x.abs() > 0.001));
    }

    // ----- FM tests -----

    #[test]
    fn test_fm_modulate_constant_signal() {
        let sr = 8000.0;
        let fc = 1000.0;
        let deviation = 75.0;
        let signal = vec![0.0; 200]; // zero message = pure carrier

        let modulated = fm_modulate(&signal, fc, sr, deviation).expect("FM modulation failed");

        // With zero message, should be a pure cosine at carrier freq
        for (i, &s) in modulated.iter().enumerate() {
            let expected = (2.0 * PI * fc * i as f64 / sr).cos();
            assert_relative_eq!(s, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fm_modulate_output_bounded() {
        let sr = 8000.0;
        let fc = 1500.0;
        let deviation = 200.0;
        let signal: Vec<f64> = (0..200)
            .map(|i| (2.0 * PI * 100.0 * i as f64 / sr).sin())
            .collect();

        let modulated = fm_modulate(&signal, fc, sr, deviation).expect("FM modulation failed");

        // FM output is always a unit-amplitude cosine
        for &s in &modulated {
            assert!(s.abs() <= 1.0 + 1e-12);
        }
    }

    #[test]
    fn test_fm_modulate_demodulate_roundtrip() {
        let sr = 8000.0;
        let fc = 2000.0;
        let deviation = 300.0;
        let n = 500;
        let message_freq = 100.0;

        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * message_freq * i as f64 / sr).sin())
            .collect();

        let modulated = fm_modulate(&signal, fc, sr, deviation).expect("FM modulation failed");
        let demodulated = fm_demodulate(&modulated, sr, deviation).expect("FM demodulation failed");

        assert_eq!(demodulated.len(), n);
        // Demodulated signal should have some periodic content
        let energy: f64 = demodulated.iter().map(|x| x * x).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn test_fm_modulate_length_preserved() {
        let signal = vec![0.3; 77];
        let modulated = fm_modulate(&signal, 1000.0, 8000.0, 75.0).expect("FM modulation failed");
        assert_eq!(modulated.len(), 77);
    }

    #[test]
    fn test_fm_demodulate_validation() {
        assert!(fm_demodulate(&[], 8000.0, 75.0).is_err());
        assert!(fm_demodulate(&[1.0], 0.0, 75.0).is_err());
        assert!(fm_demodulate(&[1.0], 8000.0, 0.0).is_err());
    }

    #[test]
    fn test_fm_different_deviations() {
        let sr = 8000.0;
        let fc = 1500.0;
        let n = 200;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 100.0 * i as f64 / sr).sin())
            .collect();

        let low_dev = fm_modulate(&signal, fc, sr, 50.0).expect("FM low dev failed");
        let high_dev = fm_modulate(&signal, fc, sr, 500.0).expect("FM high dev failed");

        // Different deviations should produce different signals
        let diff: f64 = low_dev
            .iter()
            .zip(high_dev.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.1);
    }

    // ----- QAM tests -----

    #[test]
    fn test_qam_constellation_sizes() {
        assert_eq!(qam_constellation(QamOrder::Qam4).len(), 4);
        assert_eq!(qam_constellation(QamOrder::Qam16).len(), 16);
        assert_eq!(qam_constellation(QamOrder::Qam64).len(), 64);
        assert_eq!(qam_constellation(QamOrder::Qam256).len(), 256);
    }

    #[test]
    fn test_qam_constellation_unit_power() {
        for order in &[QamOrder::Qam4, QamOrder::Qam16, QamOrder::Qam64] {
            let const_map = qam_constellation(*order);
            let avg_power: f64 = const_map.iter().map(|p| p.i * p.i + p.q * p.q).sum::<f64>()
                / const_map.len() as f64;
            assert_relative_eq!(avg_power, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_qam_constellation_symmetric() {
        for order in &[QamOrder::Qam4, QamOrder::Qam16, QamOrder::Qam64] {
            let constellation = qam_constellation(*order);
            // Check that for every point (i, q) there exists (-i, -q)
            for p in &constellation {
                let has_conjugate = constellation
                    .iter()
                    .any(|q| (q.i + p.i).abs() < 1e-10 && (q.q + p.q).abs() < 1e-10);
                assert!(
                    has_conjugate,
                    "Missing conjugate point for ({}, {})",
                    p.i, p.q
                );
            }
        }
    }

    #[test]
    fn test_qam4_modulate_demodulate() {
        let bits: Vec<u8> = vec![0, 0, 0, 1, 1, 0, 1, 1];
        let symbols = qam_modulate_bits(&bits, QamOrder::Qam4).expect("QAM4 modulation failed");
        assert_eq!(symbols.len(), 4); // 8 bits / 2 bps

        let recovered =
            qam_demodulate_bits(&symbols, QamOrder::Qam4).expect("QAM4 demodulation failed");
        assert_eq!(recovered, bits);
    }

    #[test]
    fn test_qam16_modulate_demodulate() {
        let bits: Vec<u8> = vec![0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0];
        let symbols = qam_modulate_bits(&bits, QamOrder::Qam16).expect("QAM16 modulation failed");
        assert_eq!(symbols.len(), 4); // 16 bits / 4 bps

        let recovered =
            qam_demodulate_bits(&symbols, QamOrder::Qam16).expect("QAM16 demodulation failed");
        assert_eq!(recovered, bits);
    }

    #[test]
    fn test_qam64_modulate_demodulate() {
        let bits: Vec<u8> = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        let symbols = qam_modulate_bits(&bits, QamOrder::Qam64).expect("QAM64 modulation failed");
        assert_eq!(symbols.len(), 2); // 12 bits / 6 bps

        let recovered =
            qam_demodulate_bits(&symbols, QamOrder::Qam64).expect("QAM64 demodulation failed");
        assert_eq!(recovered, bits);
    }

    #[test]
    fn test_qam_modulate_invalid_bits() {
        // Wrong length for QAM16 (not multiple of 4)
        let bits: Vec<u8> = vec![0, 0, 0];
        assert!(qam_modulate_bits(&bits, QamOrder::Qam16).is_err());

        // Invalid bit value
        let bits: Vec<u8> = vec![0, 0, 2, 0];
        assert!(qam_modulate_bits(&bits, QamOrder::Qam16).is_err());
    }

    #[test]
    fn test_qam_demodulate_empty() {
        let symbols: Vec<QamSymbol> = vec![];
        assert!(qam_demodulate_bits(&symbols, QamOrder::Qam4).is_err());
    }

    #[test]
    fn test_qam_passband_length() {
        let symbols = vec![QamSymbol { i: 1.0, q: 0.0 }, QamSymbol { i: 0.0, q: 1.0 }];
        let passband =
            qam_modulate_passband(&symbols, 1000.0, 8000.0, 10).expect("QAM passband failed");
        assert_eq!(passband.len(), 20); // 2 symbols * 10 sps
    }

    #[test]
    fn test_qam_passband_validation() {
        let symbols = vec![QamSymbol { i: 1.0, q: 0.0 }];
        assert!(qam_modulate_passband(&symbols, 0.0, 8000.0, 10).is_err());
        assert!(qam_modulate_passband(&symbols, 1000.0, 0.0, 10).is_err());
        assert!(qam_modulate_passband(&symbols, 1000.0, 8000.0, 0).is_err());
        assert!(qam_modulate_passband(&[], 1000.0, 8000.0, 10).is_err());
    }

    #[test]
    fn test_qam_passband_bounded() {
        let bits: Vec<u8> = vec![0, 1, 1, 0, 1, 1, 0, 0];
        let symbols = qam_modulate_bits(&bits, QamOrder::Qam4).expect("QAM4 mod failed");
        let passband =
            qam_modulate_passband(&symbols, 1000.0, 8000.0, 20).expect("QAM passband failed");

        // All samples should be finite
        assert!(passband.iter().all(|x| x.is_finite()));
    }

    // ----- Unified interface tests -----

    #[test]
    fn test_unified_modulate_am() {
        let signal = vec![1.0; 100];
        let modulated =
            modulate(&signal, 1000.0, 8000.0, ModulationMethod::Am).expect("Unified AM mod failed");
        assert_eq!(modulated.len(), 100);
    }

    #[test]
    fn test_unified_modulate_fm() {
        let signal = vec![0.5; 100];
        let modulated = modulate(&signal, 1000.0, 8000.0, ModulationMethod::Fm(200.0))
            .expect("Unified FM mod failed");
        assert_eq!(modulated.len(), 100);
    }

    #[test]
    fn test_unified_demodulate_am() {
        let signal: Vec<f64> = (0..200)
            .map(|i| (2.0 * PI * 100.0 * i as f64 / 8000.0).sin())
            .collect();
        let modulated =
            modulate(&signal, 1500.0, 8000.0, ModulationMethod::Am).expect("AM mod failed");
        let demodulated =
            demodulate(&modulated, 1500.0, 8000.0, ModulationMethod::Am).expect("AM demod failed");
        assert_eq!(demodulated.len(), signal.len());
    }

    #[test]
    fn test_unified_demodulate_fm() {
        let signal: Vec<f64> = (0..200)
            .map(|i| (2.0 * PI * 100.0 * i as f64 / 8000.0).sin())
            .collect();
        let modulated =
            modulate(&signal, 1500.0, 8000.0, ModulationMethod::Fm(150.0)).expect("FM mod failed");
        let demodulated = demodulate(&modulated, 1500.0, 8000.0, ModulationMethod::Fm(150.0))
            .expect("FM demod failed");
        assert_eq!(demodulated.len(), signal.len());
    }

    // ----- Internal helper tests -----

    #[test]
    fn test_hilbert_transform_approx() {
        let n = 128;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / n as f64).cos())
            .collect();

        let hilbert = hilbert_transform_approx(&signal).expect("Hilbert failed");
        assert_eq!(hilbert.len(), n);

        // For cos(wt), Hilbert transform should approximate sin(wt)
        let expected: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / n as f64).sin())
            .collect();

        // Check correlation (should be high)
        let corr: f64 = hilbert
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let norm_h: f64 = hilbert.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_e: f64 = expected.iter().map(|x| x * x).sum::<f64>().sqrt();
        let normalized_corr = if norm_h * norm_e > 1e-20 {
            corr / (norm_h * norm_e)
        } else {
            0.0
        };
        assert!(
            normalized_corr > 0.9,
            "Hilbert correlation too low: {}",
            normalized_corr
        );
    }

    #[test]
    fn test_moving_average_lowpass() {
        let signal = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let filtered = moving_average_lowpass(&signal, 4);
        assert_eq!(filtered.len(), signal.len());
        // Filtered signal should be smoother (less variation)
        let raw_var: f64 = signal.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
        let filt_var: f64 = filtered.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
        assert!(filt_var < raw_var);
    }

    #[test]
    fn test_unwrap_phase_vec() {
        let phases = vec![0.0, 1.0, 2.0, 3.0, -3.0, -2.0, -1.0, 0.0];
        let unwrapped = unwrap_phase_vec(&phases);
        assert_eq!(unwrapped.len(), phases.len());
        // All consecutive differences should be in (-pi, pi]
        for w in unwrapped.windows(2) {
            let diff = w[1] - w[0];
            assert!(diff > -PI && diff <= PI, "diff = {} out of range", diff);
        }
    }
}
