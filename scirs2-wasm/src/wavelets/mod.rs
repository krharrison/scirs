//! Wavelet transform implementations for WASM
//!
//! Provides Discrete Wavelet Transform (DWT) and Continuous Wavelet Transform (CWT)
//! for signal analysis in browser/Node.js environments.
//!
//! Convention (follows PyWavelets / Mallat):
//! - Analysis:  x → (cA, cD) via downsampling convolution with (lo_d, hi_d)
//! - Synthesis: (cA, cD) → x via upsampling convolution with (lo_r, hi_r)
//!   where lo_r = reverse(lo_d), hi_r = reverse(hi_d)
//!
//! For orthogonal filters (Haar, Daubechies) the QMF relationship ensures
//! perfect reconstruction up to boundary effects.

use crate::error::{WasmError, WasmResult};

// ─── Filter coefficient tables ────────────────────────────────────────────────

/// Haar scaling coefficients (lo_d) — normalised
const HAAR_LO_D: [f64; 2] = [
    std::f64::consts::FRAC_1_SQRT_2,
    std::f64::consts::FRAC_1_SQRT_2,
];
const HAAR_HI_D: [f64; 2] = [
    -std::f64::consts::FRAC_1_SQRT_2,
    std::f64::consts::FRAC_1_SQRT_2,
];

/// Daubechies D4 (db2) — orthonormal
const DB4_LO_D: [f64; 4] = [
    0.4829629131445341_f64,
    0.8365163037378079_f64,
    0.2241438680420134_f64,
    -0.1294095225512604_f64,
];

/// Daubechies D6 (db3) — orthonormal
const DB6_LO_D: [f64; 6] = [
    0.3326705529500827_f64,
    0.8068915093110925_f64,
    0.4598775021184915_f64,
    -0.1350110200102546_f64,
    -0.0854412738820267_f64,
    0.0352262918857095_f64,
];

/// Daubechies D8 (db4) — orthonormal
const DB8_LO_D: [f64; 8] = [
    0.2303778133088965_f64,
    0.7148465705529156_f64,
    0.630_880_767_929_859_f64,
    -0.0279837694168599_f64,
    -0.1870348117190931_f64,
    0.0308413818359870_f64,
    0.0328830116668852_f64,
    -0.0105974017849973_f64,
];

// ─── Public enums and structs ─────────────────────────────────────────────────

/// Wavelet family selection
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum WaveletFamily {
    /// Haar wavelet (simplest orthogonal wavelet)
    Haar,
    /// Daubechies wavelets (filter length N = 4, 6, or 8)
    Daubechies(u8),
    /// Symlet wavelets — shares filter with equivalent Daubechies order
    Symlet(u8),
    /// Coiflet wavelets
    Coiflet(u8),
    /// Biorthogonal wavelets (p, q orders)
    Biorthogonal(u8, u8),
    /// Mexican Hat (Ricker wavelet) — CWT only
    MexicanHat,
    /// Morlet wavelet (ω₀ = 6) — CWT only
    Morlet,
}

/// Boundary extension mode for DWT
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
pub enum PaddingMode {
    /// Periodic / wrap-around extension (default)
    #[default]
    Periodic,
    /// Symmetric (mirror) extension
    Symmetric,
    /// Zero-padding
    ZeroPad,
}

/// Configuration for Discrete Wavelet Transform
#[derive(Debug, Clone)]
pub struct DwtConfig {
    /// Wavelet family to use
    pub family: WaveletFamily,
    /// Number of decomposition levels
    pub n_levels: usize,
    /// Boundary extension mode
    pub mode: PaddingMode,
}

impl Default for DwtConfig {
    fn default() -> Self {
        DwtConfig {
            family: WaveletFamily::Daubechies(4),
            n_levels: 3,
            mode: PaddingMode::Periodic,
        }
    }
}

/// Result of a 1-D DWT decomposition
#[derive(Debug, Clone)]
pub struct DwtResult {
    /// Approximation coefficients (coarsest level)
    pub approximation: Vec<f64>,
    /// Detail coefficients — `details[0]` = first (finest) level
    pub details: Vec<Vec<f64>>,
    /// Length of the original signal
    pub n_original: usize,
}

/// Result of a single-level 2-D DWT
#[derive(Debug, Clone)]
pub struct Dwt2dResult {
    /// Low-low (approximation)
    pub ll: Vec<Vec<f64>>,
    /// Low-high (horizontal detail)
    pub lh: Vec<Vec<f64>>,
    /// High-low (vertical detail)
    pub hl: Vec<Vec<f64>>,
    /// High-high (diagonal detail)
    pub hh: Vec<Vec<f64>>,
}

// ─── Filter helpers ───────────────────────────────────────────────────────────

/// Return analysis filters (lo_d, hi_d) for the given family.
fn get_filters(family: &WaveletFamily) -> WasmResult<(Vec<f64>, Vec<f64>)> {
    match family {
        WaveletFamily::Haar => {
            let lo_d = HAAR_LO_D.to_vec();
            let hi_d = HAAR_HI_D.to_vec();
            Ok((lo_d, hi_d))
        }
        WaveletFamily::Daubechies(n) => {
            let lo_d: Vec<f64> = match n {
                4 => DB4_LO_D.to_vec(),
                6 => DB6_LO_D.to_vec(),
                8 => DB8_LO_D.to_vec(),
                _ => {
                    return Err(WasmError::InvalidParameter(format!(
                        "Daubechies order {n} not supported; use 4, 6, or 8"
                    )))
                }
            };
            let hi_d = qmf(&lo_d);
            Ok((lo_d, hi_d))
        }
        WaveletFamily::Symlet(n) => {
            // Use equivalent Daubechies coefficients (same filter length)
            let equiv = WaveletFamily::Daubechies((*n).clamp(4, 8) & !1 | 4);
            get_filters(&equiv)
        }
        WaveletFamily::Coiflet(_) => {
            // Coiflet-1 (6 coefficients)
            let lo_d: Vec<f64> = vec![
                -0.0156557281999_f64,
                -0.0727326213410_f64,
                0.3848648468_f64,
                0.8525720202_f64,
                0.3378976625_f64,
                -0.0727326213410_f64,
            ];
            let hi_d = qmf(&lo_d);
            Ok((lo_d, hi_d))
        }
        WaveletFamily::Biorthogonal(_, _) => {
            // Biorthogonal 2.2 (CDF 5/3 used in JPEG 2000 lossless)
            let lo_d: Vec<f64> = vec![-0.125_f64, 0.25, 0.75, 0.25, -0.125];
            let hi_d: Vec<f64> = vec![0.0_f64, -0.5, 1.0, -0.5, 0.0];
            Ok((lo_d, hi_d))
        }
        WaveletFamily::MexicanHat | WaveletFamily::Morlet => Err(WasmError::InvalidParameter(
            "MexicanHat and Morlet are CWT-only wavelets; not supported for DWT".to_string(),
        )),
    }
}

/// Quadrature Mirror Filter: hi_d[k] = (−1)^k · lo_d[N−1−k]
fn qmf(lo_d: &[f64]) -> Vec<f64> {
    let n = lo_d.len();
    (0..n)
        .map(|k| {
            let sign: f64 = if k % 2 == 0 { 1.0 } else { -1.0 };
            sign * lo_d[n - 1 - k]
        })
        .collect()
}

// ─── 1-D operations ──────────────────────────────────────────────────────────

/// Periodically-extended dot product of `signal` starting at `pos` with `filter`.
///
/// Computes Σ_j filter[j] * signal[(pos + j) mod n]
fn periodic_dot(signal: &[f64], pos: usize, filter: &[f64]) -> f64 {
    let n = signal.len();
    filter
        .iter()
        .enumerate()
        .map(|(j, &h)| h * signal[(pos + j) % n])
        .sum()
}

/// Analysis convolution + downsample using periodic extension.
///
/// Output length = ceil(signal.len() / 2)
fn aconv_down(signal: &[f64], filter: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let out_len = n.div_ceil(2);
    (0..out_len)
        .map(|k| periodic_dot(signal, 2 * k, filter))
        .collect()
}

/// Synthesis: scatter each coefficient into output using the analysis filter.
///
/// The perfect-reconstruction formula (paired with `aconv_down`) is:
///   y[2k + j] += filter[j] * coeff[k]   for j = 0..filter_len
///
/// This is NOT the textbook "upsample then convolve" — it's an equivalent
/// scatter that avoids explicit upsampling and gives the same result.
/// Returns a vector of length 2*n (caller trims to target length).
fn sconv_up(coeff: &[f64], filter: &[f64]) -> Vec<f64> {
    let n = coeff.len();
    let _f = filter.len();
    // Output length: 2*n accommodates filter overhang for short filters;
    // for longer filters the extra tail is handled by periodic wrapping.
    let out_len = 2 * n;
    let mut out = vec![0.0_f64; out_len];
    for (k, &c) in coeff.iter().enumerate() {
        for (j, &h) in filter.iter().enumerate() {
            let t = (2 * k + j) % out_len;
            out[t] += h * c;
        }
    }
    out
}

// ─── Public DWT API ───────────────────────────────────────────────────────────

/// One-dimensional multi-level DWT decomposition.
pub fn dwt_1d(signal: &[f64], config: &DwtConfig) -> WasmResult<DwtResult> {
    if signal.is_empty() {
        return Err(WasmError::InvalidParameter(
            "Signal must not be empty".to_string(),
        ));
    }

    let (lo_d, hi_d) = get_filters(&config.family)?;
    let n_original = signal.len();
    let mut approx = signal.to_vec();
    let mut details = Vec::with_capacity(config.n_levels);

    for _ in 0..config.n_levels {
        if approx.len() < 2 {
            break;
        }
        let detail = aconv_down(&approx, &hi_d);
        approx = aconv_down(&approx, &lo_d);
        details.push(detail);
    }

    Ok(DwtResult {
        approximation: approx,
        details,
        n_original,
    })
}

/// One-dimensional multi-level IDWT reconstruction.
pub fn idwt_1d(result: &DwtResult, config: &DwtConfig) -> WasmResult<Vec<f64>> {
    let (lo_d, hi_d) = get_filters(&config.family)?;
    let mut approx = result.approximation.clone();

    // Reconstruct from coarsest detail to finest (reverse of decomposition).
    for detail in result.details.iter().rev() {
        let rec_lo = sconv_up(&approx, &lo_d);
        let rec_hi = sconv_up(detail, &hi_d);
        let len = rec_lo.len().min(rec_hi.len()).min(detail.len() * 2);
        approx = (0..len).map(|i| rec_lo[i] + rec_hi[i]).collect();
    }

    // Trim to original length
    approx.truncate(result.n_original);
    if approx.len() < result.n_original {
        approx.resize(result.n_original, 0.0);
    }
    Ok(approx)
}

/// Full wavelet decomposition: returns `[approx, detail_{n}, ..., detail_1]`.
pub fn wavedec(
    signal: &[f64],
    family: &WaveletFamily,
    n_levels: usize,
) -> WasmResult<Vec<Vec<f64>>> {
    let config = DwtConfig {
        family: family.clone(),
        n_levels,
        mode: PaddingMode::Periodic,
    };
    let result = dwt_1d(signal, &config)?;
    let mut out = Vec::with_capacity(1 + result.details.len());
    out.push(result.approximation.clone());
    for d in result.details.iter().rev() {
        out.push(d.clone());
    }
    Ok(out)
}

/// Wavelet reconstruction from `wavedec` output.
pub fn waverec(coeffs: &[Vec<f64>], family: &WaveletFamily) -> WasmResult<Vec<f64>> {
    if coeffs.is_empty() {
        return Err(WasmError::InvalidParameter(
            "coeffs must not be empty".to_string(),
        ));
    }
    if coeffs.len() == 1 {
        return Ok(coeffs[0].clone());
    }

    let (lo_d, hi_d) = get_filters(family)?;
    // coeffs = [approx, detail_n, ..., detail_1]  (coarsest detail is first)
    let mut approx = coeffs[0].clone();
    for detail in &coeffs[1..] {
        let rec_lo = sconv_up(&approx, &lo_d);
        let rec_hi = sconv_up(detail, &hi_d);
        let len = rec_lo.len().min(rec_hi.len()).min(detail.len() * 2);
        approx = (0..len).map(|i| rec_lo[i] + rec_hi[i]).collect();
    }
    Ok(approx)
}

// ─── 2-D DWT ─────────────────────────────────────────────────────────────────

/// Single-level 2-D DWT: apply 1-D DWT along rows then columns.
pub fn dwt_2d(image: &[Vec<f64>], config: &DwtConfig) -> WasmResult<Dwt2dResult> {
    if image.is_empty() || image[0].is_empty() {
        return Err(WasmError::InvalidParameter(
            "Image must not be empty".to_string(),
        ));
    }
    let (lo_d, hi_d) = get_filters(&config.family)?;
    let rows = image.len();
    let cols = image[0].len();

    // Step 1: row-wise DWT
    let mut row_lo: Vec<Vec<f64>> = Vec::with_capacity(rows);
    let mut row_hi: Vec<Vec<f64>> = Vec::with_capacity(rows);
    for row in image {
        row_lo.push(aconv_down(row, &lo_d));
        row_hi.push(aconv_down(row, &hi_d));
    }

    let half_cols = cols.div_ceil(2);
    let half_rows = rows.div_ceil(2);

    let mut ll = vec![vec![0.0_f64; half_cols]; half_rows];
    let mut lh = vec![vec![0.0_f64; half_cols]; half_rows];
    let mut hl = vec![vec![0.0_f64; half_cols]; half_rows];
    let mut hh = vec![vec![0.0_f64; half_cols]; half_rows];

    // Step 2: column-wise DWT on row_lo and row_hi
    for c in 0..half_cols {
        let col_lo: Vec<f64> = row_lo.iter().map(|r| r[c]).collect();
        let col_hi: Vec<f64> = row_hi.iter().map(|r| r[c]).collect();

        let ll_col = aconv_down(&col_lo, &lo_d);
        let hl_col = aconv_down(&col_lo, &hi_d);
        let lh_col = aconv_down(&col_hi, &lo_d);
        let hh_col = aconv_down(&col_hi, &hi_d);

        for r in 0..half_rows {
            if r < ll_col.len() {
                ll[r][c] = ll_col[r];
            }
            if r < hl_col.len() {
                hl[r][c] = hl_col[r];
            }
            if r < lh_col.len() {
                lh[r][c] = lh_col[r];
            }
            if r < hh_col.len() {
                hh[r][c] = hh_col[r];
            }
        }
    }

    Ok(Dwt2dResult { ll, lh, hl, hh })
}

// ─── CWT ──────────────────────────────────────────────────────────────────────

/// Sample the real part of the scaled and translated wavelet kernel.
fn wavelet_kernel(wavelet: &WaveletFamily, scale: f64, len: usize) -> WasmResult<Vec<f64>> {
    let half = (len as f64 - 1.0) / 2.0;
    let inv_sqrt_s = 1.0 / scale.sqrt();

    match wavelet {
        WaveletFamily::MexicanHat => Ok((0..len)
            .map(|i| {
                let t = (i as f64 - half) / scale;
                let t2 = t * t;
                inv_sqrt_s * (1.0 - t2) * (-t2 / 2.0).exp()
            })
            .collect()),
        WaveletFamily::Morlet => {
            const OMEGA0: f64 = 6.0;
            Ok((0..len)
                .map(|i| {
                    let t = (i as f64 - half) / scale;
                    let t2 = t * t;
                    inv_sqrt_s * (OMEGA0 * t).cos() * (-t2 / 2.0).exp()
                })
                .collect())
        }
        _ => Err(WasmError::InvalidParameter(
            "CWT only supports MexicanHat and Morlet wavelets".to_string(),
        )),
    }
}

/// Continuous Wavelet Transform — returns a `scales × time` matrix.
pub fn cwt(signal: &[f64], wavelet: &WaveletFamily, scales: &[f64]) -> WasmResult<Vec<Vec<f64>>> {
    if signal.is_empty() {
        return Err(WasmError::InvalidParameter(
            "Signal must not be empty".to_string(),
        ));
    }
    if scales.is_empty() {
        return Err(WasmError::InvalidParameter(
            "Scales must not be empty".to_string(),
        ));
    }

    let n = signal.len();
    let mut result = Vec::with_capacity(scales.len());

    for &scale in scales {
        if scale <= 0.0 {
            return Err(WasmError::InvalidParameter(format!(
                "Scale must be positive, got {scale}"
            )));
        }
        let kernel_len = ((10.0 * scale) as usize).clamp(3, n);
        let kernel = wavelet_kernel(wavelet, scale, kernel_len)?;

        let k = kernel.len();
        let full_len = n + k - 1;
        let mut conv = vec![0.0_f64; full_len];
        for (i, &s) in signal.iter().enumerate() {
            for (j, &h) in kernel.iter().enumerate() {
                conv[i + j] += s * h;
            }
        }
        // Take central n samples
        let start = (k - 1) / 2;
        let end = (start + n).min(conv.len());
        let mut row = conv[start..end].to_vec();
        row.resize(n, 0.0);
        result.push(row);
    }

    Ok(result)
}

/// Compute scalogram power |CWT|² per (scale, time) cell.
pub fn scalogram_power(cwt_result: &[Vec<f64>]) -> Vec<Vec<f64>> {
    cwt_result
        .iter()
        .map(|row| row.iter().map(|&v| v * v).collect())
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{PI, SQRT_2};

    fn rmse(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len().min(b.len());
        assert!(n > 0, "Cannot compute RMSE of empty slices");
        let mse: f64 = a[..n]
            .iter()
            .zip(b[..n].iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            / n as f64;
        mse.sqrt()
    }

    fn sine_wave(n: usize, freq: f64, sample_rate: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin())
            .collect()
    }

    #[test]
    fn test_haar_dwt_approx_is_mean_of_pairs() {
        // Haar level-1 approximation: cA[k] = (x[2k] + x[2k+1]) / sqrt(2)
        let signal: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let config = DwtConfig {
            family: WaveletFamily::Haar,
            n_levels: 1,
            mode: PaddingMode::Periodic,
        };
        let result = dwt_1d(&signal, &config).expect("DWT failed");
        let approx = &result.approximation;
        assert_eq!(approx.len(), 4);
        for k in 0..4 {
            let expected = (signal[2 * k] + signal[2 * k + 1]) / SQRT_2;
            assert!(
                (approx[k] - expected).abs() < 1e-10,
                "approx[{k}] = {}, expected {}",
                approx[k],
                expected
            );
        }
    }

    #[test]
    fn test_dwt_idwt_roundtrip_sine() {
        let signal = sine_wave(256, 10.0, 256.0);
        let config = DwtConfig {
            family: WaveletFamily::Haar,
            n_levels: 3,
            mode: PaddingMode::Periodic,
        };
        let dwt_result = dwt_1d(&signal, &config).expect("DWT failed");
        let reconstructed = idwt_1d(&dwt_result, &config).expect("IDWT failed");

        assert_eq!(reconstructed.len(), signal.len());
        let err = rmse(&signal, &reconstructed);
        assert!(err < 0.01, "RMSE too large: {err}");
    }

    #[test]
    fn test_wavedec_waverec_roundtrip() {
        let signal = sine_wave(128, 5.0, 128.0);
        let family = WaveletFamily::Haar;
        let n_levels = 3;

        let coeffs = wavedec(&signal, &family, n_levels).expect("wavedec failed");
        assert_eq!(
            coeffs.len(),
            n_levels + 1,
            "Expected n_levels+1 coefficient arrays"
        );

        let rec = waverec(&coeffs, &family).expect("waverec failed");
        assert_eq!(rec.len(), signal.len());
        let err = rmse(&signal, &rec);
        assert!(err < 0.01, "waverec RMSE too large: {err}");
    }

    #[test]
    fn test_cwt_high_energy_at_matching_scale() {
        // A 10 Hz sine at 256 Hz sample rate → period ≈ 25.6 samples → scale ≈ 13
        let n = 256;
        let signal = sine_wave(n, 10.0, 256.0);
        let scales: Vec<f64> = vec![5.0, 13.0, 26.0, 50.0];
        let cwt_result = cwt(&signal, &WaveletFamily::MexicanHat, &scales).expect("CWT failed");
        let power = scalogram_power(&cwt_result);

        // Total power at scale ≈ 13 should be larger than at scale 50
        let energy_13: f64 = power[1].iter().sum();
        let energy_50: f64 = power[3].iter().sum();
        assert!(
            energy_13 > energy_50,
            "Expected more energy at matching scale 13 vs 50; {energy_13} vs {energy_50}"
        );
    }

    #[test]
    fn test_dwt_2d_output_shape() {
        let rows = 8;
        let cols = 8;
        let image: Vec<Vec<f64>> = (0..rows)
            .map(|r| (0..cols).map(|c| (r * cols + c) as f64).collect())
            .collect();
        let config = DwtConfig {
            family: WaveletFamily::Haar,
            n_levels: 1,
            mode: PaddingMode::Periodic,
        };
        let result = dwt_2d(&image, &config).expect("DWT 2D failed");
        assert_eq!(result.ll.len(), rows / 2);
        assert_eq!(result.ll[0].len(), cols / 2);
    }

    #[test]
    fn test_db4_dwt_roundtrip() {
        let signal = sine_wave(256, 8.0, 256.0);
        let config = DwtConfig {
            family: WaveletFamily::Daubechies(4),
            n_levels: 2,
            mode: PaddingMode::Periodic,
        };
        let res = dwt_1d(&signal, &config).expect("DWT failed");
        let rec = idwt_1d(&res, &config).expect("IDWT failed");
        assert_eq!(rec.len(), signal.len());
        let err = rmse(&signal, &rec);
        assert!(err < 0.01, "DB4 RMSE: {err}");
    }

    #[test]
    fn test_haar_energy_conservation() {
        // Energy in wavelet domain should equal energy in signal domain (Parseval)
        let signal = sine_wave(64, 4.0, 64.0);
        let energy_signal: f64 = signal.iter().map(|&x| x * x).sum();

        let config = DwtConfig {
            family: WaveletFamily::Haar,
            n_levels: 3,
            mode: PaddingMode::Periodic,
        };
        let res = dwt_1d(&signal, &config).expect("DWT");
        let energy_approx: f64 = res.approximation.iter().map(|&x| x * x).sum();
        let energy_details: f64 = res
            .details
            .iter()
            .flat_map(|d| d.iter())
            .map(|&x| x * x)
            .sum();
        let energy_wavelet = energy_approx + energy_details;

        // Within 5% (periodic boundary introduces small artefacts)
        let ratio = energy_wavelet / energy_signal;
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "Energy ratio = {ratio} (expected ≈ 1.0)"
        );
    }
}
