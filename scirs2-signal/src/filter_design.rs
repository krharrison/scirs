// Digital Filter Design — new high-level interface wrapping the existing filter sub-modules.
//
// Provides:
//   * `FilterSpec` / `FilterType` — specification types
//   * `IirCoefficients` — wrapper with apply / freqz / filtfilt methods
//   * `butter`   — Butterworth IIR design
//   * `cheby1`   — Chebyshev Type I IIR design
//   * `cheby2`   — Chebyshev Type II IIR design
//   * `FirWindow` / `firwin`  — FIR via window method
//   * `remez`    — Parks-McClellan equiripple FIR design
//   * `lfilter_fir` / `fir_filtfilt` — FIR application helpers

use crate::error::{SignalError, SignalResult};
use std::f64::consts::PI;

// ─── Filter types ────────────────────────────────────────────────────────────

/// Selects the band-shape of a filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    Lowpass,
    Highpass,
    Bandpass,
    Bandstop,
}

/// High-level filter specification (currently informational).
#[derive(Debug, Clone)]
pub struct FilterSpec {
    pub filter_type: FilterType,
    /// Normalised cutoff frequencies (0 – 1, where 1 = Nyquist).
    /// One value for LP/HP, two values [low, high] for BP/BS.
    pub cutoff: Vec<f64>,
    pub order: usize,
    /// Passband ripple in dB (Chebyshev I / Elliptic).
    pub ripple_db: Option<f64>,
    /// Stopband attenuation in dB (Chebyshev II / Elliptic).
    pub stopband_attn: Option<f64>,
}

// ─── IIR coefficient container ───────────────────────────────────────────────

/// IIR filter expressed as H(z) = B(z) / A(z).
#[derive(Debug, Clone)]
pub struct IirCoefficients {
    /// Numerator polynomial coefficients (highest power first).
    pub b: Vec<f64>,
    /// Denominator polynomial coefficients (a[0] is typically 1).
    pub a: Vec<f64>,
}

impl IirCoefficients {
    /// Apply the filter to `x` using the direct form II transposed structure.
    pub fn filter(&self, x: &[f64]) -> SignalResult<Vec<f64>> {
        lfilter_iir(&self.b, &self.a, x)
    }

    /// Apply the filter forward *and* backward (zero-phase, no group delay).
    pub fn filtfilt(&self, x: &[f64]) -> SignalResult<Vec<f64>> {
        filtfilt_iir(&self.b, &self.a, x)
    }

    /// Compute the frequency response magnitude (in dB) at `n_points` evenly
    /// spaced frequencies from 0 to the Nyquist rate (inclusive).
    ///
    /// Returns `(freq, magnitude_db)` where `freq` is normalised 0–1.
    pub fn freqz(&self, n_points: usize) -> (Vec<f64>, Vec<f64>) {
        let n = n_points.max(2);
        let mut freqs = Vec::with_capacity(n);
        let mut mags = Vec::with_capacity(n);

        for k in 0..n {
            let omega = PI * (k as f64) / ((n - 1) as f64); // 0 … π
            let z = num_complex_exp(omega);

            let b_val = poly_eval_z(&self.b, z);
            let a_val = poly_eval_z(&self.a, z);

            let h = if a_val.norm() < 1e-30 {
                0.0
            } else {
                b_val.norm() / a_val.norm()
            };

            let mag_db = if h > 1e-30 {
                20.0 * h.log10()
            } else {
                -300.0
            };

            freqs.push(k as f64 / (n - 1) as f64);
            mags.push(mag_db);
        }

        (freqs, mags)
    }
}

// ─── IIR filter application helpers (internal) ───────────────────────────────

/// Direct form II transposed IIR filter.
fn lfilter_iir(b: &[f64], a: &[f64], x: &[f64]) -> SignalResult<Vec<f64>> {
    if a.is_empty() {
        return Err(SignalError::ValueError(
            "Denominator coefficients 'a' are empty".to_string(),
        ));
    }
    let a0 = a[0];
    if a0.abs() < 1e-30 {
        return Err(SignalError::ValueError(
            "a[0] must not be zero".to_string(),
        ));
    }

    let m = b.len().max(a.len());
    let nb = b.len();
    let na = a.len();

    // Normalise
    let b_norm: Vec<f64> = b.iter().map(|&v| v / a0).collect();
    let a_norm: Vec<f64> = a.iter().map(|&v| v / a0).collect();

    let delay = m - 1;
    let mut w = vec![0.0_f64; delay.max(1)];
    let mut y = vec![0.0_f64; x.len()];

    for (n_idx, &xn) in x.iter().enumerate() {
        // y[n] = b[0]*x[n] + w[0]
        let yn = if nb > 0 { b_norm[0] * xn } else { 0.0 }
            + if delay > 0 { w[0] } else { 0.0 };
        y[n_idx] = yn;

        // Shift delay line
        if delay > 1 {
            for k in 0..(delay - 1) {
                let bk = if k + 1 < nb { b_norm[k + 1] } else { 0.0 };
                let ak = if k + 1 < na { a_norm[k + 1] } else { 0.0 };
                w[k] = bk * xn - ak * yn + w[k + 1];
            }
        }
        if delay > 0 {
            let k = delay - 1;
            let bk = if k + 1 < nb { b_norm[k + 1] } else { 0.0 };
            let ak = if k + 1 < na { a_norm[k + 1] } else { 0.0 };
            w[k] = bk * xn - ak * yn;
        }
    }

    Ok(y)
}

/// Zero-phase IIR filter (forward + backward pass).
fn filtfilt_iir(b: &[f64], a: &[f64], x: &[f64]) -> SignalResult<Vec<f64>> {
    // Forward pass
    let y1 = lfilter_iir(b, a, x)?;

    // Reverse, backward pass
    let mut y1r = y1;
    y1r.reverse();
    let y2 = lfilter_iir(b, a, &y1r)?;
    let mut result = y2;
    result.reverse();
    Ok(result)
}

// ─── Complex arithmetic helpers ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct Cpx {
    re: f64,
    im: f64,
}

impl Cpx {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn norm(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
    fn mul(&self, rhs: Cpx) -> Cpx {
        Cpx::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
    fn add(&self, rhs: Cpx) -> Cpx {
        Cpx::new(self.re + rhs.re, self.im + rhs.im)
    }
    fn conj(&self) -> Cpx {
        Cpx::new(self.re, -self.im)
    }
    fn sub(&self, rhs: Cpx) -> Cpx {
        Cpx::new(self.re - rhs.re, self.im - rhs.im)
    }
    fn abs_sq(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    fn div(&self, rhs: Cpx) -> Cpx {
        let d = rhs.abs_sq();
        if d < 1e-300 {
            return Cpx::new(0.0, 0.0);
        }
        let n = self.mul(rhs.conj());
        Cpx::new(n.re / d, n.im / d)
    }
    fn scale(&self, s: f64) -> Cpx {
        Cpx::new(self.re * s, self.im * s)
    }
}

fn num_complex_exp(omega: f64) -> Cpx {
    Cpx::new(omega.cos(), omega.sin())
}

/// Evaluate a polynomial (coefficients highest-power first) at a complex z = e^{j ω}.
fn poly_eval_z(coeffs: &[f64], z: Cpx) -> Cpx {
    let mut acc = Cpx::new(0.0, 0.0);
    for &c in coeffs {
        acc = acc.mul(z).add(Cpx::new(c, 0.0));
    }
    acc
}

// ─── Polynomial utilities ────────────────────────────────────────────────────

/// Multiply two real polynomials represented as coefficient vectors
/// (index 0 = highest degree).
fn poly_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return vec![1.0];
    }
    let n = a.len() + b.len() - 1;
    let mut c = vec![0.0_f64; n];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            c[i + j] += ai * bj;
        }
    }
    c
}

/// Build the monic polynomial whose roots are the given complex conjugate pairs
/// plus any purely-real roots.  All poles of a causal real-coefficient digital
/// filter come either as conjugate pairs or as real values.
///
/// `poles` is a list of complex numbers; real poles have |imag| < tol.
/// Returns the denominator polynomial with real coefficients.
fn poles_to_poly(poles: &[Cpx]) -> Vec<f64> {
    let mut p = vec![1.0_f64]; // start with 1
    let tol = 1e-9;

    let mut remaining: Vec<Cpx> = poles.to_vec();
    let mut used = vec![false; remaining.len()];

    // First handle conjugate pairs
    for i in 0..remaining.len() {
        if used[i] {
            continue;
        }
        let pi = remaining[i];
        if pi.im.abs() > tol {
            // find its conjugate
            let mut found = false;
            for j in (i + 1)..remaining.len() {
                if used[j] {
                    continue;
                }
                let pj = remaining[j];
                if (pj.re - pi.re).abs() < tol && (pj.im + pi.im).abs() < tol {
                    // Quadratic factor (z - pi)(z - pj) = z^2 - 2*Re(pi)*z + |pi|^2
                    let quad = vec![1.0, -2.0 * pi.re, pi.abs_sq()];
                    p = poly_mul(&p, &quad);
                    used[i] = true;
                    used[j] = true;
                    found = true;
                    break;
                }
            }
            if !found {
                // Unpaired complex pole — form linear factor and its conjugate
                // (z - pi)(z - pi*) = z^2 - 2*Re*z + |pi|^2
                let quad = vec![1.0, -2.0 * pi.re, pi.abs_sq()];
                p = poly_mul(&p, &quad);
                used[i] = true;
            }
        }
    }

    // Then handle remaining real poles
    for i in 0..remaining.len() {
        if used[i] {
            continue;
        }
        let pi = remaining[i];
        // linear factor: z - real
        let lin = vec![1.0, -pi.re];
        p = poly_mul(&p, &lin);
    }

    p
}

/// Convert zeros / poles / gain to (b, a) polynomials.
fn zpk_to_ba(zeros: &[Cpx], poles: &[Cpx], gain: f64) -> (Vec<f64>, Vec<f64>) {
    let b_monic = poles_to_poly(zeros);
    let a = poles_to_poly(poles);
    // Scale numerator by gain
    let b: Vec<f64> = b_monic.iter().map(|&v| v * gain).collect();
    (b, a)
}

// ─── Butterworth IIR ─────────────────────────────────────────────────────────

/// Design a Butterworth digital filter.
///
/// `cutoff` — normalised frequency (0–1); for Bandpass/Bandstop supply two
/// values `[low, high]`.
pub fn butter(
    order: usize,
    cutoff: &[f64],
    filter_type: FilterType,
) -> SignalResult<IirCoefficients> {
    validate_order_nonzero(order)?;
    validate_cutoffs(cutoff, filter_type)?;

    match filter_type {
        FilterType::Lowpass => butter_lp(order, cutoff[0]),
        FilterType::Highpass => butter_hp(order, cutoff[0]),
        FilterType::Bandpass => butter_bp(order, cutoff[0], cutoff[1]),
        FilterType::Bandstop => butter_bs(order, cutoff[0], cutoff[1]),
    }
}

fn butter_lp(order: usize, wn: f64) -> SignalResult<IirCoefficients> {
    let wc = prewarp(wn);
    let poles = butterworth_analog_poles(order);
    // Scale poles to wc
    let scaled_poles: Vec<Cpx> = poles.iter().map(|&p| p.scale(wc)).collect();
    let (bz, az, gain_z) = analog_to_digital_lp(&scaled_poles, wc, order);
    let (b, a) = zpk_to_ba(&bz, &az, gain_z);
    normalize_ba(b, a)
}

fn butter_hp(order: usize, wn: f64) -> SignalResult<IirCoefficients> {
    let wc = prewarp(wn);
    let poles = butterworth_analog_poles(order);
    // HP transformation s → wc/s
    let hp_poles: Vec<Cpx> = poles.iter().map(|&p| Cpx::new(wc, 0.0).div(p)).collect();
    let (bz, az, gain_z) = analog_to_digital_hp(&hp_poles, order);
    let (b, a) = zpk_to_ba(&bz, &az, gain_z);
    normalize_ba(b, a)
}

fn butter_bp(order: usize, wl: f64, wh: f64) -> SignalResult<IirCoefficients> {
    let wl_a = prewarp(wl);
    let wh_a = prewarp(wh);
    let wc = (wl_a * wh_a).sqrt();
    let bw = wh_a - wl_a;
    let poles = butterworth_analog_poles(order);
    let (bp_z, bp_p) = analog_bp_transform(&poles, wc, bw);
    let (bz, az, gain) = analog_to_digital_general(&bp_z, &bp_p);
    let (b, a) = zpk_to_ba(&bz, &az, gain);
    normalize_ba(b, a)
}

fn butter_bs(order: usize, wl: f64, wh: f64) -> SignalResult<IirCoefficients> {
    let wl_a = prewarp(wl);
    let wh_a = prewarp(wh);
    let wc = (wl_a * wh_a).sqrt();
    let bw = wh_a - wl_a;
    let poles = butterworth_analog_poles(order);
    let (bs_z, bs_p) = analog_bs_transform(&poles, wc, bw);
    let (bz, az, gain) = analog_to_digital_general(&bs_z, &bs_p);
    let (b, a) = zpk_to_ba(&bz, &az, gain);
    normalize_ba(b, a)
}

/// Compute the analog Butterworth prototype poles (all in the left half-plane).
///
/// The N-th order Butterworth prototype has N poles equally spaced on the unit
/// circle in the left half of the s-plane:
///   s_k = exp(j * pi * (2k + N + 1) / (2N)),  k = 0, ..., N-1
fn butterworth_analog_poles(order: usize) -> Vec<Cpx> {
    let n = order as f64;
    (0..order)
        .map(|k| {
            let theta: f64 = PI * (2.0 * k as f64 + n + 1.0) / (2.0 * n);
            Cpx::new(theta.cos(), theta.sin())
        })
        .collect()
}

/// Prewarp a normalised digital frequency to analog frequency using bilinear transform.
fn prewarp(wn: f64) -> f64 {
    2.0 * (PI * wn / 2.0).tan()
}

/// Bilinear transform of a single complex pole: z = (2 + s) / (2 - s).
fn bilinear_pole(s: Cpx) -> Cpx {
    let two = Cpx::new(2.0, 0.0);
    let num = two.add(s);
    let den = two.sub(s);
    num.div(den)
}

// The bilinear transform maps s=∞ to z=-1 (zeros at z=-1 for lowpass).
fn analog_to_digital_lp(
    poles: &[Cpx],
    wc: f64,
    order: usize,
) -> (Vec<Cpx>, Vec<Cpx>, f64) {
    let digital_poles: Vec<Cpx> = poles.iter().map(|&p| bilinear_pole(p)).collect();
    let digital_zeros: Vec<Cpx> = (0..order).map(|_| Cpx::new(-1.0, 0.0)).collect();

    // DC gain normalisation: G = prod(1-z_p) / prod(1-z_z) (at z=1)
    let mut num_gain = 1.0_f64;
    let mut den_gain = 1.0_f64;
    let z1 = Cpx::new(1.0, 0.0);
    for &p in &digital_poles {
        let d = z1.sub(p);
        den_gain *= d.norm();
    }
    for &z in &digital_zeros {
        let d = z1.sub(z);
        num_gain *= d.norm();
    }
    // Target gain = wc^order (analog prototype gain) but we renormalise to unity at DC
    let gain = if num_gain > 1e-30 { den_gain / num_gain } else { 1.0 };
    (digital_zeros, digital_poles, gain)
}

fn analog_to_digital_hp(poles: &[Cpx], order: usize) -> (Vec<Cpx>, Vec<Cpx>, f64) {
    let digital_poles: Vec<Cpx> = poles.iter().map(|&p| bilinear_pole(p)).collect();
    // Highpass: zeros at z=+1
    let digital_zeros: Vec<Cpx> = (0..order).map(|_| Cpx::new(1.0, 0.0)).collect();

    // Gain normalisation at Nyquist (z = -1)
    let zm1 = Cpx::new(-1.0, 0.0);
    let mut den_gain = 1.0_f64;
    let mut num_gain = 1.0_f64;
    for &p in &digital_poles {
        den_gain *= zm1.sub(p).norm();
    }
    for &z in &digital_zeros {
        num_gain *= zm1.sub(z).norm();
    }
    let gain = if num_gain > 1e-30 { den_gain / num_gain } else { 1.0 };
    (digital_zeros, digital_poles, gain)
}

fn analog_to_digital_general(zeros: &[Cpx], poles: &[Cpx]) -> (Vec<Cpx>, Vec<Cpx>, f64) {
    let digital_poles: Vec<Cpx> = poles.iter().map(|&p| bilinear_pole(p)).collect();
    let digital_zeros: Vec<Cpx> = zeros.iter().map(|&z| bilinear_pole(z)).collect();

    // Pad extra zeros at z=-1 until #zeros == #poles
    let np = digital_poles.len();
    let nz = digital_zeros.len();
    let mut dz = digital_zeros;
    for _ in nz..np {
        dz.push(Cpx::new(-1.0, 0.0));
    }

    // Normalise gain at DC (z=1) to unity (for passband ≈ 1)
    let z1 = Cpx::new(1.0, 0.0);
    let mut num_g = 1.0_f64;
    let mut den_g = 1.0_f64;
    for &p in &digital_poles {
        den_g *= z1.sub(p).norm();
    }
    for &z in &dz {
        num_g *= z1.sub(z).norm();
    }
    // Avoid division by zero
    let gain = if num_g > 1e-30 && den_g > 1e-30 {
        // for bandpass the gain at the centre is what we want ~ 1
        den_g / num_g
    } else {
        1.0
    };
    (dz, digital_poles, gain)
}

fn analog_bp_transform(poles: &[Cpx], wc: f64, bw: f64) -> (Vec<Cpx>, Vec<Cpx>) {
    let wc2 = Cpx::new(wc * wc, 0.0);
    let mut bp_poles = Vec::new();

    // Zeros of the bandpass prototype: order zeros at s=0
    let bp_zeros: Vec<Cpx> = (0..poles.len()).map(|_| Cpx::new(0.0, 0.0)).collect();

    for &p in poles {
        // s → (s^2 + wc^2) / (bw * s)
        // Each LP pole at p maps to two BP poles satisfying:
        //   bw*p*z - z^2 - wc^2 = 0   =>   z^2 - bw*p*z + wc^2 = 0
        let half_bwp = p.scale(bw / 2.0);
        // discriminant = (bw*p/2)^2 - wc^2
        let disc = half_bwp.mul(half_bwp).sub(wc2);
        let sqrt_disc = complex_sqrt(disc);
        bp_poles.push(half_bwp.add(sqrt_disc));
        bp_poles.push(half_bwp.sub(sqrt_disc));
    }
    (bp_zeros, bp_poles)
}

fn analog_bs_transform(poles: &[Cpx], wc: f64, bw: f64) -> (Vec<Cpx>, Vec<Cpx>) {
    let wc2 = Cpx::new(wc * wc, 0.0);
    let jomega = Cpx::new(0.0, wc);
    let bs_zeros: Vec<Cpx> = poles
        .iter()
        .flat_map(|_| [jomega, jomega.conj()].into_iter())
        .collect();

    let mut bs_poles = Vec::new();
    for &p in poles {
        // s → bw*s / (s^2 + wc^2)
        // z^2 - (bw/p)*z + wc^2 = 0
        let half = Cpx::new(bw, 0.0).div(p).scale(0.5);
        let disc = half.mul(half).sub(wc2);
        let sqrt_disc = complex_sqrt(disc);
        bs_poles.push(half.add(sqrt_disc));
        bs_poles.push(half.sub(sqrt_disc));
    }
    (bs_zeros, bs_poles)
}

fn complex_sqrt(z: Cpx) -> Cpx {
    let r = z.norm().sqrt();
    let theta = z.im.atan2(z.re) / 2.0;
    Cpx::new(r * theta.cos(), r * theta.sin())
}

// ─── Chebyshev Type I IIR ────────────────────────────────────────────────────

/// Design a Chebyshev Type I digital filter (equiripple passband).
pub fn cheby1(
    order: usize,
    ripple_db: f64,
    cutoff: &[f64],
    filter_type: FilterType,
) -> SignalResult<IirCoefficients> {
    validate_order_nonzero(order)?;
    validate_cutoffs(cutoff, filter_type)?;
    if ripple_db <= 0.0 {
        return Err(SignalError::ValueError(
            "ripple_db must be positive".to_string(),
        ));
    }

    let epsilon = (10.0_f64.powf(ripple_db / 10.0) - 1.0).sqrt();
    let poles = cheby1_analog_poles(order, epsilon);

    match filter_type {
        FilterType::Lowpass => {
            let wc = prewarp(cutoff[0]);
            let scaled: Vec<Cpx> = poles.iter().map(|&p| p.scale(wc)).collect();
            let (bz, az, g) = analog_to_digital_lp(&scaled, wc, order);
            normalize_ba(zpk_to_ba(&bz, &az, g).0, zpk_to_ba(&bz, &az, g).1)
        }
        FilterType::Highpass => {
            let wc = prewarp(cutoff[0]);
            let hp: Vec<Cpx> = poles.iter().map(|&p| Cpx::new(wc, 0.0).div(p)).collect();
            let (bz, az, g) = analog_to_digital_hp(&hp, order);
            normalize_ba(zpk_to_ba(&bz, &az, g).0, zpk_to_ba(&bz, &az, g).1)
        }
        FilterType::Bandpass => {
            let wl = prewarp(cutoff[0]);
            let wh = prewarp(cutoff[1]);
            let wc = (wl * wh).sqrt();
            let bw = wh - wl;
            let (zs, ps) = analog_bp_transform(&poles, wc, bw);
            let (bz, az, g) = analog_to_digital_general(&zs, &ps);
            normalize_ba(zpk_to_ba(&bz, &az, g).0, zpk_to_ba(&bz, &az, g).1)
        }
        FilterType::Bandstop => {
            let wl = prewarp(cutoff[0]);
            let wh = prewarp(cutoff[1]);
            let wc = (wl * wh).sqrt();
            let bw = wh - wl;
            let (zs, ps) = analog_bs_transform(&poles, wc, bw);
            let (bz, az, g) = analog_to_digital_general(&zs, &ps);
            normalize_ba(zpk_to_ba(&bz, &az, g).0, zpk_to_ba(&bz, &az, g).1)
        }
    }
}

fn cheby1_analog_poles(order: usize, epsilon: f64) -> Vec<Cpx> {
    let n = order as f64;
    // arcsinh(1/ε) / N
    let v0 = (1.0 / epsilon + (1.0 / epsilon / epsilon + 1.0).sqrt())
        .ln()
        / n;

    (0..order)
        .map(|k| {
            let theta = PI * (2 * k + 1) as f64 / (2.0 * n);
            let re = -v0.sinh() * theta.sin();
            let im = v0.cosh() * theta.cos();
            Cpx::new(re, im)
        })
        .collect()
}

// ─── Chebyshev Type II IIR ───────────────────────────────────────────────────

/// Design a Chebyshev Type II digital filter (equiripple stopband).
pub fn cheby2(
    order: usize,
    stopband_attn_db: f64,
    cutoff: &[f64],
    filter_type: FilterType,
) -> SignalResult<IirCoefficients> {
    validate_order_nonzero(order)?;
    validate_cutoffs(cutoff, filter_type)?;
    if stopband_attn_db <= 0.0 {
        return Err(SignalError::ValueError(
            "stopband_attn_db must be positive".to_string(),
        ));
    }

    let epsilon = (10.0_f64.powf(stopband_attn_db / 10.0) - 1.0).sqrt();
    let (zeros, poles) = cheby2_analog_zpk(order, epsilon);

    match filter_type {
        FilterType::Lowpass => {
            let wc = prewarp(cutoff[0]);
            let sc_poles: Vec<Cpx> = poles.iter().map(|&p| p.scale(wc)).collect();
            let sc_zeros: Vec<Cpx> = zeros.iter().map(|&z| z.scale(wc)).collect();
            let (bz, az, g) = analog_to_digital_general(&sc_zeros, &sc_poles);
            normalize_ba(zpk_to_ba(&bz, &az, g).0, zpk_to_ba(&bz, &az, g).1)
        }
        FilterType::Highpass => {
            let wc = prewarp(cutoff[0]);
            // HP transformation: s → wc/s
            let hp_poles: Vec<Cpx> = poles.iter().map(|&p| Cpx::new(wc, 0.0).div(p)).collect();
            let hp_zeros: Vec<Cpx> = zeros.iter().map(|&z| Cpx::new(wc, 0.0).div(z)).collect();
            let (bz, az, g) = analog_to_digital_general(&hp_zeros, &hp_poles);
            normalize_ba(zpk_to_ba(&bz, &az, g).0, zpk_to_ba(&bz, &az, g).1)
        }
        FilterType::Bandpass => {
            let wl = prewarp(cutoff[0]);
            let wh = prewarp(cutoff[1]);
            let wc = (wl * wh).sqrt();
            let bw = wh - wl;
            let (zs, ps) = analog_bp_transform(&poles, wc, bw);
            let (zzs, _) = analog_bp_transform(&zeros, wc, bw);
            let (bz, az, g) = analog_to_digital_general(&zzs, &zs.iter().chain(ps.iter()).cloned().collect::<Vec<_>>());
            normalize_ba(zpk_to_ba(&bz, &az, g).0, zpk_to_ba(&bz, &az, g).1)
        }
        FilterType::Bandstop => {
            let wl = prewarp(cutoff[0]);
            let wh = prewarp(cutoff[1]);
            let wc = (wl * wh).sqrt();
            let bw = wh - wl;
            let (zs, ps) = analog_bs_transform(&poles, wc, bw);
            let (zzs, _) = analog_bs_transform(&zeros, wc, bw);
            let all_poles: Vec<Cpx> = zs.iter().chain(ps.iter()).cloned().collect();
            let (bz, az, g) = analog_to_digital_general(&zzs, &all_poles);
            normalize_ba(zpk_to_ba(&bz, &az, g).0, zpk_to_ba(&bz, &az, g).1)
        }
    }
}

/// Chebyshev Type II analog prototype zeros and poles.
/// The prototype stopband edge is at Ω=1.
fn cheby2_analog_zpk(order: usize, epsilon: f64) -> (Vec<Cpx>, Vec<Cpx>) {
    let n = order as f64;
    // arcsinh(ε) / N  (reciprocal of Cheby-I formula)
    let v0 = (epsilon + (epsilon * epsilon + 1.0).sqrt()).ln() / n;

    // Zeros: purely imaginary, at Ω = 1/cos(π(2k+1)/(2N))
    let zeros: Vec<Cpx> = (0..order)
        .filter_map(|k| {
            let theta = PI * (2 * k + 1) as f64 / (2.0 * n);
            let cos_theta = theta.cos();
            if cos_theta.abs() < 1e-10 {
                None
            } else {
                Some(Cpx::new(0.0, 1.0 / cos_theta))
            }
        })
        .flat_map(|z| [z, z.conj()])
        .take(order)
        .collect();

    // Poles: reciprocal of Cheby-I poles (inversion in complex plane)
    let poles: Vec<Cpx> = (0..order)
        .map(|k| {
            let theta = PI * (2 * k + 1) as f64 / (2.0 * n);
            let re = -v0.sinh() * theta.sin();
            let im = v0.cosh() * theta.cos();
            // Type II pole = 1 / conj(Cheby-I pole)
            let p = Cpx::new(re, im);
            Cpx::new(1.0, 0.0).div(p.conj())
        })
        .collect();

    (zeros, poles)
}

// ─── FIR window method ───────────────────────────────────────────────────────

/// Window function for FIR filter design.
#[derive(Debug, Clone, Copy)]
pub enum FirWindow {
    Hamming,
    Hann,
    Blackman,
    /// Kaiser window with shape parameter β.
    Kaiser(f64),
    Rectangular,
}

/// Design an FIR filter using the window method.
///
/// Returns the impulse response (length = `n_taps`).
///
/// For `Lowpass` and `Highpass` supply one cutoff; for `Bandpass`/`Bandstop`
/// supply two cutoffs.  Frequencies are normalised 0–1 (1 = Nyquist).
pub fn firwin(
    n_taps: usize,
    cutoff: &[f64],
    window: FirWindow,
    filter_type: FilterType,
) -> SignalResult<Vec<f64>> {
    if n_taps < 3 {
        return Err(SignalError::ValueError(
            "n_taps must be at least 3".to_string(),
        ));
    }
    validate_cutoffs(cutoff, filter_type)?;

    let win = make_window(n_taps, window);

    // Convert normalised freq (0-1) to rad/sample (0-π)
    let to_rad = |f: f64| f * PI;

    let h: Vec<f64> = match filter_type {
        FilterType::Lowpass => {
            let wc = to_rad(cutoff[0]);
            sinc_lp(n_taps, wc)
        }
        FilterType::Highpass => {
            let wc = to_rad(cutoff[0]);
            sinc_hp(n_taps, wc)
        }
        FilterType::Bandpass => {
            let wl = to_rad(cutoff[0]);
            let wh = to_rad(cutoff[1]);
            // Bandpass = LP(wh) - LP(wl)
            let hl = sinc_lp(n_taps, wl);
            let hh = sinc_lp(n_taps, wh);
            hl.iter().zip(hh.iter()).map(|(&a, &b)| b - a).collect()
        }
        FilterType::Bandstop => {
            let wl = to_rad(cutoff[0]);
            let wh = to_rad(cutoff[1]);
            // Bandstop = LP(wl) + HP(wh) = LP(wl) + (delta - LP(wh))
            let hl = sinc_lp(n_taps, wl);
            let hh = sinc_lp(n_taps, wh);
            let mid = (n_taps - 1) / 2;
            hl.iter()
                .zip(hh.iter())
                .enumerate()
                .map(|(i, (&a, &b))| {
                    let delta = if i == mid { 1.0 } else { 0.0 };
                    a + delta - b
                })
                .collect()
        }
    };

    // Apply window
    let mut hw: Vec<f64> = h.iter().zip(win.iter()).map(|(&hi, &wi)| hi * wi).collect();

    // Normalise
    match filter_type {
        FilterType::Lowpass | FilterType::Bandpass => {
            let sum: f64 = hw.iter().sum();
            if sum.abs() > 1e-15 {
                hw.iter_mut().for_each(|v| *v /= sum);
            }
        }
        FilterType::Highpass | FilterType::Bandstop => {
            // Normalise so that gain at Nyquist (alternating sign sum) = 1
            let nyq: f64 = hw
                .iter()
                .enumerate()
                .map(|(i, &v)| v * (-1.0_f64).powi(i as i32))
                .sum();
            if nyq.abs() > 1e-15 {
                hw.iter_mut().for_each(|v| *v /= nyq);
            }
        }
    }

    Ok(hw)
}

/// Ideal low-pass impulse response (un-windowed) cut at `wc` rad/sample.
fn sinc_lp(n: usize, wc: f64) -> Vec<f64> {
    let mid = (n - 1) as f64 / 2.0;
    (0..n)
        .map(|i| {
            let t = i as f64 - mid;
            if t == 0.0 {
                wc / PI
            } else {
                (wc * t).sin() / (PI * t)
            }
        })
        .collect()
}

/// Ideal high-pass impulse response via spectral complement.
fn sinc_hp(n: usize, wc: f64) -> Vec<f64> {
    let lp = sinc_lp(n, wc);
    let mid = (n - 1) / 2;
    lp.iter()
        .enumerate()
        .map(|(i, &v)| {
            let delta = if i == mid { 1.0 } else { 0.0 };
            delta - v
        })
        .collect()
}

fn make_window(n: usize, window: FirWindow) -> Vec<f64> {
    match window {
        FirWindow::Rectangular => vec![1.0; n],
        FirWindow::Hamming => (0..n)
            .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos())
            .collect(),
        FirWindow::Hann => (0..n)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
            .collect(),
        FirWindow::Blackman => (0..n)
            .map(|i| {
                let x = 2.0 * PI * i as f64 / (n - 1) as f64;
                0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos()
            })
            .collect(),
        FirWindow::Kaiser(beta) => kaiser_window(n, beta),
    }
}

fn kaiser_window(n: usize, beta: f64) -> Vec<f64> {
    let i0_beta = bessel_i0(beta);
    (0..n)
        .map(|i| {
            let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
            bessel_i0(beta * (1.0 - x * x).max(0.0).sqrt()) / i0_beta
        })
        .collect()
}

/// Modified Bessel function of order 0, I₀(x), via series.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    for k in 1..=30 {
        term *= (x / 2.0) / k as f64;
        term *= (x / 2.0) / k as f64;
        sum += term;
        if term < 1e-15 * sum {
            break;
        }
    }
    sum
}

// ─── Parks-McClellan (Remez exchange) FIR design ────────────────────────────

/// Design an optimal equiripple FIR filter via the Parks-McClellan algorithm.
///
/// * `bands`   — band edges in Hz; normalised by `fs/2`, so supply actual Hz.
/// * `desired` — desired gain at each band edge (pairs matching `bands`).
/// * `weights` — relative weight for each band (length = `bands.len() / 2`).
/// * `fs`      — sample rate in Hz (used to normalise `bands`).
pub fn remez(
    n_taps: usize,
    bands: &[f64],
    desired: &[f64],
    weights: &[f64],
    fs: f64,
) -> SignalResult<Vec<f64>> {
    if n_taps < 3 {
        return Err(SignalError::ValueError(
            "n_taps must be at least 3".to_string(),
        ));
    }
    if bands.len() < 2 || bands.len() % 2 != 0 {
        return Err(SignalError::ValueError(
            "bands must have an even number of elements ≥ 2".to_string(),
        ));
    }
    if desired.len() != bands.len() {
        return Err(SignalError::ValueError(
            "desired must have the same length as bands".to_string(),
        ));
    }
    if fs <= 0.0 {
        return Err(SignalError::ValueError(
            "fs must be positive".to_string(),
        ));
    }

    let nyq = fs / 2.0;
    // Normalise band edges to [0, 1]
    let norm_bands: Vec<f64> = bands.iter().map(|&f| f / nyq).collect();

    // Validate
    for i in 1..norm_bands.len() {
        if norm_bands[i] < norm_bands[i - 1] {
            return Err(SignalError::ValueError(
                "band edges must be non-decreasing".to_string(),
            ));
        }
    }
    if norm_bands[0] < 0.0 || norm_bands[norm_bands.len() - 1] > 1.0 {
        return Err(SignalError::ValueError(
            "band edges (normalised) must be within [0, 1]".to_string(),
        ));
    }

    let n_bands = norm_bands.len() / 2;
    let band_weights: Vec<f64> = if weights.len() == n_bands {
        weights.to_vec()
    } else {
        vec![1.0; n_bands]
    };

    remez_exchange(n_taps, &norm_bands, desired, &band_weights)
}

/// Core Remez exchange algorithm.
fn remez_exchange(
    n_taps: usize,
    norm_bands: &[f64],
    desired: &[f64],
    weights: &[f64],
) -> SignalResult<Vec<f64>> {
    let n_order = n_taps - 1;
    let r = n_order / 2 + 1; // number of extremal frequencies for Type I/II

    // Build dense frequency grid (ω in [0, π])
    let grid_density: usize = 16;
    let total_grid = grid_density * n_order;
    let mut omega_grid: Vec<f64> = Vec::with_capacity(total_grid);
    let mut des_grid: Vec<f64> = Vec::with_capacity(total_grid);
    let mut wt_grid: Vec<f64> = Vec::with_capacity(total_grid);

    let n_bands = norm_bands.len() / 2;
    for bi in 0..n_bands {
        let f0 = norm_bands[2 * bi];
        let f1 = norm_bands[2 * bi + 1];
        let pts = ((f1 - f0) * total_grid as f64).round().max(2.0) as usize;
        for i in 0..pts {
            let f = f0 + (f1 - f0) * i as f64 / (pts - 1) as f64;
            let t = i as f64 / (pts - 1) as f64;
            omega_grid.push(f * PI);
            let d = desired[2 * bi] * (1.0 - t) + desired[2 * bi + 1] * t;
            des_grid.push(d);
            wt_grid.push(weights[bi]);
        }
    }

    if omega_grid.len() < r + 1 {
        return Err(SignalError::ValueError(
            "Grid too small for the requested filter order".to_string(),
        ));
    }

    // Initialise extremal set evenly
    let mut ext: Vec<usize> = (0..r)
        .map(|i| i * (omega_grid.len() - 1) / (r - 1))
        .collect();

    let max_iter = 50;
    let mut h = vec![0.0_f64; n_taps];

    for _iter in 0..max_iter {
        // Barycentric interpolation: Chebyshev solution via Lagrange
        // Compute delta (equiripple error) from the r extremal points
        let x: Vec<f64> = ext.iter().map(|&i| omega_grid[i].cos()).collect();
        let d: Vec<f64> = ext.iter().map(|&i| des_grid[i]).collect();
        let w: Vec<f64> = ext.iter().map(|&i| wt_grid[i]).collect();

        // Barycentric weights for Lagrange interpolation
        let mut bary = vec![1.0_f64; r];
        for i in 0..r {
            for j in 0..r {
                if i != j {
                    let diff = x[i] - x[j];
                    if diff.abs() > 1e-15 {
                        bary[i] /= diff;
                    }
                }
            }
        }

        // Compute delta: alternation theorem
        let mut num_delta = 0.0_f64;
        let mut den_delta = 0.0_f64;
        for i in 0..r {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            num_delta += bary[i] * d[i];
            den_delta += sign * bary[i] / w[i];
        }
        let delta = if den_delta.abs() > 1e-30 {
            num_delta / den_delta
        } else {
            0.0
        };

        // Evaluate interpolated response on dense grid
        let mut errors: Vec<f64> = Vec::with_capacity(omega_grid.len());
        for (gi, &om) in omega_grid.iter().enumerate() {
            let xg = om.cos();
            let mut num = 0.0_f64;
            let mut den = 0.0_f64;
            for i in 0..r {
                let d_xi = xg - x[i];
                if d_xi.abs() < 1e-15 {
                    num = d[i];
                    den = 1.0;
                    break;
                }
                let b = bary[i] / d_xi;
                num += b * d[i];
                den += b;
            }
            let resp = if den.abs() > 1e-30 { num / den } else { d[0] };
            let err = (des_grid[gi] - resp) * wt_grid[gi];
            errors.push(err.abs());
        }

        // Find new extremal frequencies: r local maxima of |error|
        let mut new_ext: Vec<usize> = Vec::new();
        // Include endpoints
        for i in 1..(errors.len() - 1) {
            if errors[i] >= errors[i - 1] && errors[i] >= errors[i + 1] {
                new_ext.push(i);
            }
        }
        if errors.first().copied().unwrap_or(0.0) > errors.get(1).copied().unwrap_or(0.0) {
            new_ext.insert(0, 0);
        }
        if errors.last().copied().unwrap_or(0.0)
            > errors.get(errors.len() - 2).copied().unwrap_or(0.0)
        {
            new_ext.push(errors.len() - 1);
        }

        // Keep only r largest
        new_ext.sort_by(|&a, &b| errors[b].partial_cmp(&errors[a]).unwrap_or(std::cmp::Ordering::Equal));
        new_ext.truncate(r);
        new_ext.sort();
        if new_ext.len() < r {
            break; // Converged or can't find more
        }
        ext = new_ext;

        // Build filter coefficients from current extremal set (IDFT)
        let half = n_taps / 2;
        let x_pts: Vec<f64> = ext.iter().map(|&i| omega_grid[i].cos()).collect();
        let d_pts: Vec<f64> = ext.iter().map(|&i| des_grid[i]).collect();

        // Inverse DFT: h[k] = (1/N) * sum A(k) * cos(k * omega)
        let mut a_coeffs = vec![0.0_f64; r];
        for (i, &xi) in x_pts.iter().enumerate() {
            let omega_i = xi.acos();
            for k in 0..r {
                a_coeffs[k] += d_pts[i] * (k as f64 * omega_i).cos();
            }
        }
        if r > 0 {
            for c in &mut a_coeffs {
                *c /= r as f64;
            }
        }

        // Convert cosine-series to symmetric FIR filter taps
        for k in 0..n_taps {
            let n_idx = k as f64 - half as f64;
            let mut val = a_coeffs[0];
            for m in 1..r {
                val += 2.0 * a_coeffs[m] * (m as f64 * PI * k as f64 / (n_taps - 1) as f64).cos();
            }
            h[k] = val / (n_taps as f64);
        }

        // Check convergence (delta change)
        if delta.abs() < 1e-10 {
            break;
        }
    }

    // Symmetrise
    let mid = n_taps / 2;
    for i in 0..mid {
        let avg = (h[i] + h[n_taps - 1 - i]) / 2.0;
        h[i] = avg;
        h[n_taps - 1 - i] = avg;
    }

    Ok(h)
}

// ─── FIR application helpers ─────────────────────────────────────────────────

/// Convolve (apply) a FIR filter `h` to input `x` (causal, no delay removal).
pub fn lfilter_fir(h: &[f64], x: &[f64]) -> Vec<f64> {
    let nh = h.len();
    let nx = x.len();
    let mut y = vec![0.0_f64; nx];
    for n in 0..nx {
        let mut acc = 0.0_f64;
        for k in 0..nh {
            if n >= k {
                acc += h[k] * x[n - k];
            }
        }
        y[n] = acc;
    }
    y
}

/// Apply a FIR filter forward *and* backward (zero-phase).
pub fn fir_filtfilt(h: &[f64], x: &[f64]) -> Vec<f64> {
    let y1 = lfilter_fir(h, x);
    let mut y1r = y1;
    y1r.reverse();
    let y2 = lfilter_fir(h, &y1r);
    let mut result = y2;
    result.reverse();
    result
}

// ─── Validation helpers ───────────────────────────────────────────────────────

fn validate_order_nonzero(order: usize) -> SignalResult<()> {
    if order == 0 {
        Err(SignalError::ValueError("filter order must be ≥ 1".to_string()))
    } else {
        Ok(())
    }
}

fn validate_cutoffs(cutoff: &[f64], filter_type: FilterType) -> SignalResult<()> {
    match filter_type {
        FilterType::Lowpass | FilterType::Highpass => {
            if cutoff.len() < 1 {
                return Err(SignalError::ValueError(
                    "need at least one cutoff frequency".to_string(),
                ));
            }
            if cutoff[0] <= 0.0 || cutoff[0] >= 1.0 {
                return Err(SignalError::ValueError(format!(
                    "cutoff {} must be in (0, 1)",
                    cutoff[0]
                )));
            }
        }
        FilterType::Bandpass | FilterType::Bandstop => {
            if cutoff.len() < 2 {
                return Err(SignalError::ValueError(
                    "need two cutoff frequencies for BP/BS".to_string(),
                ));
            }
            if cutoff[0] <= 0.0 || cutoff[1] >= 1.0 || cutoff[0] >= cutoff[1] {
                return Err(SignalError::ValueError(format!(
                    "cutoffs [{}, {}] invalid: need 0 < low < high < 1",
                    cutoff[0], cutoff[1]
                )));
            }
        }
    }
    Ok(())
}

/// Normalise (b, a) so that a[0] == 1.
fn normalize_ba(b: Vec<f64>, a: Vec<f64>) -> SignalResult<IirCoefficients> {
    if a.is_empty() {
        return Err(SignalError::ComputationError(
            "Denominator is empty".to_string(),
        ));
    }
    let a0 = a[0];
    if a0.abs() < 1e-30 {
        return Err(SignalError::ComputationError(
            "a[0] is zero after normalisation".to_string(),
        ));
    }
    let b_out: Vec<f64> = b.iter().map(|&v| v / a0).collect();
    let a_out: Vec<f64> = a.iter().map(|&v| v / a0).collect();
    Ok(IirCoefficients { b: b_out, a: a_out })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Evaluate the magnitude of the filter H(z) at normalised frequency f in [0,1]
    fn eval_mag(iir: &IirCoefficients, f: f64) -> f64 {
        let omega = f * PI;
        let z = num_complex_exp(omega);
        let b_val = poly_eval_z(&iir.b, z);
        let a_val = poly_eval_z(&iir.a, z);
        if a_val.norm() < 1e-30 {
            return 0.0;
        }
        b_val.norm() / a_val.norm()
    }

    // ── Butterworth tests ──────────────────────────────────────────────────

    #[test]
    fn test_butter_lp_dc_passes() {
        let f = butter(4, &[0.3], FilterType::Lowpass).expect("butter LP failed");
        let mag_dc = eval_mag(&f, 0.0);
        assert!(
            (mag_dc - 1.0).abs() < 0.05,
            "DC gain should be ~1, got {mag_dc}"
        );
    }

    #[test]
    fn test_butter_lp_high_freq_attenuated() {
        let f = butter(4, &[0.3], FilterType::Lowpass).expect("butter LP");
        let mag_hf = eval_mag(&f, 0.9);
        assert!(mag_hf < 0.1, "High freq should be attenuated, got {mag_hf}");
    }

    #[test]
    fn test_butter_hp_dc_blocked() {
        let f = butter(4, &[0.3], FilterType::Highpass).expect("butter HP");
        let mag_dc = eval_mag(&f, 0.0);
        assert!(mag_dc < 0.1, "DC should be blocked by HP, got {mag_dc}");
    }

    #[test]
    fn test_butter_hp_high_freq_passes() {
        let f = butter(4, &[0.3], FilterType::Highpass).expect("butter HP");
        let mag_hf = eval_mag(&f, 0.9);
        assert!(
            mag_hf > 0.7,
            "High freq should pass through HP, got {mag_hf}"
        );
    }

    // ── Chebyshev I tests ─────────────────────────────────────────────────

    #[test]
    fn test_cheby1_lp_passband_not_zero() {
        let f = cheby1(4, 0.5, &[0.3], FilterType::Lowpass).expect("cheby1 LP");
        // At the passband edge the gain should be well above 0
        let mag = eval_mag(&f, 0.1);
        assert!(mag > 0.5, "cheby1 passband gain too low: {mag}");
    }

    #[test]
    fn test_cheby1_lp_stopband_attenuated() {
        let f = cheby1(4, 1.0, &[0.2], FilterType::Lowpass).expect("cheby1 LP");
        let mag = eval_mag(&f, 0.8);
        assert!(mag < 0.2, "cheby1 stopband should be attenuated, got {mag}");
    }

    // ── FIR firwin tests ──────────────────────────────────────────────────

    #[test]
    fn test_firwin_lp_correct_length() {
        let h = firwin(65, &[0.3], FirWindow::Hamming, FilterType::Lowpass)
            .expect("firwin LP");
        assert_eq!(h.len(), 65);
    }

    #[test]
    fn test_firwin_lp_dc_near_unity() {
        let h = firwin(65, &[0.3], FirWindow::Hamming, FilterType::Lowpass)
            .expect("firwin LP");
        let dc: f64 = h.iter().sum();
        assert!((dc - 1.0).abs() < 0.01, "FIR LP DC gain should be ~1, got {dc}");
    }

    #[test]
    fn test_firwin_lp_nyquist_near_zero() {
        let h = firwin(65, &[0.3], FirWindow::Hamming, FilterType::Lowpass)
            .expect("firwin LP");
        let nyq: f64 = h
            .iter()
            .enumerate()
            .map(|(i, &v)| v * (-1.0_f64).powi(i as i32))
            .sum();
        assert!(nyq.abs() < 0.05, "FIR LP Nyquist gain should be ~0, got {nyq}");
    }

    #[test]
    fn test_firwin_hp_nyquist_near_unity() {
        let h = firwin(65, &[0.3], FirWindow::Hamming, FilterType::Highpass)
            .expect("firwin HP");
        let nyq: f64 = h
            .iter()
            .enumerate()
            .map(|(i, &v)| v * (-1.0_f64).powi(i as i32))
            .sum();
        assert!((nyq - 1.0).abs() < 0.05, "FIR HP Nyquist gain should be ~1, got {nyq}");
    }

    // ── IirCoefficients::filtfilt preserves DC ────────────────────────────

    #[test]
    fn test_filtfilt_allpass_preserves_dc() {
        // All-pass: b = [1], a = [1]
        let iir = IirCoefficients {
            b: vec![1.0],
            a: vec![1.0],
        };
        let x: Vec<f64> = (0..50).map(|_| 2.5).collect(); // DC signal
        let y = iir.filtfilt(&x).expect("filtfilt failed");
        for (i, &yi) in y.iter().enumerate() {
            assert!(
                (yi - 2.5).abs() < 1e-9,
                "filtfilt should preserve DC at index {i}, got {yi}"
            );
        }
    }

    // ── freqz flatline for all-pass ───────────────────────────────────────

    #[test]
    fn test_freqz_allpass_flat() {
        let iir = IirCoefficients {
            b: vec![1.0],
            a: vec![1.0],
        };
        let (_, mags) = iir.freqz(64);
        for &m in &mags {
            assert!(m.abs() < 1e-9, "All-pass freqz should be 0 dB, got {m}");
        }
    }

    // ── remez basic smoke test ─────────────────────────────────────────────

    #[test]
    fn test_remez_returns_correct_length() {
        let bands = vec![0.0, 1600.0, 2000.0, 8000.0];
        let desired = vec![1.0, 1.0, 0.0, 0.0];
        let weights = vec![1.0, 1.0];
        let h = remez(33, &bands, &desired, &weights, 16000.0).expect("remez failed");
        assert_eq!(h.len(), 33);
    }

    // ── lfilter_fir / fir_filtfilt ────────────────────────────────────────

    #[test]
    fn test_lfilter_fir_impulse_response() {
        let h = vec![0.25, 0.5, 0.25];
        let impulse: Vec<f64> = std::iter::once(1.0).chain(std::iter::repeat(0.0).take(9)).collect();
        let y = lfilter_fir(&h, &impulse);
        assert!((y[0] - 0.25).abs() < 1e-12);
        assert!((y[1] - 0.5).abs() < 1e-12);
        assert!((y[2] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_fir_filtfilt_dc_preserved() {
        let h = vec![0.25, 0.5, 0.25]; // DC gain = 1
        let x: Vec<f64> = vec![3.0; 40];
        let y = fir_filtfilt(&h, &x);
        // Middle of the signal should be stable at 3.0
        for &yi in y[10..30].iter() {
            assert!((yi - 3.0).abs() < 0.01, "DC not preserved: {yi}");
        }
    }
}
