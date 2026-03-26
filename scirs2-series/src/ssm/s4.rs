//! S4 (Structured State Space Sequence) layer implementation.
//!
//! Implements the S4 model from:
//!   Gu et al. (2021) "Efficiently Modeling Long Sequences with Structured State Spaces"
//!   <https://arxiv.org/abs/2111.00396>
//!
//! # Core SSM
//!
//! The continuous-time SSM is:
//!   x'(t) = A x(t) + B u(t)
//!   y(t)  = C x(t) + D u(t)
//!
//! where A ∈ ℝ^{N×N}, B ∈ ℝ^{N×1}, C ∈ ℝ^{1×N}, D ∈ ℝ^{1×1}.
//!
//! # Discretization (Zero-Order Hold)
//!
//! Given timestep Δ:
//!   Ā = exp(Δ·A)
//!   B̄ = (Ā − I) · A^{-1} · B
//!
//! For the diagonal approximation used here, A = diag(a₁,...,a_N):
//!   Ā = diag(exp(Δ·aₙ))
//!   B̄ = diag((exp(Δ·aₙ) − 1) / aₙ) · B
//!
//! # HiPPO-LegS Initialization
//!
//! The A matrix is initialized from the HiPPO-LegS basis:
//!   A_{nk} = −(2n+1)^{1/2} (2k+1)^{1/2}  if n > k
//!   A_{nn} = −(n+1)
//!   A_{nk} = 0                              if n < k
//!
//! The eigenvalues are separated into diagonal (stable) and low-rank correction.
//! For inference efficiency we use a diagonal approximation of the eigenvalues.
//!
//! # Convolution Mode
//!
//! Given the kernel K_L = (CB̄, CĀB̄, CĀ²B̄, ..., CĀ^{L-1}B̄),
//! the output is y = K_L * u (causal convolution).
//!
//! We compute this via FFT: Y(ω) = K(ω) · U(ω), then IFFT.

use std::f64::consts::PI;

use scirs2_core::ndarray::Array2;
use scirs2_core::random::RngExt;

use crate::error::{Result, TimeSeriesError};

use super::config::S4Config;

// ---------------------------------------------------------------------------
// HiPPO matrix construction
// ---------------------------------------------------------------------------

/// Construct the HiPPO-LegS A matrix of dimension `n × n`.
///
/// This represents the optimal polynomial projection operator for the
/// Legendre measure (decaying memory). The matrix has entries:
///   A_{nk} = -(2n+1)^{1/2} (2k+1)^{1/2}  if n > k
///   A_{nn} = -(n+1)
///   A_{nk} = 0                              if n < k
pub fn hippo_legs_matrix(n: usize) -> Array2<f64> {
    let mut a = Array2::zeros((n, n));
    for row in 0..n {
        // Diagonal: -(n+1)
        a[[row, row]] = -((row + 1) as f64);
        // Below diagonal
        for col in 0..row {
            let val = -((2 * row + 1) as f64).sqrt() * ((2 * col + 1) as f64).sqrt();
            a[[row, col]] = val;
        }
    }
    a
}

/// Extract the diagonal of the HiPPO-A matrix (used for diagonal SSM approximation).
///
/// In the NPLR representation, A = Λ - P Q^* where Λ is diagonal.
/// For the diagonal approximation we use Λ ≈ diag(A).
pub fn hippo_legs_diagonal(n: usize) -> Vec<f64> {
    (0..n).map(|k| -((k + 1) as f64)).collect()
}

/// Compute the B vector for HiPPO-LegS initialization.
///
/// B_n = (2n+1)^{1/2}, which represents the input projection coefficients
/// for the Legendre polynomial basis.
pub fn hippo_legs_b(n: usize) -> Vec<f64> {
    (0..n).map(|k| ((2 * k + 1) as f64).sqrt()).collect()
}

// ---------------------------------------------------------------------------
// S4 Layer
// ---------------------------------------------------------------------------

/// S4 (Structured State Space Sequence) layer.
///
/// Applies an SSM with HiPPO-initialized parameters to sequences.
/// In convolution mode, the entire sequence is processed in O(L log L) time
/// via FFT convolution.
///
/// # Shape Convention
///
/// Input `u` has shape `[seq_len, d_model]` (sequence-first).
/// Output `y` has the same shape.
pub struct S4Layer {
    /// Layer configuration.
    pub config: S4Config,
    /// Log of diagonal SSM eigenvalues, shape `[d_state]`.
    /// The actual eigenvalues are `-exp(a_log)` (negative for stability).
    a_log: Vec<f64>,
    /// Input projection B, shape `[d_state]`.
    b: Vec<f64>,
    /// Output projection C, shape `[d_model, d_state]`.
    c: Array2<f64>,
    /// Skip connection (D) coefficients, shape `[d_model]`.
    d: Vec<f64>,
    /// Log timestep per model dimension, shape `[d_model]`.
    log_dt: Vec<f64>,
    /// Precomputed convolution kernel `[d_model, seq_len]` (None = needs recompute).
    kernel: Option<Array2<f64>>,
}

impl S4Layer {
    /// Create a new S4Layer with HiPPO initialization.
    ///
    /// # Arguments
    /// * `config` - Layer configuration.
    /// * `rng` - Random number generator for parameter initialization.
    pub fn new(config: &S4Config, rng: &mut impl scirs2_core::random::Rng) -> Self {
        let n = config.d_state;
        let d = config.d_model;

        // A: use negative diagonal of HiPPO for stability.
        // Store log(|eigenvalue|); the sign is always negative.
        let hippo_diag = hippo_legs_diagonal(n);
        let a_log: Vec<f64> = hippo_diag.iter().map(|&v| v.abs().ln()).collect();

        // B: HiPPO input projection
        let b = hippo_legs_b(n);

        // C: random initialization N(0, 1/sqrt(N))
        let scale_c = 1.0 / (n as f64).sqrt();
        let c_data: Vec<f64> = (0..d * n)
            .map(|_| {
                let u1: f64 = rng.random();
                let u2: f64 = rng.random();
                // Box-Muller for normal sample
                let bm: f64 = (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * PI * u2).cos();
                bm * scale_c
            })
            .collect();
        let c = Array2::from_shape_vec((d, n), c_data).unwrap_or_else(|_| Array2::zeros((d, n)));

        // D: skip connection, initialized to 1
        let d_param = vec![1.0_f64; d];

        // log_dt: uniform in [log(dt_min), log(dt_max)] per channel
        let log_dt_min = config.dt_min.ln();
        let log_dt_max = config.dt_max.ln();
        let log_dt: Vec<f64> = (0..d)
            .map(|_| {
                let u: f64 = rng.random::<f64>();
                log_dt_min + u * (log_dt_max - log_dt_min)
            })
            .collect();

        S4Layer {
            config: config.clone(),
            a_log,
            b,
            c,
            d: d_param,
            log_dt,
            kernel: None,
        }
    }

    /// Compute the convolution kernel K of shape `[d_model, seq_len]`.
    ///
    /// For each channel h and time step l:
    ///   `K[h, l] = Sum_n C[h, n] * (A_bar[h,n])^l * B_bar[h,n]`
    ///
    /// where `A_bar[h,n] = exp(dt_h * (-exp(a_log[n])))`
    /// and   `B_bar[h,n] = (A_bar[h,n] - 1) / (-exp(a_log[n])) * B[n]`
    pub fn compute_kernel(&mut self) {
        let seq_len = self.config.seq_len;
        let d_model = self.config.d_model;
        let d_state = self.config.d_state;

        let mut kernel = Array2::zeros((d_model, seq_len));

        for h in 0..d_model {
            let dt = self.log_dt[h].exp();

            // Precompute per-state discrete parameters
            let mut a_bar = vec![0.0_f64; d_state];
            let mut cb_bar = vec![0.0_f64; d_state]; // C[h,n] * B_bar[n]

            for n in 0..d_state {
                let a_val = -(self.a_log[n].exp()); // negative eigenvalue
                let a_bar_n = (dt * a_val).exp(); // exp(Δ · λ_n)
                a_bar[n] = a_bar_n;

                // B_bar_n = (a_bar_n - 1) / a_val * B[n]
                let b_bar_n = if a_val.abs() > 1e-8 {
                    (a_bar_n - 1.0) / a_val * self.b[n]
                } else {
                    dt * self.b[n] // limit as a_val → 0
                };
                cb_bar[n] = self.c[[h, n]] * b_bar_n;
            }

            // K[h, l] = Σ_n  C[h,n] · Ā[h,n]^l · B̄[h,n]
            //         = Σ_n  cb_bar[n] · a_bar[n]^l
            // We compute this recurrently: power[n] starts at 1 (l=0)
            let mut power = vec![1.0_f64; d_state]; // a_bar[n]^l

            for l in 0..seq_len {
                let mut k_val = 0.0_f64;
                for n in 0..d_state {
                    k_val += cb_bar[n] * power[n];
                    // advance power for next l
                    if l + 1 < seq_len {
                        power[n] *= a_bar[n];
                    }
                }
                kernel[[h, l]] = k_val;
            }
        }

        self.kernel = Some(kernel);
    }

    /// Apply the S4 layer to an input sequence via FFT convolution.
    ///
    /// # Arguments
    /// * `u` - Input tensor of shape `[seq_len, d_model]`.
    ///
    /// # Returns
    /// Output tensor of shape `[seq_len, d_model]`.
    pub fn forward(&self, u: &Array2<f64>) -> Result<Array2<f64>> {
        let (seq_len, d_model) = u.dim();

        if d_model != self.config.d_model {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.config.d_model,
                actual: d_model,
            });
        }

        let kernel = match &self.kernel {
            Some(k) => k,
            None => {
                return Err(TimeSeriesError::ModelNotFitted(
                    "S4 kernel not computed; call compute_kernel() first".to_string(),
                ))
            }
        };

        let kernel_len = kernel.dim().1;

        // Causal convolution: for each channel, convolve u[:,h] with kernel[h,:]
        // using FFT.  We use the minimum of seq_len and kernel_len for the kernel.
        let conv_len = seq_len + kernel_len - 1;
        let fft_len = next_power_of_two(conv_len);

        let mut output = Array2::zeros((seq_len, d_model));

        for h in 0..d_model {
            // Extract input and kernel for this channel
            let u_h: Vec<f64> = (0..seq_len).map(|t| u[[t, h]]).collect();
            let k_len = kernel_len.min(seq_len);
            let k_h: Vec<f64> = (0..k_len).map(|l| kernel[[h, l]]).collect();

            // FFT-based causal convolution
            let conv_result = fft_convolve(&u_h, &k_h, fft_len);

            // Take only the first seq_len values (causal)
            for t in 0..seq_len {
                output[[t, h]] = conv_result[t];
            }
        }

        // Add skip connection: y = conv(u) + D * u
        for t in 0..seq_len {
            for h in 0..d_model {
                output[[t, h]] += self.d[h] * u[[t, h]];
            }
        }

        Ok(output)
    }

    /// Apply the S4 layer in recurrence mode (step-by-step).
    ///
    /// Equivalent to `forward` but processes one step at a time.
    /// Useful for autoregressive generation.
    ///
    /// # Arguments
    /// * `u` - Input tensor `[seq_len, d_model]`.
    ///
    /// # Returns
    /// Output `[seq_len, d_model]`.
    pub fn forward_recurrent(&self, u: &Array2<f64>) -> Result<Array2<f64>> {
        let (seq_len, d_model) = u.dim();

        if d_model != self.config.d_model {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: self.config.d_model,
                actual: d_model,
            });
        }

        let d_state = self.config.d_state;
        let mut output = Array2::zeros((seq_len, d_model));

        for h in 0..d_model {
            let dt = self.log_dt[h].exp();

            // Compute discrete parameters for this channel
            let mut a_bar = vec![0.0_f64; d_state];
            let mut b_bar = vec![0.0_f64; d_state];
            for n in 0..d_state {
                let a_val = -(self.a_log[n].exp());
                let a_bar_n = (dt * a_val).exp();
                a_bar[n] = a_bar_n;
                b_bar[n] = if a_val.abs() > 1e-8 {
                    (a_bar_n - 1.0) / a_val * self.b[n]
                } else {
                    dt * self.b[n]
                };
            }

            // SSM state: h_n = 0 initially
            let mut state = vec![0.0_f64; d_state];

            for t in 0..seq_len {
                let u_t = u[[t, h]];
                // State update: x_{t+1} = Ā x_t + B̄ u_t
                for n in 0..d_state {
                    state[n] = a_bar[n] * state[n] + b_bar[n] * u_t;
                }
                // Output: y_t = C x_t + D u_t
                let y_ssm: f64 = state
                    .iter()
                    .zip(self.c.row(h).iter())
                    .map(|(s, c)| s * c)
                    .sum();
                output[[t, h]] = y_ssm + self.d[h] * u_t;
            }
        }

        Ok(output)
    }

    /// Get the convolution kernel (None if not yet computed).
    pub fn kernel(&self) -> Option<&Array2<f64>> {
        self.kernel.as_ref()
    }

    /// Retrieve the HiPPO A matrix for inspection/testing.
    pub fn hippo_a_matrix(&self) -> Array2<f64> {
        hippo_legs_matrix(self.config.d_state)
    }

    /// Return a reference to the C (output projection) matrix.
    pub fn c_matrix(&self) -> &Array2<f64> {
        &self.c
    }

    /// Return the D (skip connection) vector.
    pub fn d_param(&self) -> &[f64] {
        &self.d
    }
}

// ---------------------------------------------------------------------------
// FFT utilities
// ---------------------------------------------------------------------------

/// Returns the smallest power of two ≥ n.
fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1_usize;
    while p < n {
        p <<= 1;
    }
    p
}

/// Perform FFT-based linear (aperiodic) convolution of `a` and `b`.
///
/// Returns the full convolution of length `a.len() + b.len() - 1`,
/// but we only allocate `fft_len` points for efficiency.
fn fft_convolve(a: &[f64], b: &[f64], fft_len: usize) -> Vec<f64> {
    // Zero-pad inputs to fft_len
    let mut fa = vec![(0.0_f64, 0.0_f64); fft_len]; // (real, imag)
    let mut fb = vec![(0.0_f64, 0.0_f64); fft_len];

    for (i, &v) in a.iter().enumerate() {
        if i < fft_len {
            fa[i].0 = v;
        }
    }
    for (i, &v) in b.iter().enumerate() {
        if i < fft_len {
            fb[i].0 = v;
        }
    }

    // Forward DFT
    let fa_freq = dft_radix2(&fa);
    let fb_freq = dft_radix2(&fb);

    // Pointwise multiply
    let mut fc: Vec<(f64, f64)> = fa_freq
        .iter()
        .zip(fb_freq.iter())
        .map(|(&(ar, ai), &(br, bi))| (ar * br - ai * bi, ar * bi + ai * br))
        .collect();

    // Inverse DFT
    // Conjugate, DFT, conjugate, divide by N
    for v in fc.iter_mut() {
        v.1 = -v.1;
    }
    let mut result = dft_radix2(&fc);
    let n_inv = 1.0 / fft_len as f64;
    for v in result.iter_mut() {
        v.0 *= n_inv;
        v.1 = -v.1 * n_inv;
    }

    // Extract real part (imaginary should be ~0 for real inputs)
    result.into_iter().map(|(r, _)| r).collect()
}

/// Cooley-Tukey radix-2 FFT for complex input.
///
/// Requires `input.len()` to be a power of two.
fn dft_radix2(input: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = input.len();
    if n == 1 {
        return vec![input[0]];
    }

    // Bit-reversal permutation
    let mut a: Vec<(f64, f64)> = bit_reversal_permute(input);

    // Cooley-Tukey butterfly
    let mut len = 2_usize;
    while len <= n {
        let half = len / 2;
        let angle_step = -2.0 * PI / len as f64;
        let w_base = (angle_step.cos(), angle_step.sin());
        let mut j = 0_usize;
        while j < n {
            let mut w = (1.0_f64, 0.0_f64);
            for k in 0..half {
                let u = a[j + k];
                let v = complex_mul(w, a[j + k + half]);
                a[j + k] = (u.0 + v.0, u.1 + v.1);
                a[j + k + half] = (u.0 - v.0, u.1 - v.1);
                w = complex_mul(w, w_base);
            }
            j += len;
        }
        len <<= 1;
    }

    a
}

/// Bit-reversal permutation for FFT.
fn bit_reversal_permute(input: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = input.len();
    let bits = n.trailing_zeros() as usize;
    let mut result = vec![(0.0_f64, 0.0_f64); n];
    for i in 0..n {
        let rev = reverse_bits(i, bits);
        result[rev] = input[i];
    }
    result
}

/// Reverse `bits`-bit representation of `n`.
fn reverse_bits(mut n: usize, bits: usize) -> usize {
    let mut result = 0_usize;
    for _ in 0..bits {
        result = (result << 1) | (n & 1);
        n >>= 1;
    }
    result
}

/// Complex multiplication.
#[inline]
fn complex_mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::SeedableRng;

    fn make_rng() -> impl scirs2_core::random::Rng {
        scirs2_core::random::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_hippo_init_diagonal() {
        let n = 8;
        let diag = hippo_legs_diagonal(n);
        assert_eq!(diag.len(), n);
        // All diagonal entries should be -(k+1)
        for (k, &v) in diag.iter().enumerate() {
            let expected = -((k + 1) as f64);
            assert!(
                (v - expected).abs() < 1e-10,
                "k={k}: got {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_hippo_matrix_structure() {
        let n = 4;
        let a = hippo_legs_matrix(n);
        // Diagonal entries
        assert!((a[[0, 0]] - (-1.0)).abs() < 1e-10);
        assert!((a[[1, 1]] - (-2.0)).abs() < 1e-10);
        // Upper triangular should be zero
        assert_eq!(a[[0, 1]], 0.0);
        assert_eq!(a[[0, 2]], 0.0);
        // Sub-diagonal should be non-zero
        assert!(a[[1, 0]] != 0.0);
    }

    #[test]
    fn test_s4_kernel_shape() {
        let config = S4Config {
            d_model: 8,
            d_state: 4,
            seq_len: 16,
            ..S4Config::default()
        };
        let mut layer = S4Layer::new(&config, &mut make_rng());
        layer.compute_kernel();
        let kernel = layer.kernel().expect("kernel should be computed");
        assert_eq!(kernel.dim(), (8, 16));
    }

    #[test]
    fn test_s4_forward_shape() {
        let config = S4Config {
            d_model: 8,
            d_state: 4,
            seq_len: 16,
            ..S4Config::default()
        };
        let mut layer = S4Layer::new(&config, &mut make_rng());
        layer.compute_kernel();

        let input = Array2::ones((16, 8));
        let output = layer.forward(&input).expect("forward should succeed");
        assert_eq!(output.dim(), (16, 8));
    }

    #[test]
    fn test_s4_conv_vs_recurrence() {
        // On short sequences, convolution and recurrence modes should agree.
        let config = S4Config {
            d_model: 4,
            d_state: 4,
            seq_len: 8,
            ..S4Config::default()
        };
        let mut layer = S4Layer::new(&config, &mut make_rng());
        layer.compute_kernel();

        // Use a deterministic input
        let mut input = Array2::zeros((8, 4));
        for t in 0..8 {
            for h in 0..4 {
                input[[t, h]] = (t * 4 + h) as f64 * 0.1;
            }
        }

        let out_conv = layer.forward(&input).expect("conv forward");
        let out_rec = layer.forward_recurrent(&input).expect("recurrent forward");

        // The two modes compute the same operation; they should agree closely.
        for t in 0..8 {
            for h in 0..4 {
                let diff = (out_conv[[t, h]] - out_rec[[t, h]]).abs();
                assert!(
                    diff < 1e-8,
                    "t={t}, h={h}: conv={:.6}, rec={:.6}, diff={:.2e}",
                    out_conv[[t, h]],
                    out_rec[[t, h]],
                    diff
                );
            }
        }
    }

    #[test]
    fn test_s4_d_model_64_state_16() {
        let config = S4Config::default(); // d_model=64, d_state=16, seq_len=128
        let mut layer = S4Layer::new(&config, &mut make_rng());
        layer.compute_kernel();
        let input = Array2::zeros((128, 64));
        let output = layer.forward(&input).expect("forward should succeed");
        assert_eq!(output.dim(), (128, 64));
    }

    #[test]
    fn test_s4_dimension_mismatch_error() {
        let config = S4Config {
            d_model: 8,
            d_state: 4,
            seq_len: 16,
            ..S4Config::default()
        };
        let mut layer = S4Layer::new(&config, &mut make_rng());
        layer.compute_kernel();
        // Wrong d_model
        let input = Array2::zeros((16, 4));
        let result = layer.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_s4_skip_connection() {
        // With a zero input, the skip connection (D * u) contributes 0.
        // The kernel contribution should also be small (no input energy).
        let config = S4Config {
            d_model: 4,
            d_state: 4,
            seq_len: 8,
            ..S4Config::default()
        };
        let mut layer = S4Layer::new(&config, &mut make_rng());
        layer.compute_kernel();
        let input = Array2::zeros((8, 4));
        let output = layer.forward(&input).expect("forward");
        // All outputs should be (near-)zero
        for t in 0..8 {
            for h in 0..4 {
                assert!(
                    output[[t, h]].abs() < 1e-12,
                    "expected zero, got {}",
                    output[[t, h]]
                );
            }
        }
    }

    #[test]
    fn test_fft_convolve_simple() {
        // [1, 1, 1] * [1, 1] = [1, 2, 2, 1]
        let a = vec![1.0, 1.0, 1.0];
        let b = vec![1.0, 1.0];
        let fft_len = next_power_of_two(a.len() + b.len() - 1);
        let result = fft_convolve(&a, &b, fft_len);
        let expected = [1.0, 2.0, 2.0, 1.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (result[i] - exp).abs() < 1e-9,
                "index {i}: got {}, expected {exp}",
                result[i]
            );
        }
    }
}
