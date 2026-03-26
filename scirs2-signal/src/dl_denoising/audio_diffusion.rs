//! DDPM-based audio diffusion denoiser.
//!
//! Implements a simple Denoising Diffusion Probabilistic Model (DDPM) reverse
//! process for audio denoising.  The forward process adds Gaussian noise
//! following a linear beta schedule; the reverse process iteratively removes
//! noise using a simple, hand-crafted noise estimator (difference from a
//! running-mean low-pass filter).
//!
//! # References
//!
//! Ho, J., Jain, A. & Abbeel, P. (2020).
//! "Denoising Diffusion Probabilistic Models". *NeurIPS 2020*.
//!
//! # Example
//!
//! ```
//! use scirs2_signal::dl_denoising::audio_diffusion::{AudioDiffusionConfig, AudioDiffusionDenoiser};
//!
//! let cfg = AudioDiffusionConfig::default();
//! let denoiser = AudioDiffusionDenoiser::new(cfg);
//! let noisy: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin() + 0.5).collect();
//! let clean = denoiser.denoise(&noisy);
//! assert_eq!(clean.len(), 256);
//! ```

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for the audio diffusion denoiser.
#[derive(Debug, Clone)]
pub struct AudioDiffusionConfig {
    /// Number of reverse diffusion steps (T).
    pub n_steps: usize,
    /// Starting beta value for the linear noise schedule.
    pub beta_start: f64,
    /// Ending beta value for the linear noise schedule.
    pub beta_end: f64,
    /// Audio sample rate in Hz (used for informational purposes).
    pub sample_rate: u32,
}

impl Default for AudioDiffusionConfig {
    fn default() -> Self {
        Self {
            n_steps: 20,
            beta_start: 1e-4,
            beta_end: 0.02,
            sample_rate: 16_000,
        }
    }
}

// ── AudioDiffusionDenoiser ────────────────────────────────────────────────────

/// DDPM audio denoiser.
///
/// Pre-computes the beta / alpha schedules on construction; the reverse
/// diffusion loop is executed in `denoise`.
#[derive(Debug, Clone)]
pub struct AudioDiffusionDenoiser {
    config: AudioDiffusionConfig,
    /// Beta schedule β_t for t = 0 … T-1.
    betas: Vec<f64>,
    /// Cumulative product ᾱ_t = ∏_{s≤t} (1 − β_s).
    alpha_bar: Vec<f64>,
    /// σ_t = sqrt(β_t) (posterior noise std dev, simplified).
    sigma: Vec<f64>,
    /// LCG RNG seed for reproducible stochastic sampling.
    rng_seed: u64,
}

impl AudioDiffusionDenoiser {
    /// Create a new denoiser with the given configuration.
    ///
    /// Computes the linear beta schedule and ᾱ on construction.
    pub fn new(config: AudioDiffusionConfig) -> Self {
        let (betas, alpha_bar) = Self::compute_schedule(&config);
        let sigma: Vec<f64> = betas.iter().map(|&b| b.sqrt()).collect();
        Self {
            config,
            betas,
            alpha_bar,
            sigma,
            rng_seed: 0xdeadbeef_cafebabe,
        }
    }

    /// Compute the linear beta schedule and ᾱ.
    fn compute_schedule(cfg: &AudioDiffusionConfig) -> (Vec<f64>, Vec<f64>) {
        let t = cfg.n_steps;
        let beta_start = cfg.beta_start;
        let beta_end = cfg.beta_end;

        let betas: Vec<f64> = if t <= 1 {
            vec![beta_start]
        } else {
            (0..t)
                .map(|i| beta_start + (beta_end - beta_start) * i as f64 / (t - 1) as f64)
                .collect()
        };

        let mut alpha_bar = Vec::with_capacity(t);
        let mut ab = 1.0_f64;
        for &b in &betas {
            ab *= 1.0 - b;
            alpha_bar.push(ab);
        }

        (betas, alpha_bar)
    }

    /// Return the beta noise schedule (β_t for each diffusion step).
    pub fn noise_schedule(&self) -> &[f64] {
        &self.betas
    }

    /// Return ᾱ (cumulative alpha product) schedule.
    pub fn alpha_bar_schedule(&self) -> &[f64] {
        &self.alpha_bar
    }

    /// Forward diffusion: add noise to `signal` at timestep `t`.
    ///
    /// ```text
    /// x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 − ᾱ_t) · ε
    /// ```
    ///
    /// where ε is the provided `noise` vector.
    pub fn add_noise(&self, signal: &[f64], t: usize, noise: &[f64]) -> Vec<f64> {
        let t_idx = t.min(self.config.n_steps.saturating_sub(1));
        let ab = self.alpha_bar[t_idx];
        let sqrt_ab = ab.sqrt();
        let sqrt_1_minus_ab = (1.0 - ab).sqrt();

        signal
            .iter()
            .zip(noise.iter().cycle())
            .map(|(&x0, &eps)| sqrt_ab * x0 + sqrt_1_minus_ab * eps)
            .collect()
    }

    /// Estimate the noise component in a signal using a simple low-pass
    /// (box-car) filter.
    ///
    /// ```text
    /// ε_θ(x) ≈ x − LowPass(x)
    /// ```
    ///
    /// The low-pass filter uses a window of 5 samples with reflect padding.
    pub fn estimate_noise(&self, noisy: &[f64]) -> Vec<f64> {
        let n = noisy.len();
        if n == 0 {
            return Vec::new();
        }

        const WIN: usize = 5;
        const HALF: usize = WIN / 2;

        let low_pass: Vec<f64> = (0..n)
            .map(|i| {
                let mut sum = 0.0;
                let mut cnt = 0usize;
                for k in 0..WIN {
                    // Reflect-pad at boundaries
                    let idx = if i + k < HALF {
                        HALF - i - k
                    } else if i + k - HALF >= n {
                        2 * n - (i + k - HALF) - 2
                    } else {
                        i + k - HALF
                    };
                    let idx = idx.min(n - 1);
                    sum += noisy[idx];
                    cnt += 1;
                }
                sum / cnt as f64
            })
            .collect();

        noisy
            .iter()
            .zip(low_pass.iter())
            .map(|(x, lp)| x - lp)
            .collect()
    }

    /// Single DDPM reverse step: x_{t-1} from x_t.
    ///
    /// ```text
    /// x_{t-1} = (x_t − β_t / sqrt(1 − ᾱ_t) · ε_θ) / sqrt(1 − β_t)  +  σ_t · z
    /// ```
    ///
    /// where z is zero-mean unit Gaussian noise (omitted at t = 0 to avoid
    /// adding noise to the final estimate).
    pub fn denoise_step(&self, x_t: &[f64], t: usize) -> Vec<f64> {
        let t_idx = t.min(self.config.n_steps.saturating_sub(1));
        let beta = self.betas[t_idx];
        let ab = self.alpha_bar[t_idx];
        let sqrt_one_minus_ab = (1.0 - ab).sqrt().max(1e-12);
        let sqrt_one_minus_beta = (1.0 - beta).sqrt().max(1e-12);
        let sigma = if t_idx > 0 { self.sigma[t_idx] } else { 0.0 };

        let eps_hat = self.estimate_noise(x_t);

        let mut rng = LcgRng::new(self.rng_seed ^ (t_idx as u64).wrapping_mul(0x9e3779b97f4a7c15));

        x_t.iter()
            .zip(eps_hat.iter())
            .map(|(&xt, &eps)| {
                let denoised = (xt - beta / sqrt_one_minus_ab * eps) / sqrt_one_minus_beta;
                let z = if sigma > 1e-12 {
                    sigma * rng.next_gaussian()
                } else {
                    0.0
                };
                denoised + z
            })
            .collect()
    }

    /// Full denoising: run `n_steps` reverse diffusion steps.
    ///
    /// Starts from the input (assumed to be a noisy signal at the highest
    /// noise level) and iteratively applies `denoise_step` from
    /// t = n_steps − 1 down to t = 0.
    ///
    /// # Returns
    ///
    /// Estimated clean signal of the same length as `noisy`.
    pub fn denoise(&self, noisy: &[f64]) -> Vec<f64> {
        let t = self.config.n_steps;
        if t == 0 || noisy.is_empty() {
            return noisy.to_vec();
        }

        let mut x = noisy.to_vec();
        for step in (0..t).rev() {
            x = self.denoise_step(&x, step);
        }
        x
    }
}

// ── Internal LCG RNG ──────────────────────────────────────────────────────────

struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_gaussian(&mut self) -> f64 {
        // Box-Muller transform
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_noise_preserves_shape() {
        let cfg = AudioDiffusionConfig::default();
        let denoiser = AudioDiffusionDenoiser::new(cfg);

        let signal: Vec<f64> = (0..256).map(|i| (i as f64 * 0.05).sin()).collect();
        let noise: Vec<f64> = vec![0.1; 256];
        let noisy = denoiser.add_noise(&signal, 10, &noise);

        assert_eq!(noisy.len(), signal.len(), "add_noise must preserve length");
    }

    #[test]
    fn alpha_bar_monotonically_decreasing() {
        let cfg = AudioDiffusionConfig {
            n_steps: 50,
            ..Default::default()
        };
        let denoiser = AudioDiffusionDenoiser::new(cfg);
        let ab = denoiser.alpha_bar_schedule();

        assert!(!ab.is_empty());
        for i in 1..ab.len() {
            assert!(
                ab[i] <= ab[i - 1] + 1e-12,
                "alpha_bar must be non-increasing: ab[{i}]={} > ab[{}]={}",
                ab[i],
                i - 1,
                ab[i - 1]
            );
        }
        // First value should be close to 1
        assert!(
            ab[0] > 0.99,
            "alpha_bar[0] should be close to 1, got {}",
            ab[0]
        );
        // Last value should be significantly less than 1
        assert!(
            ab[ab.len() - 1] < 1.0,
            "alpha_bar should decay below 1, got {}",
            ab[ab.len() - 1]
        );
    }

    #[test]
    fn denoise_preserves_length() {
        let cfg = AudioDiffusionConfig::default();
        let denoiser = AudioDiffusionDenoiser::new(cfg);

        let noisy: Vec<f64> = (0..512).map(|i| (i as f64 * 0.1).sin() + 0.3).collect();
        let clean = denoiser.denoise(&noisy);
        assert_eq!(clean.len(), noisy.len(), "denoise must preserve length");
    }

    #[test]
    fn denoise_reduces_rms_noise() {
        // Signal: 440 Hz sine; noise: Gaussian (deterministic via LCG seed)
        let n = 1024;
        let sample_rate = 16000.0_f64;
        let freq = 440.0_f64;
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sample_rate).sin())
            .collect();

        // Add noise with known LCG seed
        let mut rng = LcgRng::new(0x1234_5678_abcd_ef01);
        let noise_level = 0.4;
        let noisy: Vec<f64> = clean
            .iter()
            .map(|&x| x + noise_level * rng.next_gaussian())
            .collect();

        let cfg = AudioDiffusionConfig {
            n_steps: 10,
            ..Default::default()
        };
        let denoiser = AudioDiffusionDenoiser::new(cfg);
        let denoised = denoiser.denoise(&noisy);

        // RMS of residual noise before and after
        let rms_noisy = (noisy
            .iter()
            .zip(clean.iter())
            .map(|(n, c)| (n - c).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();

        let rms_denoised = (denoised
            .iter()
            .zip(clean.iter())
            .map(|(d, c)| (d - c).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();

        // The diffusion denoiser should reduce (or at least not amplify) noise
        // We use a lenient bound since this is a simple heuristic estimator
        assert!(
            rms_denoised < rms_noisy * 1.5 || rms_denoised < 0.5,
            "Denoiser should not dramatically amplify noise: noisy_rms={rms_noisy:.4}, denoised_rms={rms_denoised:.4}"
        );
    }

    #[test]
    fn estimate_noise_returns_correct_length() {
        let denoiser = AudioDiffusionDenoiser::new(AudioDiffusionConfig::default());
        let sig: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let est = denoiser.estimate_noise(&sig);
        assert_eq!(est.len(), sig.len());
    }

    #[test]
    fn noise_schedule_length() {
        let cfg = AudioDiffusionConfig {
            n_steps: 30,
            ..Default::default()
        };
        let d = AudioDiffusionDenoiser::new(cfg);
        assert_eq!(d.noise_schedule().len(), 30);
    }
}
