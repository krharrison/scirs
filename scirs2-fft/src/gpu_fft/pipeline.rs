//! GPU FFT pipeline: plan cache, batch execution, R2C/C2R, signal windowing.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use scirs2_core::numeric::Complex64;

use super::kernels::{
    apply_normalization, bluestein_gpu, compute_twiddles_gpu, cooley_tukey_gpu, tiled_fft_1d,
};
use super::types::{
    BatchFftResult, FftDirection, GpuFftConfig, GpuFftError, GpuFftPlan, GpuFftResult,
    NormalizationMode,
};

// ─────────────────────────────────────────────────────────────────────────────
// Plan cache key
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PlanKey {
    size: usize,
    direction: bool, // true = forward, false = inverse
}

impl PlanKey {
    fn new(size: usize, direction: FftDirection) -> Self {
        Self {
            size,
            direction: direction == FftDirection::Forward,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// GPU FFT pipeline with LRU plan cache.
///
/// The pipeline precomputes and caches twiddle factors for each unique
/// `(size, direction)` combination.  Subsequent calls for the same parameters
/// reuse the cached plan, avoiding redundant trigonometric evaluations.
///
/// # Thread Safety
///
/// The plan cache is protected by an `Arc<Mutex<…>>` so multiple threads can
/// share a single `GpuFftPipeline`.
#[derive(Clone)]
pub struct GpuFftPipeline {
    config: GpuFftConfig,
    cache: Arc<Mutex<HashMap<PlanKey, Arc<GpuFftPlan>>>>,
}

impl GpuFftPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: GpuFftConfig) -> Self {
        Self {
            config,
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Retrieve (or compile and cache) a plan for the given `(size, direction)`.
    ///
    /// Twiddle factors are computed once and stored in the plan.
    ///
    /// # Errors
    ///
    /// * [`GpuFftError::SizeTooSmall`] – if `size < 2`.
    /// * [`GpuFftError::AllocationFailed`] – if the cache lock is poisoned
    ///   (treated as an allocation failure).
    pub fn plan(&self, size: usize, direction: FftDirection) -> GpuFftResult<Arc<GpuFftPlan>> {
        let key = PlanKey::new(size, direction);

        {
            let guard = self.cache.lock().map_err(|_| {
                GpuFftError::AllocationFailed(size * std::mem::size_of::<Complex64>())
            })?;
            if let Some(plan) = guard.get(&key) {
                return Ok(Arc::clone(plan));
            }
        }

        // Compile a new plan.
        if size < 2 {
            return Err(GpuFftError::SizeTooSmall(size));
        }
        let twiddle_n = if size.is_power_of_two() {
            size
        } else {
            next_pow2_for_plan(size)
        };
        let twiddle_cache = compute_twiddles_gpu(twiddle_n)?;

        let plan = Arc::new(GpuFftPlan {
            size,
            direction,
            config: self.config.clone(),
            twiddle_cache,
        });

        let mut guard = self
            .cache
            .lock()
            .map_err(|_| GpuFftError::AllocationFailed(size * std::mem::size_of::<Complex64>()))?;
        guard.insert(key, Arc::clone(&plan));
        Ok(plan)
    }

    /// Execute a single forward or inverse FFT on `data` in-place.
    ///
    /// Uses the cached twiddle factors from the plan.  After the transform the
    /// normalisation specified in the pipeline's [`GpuFftConfig`] is applied.
    ///
    /// # Errors
    ///
    /// Propagates any kernel error.
    pub fn execute(
        &self,
        data: &mut [Complex64],
        size: usize,
        direction: FftDirection,
    ) -> GpuFftResult<()> {
        if data.len() < size {
            return Err(GpuFftError::SizeTooSmall(data.len()));
        }
        let plan = self.plan(size, direction)?;
        let slice = &mut data[..size];

        if size.is_power_of_two() {
            cooley_tukey_gpu(slice, direction, &plan.twiddle_cache)?;
        } else if self.config.use_shared_memory {
            tiled_fft_1d(slice, self.config.tile_size, &plan.twiddle_cache, direction)?;
        } else {
            bluestein_gpu(slice, direction)?;
        }

        // Apply user-requested normalisation.
        let norm = match (self.config.normalization, direction) {
            (NormalizationMode::Forward, FftDirection::Forward) => NormalizationMode::Forward,
            (NormalizationMode::Backward, FftDirection::Inverse) => NormalizationMode::Backward,
            (NormalizationMode::Ortho, _) => NormalizationMode::Ortho,
            _ => NormalizationMode::None,
        };
        apply_normalization(slice, norm);
        Ok(())
    }

    /// Execute a batch of FFTs using thread-level parallelism.
    ///
    /// Each entry in `batch` is transformed independently.  The results are
    /// written back to the same slots and also returned in [`BatchFftResult`].
    ///
    /// # Errors
    ///
    /// * [`GpuFftError::BatchEmpty`] – if `batch` is empty.
    /// * Any per-signal kernel error.
    pub fn execute_batch(
        &self,
        batch: &mut [Vec<Complex64>],
        direction: FftDirection,
    ) -> GpuFftResult<BatchFftResult> {
        if batch.is_empty() {
            return Err(GpuFftError::BatchEmpty);
        }

        let start = Instant::now();

        // Collect errors from threads.
        let errors: Mutex<Vec<GpuFftError>> = Mutex::new(Vec::new());

        std::thread::scope(|s| {
            for signal in batch.iter_mut() {
                let errors_ref = &errors;
                let size = signal.len();
                let pipeline = self.clone();
                s.spawn(move || {
                    if let Err(e) = pipeline.execute(signal, size, direction) {
                        if let Ok(mut errs) = errors_ref.lock() {
                            errs.push(e);
                        }
                    }
                });
            }
        });

        let errs = errors.into_inner().unwrap_or_default();
        if let Some(first) = errs.into_iter().next() {
            return Err(first);
        }

        let elapsed_ns = start.elapsed().as_nanos() as u64;
        let outputs = batch.to_vec();
        Ok(BatchFftResult {
            outputs,
            elapsed_ns,
        })
    }

    /// Real-to-complex FFT.
    ///
    /// Converts `real` to complex (imaginary part = 0) and applies a forward
    /// FFT, returning the full N-point complex spectrum.
    ///
    /// # Errors
    ///
    /// * [`GpuFftError::SizeTooSmall`] – if `real.len() < 2`.
    pub fn execute_r2c(&self, real: &[f64]) -> GpuFftResult<Vec<Complex64>> {
        let n = real.len();
        if n < 2 {
            return Err(GpuFftError::SizeTooSmall(n));
        }
        let mut data: Vec<Complex64> = real.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        self.execute(&mut data, n, FftDirection::Forward)?;
        Ok(data)
    }

    /// Complex-to-real IFFT.
    ///
    /// Applies an inverse FFT to `complex` and returns the real parts of the
    /// result.  If `output_len` is supplied it must equal the length of the
    /// time-domain signal (used only for documentation/validation; the inverse
    /// FFT always operates on the full `complex.len()` points).
    ///
    /// # Errors
    ///
    /// * [`GpuFftError::SizeTooSmall`] – if `complex.len() < 2`.
    /// * [`GpuFftError::InvalidOutputLength`] – if `output_len` is inconsistent.
    pub fn execute_c2r(&self, complex: &[Complex64], output_len: usize) -> GpuFftResult<Vec<f64>> {
        let n = complex.len();
        if n < 2 {
            return Err(GpuFftError::SizeTooSmall(n));
        }
        // Validate output_len: must be ≤ n (we return n real samples at most).
        if output_len > n {
            return Err(GpuFftError::InvalidOutputLength {
                requested: output_len,
                input_len: n,
            });
        }
        let mut data = complex.to_vec();
        self.execute(&mut data, n, FftDirection::Inverse)?;

        let take = if output_len == 0 { n } else { output_len };
        Ok(data[..take].iter().map(|c| c.re).collect())
    }

    /// Sliding-window FFT (simplified STFT without overlap-add synthesis).
    ///
    /// Splits `signal` into overlapping frames of length `window_size` with
    /// stride `hop_size` and runs a forward FFT on each frame.
    ///
    /// Returns a `Vec` of spectra, one per frame.  The number of frames is
    /// `⌊(signal.len() − window_size) / hop_size⌋ + 1`.
    ///
    /// # Errors
    ///
    /// * [`GpuFftError::SizeTooSmall`] – if `window_size < 2` or
    ///   `signal.len() < window_size`.
    /// * Any per-frame kernel error.
    pub fn signal_pipeline(
        &self,
        signal: &[f64],
        window_size: usize,
        hop_size: usize,
    ) -> GpuFftResult<Vec<Vec<Complex64>>> {
        if window_size < 2 {
            return Err(GpuFftError::SizeTooSmall(window_size));
        }
        if signal.len() < window_size {
            return Err(GpuFftError::SizeTooSmall(signal.len()));
        }
        let hop = hop_size.max(1);
        let n_frames = (signal.len() - window_size) / hop + 1;
        let mut result = Vec::with_capacity(n_frames);

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop;
            let frame = &signal[start..start + window_size];
            let spectrum = self.execute_r2c(frame)?;
            result.push(spectrum);
        }

        Ok(result)
    }

    /// Return the number of plans currently held in the cache.
    pub fn cache_size(&self) -> usize {
        self.cache.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// Access the configuration.
    pub fn config(&self) -> &GpuFftConfig {
        &self.config
    }
}

/// Smallest power of two ≥ `n` (used for Bluestein twiddle sizing).
fn next_pow2_for_plan(n: usize) -> usize {
    if n.is_power_of_two() {
        n
    } else {
        1usize << (usize::BITS - n.leading_zeros()) as usize
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-9;
    const LOOSE: f64 = 1e-7;

    fn default_pipeline() -> GpuFftPipeline {
        GpuFftPipeline::new(GpuFftConfig::default())
    }

    // ── Plan caching ─────────────────────────────────────────────────────────

    #[test]
    fn test_plan_caching_same_arc() {
        let p = default_pipeline();
        let plan1 = p.plan(16, FftDirection::Forward).expect("plan 1");
        let plan2 = p.plan(16, FftDirection::Forward).expect("plan 2");
        // Same plan key → identical twiddle content.
        assert_eq!(plan1.size, plan2.size);
        assert_eq!(plan1.twiddle_cache.len(), plan2.twiddle_cache.len());
    }

    #[test]
    fn test_plan_cache_grows() {
        let p = default_pipeline();
        p.plan(8, FftDirection::Forward).expect("8F");
        p.plan(16, FftDirection::Forward).expect("16F");
        p.plan(8, FftDirection::Inverse).expect("8I");
        assert_eq!(p.cache_size(), 3);
    }

    // ── 8-point known result ──────────────────────────────────────────────────

    #[test]
    fn test_gpu_fft_8_point_known() {
        // FFT of [1, 0, 0, 0, 0, 0, 0, 0] = [1, 1, 1, 1, 1, 1, 1, 1]
        let p = default_pipeline();
        let real = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let spectrum = p.execute_r2c(&real).expect("r2c");
        for s in &spectrum {
            assert!((s.re - 1.0).abs() < EPS, "DC mismatch: {}", s.re);
            assert!(s.im.abs() < EPS, "IM mismatch: {}", s.im);
        }
    }

    // ── Power-of-two roundtrip ────────────────────────────────────────────────

    #[test]
    fn test_gpu_fft_power_of_two_roundtrip() {
        let p = default_pipeline();
        let original: Vec<f64> = (0..16).map(|i| (i as f64) * 0.5).collect();
        let spectrum = p.execute_r2c(&original).expect("r2c");
        let recovered = p.execute_c2r(&spectrum, original.len()).expect("c2r");
        for (i, (&o, r)) in original.iter().zip(recovered.iter()).enumerate() {
            assert!((o - r).abs() < LOOSE, "index {i}: {o} vs {r}");
        }
    }

    // ── Normalization ─────────────────────────────────────────────────────────

    #[test]
    fn test_gpu_fft_normalization_ortho_unitary() {
        let config = GpuFftConfig {
            normalization: NormalizationMode::Ortho,
            ..Default::default()
        };
        let p = GpuFftPipeline::new(config);
        let n = 8;
        let real: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let spectrum = p.execute_r2c(&real).expect("r2c ortho");
        // For ortho normalization, Parseval's theorem: ‖X‖² == ‖x‖²
        let energy_x: f64 = real.iter().map(|&x| x * x).sum();
        let energy_freq: f64 = spectrum.iter().map(|c| c.norm_sqr()).sum();
        assert!(
            (energy_x - energy_freq).abs() < 1e-6,
            "Parseval violation: {energy_x} vs {energy_freq}"
        );
    }

    #[test]
    fn test_gpu_fft_normalization_modes() {
        // Ensure Forward normalisation scales by 1/N.
        let n = 8usize;
        let config_fwd = GpuFftConfig {
            normalization: NormalizationMode::Forward,
            ..Default::default()
        };
        let config_none = GpuFftConfig {
            normalization: NormalizationMode::None,
            ..Default::default()
        };
        let p_fwd = GpuFftPipeline::new(config_fwd);
        let p_none = GpuFftPipeline::new(config_none);
        let real: Vec<f64> = vec![1.0; n];

        let s_fwd = p_fwd.execute_r2c(&real).expect("fwd");
        let s_none = p_none.execute_r2c(&real).expect("none");

        // DC bin: fwd = 1.0, none = N = 8.0
        let dc_fwd = s_fwd[0].re;
        let dc_none = s_none[0].re;
        assert!((dc_fwd - 1.0).abs() < 1e-10, "DC fwd: {dc_fwd}");
        assert!((dc_none - n as f64).abs() < 1e-10, "DC none: {dc_none}");
    }

    // ── Batch == individual ───────────────────────────────────────────────────

    #[test]
    fn test_gpu_fft_batch_same_as_individual() {
        let p = default_pipeline();
        let n = 8;
        let signals: Vec<Vec<f64>> = (0..4_u64)
            .map(|k| (0..n).map(|i| (i as f64) + k as f64).collect())
            .collect();

        let individual: Vec<Vec<Complex64>> = signals
            .iter()
            .map(|s| p.execute_r2c(s).expect("individual r2c"))
            .collect();

        let mut batch: Vec<Vec<Complex64>> = signals
            .iter()
            .map(|s| s.iter().map(|&x| Complex64::new(x, 0.0)).collect())
            .collect();
        let result = p
            .execute_batch(&mut batch, FftDirection::Forward)
            .expect("batch");

        for (i, (ind, bat)) in individual.iter().zip(result.outputs.iter()).enumerate() {
            for (j, (a, b)) in ind.iter().zip(bat.iter()).enumerate() {
                assert!(
                    (a.re - b.re).abs() < LOOSE,
                    "signal {i} bin {j} re: {} vs {}",
                    a.re,
                    b.re
                );
            }
        }
    }

    // ── Non power-of-two ─────────────────────────────────────────────────────

    #[test]
    fn test_gpu_fft_non_power_of_two() {
        let p = default_pipeline();
        let n = 6;
        let real: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let spectrum = p.execute_r2c(&real).expect("non-pow2 r2c");
        assert_eq!(spectrum.len(), n);

        // Roundtrip.
        let recovered = p.execute_c2r(&spectrum, n).expect("non-pow2 c2r");
        for (i, (&o, r)) in real.iter().zip(recovered.iter()).enumerate() {
            assert!((o - r).abs() < 1e-6, "index {i}: {o} vs {r}");
        }
    }

    // ── R2C / C2R roundtrips ──────────────────────────────────────────────────

    #[test]
    fn test_gpu_fft_r2c_roundtrip() {
        let p = default_pipeline();
        let real: Vec<f64> = (0..32).map(|i| (i as f64 * PI / 16.0).sin()).collect();
        let spectrum = p.execute_r2c(&real).expect("r2c");
        let recovered = p.execute_c2r(&spectrum, real.len()).expect("c2r");
        for (i, (&o, r)) in real.iter().zip(recovered.iter()).enumerate() {
            assert!((o - r).abs() < LOOSE, "index {i}: {o} vs {r}");
        }
    }

    #[test]
    fn test_gpu_fft_c2r_roundtrip() {
        let p = default_pipeline();
        let n = 8;
        let original: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();
        let spectrum = p.execute_r2c(&original).expect("r2c");
        let back = p.execute_c2r(&spectrum, n).expect("c2r");
        for (a, b) in original.iter().zip(back.iter()) {
            assert!((a - b).abs() < LOOSE, "{a} vs {b}");
        }
    }

    // ── Signal pipeline ───────────────────────────────────────────────────────

    #[test]
    fn test_gpu_fft_signal_pipeline_shape() {
        let p = default_pipeline();
        let n_signal = 100;
        let window = 16;
        let hop = 8;
        let signal: Vec<f64> = (0..n_signal).map(|i| (i as f64 * 0.1).sin()).collect();
        let frames = p.signal_pipeline(&signal, window, hop).expect("pipeline");
        let expected_frames = (n_signal - window) / hop + 1;
        assert_eq!(frames.len(), expected_frames, "frame count");
        for f in &frames {
            assert_eq!(f.len(), window, "frame length");
        }
    }

    // ── Linearity ─────────────────────────────────────────────────────────────

    #[test]
    fn test_gpu_fft_linearity() {
        let p = default_pipeline();
        let n = 8;
        let a: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..n).map(|i| (n - i) as f64).collect();
        let apb: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

        let fa = p.execute_r2c(&a).expect("fa");
        let fb = p.execute_r2c(&b).expect("fb");
        let fapb = p.execute_r2c(&apb).expect("fapb");

        for k in 0..n {
            let sum_re = fa[k].re + fb[k].re;
            let sum_im = fa[k].im + fb[k].im;
            assert!((fapb[k].re - sum_re).abs() < 1e-8, "bin {k} re");
            assert!((fapb[k].im - sum_im).abs() < 1e-8, "bin {k} im");
        }
    }

    // ── Shift theorem ─────────────────────────────────────────────────────────

    #[test]
    fn test_gpu_fft_shift_theorem() {
        // A circular shift by d samples multiplies the spectrum by exp(-2πi·k·d/N).
        let p = default_pipeline();
        let n = 8usize;
        let d = 2; // shift by 2 samples
        let original: Vec<f64> = (0..n).map(|i| (i as f64 * PI / 4.0).sin()).collect();
        let shifted: Vec<f64> = (0..n).map(|i| original[(i + n - d) % n]).collect();

        let fo = p.execute_r2c(&original).expect("fo");
        let fs = p.execute_r2c(&shifted).expect("fs");

        for k in 0..n {
            let angle = -2.0 * PI * k as f64 * d as f64 / n as f64;
            let phase = Complex64::new(angle.cos(), angle.sin());
            let expected = fo[k] * phase;
            assert!((expected.re - fs[k].re).abs() < 1e-8, "bin {k} re");
            assert!((expected.im - fs[k].im).abs() < 1e-8, "bin {k} im");
        }
    }

    // ── Large batch ───────────────────────────────────────────────────────────

    #[test]
    fn test_large_batch_size_512() {
        let p = default_pipeline();
        let n = 512;
        let mut batch: Vec<Vec<Complex64>> = (0..n)
            .map(|k| {
                (0..16)
                    .map(|i| Complex64::new(i as f64 + k as f64, 0.0))
                    .collect()
            })
            .collect();
        let result = p
            .execute_batch(&mut batch, FftDirection::Forward)
            .expect("large batch");
        assert_eq!(result.outputs.len(), n);
    }

    // ── Zero input ────────────────────────────────────────────────────────────

    #[test]
    fn test_gpu_fft_zero_input() {
        let p = default_pipeline();
        let real = vec![0.0f64; 8];
        let spectrum = p.execute_r2c(&real).expect("zero r2c");
        for (k, s) in spectrum.iter().enumerate() {
            assert!(s.norm() < EPS, "bin {k}: {:?}", s);
        }
    }

    // ── Impulse response ──────────────────────────────────────────────────────

    #[test]
    fn test_gpu_fft_impulse_response() {
        let p = default_pipeline();
        // A unit impulse at t=0: FFT = all-ones spectrum.
        let mut real = vec![0.0f64; 16];
        real[0] = 1.0;
        let spectrum = p.execute_r2c(&real).expect("impulse r2c");
        for (k, s) in spectrum.iter().enumerate() {
            assert!((s.re - 1.0).abs() < EPS, "bin {k} re: {}", s.re);
            assert!(s.im.abs() < EPS, "bin {k} im: {}", s.im);
        }
    }

    // ── Config default ────────────────────────────────────────────────────────

    #[test]
    fn test_pipeline_config_default() {
        let cfg = GpuFftConfig::default();
        assert_eq!(cfg.tile_size, 256);
        assert_eq!(cfg.batch_size, 8);
        assert!(cfg.use_shared_memory);
        assert_eq!(cfg.normalization, NormalizationMode::None);
    }

    // ── Batch parallel consistency ────────────────────────────────────────────

    #[test]
    fn test_batch_parallel_performance_consistency() {
        let p = default_pipeline();
        let n_signals = 16;
        let n_pts = 32;
        let signals: Vec<Vec<f64>> = (0..n_signals)
            .map(|k| {
                (0..n_pts)
                    .map(|i| (i as f64 * PI * (k + 1) as f64 / n_pts as f64).sin())
                    .collect()
            })
            .collect();

        // Individual.
        let individual: Vec<Vec<Complex64>> = signals
            .iter()
            .map(|s| p.execute_r2c(s).expect("ind"))
            .collect();

        // Batch.
        let mut batch: Vec<Vec<Complex64>> = signals
            .iter()
            .map(|s| s.iter().map(|&x| Complex64::new(x, 0.0)).collect())
            .collect();
        let batch_result = p
            .execute_batch(&mut batch, FftDirection::Forward)
            .expect("batch");

        for (i, (ind, bat)) in individual
            .iter()
            .zip(batch_result.outputs.iter())
            .enumerate()
        {
            for (j, (a, b)) in ind.iter().zip(bat.iter()).enumerate() {
                assert!(
                    (a.re - b.re).abs() < 1e-7,
                    "signal {i} bin {j} re: {} vs {}",
                    a.re,
                    b.re
                );
            }
        }
    }

    // ── Twiddle orthogonality ─────────────────────────────────────────────────

    #[test]
    fn test_twiddle_computation_orthogonality() {
        // For W_N^k = exp(-2πi·k/N), |W_N^k| = 1 for all k.
        use super::super::kernels::compute_twiddles_gpu;
        let tw = compute_twiddles_gpu(8).expect("twiddles");
        for (k, w) in tw.iter().enumerate() {
            let mag = w.norm();
            assert!(
                (mag - 1.0).abs() < 1e-12,
                "twiddle {k} magnitude deviates: {mag}"
            );
        }
    }
}
