//! Background subtraction / foreground detection for video sequences.
//!
//! Implements several classic background-modelling approaches:
//!
//! * [`SimpleBackgroundModel`] – exponential running average.
//! * [`GaussianMixtureBackground`] – per-pixel K-component Gaussian mixture
//!   (simplified Stauffer-Grimson, single-pass EM update).
//! * [`vibe_model`] – ViBe-style sample-based background model.
//! * [`subtract_background`] – produce a binary foreground mask.
//! * [`update_background`] – incremental one-step update helpers.

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::{Array2, Array3};

// ─── FrameBuffer ──────────────────────────────────────────────────────────────

/// A circular buffer that keeps the N most recent frames.
///
/// Frames are stored as `Array2<f64>` (single-channel, 0–255 range or
/// normalised 0–1 — callers decide the convention).
#[derive(Debug, Clone)]
pub struct FrameBuffer {
    capacity: usize,
    frames: Vec<Array2<f64>>,
    write_pos: usize,
    count: usize,
}

impl FrameBuffer {
    /// Create a new, empty buffer with the given capacity.
    pub fn new(capacity: usize) -> NdimageResult<Self> {
        if capacity == 0 {
            return Err(NdimageError::InvalidInput(
                "FrameBuffer capacity must be > 0".into(),
            ));
        }
        Ok(Self {
            capacity,
            frames: Vec::with_capacity(capacity),
            write_pos: 0,
            count: 0,
        })
    }

    /// Push a frame into the buffer.  The oldest frame is silently discarded
    /// once the buffer is full.
    pub fn push(&mut self, frame: Array2<f64>) -> NdimageResult<()> {
        if self.count > 0 {
            let expected = self.frames[0].shape().to_vec();
            if frame.shape() != expected.as_slice() {
                return Err(NdimageError::DimensionError(format!(
                    "FrameBuffer frame shape mismatch: expected {:?}, got {:?}",
                    expected,
                    frame.shape()
                )));
            }
        }
        if self.count < self.capacity {
            self.frames.push(frame);
            self.count += 1;
        } else {
            self.frames[self.write_pos] = frame;
        }
        self.write_pos = (self.write_pos + 1) % self.capacity;
        Ok(())
    }

    /// Number of frames currently stored.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` if no frames have been pushed yet.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns `true` when the buffer has accumulated `capacity` frames.
    pub fn is_full(&self) -> bool {
        self.count == self.capacity
    }

    /// Borrow the i-th most recent frame (`0` = newest).
    pub fn get_recent(&self, i: usize) -> NdimageResult<&Array2<f64>> {
        if i >= self.count {
            return Err(NdimageError::InvalidInput(format!(
                "FrameBuffer index {i} out of range (len={})",
                self.count
            )));
        }
        let idx = (self.write_pos + self.capacity - 1 - i) % self.capacity;
        Ok(&self.frames[idx])
    }

    /// Compute the pixel-wise mean over all stored frames.
    pub fn mean_frame(&self) -> NdimageResult<Array2<f64>> {
        if self.count == 0 {
            return Err(NdimageError::InvalidInput(
                "FrameBuffer is empty".into(),
            ));
        }
        let shape = self.frames[0].shape();
        let rows = shape[0];
        let cols = shape[1];
        let mut acc = Array2::<f64>::zeros((rows, cols));
        for frame in &self.frames {
            acc = acc + frame;
        }
        let n = self.count as f64;
        Ok(acc / n)
    }

    /// Compute the pixel-wise variance over all stored frames.
    pub fn variance_frame(&self) -> NdimageResult<Array2<f64>> {
        if self.count < 2 {
            return Err(NdimageError::InvalidInput(
                "Need at least 2 frames to compute variance".into(),
            ));
        }
        let mean = self.mean_frame()?;
        let shape = mean.shape();
        let rows = shape[0];
        let cols = shape[1];
        let mut var = Array2::<f64>::zeros((rows, cols));
        for frame in &self.frames {
            let diff = frame - &mean;
            var = var + &diff * &diff;
        }
        Ok(var / (self.count as f64 - 1.0))
    }
}

// ─── SimpleBackgroundModel ────────────────────────────────────────────────────

/// Running-average background model.
///
/// Background is estimated as an exponential moving average:
/// `B_t = (1 - alpha) * B_{t-1} + alpha * I_t`
#[derive(Debug, Clone)]
pub struct SimpleBackgroundModel {
    /// Learning rate ∈ (0, 1].
    pub alpha: f64,
    /// Current background estimate.
    pub background: Array2<f64>,
    /// Pixel-wise running variance estimate (used for adaptive thresholding).
    pub variance: Array2<f64>,
}

impl SimpleBackgroundModel {
    /// Initialise from the first frame.
    pub fn new(first_frame: &Array2<f64>, alpha: f64) -> NdimageResult<Self> {
        if !(0.0 < alpha && alpha <= 1.0) {
            return Err(NdimageError::InvalidInput(
                "alpha must be in (0, 1]".into(),
            ));
        }
        let rows = first_frame.nrows();
        let cols = first_frame.ncols();
        Ok(Self {
            alpha,
            background: first_frame.clone(),
            variance: Array2::from_elem((rows, cols), 10.0_f64.powi(2)),
        })
    }

    /// Ingest a new frame and update the background estimate.
    pub fn update(&mut self, frame: &Array2<f64>) -> NdimageResult<()> {
        if frame.shape() != self.background.shape() {
            return Err(NdimageError::DimensionError(
                "Frame shape does not match background model".into(),
            ));
        }
        let diff = frame - &self.background;
        // Update variance: running estimate σ² ← (1-α)σ² + α*(I-B)²
        let sq_diff = &diff * &diff;
        self.variance = (1.0 - self.alpha) * &self.variance + self.alpha * sq_diff;
        // Update mean
        self.background = (1.0 - self.alpha) * &self.background + self.alpha * frame;
        Ok(())
    }

    /// Produce a binary foreground mask: pixel is foreground when
    /// |I - B| > threshold * σ.
    pub fn foreground_mask(
        &self,
        frame: &Array2<f64>,
        threshold_sigma: f64,
    ) -> NdimageResult<Array2<bool>> {
        if frame.shape() != self.background.shape() {
            return Err(NdimageError::DimensionError(
                "Frame shape does not match background model".into(),
            ));
        }
        let rows = frame.nrows();
        let cols = frame.ncols();
        let mask = Array2::from_shape_fn((rows, cols), |(r, c)| {
            let diff = (frame[[r, c]] - self.background[[r, c]]).abs();
            let sigma = self.variance[[r, c]].sqrt().max(1e-6);
            diff > threshold_sigma * sigma
        });
        Ok(mask)
    }
}

// ─── GaussianMixtureBackground ────────────────────────────────────────────────

/// Per-pixel Gaussian Mixture Model background (Stauffer-Grimson, simplified).
///
/// Each pixel is modelled as a mixture of K Gaussians.  On every frame update
/// the per-pixel distributions compete; the background is the sum of the
/// lowest-variance distributions whose total weight exceeds `background_ratio`.
#[derive(Debug, Clone)]
pub struct GaussianMixtureBackground {
    /// Number of mixture components per pixel.
    pub k: usize,
    rows: usize,
    cols: usize,
    /// Mixture weights [rows, cols, k].
    weights: Array3<f64>,
    /// Mixture means [rows, cols, k].
    means: Array3<f64>,
    /// Mixture variances (scalar per component) [rows, cols, k].
    variances: Array3<f64>,
    /// Learning rate.
    pub alpha: f64,
    /// Fraction of weight considered as background (T in the paper).
    pub background_ratio: f64,
    /// Initial variance for new components.
    pub initial_variance: f64,
}

impl GaussianMixtureBackground {
    /// Create a new GMM background model initialised from the first frame.
    ///
    /// # Arguments
    /// * `first_frame`  – shape (rows, cols), pixel values in any consistent scale.
    /// * `k`            – number of Gaussian components per pixel (typically 3–5).
    /// * `alpha`        – learning rate ∈ (0, 1].
    /// * `background_ratio` – fraction of weight to call "background" (0..1, e.g. 0.9).
    pub fn new(
        first_frame: &Array2<f64>,
        k: usize,
        alpha: f64,
        background_ratio: f64,
    ) -> NdimageResult<Self> {
        if k == 0 {
            return Err(NdimageError::InvalidInput("k must be >= 1".into()));
        }
        if !(0.0 < alpha && alpha <= 1.0) {
            return Err(NdimageError::InvalidInput("alpha must be in (0,1]".into()));
        }
        if !(0.0 < background_ratio && background_ratio <= 1.0) {
            return Err(NdimageError::InvalidInput(
                "background_ratio must be in (0,1]".into(),
            ));
        }
        let rows = first_frame.nrows();
        let cols = first_frame.ncols();
        let initial_variance = 225.0_f64; // 15² in 0-255 range

        // Initialise: first component at observed intensity, rest uniformly distributed
        let mut weights = Array3::<f64>::zeros((rows, cols, k));
        let mut means = Array3::<f64>::zeros((rows, cols, k));
        let mut variances = Array3::<f64>::from_elem((rows, cols, k), initial_variance);

        for r in 0..rows {
            for c in 0..cols {
                // First component carries most weight
                weights[[r, c, 0]] = 1.0 - (k as f64 - 1.0) * (alpha / k as f64);
                means[[r, c, 0]] = first_frame[[r, c]];
                for kk in 1..k {
                    let frac = (alpha / k as f64).max(1e-6);
                    weights[[r, c, kk]] = frac;
                    // Stagger initial means slightly
                    means[[r, c, kk]] = first_frame[[r, c]] + (kk as f64 * 5.0);
                }
                // Normalise
                let total: f64 = (0..k).map(|kk| weights[[r, c, kk]]).sum::<f64>();
                if total > 1e-9 {
                    for kk in 0..k {
                        weights[[r, c, kk]] /= total;
                    }
                }
            }
        }

        Ok(Self {
            k,
            rows,
            cols,
            weights,
            means,
            variances,
            alpha,
            background_ratio,
            initial_variance,
        })
    }

    /// Update the model with a new frame and return the foreground mask.
    ///
    /// Uses the 2.5σ Mahalanobis threshold for component matching.
    pub fn update_and_segment(&mut self, frame: &Array2<f64>) -> NdimageResult<Array2<bool>> {
        if frame.nrows() != self.rows || frame.ncols() != self.cols {
            return Err(NdimageError::DimensionError(
                "Frame shape does not match GMM model".into(),
            ));
        }
        let threshold_sq = 6.25; // (2.5)²
        let mut mask = Array2::<bool>::from_elem((self.rows, self.cols), true);

        for r in 0..self.rows {
            for c in 0..self.cols {
                let x = frame[[r, c]];
                let mut matched = false;

                for kk in 0..self.k {
                    let mu = self.means[[r, c, kk]];
                    let sigma2 = self.variances[[r, c, kk]].max(1e-6);
                    let maha_sq = (x - mu) * (x - mu) / sigma2;

                    if maha_sq < threshold_sq {
                        // This component matches; update it
                        let rho = self.alpha / self.weights[[r, c, kk]].max(1e-9);
                        self.means[[r, c, kk]] =
                            (1.0 - rho) * mu + rho * x;
                        let new_diff = x - self.means[[r, c, kk]];
                        self.variances[[r, c, kk]] =
                            ((1.0 - rho) * sigma2 + rho * new_diff * new_diff)
                                .max(1.0);
                        // Weight update: increase matched, decrease others
                        for jj in 0..self.k {
                            if jj == kk {
                                self.weights[[r, c, jj]] =
                                    (1.0 - self.alpha) * self.weights[[r, c, jj]] + self.alpha;
                            } else {
                                self.weights[[r, c, jj]] =
                                    (1.0 - self.alpha) * self.weights[[r, c, jj]];
                            }
                        }
                        matched = true;
                        break;
                    }
                }

                if !matched {
                    // Replace the least-probable component
                    let mut min_kk = 0;
                    let mut min_w = self.weights[[r, c, 0]];
                    for kk in 1..self.k {
                        let w = self.weights[[r, c, kk]];
                        if w < min_w {
                            min_w = w;
                            min_kk = kk;
                        }
                    }
                    self.means[[r, c, min_kk]] = x;
                    self.variances[[r, c, min_kk]] = self.initial_variance;
                    // Weights for all: decrease others, new one stays at min weight
                    for jj in 0..self.k {
                        if jj != min_kk {
                            self.weights[[r, c, jj]] =
                                (1.0 - self.alpha) * self.weights[[r, c, jj]];
                        }
                    }
                }

                // Normalise weights
                let total: f64 = (0..self.k).map(|kk| self.weights[[r, c, kk]]).sum::<f64>();
                if total > 1e-9 {
                    for kk in 0..self.k {
                        self.weights[[r, c, kk]] /= total;
                    }
                }

                // Determine background: sort by w/σ descending, accumulate until T reached
                let mut order: Vec<usize> = (0..self.k).collect();
                order.sort_by(|&a, &b| {
                    let wa = self.weights[[r, c, a]]
                        / self.variances[[r, c, a]].sqrt().max(1e-6);
                    let wb = self.weights[[r, c, b]]
                        / self.variances[[r, c, b]].sqrt().max(1e-6);
                    wb.partial_cmp(&wa).unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut acc_w = 0.0;
                let mut is_background = false;
                for &kk in &order {
                    acc_w += self.weights[[r, c, kk]];
                    let mu = self.means[[r, c, kk]];
                    let sigma2 = self.variances[[r, c, kk]].max(1e-6);
                    let maha_sq = (x - mu) * (x - mu) / sigma2;
                    if maha_sq < threshold_sq {
                        is_background = true;
                        break;
                    }
                    if acc_w >= self.background_ratio {
                        break;
                    }
                }
                mask[[r, c]] = !is_background;
            }
        }
        Ok(mask)
    }

    /// Compute a background image from the dominant (highest-weight) component
    /// at each pixel.
    pub fn background_image(&self) -> Array2<f64> {
        Array2::from_shape_fn((self.rows, self.cols), |(r, c)| {
            let mut best_kk = 0;
            let mut best_w = self.weights[[r, c, 0]];
            for kk in 1..self.k {
                let w = self.weights[[r, c, kk]];
                if w > best_w {
                    best_w = w;
                    best_kk = kk;
                }
            }
            self.means[[r, c, best_kk]]
        })
    }
}

// ─── subtract_background ─────────────────────────────────────────────────────

/// Generate a binary foreground mask by thresholding the absolute difference
/// between `frame` and a static `background`.
///
/// Pixel (r,c) is foreground when `|frame - background| > threshold`.
pub fn subtract_background(
    frame: &Array2<f64>,
    background: &Array2<f64>,
    threshold: f64,
) -> NdimageResult<Array2<bool>> {
    if frame.shape() != background.shape() {
        return Err(NdimageError::DimensionError(
            "frame and background must have the same shape".into(),
        ));
    }
    let rows = frame.nrows();
    let cols = frame.ncols();
    Ok(Array2::from_shape_fn((rows, cols), |(r, c)| {
        (frame[[r, c]] - background[[r, c]]).abs() > threshold
    }))
}

// ─── update_background ────────────────────────────────────────────────────────

/// Perform a single exponential moving-average update:
/// `background ← (1 - alpha) * background + alpha * frame`.
///
/// The background array is mutated in-place.
pub fn update_background(
    background: &mut Array2<f64>,
    frame: &Array2<f64>,
    alpha: f64,
) -> NdimageResult<()> {
    if frame.shape() != background.shape() {
        return Err(NdimageError::DimensionError(
            "frame and background must have the same shape".into(),
        ));
    }
    if !(0.0 < alpha && alpha <= 1.0) {
        return Err(NdimageError::InvalidInput("alpha must be in (0,1]".into()));
    }
    let rows = background.nrows();
    let cols = background.ncols();
    for r in 0..rows {
        for c in 0..cols {
            background[[r, c]] =
                (1.0 - alpha) * background[[r, c]] + alpha * frame[[r, c]];
        }
    }
    Ok(())
}

// ─── ViBe-style background model ─────────────────────────────────────────────

/// Configuration for [`vibe_model`].
#[derive(Debug, Clone)]
pub struct ViBeConfig {
    /// Number of samples per pixel.
    pub num_samples: usize,
    /// Matching threshold (Euclidean distance in pixel intensity space).
    pub match_threshold: f64,
    /// Minimum number of close samples to consider pixel as background.
    pub min_close_samples: usize,
    /// Subsampling factor: only 1/`subsample_factor` background pixels share
    /// their sample with a random neighbour.
    pub subsample_factor: usize,
}

impl Default for ViBeConfig {
    fn default() -> Self {
        Self {
            num_samples: 20,
            match_threshold: 20.0,
            min_close_samples: 2,
            subsample_factor: 16,
        }
    }
}

/// A ViBe-style sample-based background model.
///
/// Each pixel stores a small set of recently observed intensity samples.
/// A new pixel is classified as background when at least `min_close_samples`
/// of its stored samples fall within `match_threshold` of the new value.
#[derive(Debug, Clone)]
pub struct ViBeModel {
    config: ViBeConfig,
    rows: usize,
    cols: usize,
    /// Samples: [rows, cols, num_samples].
    samples: Vec<Vec<Vec<f64>>>,
    /// Internal pseudo-random state (xorshift64 per pixel is too heavy;
    /// we use a global LCG seeded per pixel for spatial subsampling).
    rng_state: u64,
}

impl ViBeModel {
    /// Initialise the ViBe model from the first frame.
    pub fn new(first_frame: &Array2<f64>, config: ViBeConfig) -> NdimageResult<Self> {
        let rows = first_frame.nrows();
        let cols = first_frame.ncols();
        if rows < 2 || cols < 2 {
            return Err(NdimageError::InvalidInput(
                "Frame must be at least 2×2".into(),
            ));
        }
        let num_samples = config.num_samples;
        // Each pixel's sample set is initialised from its neighbourhood
        let mut samples = vec![vec![vec![0.0_f64; num_samples]; cols]; rows];

        for r in 0..rows {
            for c in 0..cols {
                for s in 0..num_samples {
                    // Neighbour offset (cheap deterministic spread)
                    let dr = ((r + s) % 3).wrapping_sub(1).min(rows - 1);
                    let dc = ((c + s * 2 + 1) % 3).wrapping_sub(1).min(cols - 1);
                    let nr = r.saturating_add(dr).min(rows - 1);
                    let nc = c.saturating_add(dc).min(cols - 1);
                    samples[r][c][s] = first_frame[[nr, nc]];
                }
            }
        }

        Ok(Self {
            config,
            rows,
            cols,
            samples,
            rng_state: 0x_1234_5678_9abc_def0_u64,
        })
    }

    /// Advance the internal LCG and return a value in [0, modulus).
    fn rand_usize(&mut self, modulus: usize) -> usize {
        // LCG with Knuth's constants
        self.rng_state = self.rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.rng_state >> 33) as usize) % modulus
    }

    /// Process a new frame: update the model in-place and return a foreground
    /// mask (`true` = foreground).
    pub fn update_and_segment(
        &mut self,
        frame: &Array2<f64>,
    ) -> NdimageResult<Array2<bool>> {
        if frame.nrows() != self.rows || frame.ncols() != self.cols {
            return Err(NdimageError::DimensionError(
                "Frame shape does not match ViBe model".into(),
            ));
        }
        let thr = self.config.match_threshold;
        let min_close = self.config.min_close_samples;
        let n = self.config.num_samples;
        let sub = self.config.subsample_factor;
        let rows = self.rows;
        let cols = self.cols;
        let mut mask = Array2::<bool>::from_elem((rows, cols), false);

        for r in 0..rows {
            for c in 0..cols {
                let x = frame[[r, c]];
                // Count close samples
                let close: usize = self.samples[r][c]
                    .iter()
                    .filter(|&&s| (s - x).abs() <= thr)
                    .count();

                let is_fg = close < min_close;
                mask[[r, c]] = is_fg;

                if !is_fg {
                    // Background pixel: replace a random sample with x
                    let idx = self.rand_usize(n);
                    self.samples[r][c][idx] = x;

                    // Spatial propagation with probability 1/sub
                    if self.rand_usize(sub) == 0 {
                        // Pick a random 4-neighbour
                        let neighbours: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                        let chosen = self.rand_usize(4);
                        let (dr, dc) = neighbours[chosen];
                        let nr = (r as isize + dr).max(0).min(rows as isize - 1) as usize;
                        let nc = (c as isize + dc).max(0).min(cols as isize - 1) as usize;
                        let idx2 = self.rand_usize(n);
                        self.samples[nr][nc][idx2] = x;
                    }
                }
            }
        }
        Ok(mask)
    }
}

/// Convenience wrapper: initialise a ViBe model from `first_frame`, process
/// all subsequent `frames`, and return the foreground masks.
///
/// # Arguments
/// * `first_frame` – used to initialise the sample sets.
/// * `frames`      – subsequent frames to classify.
/// * `config`      – ViBe hyper-parameters (or [`ViBeConfig::default()`]).
pub fn vibe_model(
    first_frame: &Array2<f64>,
    frames: &[Array2<f64>],
    config: Option<ViBeConfig>,
) -> NdimageResult<Vec<Array2<bool>>> {
    let cfg = config.unwrap_or_default();
    let mut model = ViBeModel::new(first_frame, cfg)?;
    let mut results = Vec::with_capacity(frames.len());
    for frame in frames {
        let mask = model.update_and_segment(frame)?;
        results.push(mask);
    }
    Ok(results)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_frame(rows: usize, cols: usize, val: f64) -> Array2<f64> {
        Array2::from_elem((rows, cols), val)
    }

    #[test]
    fn test_frame_buffer_basic() {
        let mut buf = FrameBuffer::new(3).expect("capacity ok");
        assert!(buf.is_empty());
        buf.push(make_frame(4, 4, 1.0)).expect("push ok");
        buf.push(make_frame(4, 4, 2.0)).expect("push ok");
        buf.push(make_frame(4, 4, 3.0)).expect("push ok");
        assert!(buf.is_full());
        let newest = buf.get_recent(0).expect("get ok");
        assert!((newest[[0, 0]] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_frame_buffer_overflow() {
        let mut buf = FrameBuffer::new(2).expect("ok");
        for v in [10.0, 20.0, 30.0] {
            buf.push(make_frame(2, 2, v)).expect("push");
        }
        // Should retain last 2
        let newest = buf.get_recent(0).expect("get");
        assert!((newest[[0, 0]] - 30.0).abs() < 1e-9);
        let second = buf.get_recent(1).expect("get");
        assert!((second[[0, 0]] - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_simple_background_model() {
        let bg_val = 100.0;
        let frame = make_frame(8, 8, bg_val);
        let mut model = SimpleBackgroundModel::new(&frame, 0.1).expect("new ok");
        // After many updates with the same frame the model should converge
        for _ in 0..50 {
            model.update(&frame).expect("update ok");
        }
        let diff = (model.background[[0, 0]] - bg_val).abs();
        assert!(diff < 1.0, "background should converge, diff={diff}");
    }

    #[test]
    fn test_subtract_background_simple() {
        let bg = make_frame(4, 4, 50.0);
        let frame = make_frame(4, 4, 80.0);
        let mask = subtract_background(&frame, &bg, 20.0).expect("ok");
        // |80 - 50| = 30 > 20 → all foreground
        for r in 0..4 {
            for c in 0..4 {
                assert!(mask[[r, c]]);
            }
        }
    }

    #[test]
    fn test_gmm_background_smoke() {
        let first = make_frame(6, 6, 128.0);
        let mut gmm =
            GaussianMixtureBackground::new(&first, 3, 0.05, 0.7).expect("new ok");
        // Feed a few identical frames — should classify as background
        for _ in 0..10 {
            gmm.update_and_segment(&first).expect("update ok");
        }
        let fg = gmm.update_and_segment(&first).expect("last update");
        // Most pixels should be background after convergence
        let n_fg: usize = fg.iter().filter(|&&x| x).count();
        assert!(
            n_fg as f64 / (6.0 * 6.0) < 0.5,
            "expected mostly background, got {n_fg}/36 foreground"
        );
    }

    #[test]
    fn test_vibe_model() {
        let rows = 8;
        let cols = 8;
        let first = make_frame(rows, cols, 100.0);
        let frames: Vec<_> = (0..5).map(|_| make_frame(rows, cols, 100.0)).collect();
        let masks = vibe_model(&first, &frames, None).expect("vibe ok");
        assert_eq!(masks.len(), 5);
        // After initialisation + identical frames, most pixels should be background
        let last = &masks[4];
        let n_fg: usize = last.iter().filter(|&&x| x).count();
        assert!(
            n_fg as f64 / (rows * cols) as f64 < 0.5,
            "expected mostly background, got {n_fg} fg"
        );
    }
}
