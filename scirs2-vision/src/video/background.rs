//! Background subtraction algorithms for video processing.
//!
//! Provides several background modelling approaches for separating foreground
//! objects from a relatively static background scene.
//!
//! # Algorithms
//!
//! - **Running average** -- simple exponential moving average
//! - **Gaussian Mixture Model (GMM / MOG2)** -- per-pixel multi-Gaussian model
//! - **Median background** -- per-pixel running median approximation
//! - **Shadow detection heuristics** -- optional shadow classification
//!
//! All models operate on single-channel (grayscale) `Array2<f64>` frames where
//! pixel values are expected in `[0, 1]`.

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::{Array2, Ix2};

// ---------------------------------------------------------------------------
// Common types
// ---------------------------------------------------------------------------

/// Foreground mask produced by background subtraction.
///
/// Each pixel is classified as `Background`, `Foreground`, or optionally
/// `Shadow` when shadow detection is enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForegroundLabel {
    /// Background pixel
    Background,
    /// Foreground (moving) pixel
    Foreground,
    /// Shadow pixel (optional classification)
    Shadow,
}

/// Convert a label mask to a binary f64 image (foreground = 1, else 0).
pub fn mask_to_binary(mask: &Array2<ForegroundLabel>) -> Array2<f64> {
    mask.mapv(|l| match l {
        ForegroundLabel::Foreground => 1.0,
        _ => 0.0,
    })
}

/// Configuration shared across background subtraction models.
#[derive(Debug, Clone)]
pub struct BackgroundConfig {
    /// Learning rate (alpha) for background updates. Range `(0, 1]`.
    pub learning_rate: f64,
    /// Foreground threshold -- the minimum absolute difference between a pixel
    /// and its background model to be declared foreground.
    pub fg_threshold: f64,
    /// Enable shadow detection heuristics.
    pub detect_shadows: bool,
    /// Shadow detection parameters (only used when `detect_shadows` is true).
    pub shadow_params: ShadowParams,
}

impl Default for BackgroundConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.05,
            fg_threshold: 0.15,
            detect_shadows: false,
            shadow_params: ShadowParams::default(),
        }
    }
}

/// Parameters for the shadow detection heuristic.
///
/// A pixel is classified as shadow rather than foreground when the ratio
/// `pixel / bg` falls within `[tau_lo, tau_hi]`.  This exploits the
/// observation that shadows darken a pixel but preserve its relative
/// relationship to the background.
#[derive(Debug, Clone)]
pub struct ShadowParams {
    /// Lower ratio bound (e.g. 0.4).
    pub tau_lo: f64,
    /// Upper ratio bound (e.g. 0.9).
    pub tau_hi: f64,
}

impl Default for ShadowParams {
    fn default() -> Self {
        Self {
            tau_lo: 0.4,
            tau_hi: 0.9,
        }
    }
}

// ---------------------------------------------------------------------------
// Shadow detection helper
// ---------------------------------------------------------------------------

fn classify_pixel(
    pixel: f64,
    bg_value: f64,
    threshold: f64,
    detect_shadows: bool,
    shadow_params: &ShadowParams,
) -> ForegroundLabel {
    let diff = (pixel - bg_value).abs();
    if diff < threshold {
        return ForegroundLabel::Background;
    }
    if detect_shadows && bg_value > 1e-9 {
        let ratio = pixel / bg_value;
        if ratio >= shadow_params.tau_lo && ratio <= shadow_params.tau_hi {
            return ForegroundLabel::Shadow;
        }
    }
    ForegroundLabel::Foreground
}

// ---------------------------------------------------------------------------
// Running Average Background Model
// ---------------------------------------------------------------------------

/// Running-average background model.
///
/// The background at each pixel is maintained as an exponentially-weighted
/// moving average:
///
/// ```text
/// bg(t) = (1 - alpha) * bg(t-1) + alpha * frame(t)
/// ```
///
/// where `alpha` is the learning rate.
#[derive(Debug, Clone)]
pub struct RunningAverageBackground {
    /// Current background estimate.
    background: Option<Array2<f64>>,
    /// Configuration.
    config: BackgroundConfig,
    /// Number of frames processed so far.
    frame_count: u64,
}

impl RunningAverageBackground {
    /// Create a new running-average background model.
    pub fn new(config: BackgroundConfig) -> Result<Self> {
        if config.learning_rate <= 0.0 || config.learning_rate > 1.0 {
            return Err(VisionError::InvalidParameter(
                "learning_rate must be in (0, 1]".into(),
            ));
        }
        if config.fg_threshold <= 0.0 {
            return Err(VisionError::InvalidParameter(
                "fg_threshold must be positive".into(),
            ));
        }
        Ok(Self {
            background: None,
            config,
            frame_count: 0,
        })
    }

    /// Create with default configuration.
    pub fn default_config() -> Result<Self> {
        Self::new(BackgroundConfig::default())
    }

    /// Process the next frame and return a foreground mask.
    pub fn apply(&mut self, frame: &Array2<f64>) -> Result<Array2<ForegroundLabel>> {
        let shape = frame.raw_dim();
        match &mut self.background {
            None => {
                // First frame becomes the initial background.
                self.background = Some(frame.clone());
                self.frame_count = 1;
                Ok(Array2::from_elem(shape, ForegroundLabel::Background))
            }
            Some(bg) => {
                if bg.raw_dim() != shape {
                    return Err(VisionError::DimensionMismatch(format!(
                        "Frame shape {:?} does not match background {:?}",
                        shape,
                        bg.raw_dim()
                    )));
                }
                self.frame_count += 1;
                let alpha = self.config.learning_rate;
                let threshold = self.config.fg_threshold;
                let detect_shadows = self.config.detect_shadows;
                let shadow_params = &self.config.shadow_params;

                let rows = shape[0];
                let cols = shape[1];
                let mut mask = Array2::from_elem(shape, ForegroundLabel::Background);

                for r in 0..rows {
                    for c in 0..cols {
                        let p = frame[[r, c]];
                        let b = bg[[r, c]];
                        mask[[r, c]] =
                            classify_pixel(p, b, threshold, detect_shadows, shadow_params);
                        // Update background
                        bg[[r, c]] = (1.0 - alpha) * b + alpha * p;
                    }
                }
                Ok(mask)
            }
        }
    }

    /// Return a reference to the current background image, if available.
    pub fn background(&self) -> Option<&Array2<f64>> {
        self.background.as_ref()
    }

    /// Return the number of frames processed.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Set a new learning rate.
    pub fn set_learning_rate(&mut self, rate: f64) -> Result<()> {
        if rate <= 0.0 || rate > 1.0 {
            return Err(VisionError::InvalidParameter(
                "learning_rate must be in (0, 1]".into(),
            ));
        }
        self.config.learning_rate = rate;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Gaussian Mixture Model (GMM / MOG2) Background Subtraction
// ---------------------------------------------------------------------------

/// Per-pixel Gaussian component.
#[derive(Debug, Clone)]
struct GaussianComponent {
    mean: f64,
    variance: f64,
    weight: f64,
}

/// Gaussian Mixture Model (MOG2) background subtractor.
///
/// Each pixel is modelled by a mixture of `K` Gaussians.  A new observation
/// is matched to the closest component; if no match is found a new component
/// is created (replacing the weakest).  The components are ranked by
/// `weight / sigma` and the top components whose cumulative weight exceeds a
/// threshold are considered "background".
#[derive(Debug, Clone)]
pub struct GmmBackground {
    /// Maximum number of Gaussian components per pixel.
    max_components: usize,
    /// Per-pixel component vectors -- indexed `[row][col]`.
    models: Option<Vec<Vec<Vec<GaussianComponent>>>>,
    /// Configuration.
    config: BackgroundConfig,
    /// Mahalanobis distance threshold for match (number of std deviations).
    match_threshold: f64,
    /// Background weight threshold -- cumulative weight fraction that counts
    /// as background.
    bg_weight_threshold: f64,
    /// Minimum variance to avoid singularities.
    min_variance: f64,
    /// Frame count.
    frame_count: u64,
    /// Frame dimensions for validation.
    frame_rows: usize,
    /// Frame dimensions for validation.
    frame_cols: usize,
}

impl GmmBackground {
    /// Create a new GMM background model.
    ///
    /// # Arguments
    /// * `max_components` -- number of Gaussians per pixel (typically 3--5)
    /// * `config` -- common background configuration
    pub fn new(max_components: usize, config: BackgroundConfig) -> Result<Self> {
        if max_components == 0 {
            return Err(VisionError::InvalidParameter(
                "max_components must be >= 1".into(),
            ));
        }
        if config.learning_rate <= 0.0 || config.learning_rate > 1.0 {
            return Err(VisionError::InvalidParameter(
                "learning_rate must be in (0, 1]".into(),
            ));
        }
        Ok(Self {
            max_components,
            models: None,
            config,
            match_threshold: 2.5,
            bg_weight_threshold: 0.7,
            min_variance: 0.001,
            frame_count: 0,
            frame_rows: 0,
            frame_cols: 0,
        })
    }

    /// Create with typical defaults (5 components).
    pub fn default_config() -> Result<Self> {
        Self::new(5, BackgroundConfig::default())
    }

    /// Set the Mahalanobis match threshold.
    pub fn set_match_threshold(&mut self, t: f64) -> Result<()> {
        if t <= 0.0 {
            return Err(VisionError::InvalidParameter(
                "match_threshold must be positive".into(),
            ));
        }
        self.match_threshold = t;
        Ok(())
    }

    /// Process a frame and return foreground mask.
    pub fn apply(&mut self, frame: &Array2<f64>) -> Result<Array2<ForegroundLabel>> {
        let rows = frame.nrows();
        let cols = frame.ncols();
        let shape: Ix2 = frame.raw_dim();

        if self.models.is_none() {
            // Initialise models with one component per pixel.
            let mut models = Vec::with_capacity(rows);
            for r in 0..rows {
                let mut row_models = Vec::with_capacity(cols);
                for c in 0..cols {
                    let comp = GaussianComponent {
                        mean: frame[[r, c]],
                        variance: 0.02,
                        weight: 1.0,
                    };
                    row_models.push(vec![comp]);
                }
                models.push(row_models);
            }
            self.models = Some(models);
            self.frame_rows = rows;
            self.frame_cols = cols;
            self.frame_count = 1;
            return Ok(Array2::from_elem(shape, ForegroundLabel::Background));
        }

        if rows != self.frame_rows || cols != self.frame_cols {
            return Err(VisionError::DimensionMismatch(format!(
                "Frame ({rows}x{cols}) differs from model ({}x{})",
                self.frame_rows, self.frame_cols,
            )));
        }

        self.frame_count += 1;
        let alpha = self.config.learning_rate;
        let fg_thresh = self.config.fg_threshold;
        let detect_shadows = self.config.detect_shadows;
        let shadow_params = self.config.shadow_params.clone();
        let match_thresh = self.match_threshold;
        let max_k = self.max_components;
        let bg_wt = self.bg_weight_threshold;
        let min_var = self.min_variance;

        let models = self
            .models
            .as_mut()
            .ok_or_else(|| VisionError::OperationError("Models not initialised".into()))?;

        let mut mask = Array2::from_elem(shape, ForegroundLabel::Background);

        for r in 0..rows {
            for c in 0..cols {
                let pixel = frame[[r, c]];
                let comps = &mut models[r][c];

                // Try to match to an existing component.
                let mut matched = false;
                let mut matched_bg = false;

                // Sort by weight/sigma descending to find background components.
                comps.sort_by(|a, b| {
                    let ra = a.weight / a.variance.sqrt().max(1e-12);
                    let rb = b.weight / b.variance.sqrt().max(1e-12);
                    rb.partial_cmp(&ra).unwrap_or(std::cmp::Ordering::Equal)
                });

                // Determine which components are background.
                let mut cum_weight = 0.0;
                let mut bg_count = 0;
                for comp in comps.iter() {
                    cum_weight += comp.weight;
                    bg_count += 1;
                    if cum_weight >= bg_wt {
                        break;
                    }
                }

                for (i, comp) in comps.iter_mut().enumerate() {
                    let sigma = comp.variance.sqrt().max(1e-12);
                    let d = (pixel - comp.mean).abs() / sigma;
                    if d < match_thresh {
                        // Matched -- update this component.
                        comp.mean = (1.0 - alpha) * comp.mean + alpha * pixel;
                        let diff = pixel - comp.mean;
                        comp.variance =
                            ((1.0 - alpha) * comp.variance + alpha * diff * diff).max(min_var);
                        comp.weight = (1.0 - alpha) * comp.weight + alpha;
                        matched = true;
                        if i < bg_count {
                            matched_bg = true;
                        }
                        break;
                    }
                }

                // Decrease weights of all non-matched components.
                let mut total_w = 0.0;
                for comp in comps.iter_mut() {
                    comp.weight *= 1.0 - alpha;
                    total_w += comp.weight;
                }

                if !matched {
                    // Add or replace with a new component.
                    let new_comp = GaussianComponent {
                        mean: pixel,
                        variance: 0.02,
                        weight: alpha,
                    };
                    total_w += alpha;
                    if comps.len() < max_k {
                        comps.push(new_comp);
                    } else {
                        // Replace weakest.
                        if let Some(last) = comps.last_mut() {
                            total_w -= last.weight;
                            *last = new_comp;
                            total_w += last.weight;
                        }
                    }
                }

                // Normalise weights.
                if total_w > 0.0 {
                    for comp in comps.iter_mut() {
                        comp.weight /= total_w;
                    }
                }

                // Classify pixel.
                if matched_bg {
                    // Possibly a shadow?
                    if detect_shadows {
                        // Use the top background component mean.
                        let bg_mean = comps.first().map(|c| c.mean).unwrap_or(0.0);
                        mask[[r, c]] = classify_pixel(
                            pixel,
                            bg_mean,
                            fg_thresh,
                            detect_shadows,
                            &shadow_params,
                        );
                    } else {
                        mask[[r, c]] = ForegroundLabel::Background;
                    }
                } else {
                    mask[[r, c]] = ForegroundLabel::Foreground;
                }
            }
        }

        Ok(mask)
    }

    /// Return the estimated background image (mean of the dominant component).
    pub fn background_image(&self) -> Option<Array2<f64>> {
        let models = self.models.as_ref()?;
        let mut bg = Array2::zeros((self.frame_rows, self.frame_cols));
        for r in 0..self.frame_rows {
            for c in 0..self.frame_cols {
                if let Some(comp) = models[r][c].first() {
                    bg[[r, c]] = comp.mean;
                }
            }
        }
        Some(bg)
    }

    /// Frame count.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

// ---------------------------------------------------------------------------
// Median Background Model
// ---------------------------------------------------------------------------

/// Median background model.
///
/// Maintains a sliding window of recent pixel values and uses the approximate
/// running median as the background estimate.  The running median is updated
/// incrementally: if the new pixel is above the current median, the median is
/// increased by a small step; if below, it is decreased.
#[derive(Debug, Clone)]
pub struct MedianBackground {
    /// Running median estimate per pixel.
    median: Option<Array2<f64>>,
    /// Step size for median updates.
    step: f64,
    /// Configuration.
    config: BackgroundConfig,
    /// Frame count.
    frame_count: u64,
}

impl MedianBackground {
    /// Create a new median background model.
    ///
    /// `step` controls how fast the median adapts -- typical values are 0.001--0.01.
    pub fn new(step: f64, config: BackgroundConfig) -> Result<Self> {
        if step <= 0.0 {
            return Err(VisionError::InvalidParameter(
                "step must be positive".into(),
            ));
        }
        Ok(Self {
            median: None,
            step,
            config,
            frame_count: 0,
        })
    }

    /// Default configuration.
    pub fn default_config() -> Result<Self> {
        Self::new(0.005, BackgroundConfig::default())
    }

    /// Process a frame and return the foreground mask.
    pub fn apply(&mut self, frame: &Array2<f64>) -> Result<Array2<ForegroundLabel>> {
        let shape = frame.raw_dim();
        match &mut self.median {
            None => {
                self.median = Some(frame.clone());
                self.frame_count = 1;
                Ok(Array2::from_elem(shape, ForegroundLabel::Background))
            }
            Some(med) => {
                if med.raw_dim() != shape {
                    return Err(VisionError::DimensionMismatch(format!(
                        "Frame shape {:?} vs median {:?}",
                        shape,
                        med.raw_dim()
                    )));
                }
                self.frame_count += 1;
                let step = self.step;
                let threshold = self.config.fg_threshold;
                let detect_shadows = self.config.detect_shadows;
                let shadow_params = &self.config.shadow_params;

                let rows = shape[0];
                let cols = shape[1];
                let mut mask = Array2::from_elem(shape, ForegroundLabel::Background);

                for r in 0..rows {
                    for c in 0..cols {
                        let p = frame[[r, c]];
                        let m = med[[r, c]];
                        mask[[r, c]] =
                            classify_pixel(p, m, threshold, detect_shadows, shadow_params);

                        // Update running median.
                        if p > m {
                            med[[r, c]] = (m + step).min(1.0);
                        } else if p < m {
                            med[[r, c]] = (m - step).max(0.0);
                        }
                    }
                }
                Ok(mask)
            }
        }
    }

    /// Return the current median background estimate.
    pub fn background(&self) -> Option<&Array2<f64>> {
        self.median.as_ref()
    }

    /// Frame count.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

// ---------------------------------------------------------------------------
// Adaptive learning rate helper
// ---------------------------------------------------------------------------

/// Compute an adaptive learning rate based on the fraction of foreground
/// pixels in the most recent mask.  When the scene is mostly static the rate
/// is higher (fast convergence); when significant motion is present the rate
/// is lowered to avoid absorbing foreground into the background model.
///
/// # Arguments
/// * `mask` -- most recently computed foreground mask
/// * `base_rate` -- nominal learning rate (e.g. 0.05)
/// * `min_rate` -- minimum learning rate floor (e.g. 0.001)
pub fn adaptive_learning_rate(
    mask: &Array2<ForegroundLabel>,
    base_rate: f64,
    min_rate: f64,
) -> f64 {
    let total = mask.len() as f64;
    if total == 0.0 {
        return base_rate;
    }
    let fg_count = mask
        .iter()
        .filter(|&&l| l == ForegroundLabel::Foreground)
        .count() as f64;
    let fg_fraction = fg_count / total;
    // Linearly decrease rate as foreground fraction increases.
    let rate = base_rate * (1.0 - fg_fraction);
    rate.max(min_rate)
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Helper: create a blank 8x8 frame with a given value.
    fn uniform_frame(val: f64) -> Array2<f64> {
        Array2::from_elem((8, 8), val)
    }

    /// Helper: place a "foreground object" (bright square) on a dark frame.
    fn frame_with_object(bg: f64, fg: f64, top: usize, left: usize, size: usize) -> Array2<f64> {
        let mut f = Array2::from_elem((8, 8), bg);
        for r in top..(top + size).min(8) {
            for c in left..(left + size).min(8) {
                f[[r, c]] = fg;
            }
        }
        f
    }

    // ---- Running Average ----

    #[test]
    fn test_running_avg_first_frame_all_bg() {
        let mut model =
            RunningAverageBackground::default_config().expect("default config should succeed");
        let frame = uniform_frame(0.5);
        let mask = model.apply(&frame).expect("apply should succeed");
        assert!(mask.iter().all(|&l| l == ForegroundLabel::Background));
        assert_eq!(model.frame_count(), 1);
    }

    #[test]
    fn test_running_avg_detects_foreground() {
        let mut model = RunningAverageBackground::new(BackgroundConfig {
            learning_rate: 0.01,
            fg_threshold: 0.1,
            ..Default::default()
        })
        .expect("config ok");
        // Train on several blank frames.
        let blank = uniform_frame(0.2);
        for _ in 0..10 {
            model.apply(&blank).expect("apply");
        }
        // Now introduce an object.
        let obj = frame_with_object(0.2, 0.9, 2, 2, 3);
        let mask = model.apply(&obj).expect("apply");
        // Object pixels should be foreground.
        for r in 2..5 {
            for c in 2..5 {
                assert_eq!(mask[[r, c]], ForegroundLabel::Foreground);
            }
        }
        // Background pixels should remain background.
        assert_eq!(mask[[0, 0]], ForegroundLabel::Background);
    }

    #[test]
    fn test_running_avg_shape_mismatch() {
        let mut model = RunningAverageBackground::default_config().expect("ok");
        model.apply(&uniform_frame(0.5)).expect("first apply");
        let wrong = Array2::from_elem((4, 4), 0.5);
        let res = model.apply(&wrong);
        assert!(res.is_err());
    }

    #[test]
    fn test_running_avg_invalid_lr() {
        let res = RunningAverageBackground::new(BackgroundConfig {
            learning_rate: 0.0,
            ..Default::default()
        });
        assert!(res.is_err());
        let res2 = RunningAverageBackground::new(BackgroundConfig {
            learning_rate: 1.5,
            ..Default::default()
        });
        assert!(res2.is_err());
    }

    #[test]
    fn test_running_avg_shadow_detection() {
        let mut model = RunningAverageBackground::new(BackgroundConfig {
            learning_rate: 0.01,
            fg_threshold: 0.05,
            detect_shadows: true,
            shadow_params: ShadowParams {
                tau_lo: 0.4,
                tau_hi: 0.9,
            },
        })
        .expect("ok");
        let bg_val = 0.8;
        let blank = uniform_frame(bg_val);
        for _ in 0..20 {
            model.apply(&blank).expect("ok");
        }
        // Introduce a shadow (darker but proportional).
        let shadow_val = 0.55; // ratio = 0.55/0.8 = 0.6875 in [0.4, 0.9]
        let shadow_frame = frame_with_object(bg_val, shadow_val, 1, 1, 2);
        let mask = model.apply(&shadow_frame).expect("ok");
        for r in 1..3 {
            for c in 1..3 {
                assert_eq!(mask[[r, c]], ForegroundLabel::Shadow);
            }
        }
    }

    #[test]
    fn test_running_avg_background_converges() {
        let mut model = RunningAverageBackground::new(BackgroundConfig {
            learning_rate: 0.5,
            fg_threshold: 0.05,
            ..Default::default()
        })
        .expect("ok");
        let frame = uniform_frame(0.6);
        for _ in 0..50 {
            model.apply(&frame).expect("ok");
        }
        let bg = model.background().expect("should exist");
        for &val in bg.iter() {
            assert!((val - 0.6).abs() < 0.01, "bg should converge to 0.6");
        }
    }

    #[test]
    fn test_running_avg_set_lr() {
        let mut model = RunningAverageBackground::default_config().expect("ok");
        assert!(model.set_learning_rate(0.1).is_ok());
        assert!(model.set_learning_rate(0.0).is_err());
        assert!(model.set_learning_rate(1.5).is_err());
    }

    // ---- GMM / MOG2 ----

    #[test]
    fn test_gmm_first_frame() {
        let mut model = GmmBackground::default_config().expect("ok");
        let frame = uniform_frame(0.5);
        let mask = model.apply(&frame).expect("apply");
        assert!(mask.iter().all(|&l| l == ForegroundLabel::Background));
    }

    #[test]
    fn test_gmm_detects_foreground() {
        let mut model = GmmBackground::new(
            3,
            BackgroundConfig {
                learning_rate: 0.1,
                fg_threshold: 0.1,
                ..Default::default()
            },
        )
        .expect("ok");
        let blank = uniform_frame(0.3);
        for _ in 0..15 {
            model.apply(&blank).expect("ok");
        }
        let obj = frame_with_object(0.3, 0.95, 3, 3, 2);
        let mask = model.apply(&obj).expect("ok");
        for r in 3..5 {
            for c in 3..5 {
                assert_eq!(mask[[r, c]], ForegroundLabel::Foreground);
            }
        }
    }

    #[test]
    fn test_gmm_shape_mismatch() {
        let mut model = GmmBackground::default_config().expect("ok");
        model.apply(&uniform_frame(0.5)).expect("first ok");
        let wrong = Array2::from_elem((4, 4), 0.5);
        assert!(model.apply(&wrong).is_err());
    }

    #[test]
    fn test_gmm_background_image() {
        let mut model = GmmBackground::default_config().expect("ok");
        let frame = uniform_frame(0.5);
        model.apply(&frame).expect("ok");
        let bg = model.background_image().expect("should have bg");
        assert_eq!(bg.nrows(), 8);
        assert_eq!(bg.ncols(), 8);
    }

    #[test]
    fn test_gmm_invalid_params() {
        assert!(GmmBackground::new(0, BackgroundConfig::default()).is_err());
        assert!(GmmBackground::new(
            3,
            BackgroundConfig {
                learning_rate: -0.1,
                ..Default::default()
            }
        )
        .is_err());
    }

    #[test]
    fn test_gmm_shadow_mode() {
        let mut model = GmmBackground::new(
            3,
            BackgroundConfig {
                learning_rate: 0.1,
                fg_threshold: 0.05,
                detect_shadows: true,
                shadow_params: ShadowParams {
                    tau_lo: 0.4,
                    tau_hi: 0.9,
                },
            },
        )
        .expect("ok");
        let blank = uniform_frame(0.8);
        for _ in 0..20 {
            model.apply(&blank).expect("ok");
        }
        // Shadow frame.
        let shadow_frame = frame_with_object(0.8, 0.55, 0, 0, 2);
        let mask = model.apply(&shadow_frame).expect("ok");
        // Shadow pixels should be Shadow or Foreground (both are acceptable
        // depending on model state); they should NOT all be Background.
        let non_bg: usize = mask
            .iter()
            .filter(|&&l| l != ForegroundLabel::Background)
            .count();
        assert!(non_bg > 0, "Expected some non-background pixels");
    }

    // ---- Median Background ----

    #[test]
    fn test_median_first_frame() {
        let mut model = MedianBackground::default_config().expect("ok");
        let frame = uniform_frame(0.5);
        let mask = model.apply(&frame).expect("ok");
        assert!(mask.iter().all(|&l| l == ForegroundLabel::Background));
    }

    #[test]
    fn test_median_detects_foreground() {
        let mut model = MedianBackground::new(
            0.01,
            BackgroundConfig {
                fg_threshold: 0.1,
                ..Default::default()
            },
        )
        .expect("ok");
        let blank = uniform_frame(0.3);
        for _ in 0..20 {
            model.apply(&blank).expect("ok");
        }
        let obj = frame_with_object(0.3, 0.9, 1, 1, 3);
        let mask = model.apply(&obj).expect("ok");
        for r in 1..4 {
            for c in 1..4 {
                assert_eq!(mask[[r, c]], ForegroundLabel::Foreground);
            }
        }
    }

    #[test]
    fn test_median_shape_mismatch() {
        let mut model = MedianBackground::default_config().expect("ok");
        model.apply(&uniform_frame(0.5)).expect("ok");
        assert!(model.apply(&Array2::from_elem((4, 4), 0.5)).is_err());
    }

    #[test]
    fn test_median_invalid_step() {
        assert!(MedianBackground::new(0.0, BackgroundConfig::default()).is_err());
        assert!(MedianBackground::new(-1.0, BackgroundConfig::default()).is_err());
    }

    #[test]
    fn test_median_background_converges() {
        let mut model = MedianBackground::new(
            0.05,
            BackgroundConfig {
                fg_threshold: 0.1,
                ..Default::default()
            },
        )
        .expect("ok");
        let frame = uniform_frame(0.7);
        for _ in 0..100 {
            model.apply(&frame).expect("ok");
        }
        let bg = model.background().expect("bg");
        for &v in bg.iter() {
            assert!(
                (v - 0.7).abs() < 0.06,
                "median should approach 0.7, got {v}"
            );
        }
    }

    // ---- mask_to_binary ----

    #[test]
    fn test_mask_to_binary() {
        let mut mask = Array2::from_elem((3, 3), ForegroundLabel::Background);
        mask[[0, 0]] = ForegroundLabel::Foreground;
        mask[[1, 1]] = ForegroundLabel::Shadow;
        let bin = mask_to_binary(&mask);
        assert!((bin[[0, 0]] - 1.0).abs() < 1e-12);
        assert!(bin[[1, 1]].abs() < 1e-12);
        assert!(bin[[2, 2]].abs() < 1e-12);
    }

    // ---- Adaptive learning rate ----

    #[test]
    fn test_adaptive_lr_static_scene() {
        let mask = Array2::from_elem((4, 4), ForegroundLabel::Background);
        let rate = adaptive_learning_rate(&mask, 0.05, 0.001);
        assert!((rate - 0.05).abs() < 1e-9, "all bg => full rate");
    }

    #[test]
    fn test_adaptive_lr_high_motion() {
        let mask = Array2::from_elem((4, 4), ForegroundLabel::Foreground);
        let rate = adaptive_learning_rate(&mask, 0.05, 0.001);
        assert!(
            (rate - 0.001).abs() < 1e-9,
            "all fg => rate should be at min_rate"
        );
    }

    #[test]
    fn test_adaptive_lr_partial_fg() {
        let mut mask = Array2::from_elem((4, 4), ForegroundLabel::Background);
        // 4/16 = 25% foreground
        mask[[0, 0]] = ForegroundLabel::Foreground;
        mask[[0, 1]] = ForegroundLabel::Foreground;
        mask[[1, 0]] = ForegroundLabel::Foreground;
        mask[[1, 1]] = ForegroundLabel::Foreground;
        let rate = adaptive_learning_rate(&mask, 0.05, 0.001);
        let expected = 0.05 * (1.0 - 0.25);
        assert!(
            (rate - expected).abs() < 1e-9,
            "25% fg => rate = {expected}"
        );
    }
}
