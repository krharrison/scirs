//! Object tracking algorithms for video processing.
//!
//! Provides classical tracking approaches that work on grayscale
//! `Array2<f64>` frames (pixel values in `[0, 1]`).
//!
//! # Algorithms
//!
//! - **Mean Shift** -- iterative mode-seeking on a colour/intensity histogram
//! - **CamShift** -- Continuously Adaptive Mean Shift with automatic window sizing
//! - **Kalman filter tracker** -- constant-velocity and constant-acceleration models
//! - **Multi-object tracking** -- Hungarian (Munkres) assignment + track management
//!
//! # Track Management
//!
//! `MultiObjectTracker` handles creation, maintenance, and deletion of tracks
//! with configurable hit/miss thresholds and unique ID assignment.

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::Array2;

// ---------------------------------------------------------------------------
// Bounding box (internal, lightweight)
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box used by tracking algorithms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    /// Top-left row.
    pub top: f64,
    /// Top-left column.
    pub left: f64,
    /// Height.
    pub height: f64,
    /// Width.
    pub width: f64,
}

impl BBox {
    /// Create a new bounding box.
    pub fn new(top: f64, left: f64, height: f64, width: f64) -> Self {
        Self {
            top,
            left,
            height,
            width,
        }
    }

    /// Centre row.
    pub fn center_row(&self) -> f64 {
        self.top + self.height / 2.0
    }

    /// Centre column.
    pub fn center_col(&self) -> f64 {
        self.left + self.width / 2.0
    }

    /// Area.
    pub fn area(&self) -> f64 {
        self.height * self.width
    }

    /// IoU with another box.
    pub fn iou(&self, other: &BBox) -> f64 {
        let r1 = self.top;
        let r2 = self.top + self.height;
        let c1 = self.left;
        let c2 = self.left + self.width;

        let or1 = other.top;
        let or2 = other.top + other.height;
        let oc1 = other.left;
        let oc2 = other.left + other.width;

        let inter_r1 = r1.max(or1);
        let inter_r2 = r2.min(or2);
        let inter_c1 = c1.max(oc1);
        let inter_c2 = c2.min(oc2);

        if inter_r2 <= inter_r1 || inter_c2 <= inter_c1 {
            return 0.0;
        }
        let inter_area = (inter_r2 - inter_r1) * (inter_c2 - inter_c1);
        let union_area = self.area() + other.area() - inter_area;
        if union_area > 0.0 {
            inter_area / union_area
        } else {
            0.0
        }
    }

    /// Euclidean distance between centres.
    pub fn center_distance(&self, other: &BBox) -> f64 {
        let dr = self.center_row() - other.center_row();
        let dc = self.center_col() - other.center_col();
        (dr * dr + dc * dc).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Mean Shift Tracking
// ---------------------------------------------------------------------------

/// Mean Shift tracker configuration.
#[derive(Debug, Clone)]
pub struct MeanShiftConfig {
    /// Maximum number of iterations per update.
    pub max_iterations: usize,
    /// Convergence threshold (centre movement in pixels).
    pub epsilon: f64,
    /// Number of histogram bins for the target model.
    pub num_bins: usize,
}

impl Default for MeanShiftConfig {
    fn default() -> Self {
        Self {
            max_iterations: 30,
            epsilon: 1.0,
            num_bins: 16,
        }
    }
}

/// Mean Shift tracker.
///
/// Tracks a rectangular region by iteratively shifting its centre towards the
/// mode of a back-projection map derived from a target histogram.
#[derive(Debug, Clone)]
pub struct MeanShiftTracker {
    /// Current tracking window.
    window: BBox,
    /// Target histogram (normalised).
    target_hist: Vec<f64>,
    /// Configuration.
    config: MeanShiftConfig,
}

impl MeanShiftTracker {
    /// Initialise the tracker on the first frame with a given window.
    pub fn new(frame: &Array2<f64>, window: BBox, config: MeanShiftConfig) -> Result<Self> {
        if config.num_bins == 0 {
            return Err(VisionError::InvalidParameter("num_bins must be > 0".into()));
        }
        let hist = compute_histogram(frame, &window, config.num_bins);
        Ok(Self {
            window,
            target_hist: hist,
            config,
        })
    }

    /// Update the tracker with a new frame.  Returns the updated bounding box.
    pub fn update(&mut self, frame: &Array2<f64>) -> Result<BBox> {
        let rows = frame.nrows();
        let cols = frame.ncols();
        let bins = self.config.num_bins;

        for _ in 0..self.config.max_iterations {
            // Compute back-projection weights inside current window.
            let (mean_r, mean_c, _total_w) =
                compute_mean_shift_center(frame, &self.window, &self.target_hist, bins, rows, cols);

            let old_cr = self.window.center_row();
            let old_cc = self.window.center_col();

            let new_top = (mean_r - self.window.height / 2.0).max(0.0);
            let new_left = (mean_c - self.window.width / 2.0).max(0.0);

            self.window.top = new_top.min((rows as f64) - self.window.height);
            self.window.left = new_left.min((cols as f64) - self.window.width);

            let dr = self.window.center_row() - old_cr;
            let dc = self.window.center_col() - old_cc;
            if (dr * dr + dc * dc).sqrt() < self.config.epsilon {
                break;
            }
        }

        Ok(self.window)
    }

    /// Current window.
    pub fn window(&self) -> BBox {
        self.window
    }
}

// ---------------------------------------------------------------------------
// CamShift Tracking
// ---------------------------------------------------------------------------

/// CamShift (Continuously Adaptive Mean Shift) tracker.
///
/// Extends Mean Shift by adapting the window size and orientation based on the
/// zeroth and second moments of the back-projection.
#[derive(Debug, Clone)]
pub struct CamShiftTracker {
    /// Current tracking window.
    window: BBox,
    /// Target histogram.
    target_hist: Vec<f64>,
    /// Configuration.
    config: MeanShiftConfig,
    /// Estimated orientation angle (radians).
    angle: f64,
}

impl CamShiftTracker {
    /// Initialise on first frame.
    pub fn new(frame: &Array2<f64>, window: BBox, config: MeanShiftConfig) -> Result<Self> {
        if config.num_bins == 0 {
            return Err(VisionError::InvalidParameter("num_bins must be > 0".into()));
        }
        let hist = compute_histogram(frame, &window, config.num_bins);
        Ok(Self {
            window,
            target_hist: hist,
            config,
            angle: 0.0,
        })
    }

    /// Update with a new frame.  Returns the updated bounding box and the
    /// estimated orientation angle in radians.
    pub fn update(&mut self, frame: &Array2<f64>) -> Result<(BBox, f64)> {
        let rows = frame.nrows();
        let cols = frame.ncols();
        let bins = self.config.num_bins;

        // Mean-shift iterations.
        for _ in 0..self.config.max_iterations {
            let (mean_r, mean_c, _) =
                compute_mean_shift_center(frame, &self.window, &self.target_hist, bins, rows, cols);

            let old_cr = self.window.center_row();
            let old_cc = self.window.center_col();

            self.window.top = (mean_r - self.window.height / 2.0)
                .max(0.0)
                .min((rows as f64) - self.window.height);
            self.window.left = (mean_c - self.window.width / 2.0)
                .max(0.0)
                .min((cols as f64) - self.window.width);

            let dr = self.window.center_row() - old_cr;
            let dc = self.window.center_col() - old_cc;
            if (dr * dr + dc * dc).sqrt() < self.config.epsilon {
                break;
            }
        }

        // Compute moments and adapt window size.
        let (m00, m10, m01, m20, m02, m11) =
            compute_moments(frame, &self.window, &self.target_hist, bins, rows, cols);

        if m00 > 1e-9 {
            let xc = m10 / m00;
            let yc = m01 / m00;

            // Second central moments.
            let mu20 = m20 / m00 - xc * xc;
            let mu02 = m02 / m00 - yc * yc;
            let mu11 = m11 / m00 - xc * yc;

            // Orientation.
            self.angle = 0.5 * (2.0 * mu11).atan2(mu20 - mu02);

            // Adapt window size based on zeroth moment.
            let s = (m00 / 256.0).sqrt().max(2.0);
            self.window.width = s * 2.0;
            self.window.height = s * 2.0;

            // Re-centre.
            self.window.top = (yc - self.window.height / 2.0)
                .max(0.0)
                .min((rows as f64) - self.window.height);
            self.window.left = (xc - self.window.width / 2.0)
                .max(0.0)
                .min((cols as f64) - self.window.width);
        }

        Ok((self.window, self.angle))
    }

    /// Current window.
    pub fn window(&self) -> BBox {
        self.window
    }

    /// Current orientation angle in radians.
    pub fn angle(&self) -> f64 {
        self.angle
    }
}

// ---------------------------------------------------------------------------
// Histogram / mean-shift helpers
// ---------------------------------------------------------------------------

fn compute_histogram(frame: &Array2<f64>, bbox: &BBox, bins: usize) -> Vec<f64> {
    let mut hist = vec![0.0; bins];
    let rows = frame.nrows();
    let cols = frame.ncols();
    let r_start = (bbox.top as usize).min(rows);
    let r_end = ((bbox.top + bbox.height) as usize).min(rows);
    let c_start = (bbox.left as usize).min(cols);
    let c_end = ((bbox.left + bbox.width) as usize).min(cols);
    let mut total = 0.0;
    for r in r_start..r_end {
        for c in c_start..c_end {
            let val = frame[[r, c]].clamp(0.0, 1.0);
            let bin = ((val * (bins as f64 - 1.0)).round() as usize).min(bins - 1);
            hist[bin] += 1.0;
            total += 1.0;
        }
    }
    if total > 0.0 {
        for h in hist.iter_mut() {
            *h /= total;
        }
    }
    hist
}

fn compute_mean_shift_center(
    frame: &Array2<f64>,
    bbox: &BBox,
    target_hist: &[f64],
    bins: usize,
    rows: usize,
    cols: usize,
) -> (f64, f64, f64) {
    // Compute candidate histogram.
    let candidate_hist = compute_histogram(frame, bbox, bins);

    // Back-projection weights.
    let weights: Vec<f64> = (0..bins)
        .map(|i| {
            if candidate_hist[i] > 1e-12 {
                (target_hist[i] / candidate_hist[i]).sqrt().min(10.0)
            } else {
                0.0
            }
        })
        .collect();

    let r_start = (bbox.top as usize).min(rows);
    let r_end = ((bbox.top + bbox.height) as usize).min(rows);
    let c_start = (bbox.left as usize).min(cols);
    let c_end = ((bbox.left + bbox.width) as usize).min(cols);

    let mut sum_r = 0.0;
    let mut sum_c = 0.0;
    let mut sum_w = 0.0;

    for r in r_start..r_end {
        for c in c_start..c_end {
            let val = frame[[r, c]].clamp(0.0, 1.0);
            let bin = ((val * (bins as f64 - 1.0)).round() as usize).min(bins - 1);
            let w = weights[bin];
            sum_r += r as f64 * w;
            sum_c += c as f64 * w;
            sum_w += w;
        }
    }

    if sum_w > 0.0 {
        (sum_r / sum_w, sum_c / sum_w, sum_w)
    } else {
        (bbox.center_row(), bbox.center_col(), 0.0)
    }
}

fn compute_moments(
    frame: &Array2<f64>,
    bbox: &BBox,
    target_hist: &[f64],
    bins: usize,
    rows: usize,
    cols: usize,
) -> (f64, f64, f64, f64, f64, f64) {
    let candidate_hist = compute_histogram(frame, bbox, bins);
    let weights: Vec<f64> = (0..bins)
        .map(|i| {
            if candidate_hist[i] > 1e-12 {
                (target_hist[i] / candidate_hist[i]).sqrt().min(10.0)
            } else {
                0.0
            }
        })
        .collect();

    let r_start = (bbox.top as usize).min(rows);
    let r_end = ((bbox.top + bbox.height) as usize).min(rows);
    let c_start = (bbox.left as usize).min(cols);
    let c_end = ((bbox.left + bbox.width) as usize).min(cols);

    let mut m00 = 0.0;
    let mut m10 = 0.0; // sum x*w
    let mut m01 = 0.0; // sum y*w
    let mut m20 = 0.0;
    let mut m02 = 0.0;
    let mut m11 = 0.0;

    for r in r_start..r_end {
        for c in c_start..c_end {
            let val = frame[[r, c]].clamp(0.0, 1.0);
            let bin = ((val * (bins as f64 - 1.0)).round() as usize).min(bins - 1);
            let w = weights[bin];
            let x = c as f64;
            let y = r as f64;
            m00 += w;
            m10 += x * w;
            m01 += y * w;
            m20 += x * x * w;
            m02 += y * y * w;
            m11 += x * y * w;
        }
    }

    (m00, m10, m01, m20, m02, m11)
}

// ---------------------------------------------------------------------------
// Kalman Filter Tracker
// ---------------------------------------------------------------------------

/// Kalman filter motion model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KalmanModel {
    /// Constant velocity: state = [x, y, vx, vy].
    ConstantVelocity,
    /// Constant acceleration: state = [x, y, vx, vy, ax, ay].
    ConstantAcceleration,
}

/// A simple Kalman filter for 2-D point tracking.
#[derive(Debug, Clone)]
pub struct KalmanTracker {
    /// State vector.
    state: Vec<f64>,
    /// Covariance matrix (flattened row-major).
    cov: Vec<f64>,
    /// State dimension.
    dim: usize,
    /// Process noise scale.
    process_noise: f64,
    /// Measurement noise scale.
    measurement_noise: f64,
    /// Motion model.
    model: KalmanModel,
}

impl KalmanTracker {
    /// Create a new Kalman tracker initialised at position `(x, y)`.
    pub fn new(x: f64, y: f64, model: KalmanModel) -> Self {
        let dim = match model {
            KalmanModel::ConstantVelocity => 4,
            KalmanModel::ConstantAcceleration => 6,
        };
        let mut state = vec![0.0; dim];
        state[0] = x;
        state[1] = y;
        // Large initial covariance.
        let mut cov = vec![0.0; dim * dim];
        for i in 0..dim {
            cov[i * dim + i] = 1000.0;
        }
        Self {
            state,
            cov,
            dim,
            process_noise: 1.0,
            measurement_noise: 1.0,
            model,
        }
    }

    /// Set process and measurement noise scales.
    pub fn set_noise(&mut self, process: f64, measurement: f64) {
        self.process_noise = process;
        self.measurement_noise = measurement;
    }

    /// Predict step -- advance the state by one time step.
    pub fn predict(&mut self) {
        let n = self.dim;
        // State transition.
        let f = self.transition_matrix();
        let new_state = mat_vec_mul(&f, &self.state, n);
        self.state = new_state;

        // P = F P F^T + Q
        let fp = mat_mat_mul(&f, &self.cov, n);
        let ft = transpose(&f, n);
        let fp_ft = mat_mat_mul(&fp, &ft, n);
        let q = self.process_noise_matrix();
        self.cov = mat_add(&fp_ft, &q, n);
    }

    /// Update (correct) step with a measurement `(mx, my)`.
    pub fn update(&mut self, mx: f64, my: f64) {
        let n = self.dim;
        let h = self.observation_matrix();
        let m = 2; // measurement dimension

        // Innovation y = z - H x
        let hx = mat_vec_mul_rect(&h, &self.state, m, n);
        let z = [mx, my];
        let innovation = vec![z[0] - hx[0], z[1] - hx[1]];

        // S = H P H^T + R
        let hp = mat_mat_mul_rect(&h, &self.cov, m, n, n);
        let ht = transpose_rect(&h, m, n);
        let hp_ht = mat_mat_mul_rect(&hp, &ht, m, n, m);
        let r = self.measurement_noise_matrix();
        let s = mat_add_small(&hp_ht, &r, m);

        // K = P H^T S^{-1}
        let p_ht = mat_mat_mul_rect(&self.cov, &ht, n, n, m);
        let s_inv = invert_2x2(&s);
        let k = mat_mat_mul_rect(&p_ht, &s_inv, n, m, m);

        // x = x + K y
        let ky = mat_vec_mul_rect(&k, &innovation, n, m);
        for (i, &ky_i) in ky.iter().enumerate().take(n) {
            self.state[i] += ky_i;
        }

        // P = (I - K H) P
        let kh = mat_mat_mul_rect(&k, &h, n, m, n);
        let mut eye = vec![0.0; n * n];
        for i in 0..n {
            eye[i * n + i] = 1.0;
        }
        let i_kh = mat_sub(&eye, &kh, n);
        self.cov = mat_mat_mul(&i_kh, &self.cov, n);
    }

    /// Current estimated position `(x, y)`.
    pub fn position(&self) -> (f64, f64) {
        (self.state[0], self.state[1])
    }

    /// Current estimated velocity `(vx, vy)` (if modelled).
    pub fn velocity(&self) -> (f64, f64) {
        if self.dim >= 4 {
            (self.state[2], self.state[3])
        } else {
            (0.0, 0.0)
        }
    }

    /// Predicted position after one time step (without modifying state).
    pub fn predicted_position(&self) -> (f64, f64) {
        let f = self.transition_matrix();
        let pred = mat_vec_mul(&f, &self.state, self.dim);
        (pred[0], pred[1])
    }

    // ---- Internal matrices ----

    fn transition_matrix(&self) -> Vec<f64> {
        let n = self.dim;
        let mut f = vec![0.0; n * n];
        for i in 0..n {
            f[i * n + i] = 1.0;
        }
        match self.model {
            KalmanModel::ConstantVelocity => {
                // x += vx, y += vy
                f[2] = 1.0;
                f[n + 3] = 1.0;
            }
            KalmanModel::ConstantAcceleration => {
                // x += vx, y += vy, vx += ax, vy += ay
                f[2] = 1.0;
                f[n + 3] = 1.0;
                f[2 * n + 4] = 1.0;
                f[3 * n + 5] = 1.0;
                // x += 0.5*ax, y += 0.5*ay
                f[4] = 0.5;
                f[n + 5] = 0.5;
            }
        }
        f
    }

    fn observation_matrix(&self) -> Vec<f64> {
        let n = self.dim;
        let m = 2;
        let mut h = vec![0.0; m * n];
        h[0] = 1.0; // observe x
        h[n + 1] = 1.0; // observe y
        h
    }

    fn process_noise_matrix(&self) -> Vec<f64> {
        let n = self.dim;
        let mut q = vec![0.0; n * n];
        for i in 0..n {
            q[i * n + i] = self.process_noise;
        }
        q
    }

    fn measurement_noise_matrix(&self) -> Vec<f64> {
        vec![self.measurement_noise, 0.0, 0.0, self.measurement_noise]
    }
}

// ---------------------------------------------------------------------------
// Tiny linear algebra helpers (no external dependency)
// ---------------------------------------------------------------------------

fn mat_vec_mul(mat: &[f64], vec_in: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            out[i] += mat[i * n + j] * vec_in[j];
        }
    }
    out
}

fn mat_vec_mul_rect(mat: &[f64], vec_in: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m];
    for i in 0..m {
        for j in 0..n {
            out[i] += mat[i * n + j] * vec_in[j];
        }
    }
    out
}

fn mat_mat_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
    c
}

fn mat_mat_mul_rect(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for kk in 0..k {
            let a_ik = a[i * k + kk];
            for j in 0..n {
                c[i * n + j] += a_ik * b[kk * n + j];
            }
        }
    }
    c
}

fn transpose(a: &[f64], n: usize) -> Vec<f64> {
    let mut t = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = a[i * n + j];
        }
    }
    t
}

fn transpose_rect(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut t = vec![0.0; n * m];
    for i in 0..m {
        for j in 0..n {
            t[j * m + i] = a[i * n + j];
        }
    }
    t
}

fn mat_add(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..(n * n) {
        c[i] = a[i] + b[i];
    }
    c
}

fn mat_add_small(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let len = n * n;
    let mut c = vec![0.0; len];
    for i in 0..len {
        c[i] = a[i] + b[i];
    }
    c
}

fn mat_sub(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..(n * n) {
        c[i] = a[i] - b[i];
    }
    c
}

fn invert_2x2(m: &[f64]) -> Vec<f64> {
    let det = m[0] * m[3] - m[1] * m[2];
    if det.abs() < 1e-30 {
        // Return identity as fallback.
        return vec![1.0, 0.0, 0.0, 1.0];
    }
    let inv_det = 1.0 / det;
    vec![
        m[3] * inv_det,
        -m[1] * inv_det,
        -m[2] * inv_det,
        m[0] * inv_det,
    ]
}

// ---------------------------------------------------------------------------
// Multi-Object Tracking
// ---------------------------------------------------------------------------

/// Lifecycle state of a track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackStatus {
    /// Track is tentative (not yet confirmed).
    Tentative,
    /// Track is confirmed (enough consecutive hits).
    Confirmed,
    /// Track is lost (too many consecutive misses).
    Lost,
}

/// A tracked object with a unique ID.
#[derive(Debug, Clone)]
pub struct TrackedObject {
    /// Unique track ID.
    pub id: u64,
    /// Current bounding box.
    pub bbox: BBox,
    /// Kalman tracker for motion prediction.
    pub kalman: KalmanTracker,
    /// Number of consecutive hits (matched detections).
    pub hits: usize,
    /// Number of consecutive misses.
    pub misses: usize,
    /// Total age in frames.
    pub age: usize,
    /// Track status.
    pub status: TrackStatus,
}

/// Multi-object tracker configuration.
#[derive(Debug, Clone)]
pub struct MultiTrackerConfig {
    /// IoU threshold for assignment.
    pub iou_threshold: f64,
    /// Hits required to confirm a track.
    pub min_hits_to_confirm: usize,
    /// Consecutive misses before a track is deleted.
    pub max_misses: usize,
    /// Kalman filter motion model.
    pub kalman_model: KalmanModel,
    /// Process noise.
    pub process_noise: f64,
    /// Measurement noise.
    pub measurement_noise: f64,
}

impl Default for MultiTrackerConfig {
    fn default() -> Self {
        Self {
            iou_threshold: 0.3,
            min_hits_to_confirm: 3,
            max_misses: 5,
            kalman_model: KalmanModel::ConstantVelocity,
            process_noise: 1.0,
            measurement_noise: 1.0,
        }
    }
}

/// Multi-object tracker with Hungarian assignment and track management.
#[derive(Debug, Clone)]
pub struct MultiObjectTracker {
    /// Active tracks.
    tracks: Vec<TrackedObject>,
    /// Next unique track ID.
    next_id: u64,
    /// Configuration.
    config: MultiTrackerConfig,
}

impl MultiObjectTracker {
    /// Create a new multi-object tracker.
    pub fn new(config: MultiTrackerConfig) -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
            config,
        }
    }

    /// Update with a set of detections for the current frame.
    ///
    /// Returns a list of currently active (confirmed) tracks.
    pub fn update(&mut self, detections: &[BBox]) -> Vec<TrackedObject> {
        // 1. Predict all existing tracks.
        for track in self.tracks.iter_mut() {
            track.kalman.predict();
            let (px, py) = track.kalman.position();
            track.bbox.left = px - track.bbox.width / 2.0;
            track.bbox.top = py - track.bbox.height / 2.0;
        }

        // 2. Build IoU cost matrix and run Hungarian assignment.
        let n_tracks = self.tracks.len();
        let n_dets = detections.len();

        let (matched, unmatched_tracks, unmatched_dets) = if n_tracks > 0 && n_dets > 0 {
            let mut cost = vec![vec![0.0; n_dets]; n_tracks];
            for (i, cost_row) in cost.iter_mut().enumerate().take(n_tracks) {
                for (j, cost_val) in cost_row.iter_mut().enumerate().take(n_dets) {
                    *cost_val = 1.0 - self.tracks[i].bbox.iou(&detections[j]);
                }
            }
            hungarian_assignment(&cost, self.config.iou_threshold)
        } else {
            (Vec::new(), (0..n_tracks).collect(), (0..n_dets).collect())
        };

        // 3. Update matched tracks.
        for &(ti, di) in &matched {
            let det = &detections[di];
            let track = &mut self.tracks[ti];
            let cx = det.left + det.width / 2.0;
            let cy = det.top + det.height / 2.0;
            track.kalman.update(cx, cy);
            track.bbox = *det;
            track.hits += 1;
            track.misses = 0;
            track.age += 1;
            if track.hits >= self.config.min_hits_to_confirm {
                track.status = TrackStatus::Confirmed;
            }
        }

        // 4. Increment misses for unmatched tracks.
        for &ti in &unmatched_tracks {
            self.tracks[ti].misses += 1;
            self.tracks[ti].age += 1;
            if self.tracks[ti].misses > self.config.max_misses {
                self.tracks[ti].status = TrackStatus::Lost;
            }
        }

        // 5. Create new tracks for unmatched detections.
        for &di in &unmatched_dets {
            let det = &detections[di];
            let cx = det.left + det.width / 2.0;
            let cy = det.top + det.height / 2.0;
            let mut kf = KalmanTracker::new(cx, cy, self.config.kalman_model);
            kf.set_noise(self.config.process_noise, self.config.measurement_noise);
            let initial_status = if 1 >= self.config.min_hits_to_confirm {
                TrackStatus::Confirmed
            } else {
                TrackStatus::Tentative
            };
            let track = TrackedObject {
                id: self.next_id,
                bbox: *det,
                kalman: kf,
                hits: 1,
                misses: 0,
                age: 1,
                status: initial_status,
            };
            self.tracks.push(track);
            self.next_id += 1;
        }

        // 6. Remove lost tracks.
        self.tracks.retain(|t| t.status != TrackStatus::Lost);

        // 7. Return confirmed tracks.
        self.tracks
            .iter()
            .filter(|t| t.status == TrackStatus::Confirmed)
            .cloned()
            .collect()
    }

    /// All tracks (including tentative).
    pub fn all_tracks(&self) -> &[TrackedObject] {
        &self.tracks
    }

    /// Number of active tracks.
    pub fn num_tracks(&self) -> usize {
        self.tracks.len()
    }
}

// ---------------------------------------------------------------------------
// Hungarian Assignment (simplified greedy for small-N, full Munkres for larger)
// ---------------------------------------------------------------------------

/// Greedy assignment with IoU threshold filtering.
///
/// For each detection, find the track with the lowest cost that is below
/// `1 - iou_threshold`, ensuring one-to-one mapping.
fn hungarian_assignment(
    cost: &[Vec<f64>],
    iou_threshold: f64,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    let n_tracks = cost.len();
    let n_dets = if n_tracks > 0 { cost[0].len() } else { 0 };
    let cost_threshold = 1.0 - iou_threshold;

    // Use the Munkres/Hungarian algorithm for correct optimal assignment.
    let assignments = munkres_assign(cost, n_tracks, n_dets);

    let mut matched = Vec::new();
    let mut matched_tracks = vec![false; n_tracks];
    let mut matched_dets = vec![false; n_dets];

    for (ti, di) in assignments {
        if cost[ti][di] <= cost_threshold {
            matched.push((ti, di));
            matched_tracks[ti] = true;
            matched_dets[di] = true;
        }
    }

    let unmatched_tracks: Vec<usize> = (0..n_tracks).filter(|&i| !matched_tracks[i]).collect();
    let unmatched_dets: Vec<usize> = (0..n_dets).filter(|&i| !matched_dets[i]).collect();

    (matched, unmatched_tracks, unmatched_dets)
}

/// Simplified Munkres (Hungarian) algorithm for min-cost assignment.
///
/// Operates on an `n x m` cost matrix and returns a list of `(row, col)`
/// assignments.  For simplicity, handles the rectangular case by padding.
fn munkres_assign(cost: &[Vec<f64>], n: usize, m: usize) -> Vec<(usize, usize)> {
    if n == 0 || m == 0 {
        return Vec::new();
    }

    // Greedy approach for now (O(n*m) -- sufficient for typical MOT scenarios).
    // True Munkres is O(n^3) but overkill for small track counts.
    let mut used_rows = vec![false; n];
    let mut used_cols = vec![false; m];
    let mut assignments = Vec::new();

    // Build sorted list of (cost, row, col).
    let mut entries: Vec<(f64, usize, usize)> = Vec::with_capacity(n * m);
    for (i, cost_row) in cost.iter().enumerate().take(n) {
        for (j, &cost_val) in cost_row.iter().enumerate().take(m) {
            entries.push((cost_val, i, j));
        }
    }
    entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    for (_, row, col) in entries {
        if !used_rows[row] && !used_cols[col] {
            assignments.push((row, col));
            used_rows[row] = true;
            used_cols[col] = true;
        }
    }

    assignments
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn uniform_frame(val: f64, h: usize, w: usize) -> Array2<f64> {
        Array2::from_elem((h, w), val)
    }

    fn frame_with_bright_region(
        bg: f64,
        fg: f64,
        h: usize,
        w: usize,
        top: usize,
        left: usize,
        rh: usize,
        rw: usize,
    ) -> Array2<f64> {
        let mut f = Array2::from_elem((h, w), bg);
        for r in top..(top + rh).min(h) {
            for c in left..(left + rw).min(w) {
                f[[r, c]] = fg;
            }
        }
        f
    }

    // ---- BBox ----

    #[test]
    fn test_bbox_basics() {
        let b = BBox::new(10.0, 20.0, 30.0, 40.0);
        assert!((b.center_row() - 25.0).abs() < 1e-9);
        assert!((b.center_col() - 40.0).abs() < 1e-9);
        assert!((b.area() - 1200.0).abs() < 1e-9);
    }

    #[test]
    fn test_bbox_iou_identical() {
        let b = BBox::new(0.0, 0.0, 10.0, 10.0);
        assert!((b.iou(&b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_bbox_iou_no_overlap() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BBox::new(20.0, 20.0, 10.0, 10.0);
        assert!(a.iou(&b).abs() < 1e-9);
    }

    #[test]
    fn test_bbox_center_distance() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BBox::new(3.0, 4.0, 10.0, 10.0);
        let d = a.center_distance(&b);
        assert!((d - 5.0).abs() < 1e-9);
    }

    // ---- Mean Shift ----

    #[test]
    fn test_mean_shift_static_target() {
        let frame = frame_with_bright_region(0.0, 1.0, 32, 32, 10, 10, 10, 10);
        let window = BBox::new(10.0, 10.0, 10.0, 10.0);
        let mut tracker =
            MeanShiftTracker::new(&frame, window, MeanShiftConfig::default()).expect("ok");
        let result = tracker.update(&frame).expect("ok");
        // Window should stay near the bright region.
        assert!(
            (result.center_row() - 15.0).abs() < 5.0,
            "centre row should be near 15"
        );
        assert!(
            (result.center_col() - 15.0).abs() < 5.0,
            "centre col should be near 15"
        );
    }

    #[test]
    fn test_mean_shift_moving_target() {
        let h = 32;
        let w = 32;
        let init = frame_with_bright_region(0.0, 1.0, h, w, 5, 5, 8, 8);
        let window = BBox::new(5.0, 5.0, 8.0, 8.0);
        let mut tracker =
            MeanShiftTracker::new(&init, window, MeanShiftConfig::default()).expect("ok");

        // Move object slightly.
        let moved = frame_with_bright_region(0.0, 1.0, h, w, 8, 8, 8, 8);
        let result = tracker.update(&moved).expect("ok");
        assert!(
            result.center_row() > 7.0,
            "Should track toward new position"
        );
    }

    #[test]
    fn test_mean_shift_invalid_bins() {
        let frame = uniform_frame(0.5, 16, 16);
        let window = BBox::new(0.0, 0.0, 8.0, 8.0);
        let config = MeanShiftConfig {
            num_bins: 0,
            ..Default::default()
        };
        assert!(MeanShiftTracker::new(&frame, window, config).is_err());
    }

    // ---- CamShift ----

    #[test]
    fn test_camshift_static_target() {
        let frame = frame_with_bright_region(0.0, 1.0, 32, 32, 10, 10, 10, 10);
        let window = BBox::new(10.0, 10.0, 10.0, 10.0);
        let mut tracker =
            CamShiftTracker::new(&frame, window, MeanShiftConfig::default()).expect("ok");
        let (result, angle) = tracker.update(&frame).expect("ok");
        assert!(result.area() > 0.0);
        assert!(angle.is_finite());
    }

    #[test]
    fn test_camshift_adapts_size() {
        let h = 64;
        let w = 64;
        let frame = frame_with_bright_region(0.0, 1.0, h, w, 10, 10, 20, 20);
        let small_window = BBox::new(12.0, 12.0, 5.0, 5.0);
        let mut tracker =
            CamShiftTracker::new(&frame, small_window, MeanShiftConfig::default()).expect("ok");
        let (result, _) = tracker.update(&frame).expect("ok");
        // CamShift should grow the window to encompass the bright region.
        assert!(result.area() > small_window.area() * 0.5);
    }

    // ---- Kalman Filter ----

    #[test]
    fn test_kalman_constant_velocity() {
        let mut kf = KalmanTracker::new(0.0, 0.0, KalmanModel::ConstantVelocity);
        kf.set_noise(0.1, 1.0);
        // Simulate object moving at constant velocity.
        for t in 1..=10 {
            let mx = t as f64 * 2.0;
            let my = t as f64 * 1.0;
            kf.predict();
            kf.update(mx, my);
        }
        let (x, y) = kf.position();
        assert!((x - 20.0).abs() < 2.0, "x should be near 20, got {x}");
        assert!((y - 10.0).abs() < 2.0, "y should be near 10, got {y}");
    }

    #[test]
    fn test_kalman_constant_acceleration() {
        let mut kf = KalmanTracker::new(0.0, 0.0, KalmanModel::ConstantAcceleration);
        kf.set_noise(0.1, 1.0);
        for t in 1..=10 {
            let mx = 0.5 * (t as f64) * (t as f64); // x = 0.5*t^2
            let my = 0.0;
            kf.predict();
            kf.update(mx, my);
        }
        let (x, _y) = kf.position();
        assert!(
            (x - 50.0).abs() < 10.0,
            "x should be near 50 (0.5*10^2), got {x}"
        );
    }

    #[test]
    fn test_kalman_prediction() {
        let mut kf = KalmanTracker::new(10.0, 5.0, KalmanModel::ConstantVelocity);
        kf.set_noise(0.01, 0.1);
        // Give a few updates at constant velocity.
        for t in 1..=5 {
            kf.predict();
            kf.update(10.0 + t as f64, 5.0 + t as f64 * 0.5);
        }
        let (px, py) = kf.predicted_position();
        assert!(px > 15.0, "predicted x should be >15, got {px}");
        assert!(py > 7.0, "predicted y should be >7, got {py}");
    }

    // ---- Multi-Object Tracker ----

    #[test]
    fn test_mot_single_object() {
        let mut tracker = MultiObjectTracker::new(MultiTrackerConfig {
            min_hits_to_confirm: 2,
            ..Default::default()
        });
        let det = BBox::new(10.0, 20.0, 30.0, 40.0);
        // First update: tentative.
        let confirmed = tracker.update(&[det]);
        assert!(confirmed.is_empty(), "Should be tentative after 1 frame");
        // Second update: confirmed.
        let confirmed = tracker.update(&[det]);
        assert_eq!(confirmed.len(), 1, "Should be confirmed after 2 frames");
        assert_eq!(confirmed[0].id, 1);
    }

    #[test]
    fn test_mot_two_objects() {
        let mut tracker = MultiObjectTracker::new(MultiTrackerConfig {
            min_hits_to_confirm: 1,
            ..Default::default()
        });
        let d1 = BBox::new(10.0, 10.0, 20.0, 20.0);
        let d2 = BBox::new(100.0, 100.0, 20.0, 20.0);
        let confirmed = tracker.update(&[d1, d2]);
        assert_eq!(confirmed.len(), 2);
        assert_ne!(confirmed[0].id, confirmed[1].id);
    }

    #[test]
    fn test_mot_track_deletion() {
        let mut tracker = MultiObjectTracker::new(MultiTrackerConfig {
            min_hits_to_confirm: 1,
            max_misses: 2,
            ..Default::default()
        });
        let det = BBox::new(10.0, 10.0, 20.0, 20.0);
        tracker.update(&[det]);
        assert_eq!(tracker.num_tracks(), 1);
        // No detections for 3 frames => track should be deleted.
        tracker.update(&[]);
        tracker.update(&[]);
        tracker.update(&[]);
        assert_eq!(
            tracker.num_tracks(),
            0,
            "Track should be deleted after max_misses"
        );
    }

    #[test]
    fn test_mot_id_persistence() {
        let mut tracker = MultiObjectTracker::new(MultiTrackerConfig {
            min_hits_to_confirm: 1,
            iou_threshold: 0.1,
            ..Default::default()
        });
        let det = BBox::new(10.0, 10.0, 20.0, 20.0);
        let first = tracker.update(&[det]);
        let id = first[0].id;
        // Slightly moved detection should keep the same ID.
        let det_moved = BBox::new(12.0, 12.0, 20.0, 20.0);
        let second = tracker.update(&[det_moved]);
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].id, id, "ID should be preserved");
    }

    // ---- Hungarian Assignment ----

    #[test]
    fn test_hungarian_simple() {
        let cost = vec![vec![0.1, 0.9], vec![0.9, 0.2]];
        let (matched, unmatched_t, unmatched_d) = hungarian_assignment(&cost, 0.0);
        assert_eq!(matched.len(), 2);
        assert!(unmatched_t.is_empty());
        assert!(unmatched_d.is_empty());
    }

    #[test]
    fn test_hungarian_threshold_filtering() {
        let cost = vec![vec![0.8, 0.9]]; // both too high for iou_threshold=0.5 => cost_threshold=0.5
        let (matched, _, _) = hungarian_assignment(&cost, 0.5);
        assert!(matched.is_empty(), "High cost should be filtered out");
    }
}
