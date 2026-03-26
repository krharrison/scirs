//! Kalman filter for bounding-box tracking.
//!
//! State vector: `[cx, cy, s, r, vx, vy, vs]` where
//! - `cx`, `cy` = centre coordinates
//! - `s` = area (scale)
//! - `r` = aspect ratio (w/h, assumed constant)
//! - `vx`, `vy`, `vs` = velocities of centre and area
//!
//! Measurement vector: `[cx, cy, s, r]` (first four state components).
//!
//! The model uses a **constant-velocity** assumption.

use crate::tracking::types::BoundingBox;

/// Dimension of the state vector.
const STATE_DIM: usize = 7;
/// Dimension of the measurement vector.
const MEAS_DIM: usize = 4;

// ── tiny fixed-size matrix helpers ──────────────────────────────────────────

/// Column-major 7×7 matrix stored as a flat `[f32; 49]` array.
type Mat7 = [f32; 49];
/// Column-major 4×7 matrix.
type Mat4x7 = [f32; 28];
/// Column-major 7×4 matrix.
type Mat7x4 = [f32; 28];
/// 4×4 matrix.
type Mat4 = [f32; 16];

/// Zero 7×7 matrix.
#[inline]
fn zero7() -> Mat7 {
    [0.0; 49]
}

/// Identity 7×7 matrix.
#[inline]
fn eye7() -> Mat7 {
    let mut m = zero7();
    for i in 0..STATE_DIM {
        m[i * STATE_DIM + i] = 1.0;
    }
    m
}

/// Zero 4×4 matrix.
#[inline]
fn zero4() -> Mat4 {
    [0.0; 16]
}

/// Multiply A (r×n) by B (n×c), result (r×c).
fn mat_mul_general(a: &[f32], r: usize, n: usize, b: &[f32], c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += a[i * n + k] * b[k * c + j];
            }
            out[i * c + j] = sum;
        }
    }
    out
}

/// Transpose a matrix (r×c) → (c×r).
fn transpose(a: &[f32], r: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = a[i * c + j];
        }
    }
    out
}

/// Invert a 4×4 matrix using Gauss-Jordan elimination.
/// Returns `None` when the matrix is singular.
fn invert4(m: &Mat4) -> Option<Mat4> {
    let mut aug = [0.0f32; 32]; // 4 × 8 augmented matrix
    for i in 0..4 {
        for j in 0..4 {
            aug[i * 8 + j] = m[i * 4 + j];
        }
        aug[i * 8 + 4 + i] = 1.0; // identity on right half
    }

    for col in 0..4 {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col * 8 + col].abs();
        for row in (col + 1)..4 {
            let v = aug[row * 8 + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        // Swap rows
        if max_row != col {
            for j in 0..8 {
                aug.swap(col * 8 + j, max_row * 8 + j);
            }
        }
        // Scale pivot row
        let pivot = aug[col * 8 + col];
        for j in 0..8 {
            aug[col * 8 + j] /= pivot;
        }
        // Eliminate column
        for row in 0..4 {
            if row == col {
                continue;
            }
            let factor = aug[row * 8 + col];
            for j in 0..8 {
                let val = aug[col * 8 + j];
                aug[row * 8 + j] -= factor * val;
            }
        }
    }

    let mut inv = [0.0f32; 16];
    for i in 0..4 {
        for j in 0..4 {
            inv[i * 4 + j] = aug[i * 8 + 4 + j];
        }
    }
    Some(inv)
}

// ── KalmanBoxTracker ─────────────────────────────────────────────────────────

/// Kalman filter tracker that wraps a single bounding-box trajectory.
///
/// State:  `x = [cx, cy, s, r, vx, vy, vs]`
/// Measure: `z = [cx, cy, s, r]`
pub struct KalmanBoxTracker {
    /// State estimate.
    x: [f32; STATE_DIM],
    /// State covariance.
    p: Mat7,
    /// State-transition matrix (constant-velocity).
    f: Mat7,
    /// Measurement matrix (H).
    h: Mat4x7,
    /// Process noise covariance.
    q: Mat7,
    /// Measurement noise covariance.
    r: Mat4,
    /// Number of times `predict` has been called without an `update`.
    pub time_since_update: usize,
    /// Total number of `predict` calls.
    pub count: usize,
}

impl KalmanBoxTracker {
    /// Create a new tracker initialised at `bbox`.
    pub fn new(bbox: &BoundingBox) -> Self {
        let (cx, cy, w, h) = bbox.to_xywh();
        let s = (w * h).max(1.0);
        let r = if h > 0.0 { w / h } else { 1.0 };

        let mut x = [0.0f32; STATE_DIM];
        x[0] = cx;
        x[1] = cy;
        x[2] = s;
        x[3] = r;
        // velocities start at zero

        // State-transition: [I | dt*I] for position/scale; dt=1
        let mut f = eye7();
        // cx += vx
        f[4] = 1.0; // cx += vx  (row 0, col 4)
                    // cy += vy
        f[STATE_DIM + 5] = 1.0;
        // s  += vs
        f[2 * STATE_DIM + 6] = 1.0;

        // Measurement matrix: extracts first 4 components
        let mut hm = [0.0f32; MEAS_DIM * STATE_DIM];
        for i in 0..MEAS_DIM {
            hm[i * STATE_DIM + i] = 1.0;
        }

        // Process noise Q (small)
        let mut q = zero7();
        let qdiag: [f32; STATE_DIM] = [1.0, 1.0, 10.0, 0.01, 0.01, 0.01, 0.0001];
        for i in 0..STATE_DIM {
            q[i * STATE_DIM + i] = qdiag[i];
        }

        // Measurement noise R
        let mut rm = zero4();
        let rdiag: [f32; MEAS_DIM] = [1.0, 1.0, 10.0, 0.01];
        for i in 0..MEAS_DIM {
            rm[i * MEAS_DIM + i] = rdiag[i];
        }

        // Initial covariance: large uncertainty in velocity
        let mut p = eye7();
        let pdiag: [f32; STATE_DIM] = [10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4];
        for i in 0..STATE_DIM {
            p[i * STATE_DIM + i] = pdiag[i];
        }

        Self {
            x,
            p,
            f,
            h: hm,
            q,
            r: rm,
            time_since_update: 0,
            count: 0,
        }
    }

    // -- Kalman predict step -------------------------------------------------

    /// Advance the state by one time step and return the predicted bounding box.
    pub fn predict(&mut self) -> BoundingBox {
        // Clip area to avoid negative
        if self.x[2] + self.x[6] < 0.0 {
            self.x[6] = 0.0;
        }

        // x = F * x
        let fx = mat_mul_general(&self.f, STATE_DIM, STATE_DIM, &self.x, 1);
        self.x.copy_from_slice(&fx);

        // P = F * P * F' + Q
        let fp = mat_mul_general(&self.f, STATE_DIM, STATE_DIM, &self.p, STATE_DIM);
        let ft = transpose(&self.f, STATE_DIM, STATE_DIM);
        let fpft = mat_mul_general(&fp, STATE_DIM, STATE_DIM, &ft, STATE_DIM);
        for (i, &fpft_val) in fpft.iter().enumerate().take(STATE_DIM * STATE_DIM) {
            self.p[i] = fpft_val + self.q[i];
        }

        self.time_since_update += 1;
        self.count += 1;
        self.get_state()
    }

    // -- Kalman update step --------------------------------------------------

    /// Update the filter with a new measurement (matched detection).
    pub fn update(&mut self, bbox: &BoundingBox) {
        let (cx, cy, w, h) = bbox.to_xywh();
        let s = (w * h).max(1.0);
        let r = if h > 0.0 { w / h } else { 1.0 };
        let z = [cx, cy, s, r];

        // y = z - H * x  (innovation)
        let hx = mat_mul_general(&self.h, MEAS_DIM, STATE_DIM, &self.x, 1);
        let y: Vec<f32> = (0..MEAS_DIM).map(|i| z[i] - hx[i]).collect();

        // S = H * P * H' + R
        let hp = mat_mul_general(&self.h, MEAS_DIM, STATE_DIM, &self.p, STATE_DIM);
        let ht: Mat7x4 = {
            let v = transpose(&self.h, MEAS_DIM, STATE_DIM);
            let mut arr = [0.0f32; STATE_DIM * MEAS_DIM];
            arr.copy_from_slice(&v);
            arr
        };
        let hpht_raw = mat_mul_general(&hp, MEAS_DIM, STATE_DIM, &ht, MEAS_DIM);
        let mut s_mat: Mat4 = [0.0f32; 16];
        for i in 0..16 {
            s_mat[i] = hpht_raw[i] + self.r[i];
        }

        // K = P * H' * S⁻¹
        let ph = mat_mul_general(&self.p, STATE_DIM, STATE_DIM, &ht, MEAS_DIM);

        let s_inv = match invert4(&s_mat) {
            Some(inv) => inv,
            None => {
                // Degenerate case: skip update
                self.time_since_update = 0;
                return;
            }
        };
        let k_raw = mat_mul_general(&ph, STATE_DIM, MEAS_DIM, &s_inv, MEAS_DIM);

        // x = x + K * y
        let ky = mat_mul_general(&k_raw, STATE_DIM, MEAS_DIM, &y, 1);
        for (i, &ky_val) in ky.iter().enumerate().take(STATE_DIM) {
            self.x[i] += ky_val;
        }

        // P = (I - K*H) * P
        let kh = mat_mul_general(&k_raw, STATE_DIM, MEAS_DIM, &self.h, STATE_DIM);
        let mut i_minus_kh = eye7();
        for i in 0..(STATE_DIM * STATE_DIM) {
            i_minus_kh[i] -= kh[i];
        }
        let new_p = mat_mul_general(&i_minus_kh, STATE_DIM, STATE_DIM, &self.p, STATE_DIM);
        self.p.copy_from_slice(&new_p);

        self.time_since_update = 0;
    }

    // -- State accessor -------------------------------------------------------

    /// Return the current state as a `BoundingBox`.
    pub fn get_state(&self) -> BoundingBox {
        let cx = self.x[0];
        let cy = self.x[1];
        let s = self.x[2].max(0.0);
        let r = self.x[3].max(1e-4);
        // s = w * h,  r = w / h  →  h = sqrt(s/r),  w = sqrt(s*r)
        let w = (s * r).sqrt();
        let h = if r > 0.0 { (s / r).sqrt() } else { 1.0 };
        BoundingBox::new(
            cx - w * 0.5,
            cy - h * 0.5,
            cx + w * 0.5,
            cy + h * 0.5,
            0.0,
            None,
        )
    }
}
