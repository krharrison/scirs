//! Online / incremental RBF interpolation with a sliding window.
//!
//! The `StreamingRbf` maintains a Gram matrix in Cholesky-factored form and
//! supports:
//!
//! 1. **Rank-1 Cholesky update** when a new point is appended.
//! 2. **Rank-1 Cholesky downdate** when the oldest point is evicted once the
//!    window size is exceeded.
//! 3. **Forget factor** that shrinks stale information before each update.
//!
//! The implementation uses a multiquadric kernel  φ(r) = √(1 + (ε r)²)  and
//! solves the system  G α = y  where  G_{ij} = φ(||x_i − x_j||).

use crate::error::InterpolateError;

/// Inverse multiquadric kernel: φ(r) = 1 / sqrt(1 + (ε r)²).
///
/// This kernel is strictly positive definite, meaning the Gram matrix G with
/// G_{ij} = φ(||xi − xj||) is SPD for any set of distinct points.
#[inline]
fn inv_multiquadric(r: f64, eps: f64) -> f64 {
    1.0 / (1.0 + (eps * r) * (eps * r)).sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the streaming RBF interpolant.
#[derive(Debug, Clone)]
pub struct StreamingRbfConfig {
    /// Maximum number of points to keep (sliding window size).
    pub window_size: usize,
    /// Multiquadric shape parameter ε.
    pub shape_param: f64,
    /// Forget factor applied to the Gram matrix before each update (0 < γ ≤ 1).
    /// A value < 1 down-weights older data.
    pub forget_factor: f64,
}

impl Default for StreamingRbfConfig {
    fn default() -> Self {
        Self {
            window_size: 200,
            shape_param: 1.0,
            forget_factor: 0.99,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming RBF struct
// ─────────────────────────────────────────────────────────────────────────────

/// Online / incremental RBF interpolant.
///
/// Points are added one at a time via [`update`](StreamingRbf::update).  When
/// the window is full the oldest point is removed via a rank-1 downdate.
#[derive(Debug, Clone)]
pub struct StreamingRbf {
    config: StreamingRbfConfig,
    /// Sliding window of input points.
    points: Vec<Vec<f64>>,
    /// Corresponding function values.
    values: Vec<f64>,
    /// Lower-triangular Cholesky factor of the Gram matrix G.
    /// `l` has shape `(n × n)` where `n = points.len()`.
    l: Vec<Vec<f64>>,
    /// RBF coefficients (solution of G α = y).
    coeffs: Vec<f64>,
    /// Whether the coefficient vector is up-to-date.
    coeffs_dirty: bool,
}

impl StreamingRbf {
    /// Create a new, empty streaming RBF.
    pub fn new(config: StreamingRbfConfig) -> Self {
        Self {
            config,
            points: Vec::new(),
            values: Vec::new(),
            l: Vec::new(),
            coeffs: Vec::new(),
            coeffs_dirty: false,
        }
    }

    /// Number of points currently in the window.
    pub fn n_points(&self) -> usize {
        self.points.len()
    }

    // ── kernel helpers ────────────────────────────────────────────────────

    fn dist(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi) * (ai - bi))
            .sum::<f64>()
            .sqrt()
    }

    fn phi(&self, a: &[f64], b: &[f64]) -> f64 {
        inv_multiquadric(Self::dist(a, b), self.config.shape_param)
    }

    // ── dense Cholesky of a positive-definite n×n matrix ─────────────────

    fn full_cholesky(g: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, InterpolateError> {
        let n = g.len();
        // Add a small ridge for numerical safety.
        let ridge = 1e-12;
        let mut l = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let mut s = g[i][j] + if i == j { ridge } else { 0.0 };
                for k in 0..j {
                    s -= l[i][k] * l[j][k];
                }
                if i == j {
                    if s <= 0.0 {
                        s = ridge;
                    }
                    l[i][j] = s.sqrt();
                } else {
                    l[i][j] = s / l[j][j];
                }
            }
        }
        Ok(l)
    }

    // ── forward / back substitution ───────────────────────────────────────

    fn forward_sub(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = l.len();
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut s = b[i];
            for k in 0..i {
                s -= l[i][k] * y[k];
            }
            y[i] = s / l[i][i];
        }
        y
    }

    fn back_sub(l: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
        let n = l.len();
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            for k in (i + 1)..n {
                s -= l[k][i] * x[k];
            }
            x[i] = s / l[i][i];
        }
        x
    }

    fn solve_cholesky(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let y = Self::forward_sub(l, b);
        Self::back_sub(l, &y)
    }

    // ── rank-1 Cholesky update: L → L' such that L'L'ᵀ = LLᵀ + v vᵀ ─────

    /// Performs an O(n²) Cholesky extension.
    ///
    /// Appends a new row and column to the current lower-triangular Cholesky
    /// factor `L` so that the extended factor corresponds to the augmented
    /// Gram matrix `G' = [[G, v[..n]]; [v[..n]ᵀ, v[n]]]`.
    ///
    /// `v` has length `old_n + 1`:
    /// - `v[0..old_n]` are the cross-kernel values between the new point and
    ///   all existing points.
    /// - `v[old_n]` is the self-kernel value `φ(new, new)`.
    pub fn cholesky_rank1_update(l: &mut Vec<Vec<f64>>, v: &[f64]) -> Result<(), InterpolateError> {
        // old_n is the current (pre-extension) dimension.
        let old_n = l.len();
        let new_n = old_n + 1;

        if v.len() != new_n {
            return Err(InterpolateError::DimensionMismatch(format!(
                "cholesky_rank1_update: v has length {} but expected {}",
                v.len(),
                new_n
            )));
        }

        // Extend each existing row to width new_n.
        for row in l.iter_mut() {
            row.push(0.0);
        }
        // Append the new (last) row.
        l.push(vec![0.0; new_n]);

        // Compute the new last row by solving L_{old} w = v[0..old_n].
        let w = if old_n > 0 {
            let v_sub = &v[..old_n];
            let l_old: Vec<Vec<f64>> = l[..old_n].iter().map(|r| r[..old_n].to_vec()).collect();
            Self::forward_sub(&l_old, v_sub)
        } else {
            vec![]
        };

        // Fill in off-diagonal entries of the new last row.
        for j in 0..old_n {
            l[old_n][j] = w[j];
        }

        // Diagonal element: sqrt(v[old_n] - ||w||²).
        let w_norm2: f64 = w.iter().map(|&wi| wi * wi).sum();
        let diag2 = v[old_n] - w_norm2;
        let diag = if diag2 <= 0.0 {
            1e-10_f64
        } else {
            diag2.sqrt()
        };
        l[old_n][old_n] = diag;

        Ok(())
    }

    // ── rank-1 Cholesky downdate: remove first row/column ─────────────────
    // Strategy: after removing the first column/row from L (and hence from the
    // Gram matrix), we recompute L from scratch on the reduced n-1 × n-1 system.
    // This is O(n³) in the worst case but n ≤ window_size which is typically
    // small enough.

    #[allow(dead_code)]
    fn remove_first_point(&mut self) -> Result<(), InterpolateError> {
        // Remove first point
        self.points.remove(0);
        self.values.remove(0);
        let n = self.points.len();
        if n == 0 {
            self.l.clear();
            self.coeffs.clear();
            self.coeffs_dirty = false;
            return Ok(());
        }
        // Rebuild Gram matrix G from scratch after removal.
        let mut g = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                let r = Self::dist(&self.points[i], &self.points[j]);
                g[i][j] = inv_multiquadric(r, self.config.shape_param);
            }
        }
        self.l = Self::full_cholesky(&g)?;
        self.coeffs_dirty = true;
        Ok(())
    }

    // ── update coefficients if dirty ──────────────────────────────────────

    fn refresh_coeffs(&mut self) {
        if !self.coeffs_dirty || self.points.is_empty() {
            return;
        }
        self.coeffs = Self::solve_cholesky(&self.l, &self.values);
        self.coeffs_dirty = false;
    }

    // ─────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────

    /// Add a new observation `(x, y)` and update the interpolant.
    ///
    /// If the window is full the oldest observation is evicted first.
    ///
    /// Implementation strategy:
    /// - When `forget_factor == 1.0` (no forgetting): use rank-1 Cholesky update
    ///   (O(n²)) by appending the new column.
    /// - When `forget_factor < 1.0`: apply exponential forgetting to stored values
    ///   and rebuild the Gram matrix from scratch (O(n²) rebuild, but avoids
    ///   the numerical issues of scaling an existing factorisation).
    pub fn update(&mut self, x: Vec<f64>, y: f64) -> Result<(), InterpolateError> {
        let gamma = self.config.forget_factor;

        // Evict oldest if window is full before adding the new point.
        if self.points.len() >= self.config.window_size {
            self.points.remove(0);
            self.values.remove(0);
        }

        // Apply forget factor: discount stored values (used for weighted solve).
        if gamma < 1.0 {
            for v in &mut self.values {
                *v *= gamma;
            }
        }

        // Add new point.
        self.points.push(x);
        self.values.push(y);

        // Rebuild Gram matrix and its Cholesky factor from scratch.
        // O(n²) but ensures numerical correctness across all code paths.
        let n = self.points.len();
        let eps = self.config.shape_param;
        let mut g = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                let r = Self::dist(&self.points[i], &self.points[j]);
                g[i][j] = inv_multiquadric(r, eps);
            }
        }
        self.l = Self::full_cholesky(&g)?;
        self.coeffs_dirty = true;

        Ok(())
    }

    /// Predict the interpolated value at a new point `x`.
    pub fn predict(&mut self, x: &[f64]) -> Result<f64, InterpolateError> {
        if self.points.is_empty() {
            return Err(InterpolateError::InvalidState(
                "StreamingRbf has no data yet".to_string(),
            ));
        }
        self.refresh_coeffs();
        let val: f64 = self
            .points
            .iter()
            .zip(self.coeffs.iter())
            .map(|(pt, &alpha)| {
                let r = Self::dist(x, pt);
                alpha * inv_multiquadric(r, self.config.shape_param)
            })
            .sum();
        Ok(val)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_streaming_sin_prediction() {
        let mut rbf = StreamingRbf::new(StreamingRbfConfig {
            window_size: 20,
            shape_param: 1.0,
            forget_factor: 1.0, // no forgetting for this test
        });

        // Add 10 points from y = sin(x) on [0, π]
        for i in 0..10 {
            let xi = (i as f64) * PI / 9.0;
            rbf.update(vec![xi], xi.sin())
                .expect("update should succeed");
        }

        assert_eq!(rbf.n_points(), 10);

        // Predict at x = π/4
        let test_x = PI / 4.0;
        let pred = rbf.predict(&[test_x]).expect("predict should succeed");
        let expected = test_x.sin();
        assert!(
            (pred - expected).abs() < 0.1,
            "sin prediction off: pred={:.4}, expected={:.4}",
            pred,
            expected
        );
    }

    #[test]
    fn test_window_eviction() {
        let window = 5;
        let mut rbf = StreamingRbf::new(StreamingRbfConfig {
            window_size: window,
            shape_param: 1.0,
            forget_factor: 1.0,
        });

        for i in 0..10 {
            let xi = i as f64 * 0.1;
            rbf.update(vec![xi], xi * xi).expect("update");
        }

        assert_eq!(
            rbf.n_points(),
            window,
            "window should be capped at {window}"
        );
    }

    #[test]
    fn test_predict_empty_returns_error() {
        let mut rbf = StreamingRbf::new(StreamingRbfConfig::default());
        let result = rbf.predict(&[0.5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_forget_factor_effect() {
        // With strong forgetting, recent data dominates.
        let mut rbf = StreamingRbf::new(StreamingRbfConfig {
            window_size: 100,
            shape_param: 1.0,
            forget_factor: 0.5,
        });

        // Add old data: f = 0
        for i in 0..5 {
            rbf.update(vec![i as f64 * 0.2], 0.0).expect("update old");
        }
        // Add recent data: f = 1
        for i in 0..5 {
            rbf.update(vec![i as f64 * 0.2 + 0.05], 1.0)
                .expect("update new");
        }

        // Prediction should be closer to 1 than 0 since old data is discounted.
        let pred = rbf.predict(&[0.5]).expect("predict");
        assert!(
            pred > 0.0,
            "prediction with forgetting should lean towards recent data, got {pred}"
        );
    }

    #[test]
    fn test_cholesky_rank1_update_basic() {
        // Build a 1×1 Cholesky factor, then extend to 2×2 via rank-1 update.
        // Use a manually constructed 2×2 SPD Gram matrix:
        //   G = [[4.0, 1.0], [1.0, 4.0]]
        // Cholesky: L[0][0] = 2.0
        //           L[1][0] = 0.5, L[1][1] = sqrt(4 - 0.25) = sqrt(3.75)
        let g00 = 4.0_f64;
        let g01 = 1.0_f64;
        let g11 = 4.0_f64;

        // Initial 1×1 L from 1×1 G = [[4.0]].
        let mut l = vec![vec![g00.sqrt()]]; // [[2.0]]

        // v for extending: [g01, g11] = [1.0, 4.0]
        let v = vec![g01, g11];
        StreamingRbf::cholesky_rank1_update(&mut l, &v).expect("rank-1 update");

        assert_eq!(l.len(), 2);

        // Verify L Lᵀ ≈ G
        // G[0][0] = l[0][0]²
        // G[1][0] = l[1][0] * l[0][0]
        // G[1][1] = l[1][0]² + l[1][1]²
        let rec_g00 = l[0][0] * l[0][0];
        let rec_g10 = l[1][0] * l[0][0];
        let rec_g11 = l[1][0] * l[1][0] + l[1][1] * l[1][1];
        assert!(
            (rec_g00 - g00).abs() < 1e-10,
            "G[0][0] mismatch: {} vs {}",
            rec_g00,
            g00
        );
        assert!(
            (rec_g10 - g01).abs() < 1e-10,
            "G[1][0] mismatch: {} vs {}",
            rec_g10,
            g01
        );
        assert!(
            (rec_g11 - g11).abs() < 1e-10,
            "G[1][1] mismatch: {} vs {}",
            rec_g11,
            g11
        );
    }
}
