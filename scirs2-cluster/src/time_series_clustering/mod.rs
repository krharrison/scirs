//! Time-series-specific clustering algorithms.
//!
//! Provides algorithms that exploit temporal structure in data:
//!
//! - [`DTWDistance`]: Dynamic Time Warping with Sakoe-Chiba band.
//! - [`KMeansDTW`]: K-Means clustering with DTW distance and DBA centroids.
//! - [`DBABarycenter`]: DTW Barycenter Averaging (iterative).
//! - [`ShapeBasedDistance`]: Normalised cross-correlation (SBD) distance.
//!
//! # References
//!
//! - Sakoe, H. & Chiba, S. (1978). "Dynamic programming algorithm optimization
//!   for spoken word recognition." *IEEE Trans. Acoust.*, 26(1):43-49.
//! - Petitjean, F., Ketterlin, A., & Gançarski, P. (2011).
//!   "A global averaging method for dynamic time warping, with applications to
//!   clustering." *Pattern Recognition*, 44(3):678-693. (DBA)
//! - Paparrizos, J. & Gravano, L. (2015).
//!   "k-Shape: Efficient and Accurate Clustering of Time Series."
//!   *SIGMOD 2015*. (SBD)
//!
//! # Example
//!
//! ```rust
//! use scirs2_cluster::time_series_clustering::{DTWDistance, KMeansDTW};
//!
//! let s1 = vec![1.0_f64, 2.0, 3.0, 2.0, 1.0];
//! let s2 = vec![1.0_f64, 2.0, 2.0, 3.0, 2.0, 1.0];
//!
//! let d = DTWDistance::compute(&s1, &s2, None).expect("dtw ok");
//! assert!(d >= 0.0);
//!
//! let series = vec![s1.clone(), s2.clone(),
//!     vec![8.0, 8.0, 9.0, 8.0], vec![8.0, 9.0, 9.0, 8.0]];
//! let kmeans = KMeansDTW { k: 2, max_iter: 10, window: None };
//! let labels = kmeans.fit(&series).expect("kmeans_dtw ok");
//! assert_eq!(labels.len(), 4);
//! ```

use std::f64;

use crate::error::{ClusteringError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// DTW Distance
// ─────────────────────────────────────────────────────────────────────────────

/// Dynamic Time Warping (DTW) distance computer.
pub struct DTWDistance;

impl DTWDistance {
    /// Compute the DTW distance between two time series.
    ///
    /// Uses the classic DP formulation with an optional Sakoe-Chiba
    /// band constraint to limit the warping path width.
    ///
    /// # Parameters
    ///
    /// - `a`: first series.
    /// - `b`: second series.
    /// - `window`: Sakoe-Chiba band half-width (`None` = unconstrained).
    ///
    /// # Returns
    ///
    /// DTW distance ≥ 0.
    pub fn compute(a: &[f64], b: &[f64], window: Option<usize>) -> Result<f64> {
        let n = a.len();
        let m = b.len();
        if n == 0 || m == 0 {
            return Err(ClusteringError::InvalidInput(
                "Time series must be non-empty".to_string(),
            ));
        }

        // Flatten into a (n+1) × (m+1) cost matrix, initialised to ∞.
        let stride = m + 1;
        let mut dtw = vec![f64::INFINITY; (n + 1) * stride];
        dtw[0] = 0.0;

        let effective_window = window.unwrap_or(m.max(n) + 1);

        for i in 1..=n {
            let w_start = if effective_window + 1 <= i {
                i - effective_window
            } else {
                1
            };
            let w_end = (i + effective_window).min(m);

            for j in w_start..=w_end {
                let cost = (a[i - 1] - b[j - 1]).abs();
                let prev = [
                    dtw[(i - 1) * stride + j],     // deletion
                    dtw[i * stride + (j - 1)],     // insertion
                    dtw[(i - 1) * stride + (j - 1)], // match
                ]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
                dtw[i * stride + j] = cost + prev;
            }
        }

        let result = dtw[n * stride + m];
        if result.is_infinite() {
            Err(ClusteringError::ComputationError(
                "DTW path not found (window too small)".to_string(),
            ))
        } else {
            Ok(result)
        }
    }

    /// Compute a pairwise DTW distance matrix for a collection of series.
    pub fn pairwise(series: &[Vec<f64>], window: Option<usize>) -> Result<Vec<Vec<f64>>> {
        let n = series.len();
        let mut matrix = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = Self::compute(&series[i], &series[j], window)?;
                matrix[i][j] = d;
                matrix[j][i] = d;
            }
        }
        Ok(matrix)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DBA Barycenter
// ─────────────────────────────────────────────────────────────────────────────

/// DTW Barycenter Averaging (DBA) — compute a representative centroid.
pub struct DBABarycenter;

impl DBABarycenter {
    /// Compute the DBA centroid of a set of time series.
    ///
    /// Iteratively:
    /// 1. Align each series to the current centroid via DTW.
    /// 2. Average aligned associations at each centroid position.
    ///
    /// # Parameters
    ///
    /// - `series`: collection of time series (may have different lengths).
    /// - `weights`: per-series weight (must be length = `series.len()`).
    ///   If empty, uniform weights are used.
    /// - `n_iter`: number of DBA iterations (default 10).
    /// - `window`: DTW window constraint.
    ///
    /// # Returns
    ///
    /// The DBA centroid as a `Vec<f64>`.  The centroid length is taken from
    /// the first series (medoid initialisation).
    pub fn compute(
        series: &[Vec<f64>],
        weights: &[f64],
        n_iter: usize,
        window: Option<usize>,
    ) -> Result<Vec<f64>> {
        if series.is_empty() {
            return Err(ClusteringError::InvalidInput(
                "Series set must be non-empty".to_string(),
            ));
        }

        let use_weights: Vec<f64> = if weights.is_empty() {
            vec![1.0; series.len()]
        } else {
            if weights.len() != series.len() {
                return Err(ClusteringError::InvalidInput(
                    "weights length must equal series count".to_string(),
                ));
            }
            weights.to_vec()
        };

        // Initialise centroid as the first series.
        let centroid_len = series[0].len();
        if centroid_len == 0 {
            return Err(ClusteringError::InvalidInput(
                "Series must be non-empty".to_string(),
            ));
        }

        let mut centroid: Vec<f64> = series[0].clone();

        let n_iterations = if n_iter == 0 { 10 } else { n_iter };

        for _ in 0..n_iterations {
            let mut assoc_sum = vec![0.0f64; centroid_len];
            let mut assoc_count = vec![0.0f64; centroid_len];

            for (s_idx, s) in series.iter().enumerate() {
                if s.is_empty() {
                    continue;
                }
                let w = use_weights[s_idx];

                // Get DTW alignment path (warp path).
                let path = dtw_path(&centroid, s, window)?;

                // Accumulate associations.
                for &(ci, si) in &path {
                    assoc_sum[ci] += w * s[si];
                    assoc_count[ci] += w;
                }
            }

            // Update centroid.
            for t in 0..centroid_len {
                if assoc_count[t] > 1e-15 {
                    centroid[t] = assoc_sum[t] / assoc_count[t];
                }
            }
        }

        Ok(centroid)
    }
}

/// Compute the DTW alignment path between `a` (centroid) and `b` (series).
///
/// Returns a list of `(i, j)` pairs tracing the optimal warp path
/// from `(0, 0)` to `(n-1, m-1)`.
fn dtw_path(a: &[f64], b: &[f64], window: Option<usize>) -> Result<Vec<(usize, usize)>> {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return Ok(Vec::new());
    }

    let stride = m + 1;
    let mut dtw = vec![f64::INFINITY; (n + 1) * stride];
    dtw[0] = 0.0;

    let effective_window = window.unwrap_or(m.max(n) + 1);

    for i in 1..=n {
        let w_start = if effective_window + 1 <= i {
            i - effective_window
        } else {
            1
        };
        let w_end = (i + effective_window).min(m);
        for j in w_start..=w_end {
            let cost = (a[i - 1] - b[j - 1]).abs();
            let prev = [
                dtw[(i - 1) * stride + j],
                dtw[i * stride + (j - 1)],
                dtw[(i - 1) * stride + (j - 1)],
            ]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
            dtw[i * stride + j] = cost + prev;
        }
    }

    // Traceback.
    let mut path = Vec::new();
    let mut i = n;
    let mut j = m;
    while i > 0 && j > 0 {
        path.push((i - 1, j - 1));
        let up = dtw[(i - 1) * stride + j];
        let left = dtw[i * stride + (j - 1)];
        let diag = dtw[(i - 1) * stride + (j - 1)];

        if diag <= up && diag <= left {
            i -= 1;
            j -= 1;
        } else if up <= left {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    // Handle remaining.
    while i > 0 {
        path.push((i - 1, 0));
        i -= 1;
    }
    while j > 0 {
        path.push((0, j - 1));
        j -= 1;
    }
    path.reverse();
    Ok(path)
}

// ─────────────────────────────────────────────────────────────────────────────
// K-Means DTW
// ─────────────────────────────────────────────────────────────────────────────

/// K-Means clustering using DTW distance and DBA centroids.
#[derive(Debug, Clone)]
pub struct KMeansDTW {
    /// Number of clusters K.
    pub k: usize,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// DTW Sakoe-Chiba window (None = unconstrained).
    pub window: Option<usize>,
}

impl KMeansDTW {
    /// Fit K-Means (DTW) to a collection of time series.
    ///
    /// Uses K-Means++ initialisation for centroids, then iterates:
    /// 1. Assign each series to the nearest centroid (DTW distance).
    /// 2. Update centroids with DBA.
    ///
    /// # Returns
    ///
    /// Cluster label vector of length `series.len()`.
    pub fn fit(&self, series: &[Vec<f64>]) -> Result<Vec<usize>> {
        let n = series.len();
        if n == 0 {
            return Err(ClusteringError::InvalidInput(
                "Series set must be non-empty".to_string(),
            ));
        }
        if self.k == 0 {
            return Err(ClusteringError::InvalidInput(
                "k must be > 0".to_string(),
            ));
        }
        if self.k > n {
            return Err(ClusteringError::InvalidInput(format!(
                "k ({}) must be <= number of series ({})",
                self.k, n
            )));
        }

        // K-Means++ initialisation.
        let mut centroids = self.kmeans_plus_plus_init(series)?;

        let mut labels = vec![0usize; n];

        for _iter in 0..self.max_iter {
            // Assignment step.
            let new_labels = self.assign(series, &centroids)?;

            let changed = new_labels.iter().zip(labels.iter()).any(|(a, b)| a != b);
            labels = new_labels;

            if !changed {
                break;
            }

            // Update step: DBA centroids.
            for k_idx in 0..self.k {
                let members: Vec<Vec<f64>> = series
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| labels[*i] == k_idx)
                    .map(|(_, s)| s.clone())
                    .collect();

                if members.is_empty() {
                    // Empty cluster: keep old centroid.
                    continue;
                }

                centroids[k_idx] =
                    DBABarycenter::compute(&members, &[], 5, self.window)?;
            }
        }

        Ok(labels)
    }

    /// Assign each series to the nearest centroid.
    fn assign(&self, series: &[Vec<f64>], centroids: &[Vec<f64>]) -> Result<Vec<usize>> {
        let mut labels = Vec::with_capacity(series.len());
        for s in series {
            let mut best_k = 0;
            let mut best_d = f64::INFINITY;
            for (k_idx, c) in centroids.iter().enumerate() {
                let d = DTWDistance::compute(s, c, self.window)?;
                if d < best_d {
                    best_d = d;
                    best_k = k_idx;
                }
            }
            labels.push(best_k);
        }
        Ok(labels)
    }

    /// K-Means++ initialisation: select k seeds, biased towards distant points.
    fn kmeans_plus_plus_init(&self, series: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n = series.len();
        // Deterministic seed selection: first centroid = index 0.
        let mut centroids: Vec<Vec<f64>> = vec![series[0].clone()];
        let mut dists: Vec<f64> = vec![f64::INFINITY; n];

        for _ in 1..self.k {
            // Update min distances to current centroid set.
            let last_c = centroids.last().ok_or_else(|| {
                ClusteringError::ComputationError("No centroids".to_string())
            })?;
            for (i, s) in series.iter().enumerate() {
                let d = DTWDistance::compute(s, last_c, self.window)?;
                if d < dists[i] {
                    dists[i] = d;
                }
            }

            // Pick next centroid: max distance (greedy, deterministic).
            let next_idx = dists
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .ok_or_else(|| ClusteringError::ComputationError("Empty series".to_string()))?;
            centroids.push(series[next_idx].clone());
        }

        Ok(centroids)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shape-Based Distance (SBD)
// ─────────────────────────────────────────────────────────────────────────────

/// Shape-Based Distance (SBD): shift-invariant similarity via normalised
/// cross-correlation.
///
/// SBD(a, b) = 1 - max_s NCC_c(a, b, s)
///
/// where NCC_c is the coefficient-normalised cross-correlation.
pub struct ShapeBasedDistance;

impl ShapeBasedDistance {
    /// Compute the SBD between two time series.
    ///
    /// Both series are first z-normalised to make the metric amplitude-invariant.
    ///
    /// # Returns
    ///
    /// SBD ∈ [0, 2] (0 = identical shape).
    pub fn compute(a: &[f64], b: &[f64]) -> Result<f64> {
        if a.is_empty() || b.is_empty() {
            return Err(ClusteringError::InvalidInput(
                "Series must be non-empty".to_string(),
            ));
        }

        // Z-normalise.
        let a_norm = z_normalise(a);
        let b_norm = z_normalise(b);

        // Compute normalised cross-correlation over all shifts.
        let max_ncc = normalised_cross_correlation_max(&a_norm, &b_norm);

        Ok(1.0 - max_ncc)
    }

    /// Compute pairwise SBD matrix.
    pub fn pairwise(series: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n = series.len();
        let mut matrix = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = Self::compute(&series[i], &series[j])?;
                matrix[i][j] = d;
                matrix[j][i] = d;
            }
        }
        Ok(matrix)
    }
}

/// Z-normalise a series (zero mean, unit variance).
fn z_normalise(series: &[f64]) -> Vec<f64> {
    let n = series.len() as f64;
    let mean: f64 = series.iter().sum::<f64>() / n;
    let variance: f64 = series.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    if std_dev < 1e-10 {
        return vec![0.0; series.len()];
    }
    series.iter().map(|x| (x - mean) / std_dev).collect()
}

/// Compute the maximum normalised cross-correlation coefficient over all shifts.
///
/// Uses a direct O(n²) computation (no FFT required for moderate-length series).
fn normalised_cross_correlation_max(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let m = b.len();

    // Norms.
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    let denom = norm_a * norm_b;
    if denom < 1e-10 {
        return 1.0; // Both zero → same shape.
    }

    let len = n.max(m);
    let shifts = 2 * len - 1;

    // Shift range: -(len-1) .. +(len-1)
    let mut max_ncc = f64::NEG_INFINITY;

    for shift_idx in 0..shifts {
        let shift = shift_idx as isize - (len as isize - 1);
        // Cross-correlation at this shift: Σ_t a[t] * b[t - shift]
        let cc: f64 = (0..n)
            .filter_map(|t| {
                let t_b = t as isize - shift;
                if t_b >= 0 && (t_b as usize) < m {
                    Some(a[t] * b[t_b as usize])
                } else {
                    None
                }
            })
            .sum();
        let ncc = cc / denom;
        if ncc > max_ncc {
            max_ncc = ncc;
        }
    }

    max_ncc.clamp(-1.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── DTW Tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_dtw_identical() {
        let s = vec![1.0, 2.0, 3.0];
        let d = DTWDistance::compute(&s, &s, None).expect("dtw ok");
        assert!(d.abs() < 1e-10, "DTW(s, s) should be 0, got {}", d);
    }

    #[test]
    fn test_dtw_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 2.0, 3.0];
        let d = DTWDistance::compute(&a, &b, None).expect("dtw ok");
        assert!(d >= 0.0);
    }

    #[test]
    fn test_dtw_different_lengths() {
        let a = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let b = vec![1.0, 2.0, 2.0, 3.0, 2.0, 1.0];
        let d = DTWDistance::compute(&a, &b, None).expect("dtw ok");
        assert!(d >= 0.0);
    }

    #[test]
    fn test_dtw_window_constraint() {
        let a = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        // Should work with window = 2.
        let d = DTWDistance::compute(&a, &b, Some(2)).expect("dtw with window");
        assert!(d.abs() < 1e-10, "DTW of identical series should be 0, got {}", d);
    }

    #[test]
    fn test_dtw_empty_error() {
        let result = DTWDistance::compute(&[], &[1.0, 2.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_dtw_pairwise() {
        let series = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
            vec![9.0, 8.0, 7.0],
        ];
        let mat = DTWDistance::pairwise(&series, None).expect("pairwise dtw");
        assert_eq!(mat.len(), 3);
        assert_eq!(mat[0][0], 0.0);
        assert!((mat[0][1]).abs() < 1e-10); // Identical series.
        assert!(mat[0][2] > mat[0][1]);     // Far series should be farther.
    }

    // ── DBA Tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_dba_single_series() {
        let series = vec![vec![1.0, 2.0, 3.0]];
        let centroid = DBABarycenter::compute(&series, &[], 5, None)
            .expect("dba single series");
        assert_eq!(centroid.len(), 3);
        assert!((centroid[0] - 1.0).abs() < 1e-10);
        assert!((centroid[1] - 2.0).abs() < 1e-10);
        assert!((centroid[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dba_two_identical_series() {
        let s = vec![1.0, 2.0, 3.0];
        let series = vec![s.clone(), s.clone()];
        let centroid = DBABarycenter::compute(&series, &[], 3, None)
            .expect("dba identical");
        assert_eq!(centroid.len(), 3);
        for (c, expected) in centroid.iter().zip([1.0, 2.0, 3.0].iter()) {
            assert!((c - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_dba_weighted() {
        let series = vec![
            vec![0.0, 0.0, 0.0],
            vec![2.0, 2.0, 2.0],
        ];
        let weights = vec![1.0, 0.0]; // Only first series.
        let centroid = DBABarycenter::compute(&series, &weights, 5, None)
            .expect("dba weighted");
        // Centroid should be close to first series.
        assert_eq!(centroid.len(), 3);
    }

    // ── K-Means DTW Tests ────────────────────────────────────────────────────

    #[test]
    fn test_kmeans_dtw_basic() {
        let series = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 1.9, 3.1],
            vec![9.0, 8.0, 7.0],
            vec![8.9, 8.1, 7.1],
        ];
        let kmeans = KMeansDTW { k: 2, max_iter: 20, window: None };
        let labels = kmeans.fit(&series).expect("kmeans_dtw ok");
        assert_eq!(labels.len(), 4);
        // Two distinct clusters.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_kmeans_dtw_k_gt_n_error() {
        let series = vec![vec![1.0, 2.0]];
        let kmeans = KMeansDTW { k: 5, max_iter: 10, window: None };
        assert!(kmeans.fit(&series).is_err());
    }

    // ── SBD Tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_sbd_identical() {
        let s = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let d = ShapeBasedDistance::compute(&s, &s).expect("sbd ok");
        // SBD of identical series should be near 0.
        assert!(d.abs() < 1e-5, "SBD(s, s) should be ~0, got {}", d);
    }

    #[test]
    fn test_sbd_range() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let d = ShapeBasedDistance::compute(&a, &b).expect("sbd ok");
        assert!(d >= 0.0 && d <= 2.0, "SBD out of range: {}", d);
    }

    #[test]
    fn test_sbd_constant_series() {
        // Constant series → zero std → z-normalised to zero → same shape.
        let a = vec![3.0, 3.0, 3.0];
        let b = vec![5.0, 5.0, 5.0];
        let d = ShapeBasedDistance::compute(&a, &b).expect("sbd constant");
        assert!(d.abs() < 1e-5, "SBD of constant series should be ~0, got {}", d);
    }

    #[test]
    fn test_sbd_pairwise() {
        let series = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
            vec![3.0, 2.0, 1.0],
        ];
        let mat = ShapeBasedDistance::pairwise(&series).expect("sbd pairwise");
        assert_eq!(mat.len(), 3);
        assert!(mat[0][0].abs() < 1e-5);
        assert!(mat[0][1].abs() < 1e-5); // Identical.
    }

    #[test]
    fn test_z_normalise() {
        let s = vec![2.0, 4.0, 6.0];
        let z = z_normalise(&s);
        let mean: f64 = z.iter().sum::<f64>() / z.len() as f64;
        let var: f64 = z.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / z.len() as f64;
        assert!(mean.abs() < 1e-10, "z-norm mean should be 0");
        assert!((var - 1.0).abs() < 1e-6, "z-norm var should be 1");
    }
}
