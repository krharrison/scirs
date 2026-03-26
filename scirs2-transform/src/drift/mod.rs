//! Data drift and distribution shift detection
//!
//! This module provides multivariate drift detection for monitoring feature distributions
//! over time between a reference dataset and a current (production) dataset.
//!
//! ## Methods
//!
//! | Method | Description |
//! |--------|-------------|
//! | [`DriftMethod::KolmogorovSmirnov`] | 2-sample KS test per feature with asymptotic p-value |
//! | [`DriftMethod::PopulationStabilityIndex`] | Binning-based PSI score |
//! | [`DriftMethod::Wasserstein`] | W1 (Earth mover's) distance per feature |
//! | [`DriftMethod::MaximumMeanDiscrepancy`] | MMD² with RBF kernel (multivariate) |
//!
//! ## Example
//!
//! ```rust
//! use scirs2_transform::drift::{DriftDetector, DriftDetectorConfig, DriftMethod};
//! use scirs2_core::ndarray::Array2;
//!
//! let reference = Array2::<f64>::zeros((100, 3));
//! let config = DriftDetectorConfig::default();
//! let detector = DriftDetector::fit(&reference, config);
//!
//! let current = Array2::<f64>::zeros((80, 3));
//! let report = detector.detect(&current).expect("detection should succeed");
//! assert!(!report.drifted, "identical distributions should not drift");
//! ```

use scirs2_core::ndarray::{Array1, Array2, Axis};

use crate::error::{Result, TransformError};

// ---------------------------------------------------------------------------
// DriftMethod enum
// ---------------------------------------------------------------------------

/// Method used for drift detection.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum DriftMethod {
    /// Two-sample Kolmogorov-Smirnov test per feature.
    ///
    /// The KS statistic D = sup|F1(x) − F2(x)| is computed per feature.
    /// A p-value is derived from the asymptotic Kolmogorov distribution.
    KolmogorovSmirnov,
    /// Population Stability Index (PSI) via reference-bin comparison.
    ///
    /// PSI = Σ (ref_pct − curr_pct) × ln(ref_pct / curr_pct)
    /// Thresholds: PSI < 0.1 (stable), 0.1–0.2 (moderate), > 0.2 (severe).
    PopulationStabilityIndex,
    /// Wasserstein-1 (Earth mover's) distance per feature.
    ///
    /// Computed exactly in 1D via sorted CDF integration.
    Wasserstein,
    /// Maximum Mean Discrepancy with RBF kernel.
    ///
    /// MMD² = E[k(x,x')] + E[k(y,y')] − 2E[k(x,y)] using U-statistic estimator.
    /// Bandwidth is set via median heuristic if not specified.
    MaximumMeanDiscrepancy,
}

// ---------------------------------------------------------------------------
// DriftDetectorConfig
// ---------------------------------------------------------------------------

/// Configuration for a [`DriftDetector`].
#[derive(Debug, Clone)]
pub struct DriftDetectorConfig {
    /// Detection method.
    pub method: DriftMethod,
    /// Significance level for hypothesis-test-based methods (KS). Default: 0.05.
    pub significance_level: f64,
    /// Number of bins for PSI. Default: 10.
    pub n_bins: usize,
    /// RBF kernel bandwidth for MMD. `None` uses the median pairwise distance heuristic.
    pub mmd_bandwidth: Option<f64>,
    /// W1 distance threshold for Wasserstein method. Default: 0.1.
    pub wasserstein_threshold: f64,
    /// PSI threshold above which drift is flagged as severe. Default: 0.2.
    pub psi_threshold: f64,
}

impl Default for DriftDetectorConfig {
    fn default() -> Self {
        Self {
            method: DriftMethod::KolmogorovSmirnov,
            significance_level: 0.05,
            n_bins: 10,
            mmd_bandwidth: None,
            wasserstein_threshold: 0.1,
            psi_threshold: 0.2,
        }
    }
}

// ---------------------------------------------------------------------------
// DriftReport
// ---------------------------------------------------------------------------

/// Report produced by [`DriftDetector::detect`].
#[derive(Debug, Clone)]
pub struct DriftReport {
    /// Per-feature drift scores (interpretation depends on method).
    pub feature_scores: Vec<f64>,
    /// Per-feature drift flags.
    pub feature_drifted: Vec<bool>,
    /// Aggregate drift score (mean of per-feature scores).
    pub overall_score: f64,
    /// Whether overall drift is detected.
    pub drifted: bool,
    /// Method used for detection.
    pub method: DriftMethod,
}

// ---------------------------------------------------------------------------
// DriftDetector
// ---------------------------------------------------------------------------

/// Multivariate drift detector that compares a reference distribution against
/// a current (test) distribution feature by feature.
pub struct DriftDetector {
    config: DriftDetectorConfig,
    /// Reference distribution: shape (n_ref × n_features).
    reference: Array2<f64>,
}

impl DriftDetector {
    /// Fit the detector by storing the reference distribution.
    ///
    /// # Arguments
    /// * `reference` – Reference dataset with shape (n_samples × n_features).
    /// * `config`    – Detection configuration.
    pub fn fit(reference: &Array2<f64>, config: DriftDetectorConfig) -> Self {
        Self {
            config,
            reference: reference.to_owned(),
        }
    }

    /// Detect drift between the stored reference and a new current dataset.
    ///
    /// # Arguments
    /// * `current` – Current dataset with shape (n_samples × n_features).
    ///
    /// # Errors
    /// Returns [`TransformError::InvalidInput`] if the number of features differs
    /// or if either dataset is empty.
    pub fn detect(&self, current: &Array2<f64>) -> Result<DriftReport> {
        let n_ref = self.reference.nrows();
        let n_cur = current.nrows();
        let n_features = self.reference.ncols();

        if n_ref == 0 {
            return Err(TransformError::InvalidInput(
                "Reference dataset is empty".to_string(),
            ));
        }
        if n_cur == 0 {
            return Err(TransformError::InvalidInput(
                "Current dataset is empty".to_string(),
            ));
        }
        if current.ncols() != n_features {
            return Err(TransformError::InvalidInput(format!(
                "Feature dimension mismatch: reference has {n_features} features, current has {}",
                current.ncols()
            )));
        }

        match &self.config.method {
            DriftMethod::KolmogorovSmirnov => self.detect_ks(current),
            DriftMethod::PopulationStabilityIndex => self.detect_psi(current),
            DriftMethod::Wasserstein => self.detect_wasserstein(current),
            DriftMethod::MaximumMeanDiscrepancy => self.detect_mmd(current),
        }
    }

    /// Replace the reference distribution (e.g. for sliding window monitoring).
    pub fn update_reference(&mut self, new_reference: &Array2<f64>) {
        self.reference = new_reference.to_owned();
    }

    // -----------------------------------------------------------------------
    // KS per-feature detection
    // -----------------------------------------------------------------------

    fn detect_ks(&self, current: &Array2<f64>) -> Result<DriftReport> {
        let n_features = self.reference.ncols();
        let mut scores = Vec::with_capacity(n_features);
        let mut drifted_flags = Vec::with_capacity(n_features);

        for f in 0..n_features {
            let ref_col: Vec<f64> = self.reference.column(f).iter().copied().collect();
            let cur_col: Vec<f64> = current.column(f).iter().copied().collect();

            let (ks_stat, p_value) = ks_2samp(&ref_col, &cur_col)?;
            scores.push(ks_stat);
            drifted_flags.push(p_value < self.config.significance_level);
        }

        let overall_score = scores.iter().copied().sum::<f64>() / scores.len() as f64;
        let drifted = drifted_flags.iter().any(|&d| d);

        Ok(DriftReport {
            feature_scores: scores,
            feature_drifted: drifted_flags,
            overall_score,
            drifted,
            method: DriftMethod::KolmogorovSmirnov,
        })
    }

    // -----------------------------------------------------------------------
    // PSI per-feature detection
    // -----------------------------------------------------------------------

    fn detect_psi(&self, current: &Array2<f64>) -> Result<DriftReport> {
        let n_features = self.reference.ncols();
        let mut scores = Vec::with_capacity(n_features);
        let mut drifted_flags = Vec::with_capacity(n_features);

        for f in 0..n_features {
            let ref_col: Vec<f64> = self.reference.column(f).iter().copied().collect();
            let cur_col: Vec<f64> = current.column(f).iter().copied().collect();

            let psi = compute_psi(&ref_col, &cur_col, self.config.n_bins)?;
            scores.push(psi);
            drifted_flags.push(psi > self.config.psi_threshold);
        }

        let overall_score = scores.iter().copied().sum::<f64>() / scores.len() as f64;
        let drifted = drifted_flags.iter().any(|&d| d);

        Ok(DriftReport {
            feature_scores: scores,
            feature_drifted: drifted_flags,
            overall_score,
            drifted,
            method: DriftMethod::PopulationStabilityIndex,
        })
    }

    // -----------------------------------------------------------------------
    // Wasserstein-1 per-feature detection
    // -----------------------------------------------------------------------

    fn detect_wasserstein(&self, current: &Array2<f64>) -> Result<DriftReport> {
        let n_features = self.reference.ncols();
        let mut scores = Vec::with_capacity(n_features);
        let mut drifted_flags = Vec::with_capacity(n_features);

        for f in 0..n_features {
            let ref_col: Vec<f64> = self.reference.column(f).iter().copied().collect();
            let cur_col: Vec<f64> = current.column(f).iter().copied().collect();

            let w1 = wasserstein_1d_distance(&ref_col, &cur_col)?;
            scores.push(w1);
            drifted_flags.push(w1 > self.config.wasserstein_threshold);
        }

        let overall_score = scores.iter().copied().sum::<f64>() / scores.len() as f64;
        let drifted = drifted_flags.iter().any(|&d| d);

        Ok(DriftReport {
            feature_scores: scores,
            feature_drifted: drifted_flags,
            overall_score,
            drifted,
            method: DriftMethod::Wasserstein,
        })
    }

    // -----------------------------------------------------------------------
    // MMD multivariate detection
    // -----------------------------------------------------------------------

    fn detect_mmd(&self, current: &Array2<f64>) -> Result<DriftReport> {
        let n_features = self.reference.ncols();

        // Determine bandwidth: median heuristic on reference if not specified
        let bandwidth = match self.config.mmd_bandwidth {
            Some(bw) => {
                if bw <= 0.0 {
                    return Err(TransformError::InvalidInput(
                        "mmd_bandwidth must be positive".to_string(),
                    ));
                }
                bw
            }
            None => median_heuristic_bandwidth(&self.reference)?,
        };

        // Per-feature MMD scores
        let mut scores = Vec::with_capacity(n_features);
        let mut drifted_flags = Vec::with_capacity(n_features);

        for f in 0..n_features {
            let ref_col: Vec<f64> = self.reference.column(f).iter().copied().collect();
            let cur_col: Vec<f64> = current.column(f).iter().copied().collect();

            let mmd2 = mmd_u_statistic_1d(&ref_col, &cur_col, bandwidth)?;
            scores.push(mmd2);
            // Threshold: mmd2 > 2 * bandwidth_scale (simple heuristic)
            // A positive MMD² >> 0 indicates drift; threshold chosen as function of n.
            let n_eff = (ref_col.len().min(cur_col.len()) as f64).max(1.0);
            let threshold = 4.0 / n_eff.sqrt();
            drifted_flags.push(mmd2 > threshold);
        }

        let overall_score = scores.iter().copied().sum::<f64>() / scores.len() as f64;
        let drifted = drifted_flags.iter().any(|&d| d);

        Ok(DriftReport {
            feature_scores: scores,
            feature_drifted: drifted_flags,
            overall_score,
            drifted,
            method: DriftMethod::MaximumMeanDiscrepancy,
        })
    }
}

// ---------------------------------------------------------------------------
// KS 2-sample statistic and asymptotic p-value
// ---------------------------------------------------------------------------

/// Compute the two-sample KS statistic D and an asymptotic p-value.
///
/// Uses the Kolmogorov asymptotic approximation:
/// p ≈ 2 Σ_{j=1}^{∞} (−1)^{j+1} exp(−2 j² λ²)  where  λ = D √(n1 n2 / (n1+n2))
fn ks_2samp(x: &[f64], y: &[f64]) -> Result<(f64, f64)> {
    if x.is_empty() || y.is_empty() {
        return Err(TransformError::InvalidInput(
            "KS samples must be non-empty".to_string(),
        ));
    }

    let mut xs = x.to_vec();
    let mut ys = y.to_vec();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n1 = xs.len();
    let n2 = ys.len();
    let n1f = n1 as f64;
    let n2f = n2 as f64;

    // Two-pointer sweep over merged sorted values
    let mut i = 0usize;
    let mut j = 0usize;
    let mut d_max: f64 = 0.0;

    while i < n1 || j < n2 {
        let xv = if i < n1 { xs[i] } else { f64::INFINITY };
        let yv = if j < n2 { ys[j] } else { f64::INFINITY };

        // Advance both pointers past the current value
        let cur = xv.min(yv);
        while i < n1 && xs[i] <= cur {
            i += 1;
        }
        while j < n2 && ys[j] <= cur {
            j += 1;
        }

        let cdf1 = i as f64 / n1f;
        let cdf2 = j as f64 / n2f;
        d_max = d_max.max((cdf1 - cdf2).abs());
    }

    // Asymptotic Kolmogorov distribution
    let lambda = d_max * ((n1f * n2f / (n1f + n2f)).sqrt());
    let p_value = kolmogorov_p_value(lambda);

    Ok((d_max, p_value))
}

/// Asymptotic KS p-value: P(D_n > d) ≈ 2 Σ_{j=1}^{K} (−1)^{j+1} exp(−2 j² λ²)
fn kolmogorov_p_value(lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 1.0;
    }
    if lambda > 4.0 {
        return 0.0;
    }

    let mut sum = 0.0;
    // Converges quickly; 20 terms is more than sufficient
    for j in 1usize..=20 {
        let jf = j as f64;
        let term = (-2.0 * jf * jf * lambda * lambda).exp();
        if j % 2 == 0 {
            sum -= term;
        } else {
            sum += term;
        }
        if term < 1e-15 {
            break;
        }
    }
    (2.0 * sum).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// PSI
// ---------------------------------------------------------------------------

/// Compute the Population Stability Index between two 1D distributions.
///
/// PSI = Σ (ref_pct − curr_pct) × ln(ref_pct / curr_pct)
///
/// Bins are derived from the reference distribution quantiles.
fn compute_psi(reference: &[f64], current: &[f64], n_bins: usize) -> Result<f64> {
    if reference.is_empty() || current.is_empty() {
        return Err(TransformError::InvalidInput(
            "PSI samples must be non-empty".to_string(),
        ));
    }
    if n_bins == 0 {
        return Err(TransformError::InvalidInput(
            "n_bins must be at least 1".to_string(),
        ));
    }

    let mut ref_sorted: Vec<f64> = reference.to_vec();
    ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Build bin edges from reference quantiles
    let mut edges: Vec<f64> = Vec::with_capacity(n_bins + 1);
    edges.push(f64::NEG_INFINITY);
    for i in 1..n_bins {
        let q = i as f64 / n_bins as f64;
        let idx = ((q * reference.len() as f64) as usize).min(reference.len() - 1);
        edges.push(ref_sorted[idx]);
    }
    edges.push(f64::INFINITY);

    // Deduplicate edges (handles constant features)
    edges.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);

    let actual_bins = edges.len() - 1;
    if actual_bins == 0 {
        // Perfectly constant feature — no drift possible
        return Ok(0.0);
    }

    let ref_n = reference.len() as f64;
    let cur_n = current.len() as f64;
    let epsilon = 1e-8; // smoothing to avoid log(0)

    let mut ref_counts = vec![0u64; actual_bins];
    let mut cur_counts = vec![0u64; actual_bins];

    for &v in reference {
        let bin = find_bin(v, &edges);
        ref_counts[bin] += 1;
    }
    for &v in current {
        let bin = find_bin(v, &edges);
        cur_counts[bin] += 1;
    }

    let mut psi = 0.0_f64;
    for b in 0..actual_bins {
        let ref_pct = (ref_counts[b] as f64 / ref_n + epsilon).min(1.0);
        let cur_pct = (cur_counts[b] as f64 / cur_n + epsilon).min(1.0);
        psi += (ref_pct - cur_pct) * (ref_pct / cur_pct).ln();
    }

    Ok(psi.max(0.0))
}

/// Find which bin index a value falls in using the bin edges.
fn find_bin(value: f64, edges: &[f64]) -> usize {
    let n_bins = edges.len() - 1;
    // Binary search: find rightmost edge that is <= value
    let mut lo = 1usize; // skip −∞
    let mut hi = n_bins;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if edges[mid] <= value {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    (lo - 1).min(n_bins - 1)
}

// ---------------------------------------------------------------------------
// Wasserstein-1D
// ---------------------------------------------------------------------------

/// Exact 1D Wasserstein-1 distance via sorted CDF sweep.
///
/// W₁(p, q) = ∫|F_p(x) − F_q(x)| dx  ≡  (1/n) Σ |x_i − y_i| after sorting
/// (when both have equal weight; otherwise we do the general CDF area integral).
fn wasserstein_1d_distance(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.is_empty() || y.is_empty() {
        return Err(TransformError::InvalidInput(
            "Wasserstein samples must be non-empty".to_string(),
        ));
    }

    let mut xs: Vec<f64> = x.to_vec();
    let mut ys: Vec<f64> = y.to_vec();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = xs.len();
    let m = ys.len();
    let nf = n as f64;
    let mf = m as f64;

    // Build merged sorted event points
    let mut events: Vec<f64> = xs.iter().chain(ys.iter()).copied().collect();
    events.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    events.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON * a.abs().max(1.0));

    let mut dist = 0.0_f64;
    let mut ix = 0usize;
    let mut iy = 0usize;

    for w in events.windows(2) {
        let lo = w[0];
        let hi = w[1];
        let dx = hi - lo;

        // Count elements <= lo in xs and ys
        while ix < n && xs[ix] <= lo {
            ix += 1;
        }
        while iy < m && ys[iy] <= lo {
            iy += 1;
        }

        let cdf_x = ix as f64 / nf;
        let cdf_y = iy as f64 / mf;
        dist += (cdf_x - cdf_y).abs() * dx;
    }

    Ok(dist)
}

// ---------------------------------------------------------------------------
// MMD U-statistic (1D)
// ---------------------------------------------------------------------------

/// U-statistic estimator of MMD² for 1D samples with RBF kernel.
///
/// MMD²_u = (1/(n(n-1))) Σ_{i≠j} k(x_i,x_j)
///         + (1/(m(m-1))) Σ_{i≠j} k(y_i,y_j)
///         − (2/(nm))     Σ_{i,j} k(x_i,y_j)
///
/// where k(a,b) = exp(−(a−b)² / (2σ²)).
fn mmd_u_statistic_1d(x: &[f64], y: &[f64], bandwidth: f64) -> Result<f64> {
    if x.is_empty() || y.is_empty() {
        return Err(TransformError::InvalidInput(
            "MMD samples must be non-empty".to_string(),
        ));
    }
    if bandwidth <= 0.0 {
        return Err(TransformError::InvalidInput(
            "MMD bandwidth must be positive".to_string(),
        ));
    }

    let n = x.len();
    let m = y.len();
    let gamma = 1.0 / (2.0 * bandwidth * bandwidth);

    // Term 1: (1/(n(n-1))) Σ_{i≠j} k(x_i, x_j)
    let kxx = if n > 1 {
        let mut sum = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = x[i] - x[j];
                sum += (-gamma * d * d).exp();
            }
        }
        2.0 * sum / (n * (n - 1)) as f64
    } else {
        0.0
    };

    // Term 2: (1/(m(m-1))) Σ_{i≠j} k(y_i, y_j)
    let kyy = if m > 1 {
        let mut sum = 0.0_f64;
        for i in 0..m {
            for j in (i + 1)..m {
                let d = y[i] - y[j];
                sum += (-gamma * d * d).exp();
            }
        }
        2.0 * sum / (m * (m - 1)) as f64
    } else {
        0.0
    };

    // Term 3: (2/(nm)) Σ_{i,j} k(x_i, y_j)
    let mut kxy_sum = 0.0_f64;
    for &xi in x {
        for &yi in y {
            let d = xi - yi;
            kxy_sum += (-gamma * d * d).exp();
        }
    }
    let kxy = 2.0 * kxy_sum / (n * m) as f64;

    Ok((kxx + kyy - kxy).max(0.0))
}

// ---------------------------------------------------------------------------
// Median heuristic for MMD bandwidth
// ---------------------------------------------------------------------------

/// Estimate RBF bandwidth via the median of pairwise Euclidean distances in the
/// reference dataset (subsampled to at most 500 rows for efficiency).
fn median_heuristic_bandwidth(data: &Array2<f64>) -> Result<f64> {
    let n = data.nrows();
    if n == 0 {
        return Err(TransformError::InvalidInput(
            "Cannot compute bandwidth on empty data".to_string(),
        ));
    }

    // Subsample to limit O(n²) cost
    let max_samples = 500usize;
    let step = if n > max_samples { n / max_samples } else { 1 };
    let indices: Vec<usize> = (0..n).step_by(step).collect();
    let k = indices.len();

    let mut dists: Vec<f64> = Vec::with_capacity(k * (k - 1) / 2);
    for i in 0..k {
        for j in (i + 1)..k {
            let row_i = data.row(indices[i]);
            let row_j = data.row(indices[j]);
            let sq_dist: f64 = row_i
                .iter()
                .zip(row_j.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            dists.push(sq_dist.sqrt());
        }
    }

    if dists.is_empty() {
        return Ok(1.0); // single sample: use unit bandwidth
    }

    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = dists[dists.len() / 2];
    Ok(if median > 0.0 { median } else { 1.0 })
}

// ---------------------------------------------------------------------------
// Public convenience functions
// ---------------------------------------------------------------------------

/// Compute the 2-sample KS statistic and asymptotic p-value for two 1D samples.
///
/// Returns `(ks_statistic, p_value)`.
pub fn ks_test(x: &[f64], y: &[f64]) -> Result<(f64, f64)> {
    ks_2samp(x, y)
}

/// Compute PSI between a reference and current 1D distribution.
pub fn psi(reference: &[f64], current: &[f64], n_bins: usize) -> Result<f64> {
    compute_psi(reference, current, n_bins)
}

/// Compute the W1 (Wasserstein-1) distance between two 1D empirical distributions.
pub fn wasserstein_distance_1d(x: &[f64], y: &[f64]) -> Result<f64> {
    wasserstein_1d_distance(x, y)
}

/// Compute MMD² with RBF kernel for two 1D sample arrays.
pub fn mmd_rbf(x: &[f64], y: &[f64], bandwidth: f64) -> Result<f64> {
    mmd_u_statistic_1d(x, y, bandwidth)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn zeros_matrix(rows: usize, cols: usize) -> Array2<f64> {
        Array2::<f64>::zeros((rows, cols))
    }

    fn linspace_col(start: f64, end: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| start + (end - start) * (i as f64) / ((n - 1) as f64))
            .collect()
    }

    // ------------------------------------------------------------------
    // KS tests
    // ------------------------------------------------------------------

    #[test]
    fn test_ks_no_drift() {
        // Same distribution → KS statistic ≈ 0, high p-value, no drift
        let data: Vec<f64> = linspace_col(0.0, 1.0, 100);
        let config = DriftDetectorConfig {
            method: DriftMethod::KolmogorovSmirnov,
            ..Default::default()
        };
        let ref_mat = Array2::from_shape_vec((100, 1), data.clone()).expect("shape ok");
        let cur_mat = Array2::from_shape_vec((100, 1), data).expect("shape ok");

        let detector = DriftDetector::fit(&ref_mat, config);
        let report = detector.detect(&cur_mat).expect("detect ok");
        assert!(!report.drifted, "identical distributions should not drift");
        assert!(
            report.overall_score < 0.01,
            "KS statistic should be near 0, got {}",
            report.overall_score
        );
    }

    #[test]
    fn test_ks_drift_detected() {
        // Reference: [0, 1], Current: [10, 11] — clearly different
        let ref_data: Vec<f64> = linspace_col(0.0, 1.0, 100);
        let cur_data: Vec<f64> = linspace_col(10.0, 11.0, 100);
        let config = DriftDetectorConfig {
            method: DriftMethod::KolmogorovSmirnov,
            ..Default::default()
        };
        let ref_mat = Array2::from_shape_vec((100, 1), ref_data).expect("shape ok");
        let cur_mat = Array2::from_shape_vec((100, 1), cur_data).expect("shape ok");

        let detector = DriftDetector::fit(&ref_mat, config);
        let report = detector.detect(&cur_mat).expect("detect ok");
        assert!(report.drifted, "clearly shifted distributions should drift");
        assert!(
            report.overall_score > 0.9,
            "KS stat should be close to 1.0, got {}",
            report.overall_score
        );
    }

    #[test]
    fn test_ks_drift_multifeature() {
        // 3-feature dataset; only feature 0 drifts
        let n = 100usize;
        let mut ref_data = vec![0.0f64; n * 3];
        let mut cur_data = vec![0.0f64; n * 3];

        for i in 0..n {
            // Feature 0: reference [0,1], current [5,6]
            ref_data[i * 3] = i as f64 / n as f64;
            cur_data[i * 3] = 5.0 + i as f64 / n as f64;
            // Feature 1 & 2: same
            ref_data[i * 3 + 1] = i as f64 / n as f64;
            cur_data[i * 3 + 1] = i as f64 / n as f64;
            ref_data[i * 3 + 2] = i as f64 / n as f64;
            cur_data[i * 3 + 2] = i as f64 / n as f64;
        }

        let ref_mat = Array2::from_shape_vec((n, 3), ref_data).expect("shape ok");
        let cur_mat = Array2::from_shape_vec((n, 3), cur_data).expect("shape ok");
        let config = DriftDetectorConfig {
            method: DriftMethod::KolmogorovSmirnov,
            ..Default::default()
        };

        let detector = DriftDetector::fit(&ref_mat, config);
        let report = detector.detect(&cur_mat).expect("detect ok");
        assert!(report.drifted, "overall should be drifted");
        assert!(report.feature_drifted[0], "feature 0 should drift");
        assert!(!report.feature_drifted[1], "feature 1 should not drift");
        assert!(!report.feature_drifted[2], "feature 2 should not drift");
    }

    // ------------------------------------------------------------------
    // PSI tests
    // ------------------------------------------------------------------

    #[test]
    fn test_psi_no_drift() {
        // Same distribution → PSI ≈ 0
        let data: Vec<f64> = linspace_col(0.0, 1.0, 100);
        let config = DriftDetectorConfig {
            method: DriftMethod::PopulationStabilityIndex,
            ..Default::default()
        };
        let ref_mat = Array2::from_shape_vec((100, 1), data.clone()).expect("shape ok");
        let cur_mat = Array2::from_shape_vec((100, 1), data).expect("shape ok");

        let detector = DriftDetector::fit(&ref_mat, config);
        let report = detector.detect(&cur_mat).expect("detect ok");
        assert!(
            !report.drifted,
            "same distribution should not trigger PSI drift"
        );
    }

    #[test]
    fn test_psi_severe_drift() {
        // PSI > 0.2 (severe drift)
        let ref_data: Vec<f64> = linspace_col(0.0, 1.0, 200);
        let cur_data: Vec<f64> = linspace_col(5.0, 6.0, 200);
        let config = DriftDetectorConfig {
            method: DriftMethod::PopulationStabilityIndex,
            psi_threshold: 0.2,
            ..Default::default()
        };
        let ref_mat = Array2::from_shape_vec((200, 1), ref_data).expect("shape ok");
        let cur_mat = Array2::from_shape_vec((200, 1), cur_data).expect("shape ok");

        let detector = DriftDetector::fit(&ref_mat, config);
        let report = detector.detect(&cur_mat).expect("detect ok");
        assert!(report.drifted, "severe shift should trigger PSI drift");
        assert!(
            report.overall_score > 0.2,
            "PSI should exceed threshold, got {}",
            report.overall_score
        );
    }

    // ------------------------------------------------------------------
    // Wasserstein tests
    // ------------------------------------------------------------------

    #[test]
    fn test_wasserstein_drift() {
        // Reference: [0,1], Current: [0.5, 1.5]
        // W1 ≈ 0.5 > threshold 0.1
        let ref_data: Vec<f64> = linspace_col(0.0, 1.0, 100);
        let cur_data: Vec<f64> = linspace_col(0.5, 1.5, 100);
        let config = DriftDetectorConfig {
            method: DriftMethod::Wasserstein,
            wasserstein_threshold: 0.1,
            ..Default::default()
        };
        let ref_mat = Array2::from_shape_vec((100, 1), ref_data).expect("shape ok");
        let cur_mat = Array2::from_shape_vec((100, 1), cur_data).expect("shape ok");

        let detector = DriftDetector::fit(&ref_mat, config);
        let report = detector.detect(&cur_mat).expect("detect ok");
        assert!(
            report.drifted,
            "shifted distribution should trigger W1 drift"
        );
        assert!(
            (report.overall_score - 0.5).abs() < 0.05,
            "W1 distance should be ~0.5, got {}",
            report.overall_score
        );
    }

    #[test]
    fn test_wasserstein_no_drift() {
        // Same distribution → W1 = 0
        let data: Vec<f64> = linspace_col(0.0, 1.0, 100);
        let config = DriftDetectorConfig {
            method: DriftMethod::Wasserstein,
            wasserstein_threshold: 0.1,
            ..Default::default()
        };
        let ref_mat = Array2::from_shape_vec((100, 1), data.clone()).expect("shape ok");
        let cur_mat = Array2::from_shape_vec((100, 1), data).expect("shape ok");

        let detector = DriftDetector::fit(&ref_mat, config);
        let report = detector.detect(&cur_mat).expect("detect ok");
        assert!(!report.drifted, "identical distributions should not drift");
        assert!(
            report.overall_score < 1e-10,
            "W1 should be 0, got {}",
            report.overall_score
        );
    }

    // ------------------------------------------------------------------
    // MMD tests
    // ------------------------------------------------------------------

    #[test]
    fn test_mmd_identical() {
        // MMD(P, P) should be ≈ 0
        let data: Vec<f64> = linspace_col(0.0, 1.0, 50);
        let ref_mat = Array2::from_shape_vec((50, 1), data.clone()).expect("shape ok");
        let cur_mat = Array2::from_shape_vec((50, 1), data).expect("shape ok");
        let config = DriftDetectorConfig {
            method: DriftMethod::MaximumMeanDiscrepancy,
            mmd_bandwidth: Some(0.5),
            ..Default::default()
        };

        let detector = DriftDetector::fit(&ref_mat, config);
        let report = detector.detect(&cur_mat).expect("detect ok");
        assert!(
            report.overall_score < 1e-6,
            "MMD of identical distributions should be near 0, got {}",
            report.overall_score
        );
    }

    #[test]
    fn test_mmd_different() {
        // N(0,1) vs N(5,1) → large MMD
        let ref_data: Vec<f64> = linspace_col(-3.0, 3.0, 60);
        let cur_data: Vec<f64> = linspace_col(2.0, 8.0, 60);
        let ref_mat = Array2::from_shape_vec((60, 1), ref_data).expect("shape ok");
        let cur_mat = Array2::from_shape_vec((60, 1), cur_data).expect("shape ok");
        let config = DriftDetectorConfig {
            method: DriftMethod::MaximumMeanDiscrepancy,
            mmd_bandwidth: Some(1.0),
            ..Default::default()
        };

        let detector = DriftDetector::fit(&ref_mat, config);
        let report = detector.detect(&cur_mat).expect("detect ok");
        assert!(
            report.overall_score > 0.01,
            "MMD between N(0,1) and N(5,1) should be positive, got {}",
            report.overall_score
        );
    }

    #[test]
    fn test_mmd_median_heuristic() {
        // Should work with automatic bandwidth selection
        let ref_data: Vec<f64> = linspace_col(0.0, 1.0, 40);
        let cur_data: Vec<f64> = linspace_col(5.0, 6.0, 40);
        let ref_mat = Array2::from_shape_vec((40, 1), ref_data).expect("shape ok");
        let cur_mat = Array2::from_shape_vec((40, 1), cur_data).expect("shape ok");
        let config = DriftDetectorConfig {
            method: DriftMethod::MaximumMeanDiscrepancy,
            mmd_bandwidth: None,
            ..Default::default()
        };

        let detector = DriftDetector::fit(&ref_mat, config);
        let report = detector.detect(&cur_mat).expect("detect ok");
        // Just verify it runs without error and produces a non-negative score
        assert!(report.overall_score >= 0.0, "MMD² must be non-negative");
    }

    // ------------------------------------------------------------------
    // update_reference
    // ------------------------------------------------------------------

    #[test]
    fn test_update_reference() {
        let initial_ref = zeros_matrix(50, 2);
        let config = DriftDetectorConfig::default();
        let mut detector = DriftDetector::fit(&initial_ref, config);

        // Update reference to drifted data
        let shifted_ref = Array2::from_elem((50, 2), 5.0);
        detector.update_reference(&shifted_ref);

        // Current is same as new reference → no drift
        let current = Array2::from_elem((50, 2), 5.0);
        let report = detector.detect(&current).expect("detect ok");
        assert!(!report.drifted, "after update, same data should not drift");
    }

    // ------------------------------------------------------------------
    // Edge cases / error handling
    // ------------------------------------------------------------------

    #[test]
    fn test_dimension_mismatch_error() {
        let ref_mat = zeros_matrix(50, 3);
        let cur_mat = zeros_matrix(50, 2);
        let config = DriftDetectorConfig::default();
        let detector = DriftDetector::fit(&ref_mat, config);
        let result = detector.detect(&cur_mat);
        assert!(result.is_err(), "should error on dimension mismatch");
    }

    #[test]
    fn test_ks_test_function() {
        let x: Vec<f64> = linspace_col(0.0, 1.0, 50);
        let y: Vec<f64> = linspace_col(0.0, 1.0, 50);
        let (d, p) = ks_test(&x, &y).expect("ks test ok");
        assert!(d < 0.05, "KS stat should be near 0");
        assert!(p > 0.5, "p-value should be high");
    }

    #[test]
    fn test_psi_function() {
        let ref_data: Vec<f64> = linspace_col(0.0, 1.0, 100);
        let cur_data: Vec<f64> = linspace_col(0.0, 1.0, 100);
        let score = psi(&ref_data, &cur_data, 10).expect("psi ok");
        assert!(score < 0.01, "PSI should be near 0 for same distribution");
    }

    #[test]
    fn test_wasserstein_distance_1d_function() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let d = wasserstein_distance_1d(&x, &y).expect("w1 ok");
        assert!((d - 1.0).abs() < 0.05, "W1 should be ~1.0, got {d}");
    }

    #[test]
    fn test_mmd_rbf_function() {
        let x = vec![0.0f64; 10];
        let y = vec![0.0f64; 10];
        let mmd2 = mmd_rbf(&x, &y, 1.0).expect("mmd ok");
        assert!(mmd2 < 1e-8, "MMD of identical samples should be 0");
    }
}
