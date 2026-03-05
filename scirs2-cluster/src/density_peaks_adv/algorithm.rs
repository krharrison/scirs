//! Core density peaks clustering algorithm.
//!
//! Implements the Rodriguez & Laio (2014) density peaks algorithm with both
//! Gaussian and cutoff kernel density estimators.

use crate::error::{ClusteringError, Result};

/// Result of density peaks clustering.
#[derive(Debug, Clone)]
pub struct DensityPeaksAdvResult {
    /// Cluster label for each data point (-1 = unassigned/noise if no center found).
    pub labels: Vec<i64>,
    /// Indices of detected cluster centers.
    pub centers: Vec<usize>,
    /// Local density rho for each point.
    pub rho: Vec<f64>,
    /// Distance delta to nearest point with higher density.
    pub delta: Vec<f64>,
    /// Total number of clusters found.
    pub n_clusters: usize,
}

/// Kernel type for local density estimation.
#[derive(Debug, Clone, PartialEq)]
pub enum KernelType {
    /// Gaussian kernel: rho_i = sum_j exp(-d_{ij}^2 / (2 * d_c^2))
    Gaussian,
    /// Step/cutoff kernel: rho_i = #{j : d_{ij} < d_c} - 1
    Cutoff,
}

/// Density Peaks clustering builder.
///
/// # Example
///
/// ```
/// use scirs2_cluster::density_peaks_adv::{DensityPeaksAdv, KernelType};
///
/// let data = vec![
///     vec![0.0, 0.0], vec![0.1, 0.0], vec![0.0, 0.1],
///     vec![5.0, 5.0], vec![5.1, 5.0], vec![5.0, 5.1],
/// ];
/// let result = DensityPeaksAdv::new()
///     .with_n_clusters(2)
///     .fit(&data)
///     .expect("operation should succeed");
/// assert_eq!(result.n_clusters, 2);
/// ```
pub struct DensityPeaksAdv {
    /// Cutoff distance d_c (None = auto-select at 2nd percentile of pairwise distances).
    pub d_c: Option<f64>,
    /// Kernel type for density estimation.
    pub kernel: KernelType,
    /// If Some(k), automatically select the top-k centers by gamma = rho * delta.
    pub n_clusters: Option<usize>,
    /// Density threshold for center selection (threshold-based mode).
    pub rho_threshold: f64,
    /// Delta threshold for center selection (threshold-based mode).
    pub delta_threshold: f64,
}

impl DensityPeaksAdv {
    /// Create a new `DensityPeaksAdv` with default parameters.
    pub fn new() -> Self {
        Self {
            d_c: None,
            kernel: KernelType::Gaussian,
            n_clusters: None,
            rho_threshold: 0.0,
            delta_threshold: 0.0,
        }
    }

    /// Set the cutoff distance d_c explicitly.
    pub fn with_d_c(mut self, d_c: f64) -> Self {
        self.d_c = Some(d_c);
        self
    }

    /// Set the kernel type.
    pub fn with_kernel(mut self, kernel: KernelType) -> Self {
        self.kernel = kernel;
        self
    }

    /// Automatically select the top-k centers by gamma = rho * delta.
    pub fn with_n_clusters(mut self, k: usize) -> Self {
        self.n_clusters = Some(k);
        self
    }

    /// Use threshold-based center selection.
    pub fn with_thresholds(mut self, rho_t: f64, delta_t: f64) -> Self {
        self.rho_threshold = rho_t;
        self.delta_threshold = delta_t;
        self
    }

    /// Fit density peaks clustering on a slice of feature vectors.
    ///
    /// Each inner `Vec<f64>` is one data point. All points must have the same
    /// dimensionality.
    ///
    /// # Errors
    ///
    /// Returns [`ClusteringError::InvalidInput`] if `data` is empty or if
    /// `n_clusters` exceeds the number of data points.
    pub fn fit(&self, data: &[Vec<f64>]) -> Result<DensityPeaksAdvResult> {
        let n = data.len();
        if n == 0 {
            return Err(ClusteringError::InvalidInput(
                "input data must not be empty".to_string(),
            ));
        }

        // Validate uniform dimensionality
        let dim = data[0].len();
        for (i, point) in data.iter().enumerate() {
            if point.len() != dim {
                return Err(ClusteringError::InvalidInput(format!(
                    "point {} has dimension {} but expected {}",
                    i,
                    point.len(),
                    dim
                )));
            }
        }

        // Compute full n×n pairwise Euclidean distance matrix (upper-tri then mirror).
        let dists = compute_pairwise_distances(data);

        // Auto-select d_c: 2nd percentile of all pairwise distances.
        let d_c = match self.d_c {
            Some(v) => {
                if v <= 0.0 {
                    return Err(ClusteringError::InvalidInput(
                        "d_c must be positive".to_string(),
                    ));
                }
                v
            }
            None => auto_select_dc(&dists, n),
        };

        // Compute local density rho for each point.
        let rho = compute_rho(&dists, n, d_c, &self.kernel);

        // Sort indices by decreasing density.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            rho[b]
                .partial_cmp(&rho[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // For each point: find distance to nearest point with higher density.
        let (delta, nearest_higher) = compute_delta(&dists, n, &rho, &order);

        // Select cluster centers.
        let centers = select_centers(n, &rho, &delta, self.n_clusters, self.rho_threshold, self.delta_threshold)?;

        // Propagate cluster assignments following density order.
        let labels = propagate_labels(n, &centers, &nearest_higher, &order);

        let n_clusters = centers.len();
        Ok(DensityPeaksAdvResult {
            labels,
            centers,
            rho,
            delta,
            n_clusters,
        })
    }
}

impl Default for DensityPeaksAdv {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the n×n Euclidean distance matrix stored as a flat Vec (row-major).
fn compute_pairwise_distances(data: &[Vec<f64>]) -> Vec<f64> {
    let n = data.len();
    let mut dists = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d: f64 = data[i]
                .iter()
                .zip(data[j].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            dists[i * n + j] = d;
            dists[j * n + i] = d;
        }
    }
    dists
}

/// Auto-select d_c as the 2nd percentile of all (n*(n-1)/2) pairwise distances.
fn auto_select_dc(dists: &[f64], n: usize) -> f64 {
    let mut upper: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            upper.push(dists[i * n + j]);
        }
    }
    upper.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((upper.len() as f64) * 0.02) as usize;
    upper.get(idx).copied().unwrap_or(1.0).max(1e-10)
}

/// Compute local density rho for each point given a flat distance matrix.
fn compute_rho(dists: &[f64], n: usize, d_c: f64, kernel: &KernelType) -> Vec<f64> {
    (0..n)
        .map(|i| match kernel {
            KernelType::Gaussian => {
                let two_dc2 = 2.0 * d_c * d_c;
                (0..n)
                    .filter(|&j| j != i)
                    .map(|j| {
                        let d = dists[i * n + j];
                        (-(d * d) / two_dc2).exp()
                    })
                    .sum()
            }
            KernelType::Cutoff => {
                (0..n)
                    .filter(|&j| j != i && dists[i * n + j] < d_c)
                    .count() as f64
            }
        })
        .collect()
}

/// For each point, compute delta (min dist to a higher-density point) and
/// the index of that nearest higher-density neighbor.
///
/// The point with maximum density gets delta = max pairwise distance in the dataset.
fn compute_delta(
    dists: &[f64],
    n: usize,
    rho: &[f64],
    order: &[usize],
) -> (Vec<f64>, Vec<usize>) {
    let mut delta = vec![f64::INFINITY; n];
    let mut nearest_higher = vec![0usize; n];

    for (rank, &i) in order.iter().enumerate() {
        if rank == 0 {
            // Highest-density point: delta = max distance to any other point.
            let max_d = (0..n)
                .filter(|&j| j != i)
                .map(|j| dists[i * n + j])
                .fold(f64::NEG_INFINITY, f64::max);
            delta[i] = if max_d.is_finite() { max_d } else { 0.0 };
            nearest_higher[i] = i;
        } else {
            // Find nearest point among those with strictly higher density.
            for &j in order[..rank].iter() {
                let d = dists[i * n + j];
                if d < delta[i] {
                    delta[i] = d;
                    nearest_higher[i] = j;
                }
            }
            // Handle case where delta was never updated (all higher-density
            // points are at the same index — should not happen, but defensive).
            if delta[i].is_infinite() {
                delta[i] = 0.0;
            }
        }
    }

    (delta, nearest_higher)
}

/// Select cluster centers either by top-k gamma or by (rho, delta) thresholds.
fn select_centers(
    n: usize,
    rho: &[f64],
    delta: &[f64],
    n_clusters: Option<usize>,
    rho_threshold: f64,
    delta_threshold: f64,
) -> Result<Vec<usize>> {
    match n_clusters {
        Some(k) => {
            if k > n {
                return Err(ClusteringError::InvalidInput(format!(
                    "n_clusters ({}) exceeds number of data points ({})",
                    k, n
                )));
            }
            // Sort by gamma = rho * delta descending; take top-k.
            let mut gamma: Vec<(f64, usize)> = (0..n).map(|i| (rho[i] * delta[i], i)).collect();
            gamma.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            Ok(gamma[..k].iter().map(|(_, idx)| *idx).collect())
        }
        None => {
            // Threshold-based.
            let centers: Vec<usize> = (0..n)
                .filter(|&i| rho[i] > rho_threshold && delta[i] > delta_threshold)
                .collect();
            Ok(centers)
        }
    }
}

/// Assign labels to all points by propagating cluster assignments along the
/// decreasing density order.
fn propagate_labels(
    n: usize,
    centers: &[usize],
    nearest_higher: &[usize],
    order: &[usize],
) -> Vec<i64> {
    let mut labels = vec![-1i64; n];

    // Seed cluster centers with their cluster id.
    for (cluster_id, &center) in centers.iter().enumerate() {
        labels[center] = cluster_id as i64;
    }

    // Propagate: walk density order (highest to lowest). Each unlabeled point
    // inherits the label of its nearest higher-density neighbor which was
    // processed earlier (has higher density, so appeared earlier in `order`).
    for &i in order.iter() {
        if labels[i] == -1 {
            labels[i] = labels[nearest_higher[i]];
        }
    }

    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_two_gaussians() -> Vec<Vec<f64>> {
        // Two tight Gaussian-like clusters: cluster A near (0,0), cluster B near (10,10)
        vec![
            vec![0.0, 0.0],
            vec![0.2, 0.1],
            vec![-0.1, 0.2],
            vec![0.1, -0.1],
            vec![0.3, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 9.9],
            vec![9.9, 10.1],
            vec![10.2, 10.0],
            vec![10.0, 9.8],
        ]
    }

    #[test]
    fn test_density_peaks_two_clusters_n_clusters() {
        let data = make_two_gaussians();
        let result = DensityPeaksAdv::new()
            .with_n_clusters(2)
            .fit(&data)
            .expect("fit should succeed");

        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.labels.len(), 10);
        // All points should be assigned (no -1 remaining for n_clusters mode
        // when two centers exist and all neighbors are reachable).
        for &lbl in &result.labels {
            assert!(lbl >= 0, "label should be non-negative");
        }
    }

    #[test]
    fn test_density_peaks_gaussian_kernel() {
        let data = make_two_gaussians();
        let result = DensityPeaksAdv::new()
            .with_kernel(KernelType::Gaussian)
            .with_n_clusters(2)
            .fit(&data)
            .expect("Gaussian kernel fit should succeed");

        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.rho.len(), 10);
        assert_eq!(result.delta.len(), 10);
    }

    #[test]
    fn test_density_peaks_cutoff_kernel() {
        let data = make_two_gaussians();
        let result = DensityPeaksAdv::new()
            .with_kernel(KernelType::Cutoff)
            .with_n_clusters(2)
            .fit(&data)
            .expect("Cutoff kernel fit should succeed");

        assert_eq!(result.n_clusters, 2);
    }

    #[test]
    fn test_density_peaks_explicit_dc() {
        let data = make_two_gaussians();
        let result = DensityPeaksAdv::new()
            .with_d_c(0.5)
            .with_n_clusters(2)
            .fit(&data)
            .expect("fit with explicit d_c should succeed");

        assert_eq!(result.n_clusters, 2);
    }

    #[test]
    fn test_density_peaks_threshold_mode() {
        let data = make_two_gaussians();
        // With very low thresholds, many centers can be selected.
        let result = DensityPeaksAdv::new()
            .with_thresholds(0.0, 0.0)
            .fit(&data)
            .expect("threshold mode fit should succeed");

        // At least some centers should be selected.
        assert!(result.n_clusters > 0);
    }

    #[test]
    fn test_density_peaks_empty_input() {
        let data: Vec<Vec<f64>> = vec![];
        let err = DensityPeaksAdv::new().fit(&data);
        assert!(err.is_err());
    }

    #[test]
    fn test_density_peaks_invalid_n_clusters() {
        let data = make_two_gaussians();
        let err = DensityPeaksAdv::new().with_n_clusters(100).fit(&data);
        assert!(err.is_err());
    }

    #[test]
    fn test_auto_select_dc() {
        // Test the auto d_c selection with a known dataset
        let data = make_two_gaussians();
        let dists = compute_pairwise_distances(&data);
        let dc = auto_select_dc(&dists, data.len());
        assert!(dc > 0.0, "d_c must be positive");
    }

    #[test]
    fn test_same_cluster_labeling() {
        let data = make_two_gaussians();
        let result = DensityPeaksAdv::new()
            .with_n_clusters(2)
            .fit(&data)
            .expect("fit should succeed");

        // Points 0-4 should share the same label.
        let label_a = result.labels[0];
        for i in 1..5 {
            assert_eq!(result.labels[i], label_a, "cluster A consistency");
        }
        // Points 5-9 should share the same label.
        let label_b = result.labels[5];
        for i in 6..10 {
            assert_eq!(result.labels[i], label_b, "cluster B consistency");
        }
        assert_ne!(label_a, label_b, "two clusters must differ");
    }
}
