//! Internal cluster validation indices
//!
//! This module provides comprehensive internal validation metrics that evaluate
//! clustering quality using only the data and cluster assignments (no ground truth).
//!
//! # Indices
//!
//! - **Silhouette coefficient**: Per-point and average measure of cluster cohesion/separation
//! - **Calinski-Harabasz index**: Variance ratio criterion (higher is better)
//! - **Davies-Bouldin index**: Average similarity ratio (lower is better)
//! - **Dunn index**: Ratio of min inter-cluster to max intra-cluster distance
//! - **Gap statistic**: Comparison with uniform reference distribution via bootstrap
//! - **Elbow method**: Within-cluster sum of squares for varying k

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::random::prelude::*;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// Silhouette coefficient
// ---------------------------------------------------------------------------

/// Per-point silhouette scores.
///
/// For each point i the silhouette is `(b(i) - a(i)) / max(a(i), b(i))` where
/// `a(i)` is the mean intra-cluster distance and `b(i)` the mean nearest-cluster
/// distance.
///
/// # Errors
///
/// Returns an error when data/labels sizes mismatch or fewer than 2 clusters exist.
pub fn silhouette_samples_internal<F>(
    data: ArrayView2<F>,
    labels: ArrayView1<usize>,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = data.nrows();
    if n != labels.len() {
        return Err(ClusteringError::InvalidInput(
            "data rows and labels length must match".into(),
        ));
    }

    let k = unique_count(&labels);
    if k < 2 {
        return Err(ClusteringError::InvalidInput(
            "silhouette requires at least 2 clusters".into(),
        ));
    }

    let n_features = data.ncols();
    let mut scores = Array1::<F>::zeros(n);

    for i in 0..n {
        let ci = labels[i];
        let mut intra_sum = F::zero();
        let mut intra_cnt: usize = 0;

        // per-cluster distance accumulators
        let mut cluster_sum: std::collections::HashMap<usize, F> = std::collections::HashMap::new();
        let mut cluster_cnt: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();

        for j in 0..n {
            if i == j {
                continue;
            }
            let d = euclidean_dist(data.row(i), data.row(j), n_features);
            let cj = labels[j];
            if cj == ci {
                intra_sum = intra_sum + d;
                intra_cnt += 1;
            } else {
                *cluster_sum.entry(cj).or_insert_with(F::zero) =
                    *cluster_sum.entry(cj).or_insert_with(F::zero) + d;
                *cluster_cnt.entry(cj).or_insert(0) += 1;
            }
        }

        let a_i = if intra_cnt > 0 {
            intra_sum / from_usize::<F>(intra_cnt)?
        } else {
            F::zero()
        };

        let b_i = cluster_sum
            .iter()
            .filter_map(|(cj, &s)| {
                let cnt = cluster_cnt.get(cj).copied().unwrap_or(0);
                if cnt > 0 {
                    Some(s / from_usize::<F>(cnt).unwrap_or(F::one()))
                } else {
                    None
                }
            })
            .fold(F::infinity(), |acc, v| if v < acc { v } else { acc });

        let denom = a_i.max(b_i);
        scores[i] = if denom > F::zero() {
            (b_i - a_i) / denom
        } else {
            F::zero()
        };
    }

    Ok(scores)
}

/// Mean silhouette coefficient.
pub fn silhouette_score_internal<F>(data: ArrayView2<F>, labels: ArrayView1<usize>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let s = silhouette_samples_internal(data, labels)?;
    let n = s.len();
    if n == 0 {
        return Ok(F::zero());
    }
    let sum: F = s.iter().fold(F::zero(), |a, &v| a + v);
    Ok(sum / from_usize::<F>(n)?)
}

// ---------------------------------------------------------------------------
// Calinski-Harabasz index
// ---------------------------------------------------------------------------

/// Calinski-Harabasz index (variance ratio criterion).
///
/// `CH = (SSB / (k-1)) / (SSW / (n-k))`
///
/// Higher values indicate better-defined clusters.
pub fn calinski_harabasz_internal<F>(data: ArrayView2<F>, labels: ArrayView1<usize>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = data.nrows();
    let d = data.ncols();
    if n != labels.len() {
        return Err(ClusteringError::InvalidInput(
            "data rows and labels length must match".into(),
        ));
    }

    let (unique, cluster_sizes) = unique_labels_with_sizes(&labels);
    let k = unique.len();
    if k < 2 || k >= n {
        return Err(ClusteringError::InvalidInput(
            "Calinski-Harabasz requires 2 <= k < n".into(),
        ));
    }

    // Global mean
    let mut global_mean = Array1::<F>::zeros(d);
    for i in 0..n {
        for j in 0..d {
            global_mean[j] = global_mean[j] + data[[i, j]];
        }
    }
    let n_f = from_usize::<F>(n)?;
    global_mean.mapv_inplace(|v| v / n_f);

    // Cluster centroids
    let mut centroids = Array2::<F>::zeros((k, d));
    for i in 0..n {
        let ci = label_index(&unique, labels[i]);
        for j in 0..d {
            centroids[[ci, j]] = centroids[[ci, j]] + data[[i, j]];
        }
    }
    for ci in 0..k {
        let sz = from_usize::<F>(cluster_sizes[ci])?;
        for j in 0..d {
            centroids[[ci, j]] = centroids[[ci, j]] / sz;
        }
    }

    // SSB and SSW
    let mut ssb = F::zero();
    for ci in 0..k {
        let mut sq = F::zero();
        for j in 0..d {
            let diff = centroids[[ci, j]] - global_mean[j];
            sq = sq + diff * diff;
        }
        ssb = ssb + from_usize::<F>(cluster_sizes[ci])? * sq;
    }

    let mut ssw = F::zero();
    for i in 0..n {
        let ci = label_index(&unique, labels[i]);
        for j in 0..d {
            let diff = data[[i, j]] - centroids[[ci, j]];
            ssw = ssw + diff * diff;
        }
    }

    if ssw <= F::zero() {
        return Ok(F::infinity());
    }

    let numerator = ssb / from_usize::<F>(k - 1)?;
    let denominator = ssw / from_usize::<F>(n - k)?;

    Ok(numerator / denominator)
}

// ---------------------------------------------------------------------------
// Davies-Bouldin index
// ---------------------------------------------------------------------------

/// Davies-Bouldin index.
///
/// Average maximum ratio of within-cluster scatter to between-cluster distance.
/// Lower values indicate better clustering.
pub fn davies_bouldin_internal<F>(data: ArrayView2<F>, labels: ArrayView1<usize>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = data.nrows();
    let d = data.ncols();
    if n != labels.len() {
        return Err(ClusteringError::InvalidInput(
            "data rows and labels length must match".into(),
        ));
    }

    let (unique, cluster_sizes) = unique_labels_with_sizes(&labels);
    let k = unique.len();
    if k < 2 {
        return Err(ClusteringError::InvalidInput(
            "Davies-Bouldin requires at least 2 clusters".into(),
        ));
    }

    // Cluster centroids
    let mut centroids = Array2::<F>::zeros((k, d));
    for i in 0..n {
        let ci = label_index(&unique, labels[i]);
        for j in 0..d {
            centroids[[ci, j]] = centroids[[ci, j]] + data[[i, j]];
        }
    }
    for ci in 0..k {
        let sz = from_usize::<F>(cluster_sizes[ci])?;
        for j in 0..d {
            centroids[[ci, j]] = centroids[[ci, j]] / sz;
        }
    }

    // Scatter: average distance to centroid for each cluster
    let mut scatter = vec![F::zero(); k];
    for i in 0..n {
        let ci = label_index(&unique, labels[i]);
        let dist = euclidean_dist(data.row(i), centroids.row(ci), d);
        scatter[ci] = scatter[ci] + dist;
    }
    for ci in 0..k {
        if cluster_sizes[ci] > 0 {
            scatter[ci] = scatter[ci] / from_usize::<F>(cluster_sizes[ci])?;
        }
    }

    // DB = (1/k) * sum_i max_{j!=i} (S_i + S_j) / d(c_i, c_j)
    let mut db = F::zero();
    for i in 0..k {
        let mut max_ratio = F::zero();
        for j in 0..k {
            if i == j {
                continue;
            }
            let dist_ij = euclidean_dist(centroids.row(i), centroids.row(j), d);
            if dist_ij > F::zero() {
                let ratio = (scatter[i] + scatter[j]) / dist_ij;
                if ratio > max_ratio {
                    max_ratio = ratio;
                }
            }
        }
        db = db + max_ratio;
    }

    Ok(db / from_usize::<F>(k)?)
}

// ---------------------------------------------------------------------------
// Dunn index
// ---------------------------------------------------------------------------

/// Dunn index.
///
/// `D = min inter-cluster distance / max intra-cluster diameter`.
/// Higher values indicate better clustering.
pub fn dunn_index_internal<F>(data: ArrayView2<F>, labels: ArrayView1<usize>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = data.nrows();
    let d = data.ncols();
    if n != labels.len() {
        return Err(ClusteringError::InvalidInput(
            "data rows and labels length must match".into(),
        ));
    }

    let (unique, _) = unique_labels_with_sizes(&labels);
    let k = unique.len();
    if k < 2 {
        return Err(ClusteringError::InvalidInput(
            "Dunn index requires at least 2 clusters".into(),
        ));
    }

    // Min inter-cluster distance
    let mut min_inter = F::infinity();
    for i in 0..n {
        for j in (i + 1)..n {
            if labels[i] != labels[j] {
                let dist = euclidean_dist(data.row(i), data.row(j), d);
                if dist < min_inter {
                    min_inter = dist;
                }
            }
        }
    }

    // Max intra-cluster diameter
    let mut max_intra = F::zero();
    for &cl in &unique {
        let indices: Vec<usize> = (0..n).filter(|&i| labels[i] == cl).collect();
        for a in 0..indices.len() {
            for b in (a + 1)..indices.len() {
                let dist = euclidean_dist(data.row(indices[a]), data.row(indices[b]), d);
                if dist > max_intra {
                    max_intra = dist;
                }
            }
        }
    }

    if max_intra <= F::zero() {
        return Ok(F::infinity());
    }

    Ok(min_inter / max_intra)
}

// ---------------------------------------------------------------------------
// Gap statistic
// ---------------------------------------------------------------------------

/// Result of the gap statistic computation.
#[derive(Debug, Clone)]
pub struct GapStatisticResult<F: Float> {
    /// Gap values for each k tested.
    pub gap_values: Vec<F>,
    /// Standard deviations of the gap.
    pub gap_std: Vec<F>,
    /// The k values tested.
    pub k_values: Vec<usize>,
    /// Optimal k selected by the gap statistic criterion.
    pub optimal_k: usize,
}

/// Gap statistic with bootstrap reference distribution.
///
/// Compares within-cluster dispersion against that expected under a
/// uniform reference distribution. Uses `n_references` bootstrap samples.
///
/// # Arguments
///
/// * `data` - Input data (n x d)
/// * `k_range` - Range of cluster counts to try (inclusive)
/// * `n_references` - Number of bootstrap reference datasets (default 10)
/// * `seed` - Optional random seed
pub fn gap_statistic<F>(
    data: ArrayView2<F>,
    k_range: (usize, usize),
    n_references: usize,
    seed: Option<u64>,
) -> Result<GapStatisticResult<F>>
where
    F: Float + FromPrimitive + Debug + 'static + Send + Sync + std::iter::Sum + std::fmt::Display,
{
    let n = data.nrows();
    let d = data.ncols();
    if k_range.0 < 1 || k_range.0 > k_range.1 {
        return Err(ClusteringError::InvalidInput(
            "k_range must satisfy 1 <= k_min <= k_max".into(),
        ));
    }

    // Bounding box of data for reference generation
    let mut mins = Array1::<F>::from_elem(d, F::infinity());
    let mut maxs = Array1::<F>::from_elem(d, F::neg_infinity());
    for i in 0..n {
        for j in 0..d {
            let v = data[[i, j]];
            if v < mins[j] {
                mins[j] = v;
            }
            if v > maxs[j] {
                maxs[j] = v;
            }
        }
    }

    let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(seed.unwrap_or(42));
    let n_refs = if n_references == 0 { 10 } else { n_references };

    let mut gap_values = Vec::new();
    let mut gap_std_values = Vec::new();
    let mut k_values = Vec::new();

    for k in k_range.0..=k_range.1 {
        if k >= n {
            break;
        }

        // W_k for observed data
        let labels = run_simple_kmeans(data, k, Some(seed.unwrap_or(42)))?;
        let w_obs = log_within_cluster_dispersion(data, &labels)?;

        // W_k for reference datasets
        let mut ref_logs = Vec::with_capacity(n_refs);
        for _r in 0..n_refs {
            let ref_data = generate_uniform_reference(n, d, &mins, &maxs, &mut rng)?;
            let ref_labels = run_simple_kmeans(ref_data.view(), k, Some(rng.random::<u64>()))?;
            let w_ref = log_within_cluster_dispersion(ref_data.view(), &ref_labels)?;
            ref_logs.push(w_ref);
        }

        let mean_ref: F = ref_logs.iter().fold(F::zero(), |a, &v| a + v) / from_usize::<F>(n_refs)?;
        let gap = mean_ref - w_obs;

        // Standard deviation of reference log dispersions
        let var: F = ref_logs
            .iter()
            .fold(F::zero(), |a, &v| a + (v - mean_ref) * (v - mean_ref))
            / from_usize::<F>(n_refs)?;
        let std_dev = var.sqrt() * (F::one() + F::one() / from_usize::<F>(n_refs)?).sqrt();

        gap_values.push(gap);
        gap_std_values.push(std_dev);
        k_values.push(k);
    }

    // Select optimal k: first k where Gap(k) >= Gap(k+1) - s(k+1)
    let mut optimal_k = k_values.last().copied().unwrap_or(1);
    for idx in 0..(gap_values.len().saturating_sub(1)) {
        if gap_values[idx] >= gap_values[idx + 1] - gap_std_values[idx + 1] {
            optimal_k = k_values[idx];
            break;
        }
    }

    Ok(GapStatisticResult {
        gap_values,
        gap_std: gap_std_values,
        k_values,
        optimal_k,
    })
}

// ---------------------------------------------------------------------------
// Elbow method (WCSS)
// ---------------------------------------------------------------------------

/// Result of the elbow method.
#[derive(Debug, Clone)]
pub struct ElbowResult<F: Float> {
    /// Within-cluster sum of squares for each k.
    pub wcss_values: Vec<F>,
    /// The k values tested.
    pub k_values: Vec<usize>,
    /// Suggested elbow point (using maximum second derivative).
    pub elbow_k: usize,
}

/// Elbow method: compute WCSS for a range of k and detect the elbow.
///
/// The elbow is found via the maximum of the discrete second derivative
/// of the WCSS curve.
///
/// # Arguments
///
/// * `data` - Input data (n x d)
/// * `k_range` - Range of cluster counts (inclusive)
/// * `seed` - Optional random seed
pub fn elbow_method<F>(
    data: ArrayView2<F>,
    k_range: (usize, usize),
    seed: Option<u64>,
) -> Result<ElbowResult<F>>
where
    F: Float + FromPrimitive + Debug + 'static + Send + Sync + std::iter::Sum + std::fmt::Display,
{
    let n = data.nrows();
    if k_range.0 < 1 || k_range.0 > k_range.1 {
        return Err(ClusteringError::InvalidInput(
            "k_range must satisfy 1 <= k_min <= k_max".into(),
        ));
    }

    let mut wcss_values = Vec::new();
    let mut k_values = Vec::new();

    for k in k_range.0..=k_range.1 {
        if k >= n {
            break;
        }
        let labels = run_simple_kmeans(data, k, seed)?;
        let w = within_cluster_sum_of_squares(data, &labels)?;
        wcss_values.push(w);
        k_values.push(k);
    }

    // Detect elbow via max second derivative
    let elbow_k = detect_elbow(&wcss_values, &k_values);

    Ok(ElbowResult {
        wcss_values,
        k_values,
        elbow_k,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn from_usize<F: Float + FromPrimitive>(v: usize) -> Result<F> {
    F::from(v).ok_or_else(|| ClusteringError::ComputationError("float conversion failed".into()))
}

fn euclidean_dist<F: Float>(a: ArrayView1<F>, b: ArrayView1<F>, d: usize) -> F {
    let mut sq = F::zero();
    for j in 0..d {
        let diff = a[j] - b[j];
        sq = sq + diff * diff;
    }
    sq.sqrt()
}

fn unique_count(labels: &ArrayView1<usize>) -> usize {
    let mut seen = std::collections::HashSet::new();
    for &l in labels.iter() {
        seen.insert(l);
    }
    seen.len()
}

fn unique_labels_with_sizes(labels: &ArrayView1<usize>) -> (Vec<usize>, Vec<usize>) {
    let mut map: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for &l in labels.iter() {
        *map.entry(l).or_insert(0) += 1;
    }
    let unique: Vec<usize> = map.keys().copied().collect();
    let sizes: Vec<usize> = unique.iter().map(|k| map[k]).collect();
    (unique, sizes)
}

fn label_index(unique: &[usize], label: usize) -> usize {
    unique.iter().position(|&u| u == label).unwrap_or(0)
}

/// Very simple k-means for internal use in gap statistic / elbow.
fn run_simple_kmeans<F>(data: ArrayView2<F>, k: usize, seed: Option<u64>) -> Result<Array1<usize>>
where
    F: Float + FromPrimitive + Debug + 'static + Send + Sync + std::iter::Sum + std::fmt::Display,
{
    let n = data.nrows();
    let d = data.ncols();
    if k == 0 {
        return Err(ClusteringError::InvalidInput("k must be > 0".into()));
    }
    if k >= n {
        // Each point is its own cluster
        return Ok(Array1::from_vec((0..n).collect()));
    }

    let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(seed.unwrap_or(42));

    // K-means++ initialisation
    let mut centroids = Array2::<F>::zeros((k, d));
    let first_idx: usize = rng.random_range(0..n);
    centroids.row_mut(0).assign(&data.row(first_idx));

    for c in 1..k {
        let mut dists = Array1::<F>::zeros(n);
        for i in 0..n {
            let mut min_d = F::infinity();
            for prev in 0..c {
                let dist = euclidean_dist(data.row(i), centroids.row(prev), d);
                if dist < min_d {
                    min_d = dist;
                }
            }
            dists[i] = min_d * min_d;
        }
        let total: F = dists.iter().fold(F::zero(), |a, &v| a + v);
        if total <= F::zero() {
            // All points are at centroid positions already
            centroids.row_mut(c).assign(&data.row(c.min(n - 1)));
            continue;
        }
        let r: f64 = rng.random::<f64>();
        let threshold = F::from(r).unwrap_or(F::zero()) * total;
        let mut cumsum = F::zero();
        let mut chosen = 0;
        for i in 0..n {
            cumsum = cumsum + dists[i];
            if cumsum >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.row_mut(c).assign(&data.row(chosen));
    }

    // Lloyd iterations
    let max_iter = 100;
    let mut labels = Array1::<usize>::zeros(n);
    for _iter in 0..max_iter {
        let mut changed = false;
        // Assign
        for i in 0..n {
            let mut best_c = 0;
            let mut best_d = F::infinity();
            for c in 0..k {
                let dist = euclidean_dist(data.row(i), centroids.row(c), d);
                if dist < best_d {
                    best_d = dist;
                    best_c = c;
                }
            }
            if labels[i] != best_c {
                labels[i] = best_c;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update centroids
        let mut new_centroids = Array2::<F>::zeros((k, d));
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = labels[i];
            counts[c] += 1;
            for j in 0..d {
                new_centroids[[c, j]] = new_centroids[[c, j]] + data[[i, j]];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let sz = from_usize::<F>(counts[c])?;
                for j in 0..d {
                    new_centroids[[c, j]] = new_centroids[[c, j]] / sz;
                }
            }
        }
        centroids = new_centroids;
    }

    Ok(labels)
}

fn within_cluster_sum_of_squares<F>(data: ArrayView2<F>, labels: &Array1<usize>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = data.nrows();
    let d = data.ncols();
    let (unique, sizes) = unique_labels_with_sizes(&labels.view());
    let k = unique.len();

    let mut centroids = Array2::<F>::zeros((k, d));
    for i in 0..n {
        let ci = label_index(&unique, labels[i]);
        for j in 0..d {
            centroids[[ci, j]] = centroids[[ci, j]] + data[[i, j]];
        }
    }
    for ci in 0..k {
        let sz = from_usize::<F>(sizes[ci])?;
        for j in 0..d {
            centroids[[ci, j]] = centroids[[ci, j]] / sz;
        }
    }

    let mut wcss = F::zero();
    for i in 0..n {
        let ci = label_index(&unique, labels[i]);
        for j in 0..d {
            let diff = data[[i, j]] - centroids[[ci, j]];
            wcss = wcss + diff * diff;
        }
    }
    Ok(wcss)
}

fn log_within_cluster_dispersion<F>(data: ArrayView2<F>, labels: &Array1<usize>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let w = within_cluster_sum_of_squares(data, labels)?;
    if w <= F::zero() {
        // log(0) is -inf; clamp to a small positive value
        let eps = F::from(1e-30).unwrap_or(F::epsilon());
        Ok(eps.ln())
    } else {
        Ok(w.ln())
    }
}

fn generate_uniform_reference<F>(
    n: usize,
    d: usize,
    mins: &Array1<F>,
    maxs: &Array1<F>,
    rng: &mut scirs2_core::random::rngs::StdRng,
) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let mut data = Array2::<F>::zeros((n, d));
    for i in 0..n {
        for j in 0..d {
            let r: f64 = rng.random::<f64>();
            let range = maxs[j] - mins[j];
            data[[i, j]] = mins[j] + F::from(r).unwrap_or(F::zero()) * range;
        }
    }
    Ok(data)
}

fn detect_elbow<F: Float + FromPrimitive>(wcss: &[F], k_values: &[usize]) -> usize {
    if wcss.len() < 3 {
        return k_values.first().copied().unwrap_or(1);
    }

    // Second derivative: d2[i] = wcss[i+1] - 2*wcss[i] + wcss[i-1]
    let two = F::from(2.0).unwrap_or(F::one() + F::one());
    let mut best_idx = 0;
    let mut best_d2 = F::neg_infinity();
    for i in 1..(wcss.len() - 1) {
        let d2 = wcss[i - 1] - two * wcss[i] + wcss[i + 1];
        if d2 > best_d2 {
            best_d2 = d2;
            best_idx = i;
        }
    }

    k_values[best_idx]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr1, Array2};

    fn well_separated_data() -> (Array2<f64>, Array1<usize>) {
        let data = Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.0, 5.0, 5.2,
            ],
        )
        .expect("test data");
        let labels = arr1(&[0, 0, 0, 0, 1, 1, 1, 1]);
        (data, labels)
    }

    #[test]
    fn test_silhouette_samples_well_separated() {
        let (data, labels) = well_separated_data();
        let scores =
            silhouette_samples_internal(data.view(), labels.view()).expect("silhouette samples");
        for &s in scores.iter() {
            assert!(s > 0.5, "silhouette should be high: {}", s);
        }
    }

    #[test]
    fn test_silhouette_score_well_separated() {
        let (data, labels) = well_separated_data();
        let score: f64 =
            silhouette_score_internal(data.view(), labels.view()).expect("silhouette score");
        assert!(score > 0.8, "mean silhouette: {}", score);
    }

    #[test]
    fn test_silhouette_error_single_cluster() {
        let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
            .expect("data");
        let labels = arr1(&[0, 0, 0, 0]);
        assert!(silhouette_score_internal::<f64>(data.view(), labels.view()).is_err());
    }

    #[test]
    fn test_calinski_harabasz_well_separated() {
        let (data, labels) = well_separated_data();
        let ch: f64 = calinski_harabasz_internal(data.view(), labels.view()).expect("ch index");
        assert!(ch > 10.0, "CH should be high: {}", ch);
    }

    #[test]
    fn test_calinski_harabasz_error_single_cluster() {
        let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
            .expect("data");
        let labels = arr1(&[0, 0, 0, 0]);
        assert!(calinski_harabasz_internal::<f64>(data.view(), labels.view()).is_err());
    }

    #[test]
    fn test_davies_bouldin_well_separated() {
        let (data, labels) = well_separated_data();
        let db: f64 = davies_bouldin_internal(data.view(), labels.view()).expect("db index");
        assert!(db < 0.5, "DB should be low: {}", db);
        assert!(db >= 0.0, "DB should be non-negative");
    }

    #[test]
    fn test_dunn_index_well_separated() {
        let (data, labels) = well_separated_data();
        let dunn: f64 = dunn_index_internal(data.view(), labels.view()).expect("dunn index");
        assert!(dunn > 1.0, "Dunn should be high: {}", dunn);
    }

    #[test]
    fn test_dunn_index_error_single_cluster() {
        let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
            .expect("data");
        let labels = arr1(&[0, 0, 0, 0]);
        assert!(dunn_index_internal::<f64>(data.view(), labels.view()).is_err());
    }

    #[test]
    fn test_gap_statistic() {
        let (data, _) = well_separated_data();
        let result = gap_statistic(data.view(), (1, 4), 5, Some(123)).expect("gap statistic");
        assert!(!result.gap_values.is_empty());
        assert!(!result.k_values.is_empty());
        assert!(result.optimal_k >= 1);
    }

    #[test]
    fn test_elbow_method() {
        let (data, _) = well_separated_data();
        let result = elbow_method(data.view(), (1, 5), Some(42)).expect("elbow method");
        assert!(!result.wcss_values.is_empty());
        assert!(result.elbow_k >= 1);
        // WCSS should decrease with increasing k
        for i in 1..result.wcss_values.len() {
            assert!(
                result.wcss_values[i] <= result.wcss_values[i - 1] + 1e-6,
                "WCSS should be non-increasing"
            );
        }
    }

    #[test]
    fn test_elbow_three_clusters() {
        let data = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.0, 5.0, 5.2,
                10.0, 0.0, 10.1, 0.1, 10.2, 0.0, 10.0, 0.2,
            ],
        )
        .expect("data");
        let result = elbow_method(data.view(), (1, 6), Some(99)).expect("elbow");
        assert!(result.elbow_k >= 2 && result.elbow_k <= 4);
    }

    #[test]
    fn test_gap_statistic_optimal_k() {
        let data = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.0, 5.0, 5.2,
                10.0, 0.0, 10.1, 0.1, 10.2, 0.0, 10.0, 0.2,
            ],
        )
        .expect("data");
        let result = gap_statistic(data.view(), (1, 6), 10, Some(77)).expect("gap");
        assert!(result.optimal_k >= 1 && result.optimal_k <= 6);
    }

    #[test]
    fn test_run_simple_kmeans() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 5.0, 5.0, 5.1, 5.1, 5.2, 5.0],
        )
        .expect("data");
        let labels = run_simple_kmeans(data.view(), 2, Some(42)).expect("kmeans");
        assert_eq!(labels.len(), 6);
        // Points in same cluster should share labels
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }
}
