//! Structured dataset generators
//!
//! Provides generators for structured data matrices including bicluster
//! patterns, symmetric positive definite matrices, and sparse coded signals.
//! These are useful for testing biclustering algorithms, covariance estimation,
//! and dictionary learning methods.

use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution;

/// Helper to create an RNG from an optional seed
fn create_rng(randomseed: Option<u64>) -> StdRng {
    match randomseed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = thread_rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    }
}

/// Generate a bicluster data matrix
///
/// Creates a matrix with embedded bicluster structure: rectangular regions
/// of elevated values in an otherwise noisy background. Useful for testing
/// biclustering algorithms (e.g., Spectral Co-Clustering).
///
/// # Arguments
///
/// * `shape` - (n_rows, n_cols) of the output matrix
/// * `n_clusters` - Number of non-overlapping biclusters to embed
/// * `noise` - Standard deviation of background Gaussian noise
/// * `shuffle` - Whether to shuffle rows and columns
/// * `random_state` - Optional random seed
///
/// # Returns
///
/// A `Dataset` whose `data` is the bicluster matrix and `target` contains
/// the row cluster assignments.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::structured::make_biclusters;
///
/// let ds = make_biclusters((100, 50), 3, 0.5, true, Some(42)).expect("ok");
/// assert_eq!(ds.n_samples(), 100);
/// assert_eq!(ds.n_features(), 50);
/// ```
pub fn make_biclusters(
    shape: (usize, usize),
    n_clusters: usize,
    noise: f64,
    shuffle: bool,
    random_state: Option<u64>,
) -> Result<Dataset> {
    let (n_rows, n_cols) = shape;
    if n_rows == 0 || n_cols == 0 {
        return Err(DatasetsError::InvalidFormat(
            "shape dimensions must be > 0".to_string(),
        ));
    }
    if n_clusters == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_clusters must be > 0".to_string(),
        ));
    }
    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0".to_string(),
        ));
    }

    let mut rng = create_rng(random_state);
    let normal = scirs2_core::random::Normal::new(0.0, noise.max(1e-30)).map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create normal dist: {e}"))
    })?;

    // Start with noise background
    let mut data = Array2::zeros((n_rows, n_cols));
    if noise > 0.0 {
        for i in 0..n_rows {
            for j in 0..n_cols {
                data[[i, j]] = normal.sample(&mut rng);
            }
        }
    }

    let mut row_labels = Array1::zeros(n_rows);

    // Divide rows and columns into n_clusters groups
    let rows_per_cluster = n_rows / n_clusters;
    let cols_per_cluster = n_cols / n_clusters;

    for c in 0..n_clusters {
        let row_start = c * rows_per_cluster;
        let row_end = if c == n_clusters - 1 {
            n_rows
        } else {
            (c + 1) * rows_per_cluster
        };
        let col_start = c * cols_per_cluster;
        let col_end = if c == n_clusters - 1 {
            n_cols
        } else {
            (c + 1) * cols_per_cluster
        };

        // Generate a random bicluster value (elevated signal)
        let signal = 5.0 + 3.0 * rng.random::<f64>();

        for i in row_start..row_end {
            row_labels[i] = c as f64;
            for j in col_start..col_end {
                data[[i, j]] += signal;
            }
        }
    }

    // Shuffle if requested
    if shuffle {
        // Shuffle rows
        for i in (1..n_rows).rev() {
            let j = rng.random_range(0..=i);
            if i != j {
                for col in 0..n_cols {
                    let tmp = data[[i, col]];
                    data[[i, col]] = data[[j, col]];
                    data[[j, col]] = tmp;
                }
                let tmp = row_labels[i];
                row_labels[i] = row_labels[j];
                row_labels[j] = tmp;
            }
        }
        // Shuffle columns
        for j in (1..n_cols).rev() {
            let k = rng.random_range(0..=j);
            if j != k {
                for i in 0..n_rows {
                    let tmp = data[[i, j]];
                    data[[i, j]] = data[[i, k]];
                    data[[i, k]] = tmp;
                }
            }
        }
    }

    let feature_names: Vec<String> = (0..n_cols).map(|j| format!("col_{j}")).collect();

    let dataset = Dataset::new(data, Some(row_labels))
        .with_featurenames(feature_names)
        .with_description(format!(
            "Bicluster matrix ({n_rows}x{n_cols}) with {n_clusters} embedded biclusters"
        ))
        .with_metadata("generator", "make_biclusters")
        .with_metadata("n_clusters", &n_clusters.to_string())
        .with_metadata("noise", &noise.to_string());

    Ok(dataset)
}

/// Generate a checkerboard bicluster pattern
///
/// Creates a matrix with a checkerboard pattern of alternating high and low
/// values. This produces a bicluster structure where adjacent blocks have
/// different mean values.
///
/// # Arguments
///
/// * `shape` - (n_rows, n_cols) of the output matrix
/// * `n_clusters` - (n_row_clusters, n_col_clusters)
/// * `noise` - Standard deviation of Gaussian noise
/// * `random_state` - Optional random seed
///
/// # Returns
///
/// A `Dataset` with the checkerboard matrix
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::structured::make_checkerboard;
///
/// let ds = make_checkerboard((100, 80), (4, 5), 0.5, Some(42)).expect("ok");
/// assert_eq!(ds.n_samples(), 100);
/// assert_eq!(ds.n_features(), 80);
/// ```
pub fn make_checkerboard(
    shape: (usize, usize),
    n_clusters: (usize, usize),
    noise: f64,
    random_state: Option<u64>,
) -> Result<Dataset> {
    let (n_rows, n_cols) = shape;
    let (n_row_clusters, n_col_clusters) = n_clusters;

    if n_rows == 0 || n_cols == 0 {
        return Err(DatasetsError::InvalidFormat(
            "shape dimensions must be > 0".to_string(),
        ));
    }
    if n_row_clusters == 0 || n_col_clusters == 0 {
        return Err(DatasetsError::InvalidFormat(
            "cluster dimensions must be > 0".to_string(),
        ));
    }
    if noise < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "noise must be >= 0".to_string(),
        ));
    }

    let mut rng = create_rng(random_state);
    let normal = scirs2_core::random::Normal::new(0.0, noise.max(1e-30)).map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create normal dist: {e}"))
    })?;

    let mut data = Array2::zeros((n_rows, n_cols));
    let mut row_labels = Array1::zeros(n_rows);

    let rows_per_cluster = n_rows / n_row_clusters;
    let cols_per_cluster = n_cols / n_col_clusters;

    for i in 0..n_rows {
        let row_cluster = (i / rows_per_cluster).min(n_row_clusters - 1);
        row_labels[i] = row_cluster as f64;

        for j in 0..n_cols {
            let col_cluster = (j / cols_per_cluster).min(n_col_clusters - 1);

            // Checkerboard: value depends on parity of (row_cluster + col_cluster)
            let base = if (row_cluster + col_cluster) % 2 == 0 {
                5.0
            } else {
                -5.0
            };

            data[[i, j]] = base;
            if noise > 0.0 {
                data[[i, j]] += normal.sample(&mut rng);
            }
        }
    }

    let feature_names: Vec<String> = (0..n_cols).map(|j| format!("col_{j}")).collect();

    let dataset = Dataset::new(data, Some(row_labels))
        .with_featurenames(feature_names)
        .with_description(format!(
            "Checkerboard bicluster matrix ({n_rows}x{n_cols}) with ({n_row_clusters}x{n_col_clusters}) pattern"
        ))
        .with_metadata("generator", "make_checkerboard")
        .with_metadata("n_row_clusters", &n_row_clusters.to_string())
        .with_metadata("n_col_clusters", &n_col_clusters.to_string())
        .with_metadata("noise", &noise.to_string());

    Ok(dataset)
}

/// Generate a random symmetric positive definite (SPD) matrix
///
/// Creates a dense SPD matrix by generating A = B * B^T + alpha * I where
/// B has entries from a standard normal distribution and alpha ensures strong
/// positive definiteness.
///
/// # Arguments
///
/// * `n_dim` - Matrix dimension
/// * `random_state` - Optional random seed
///
/// # Returns
///
/// A `Dataset` whose `data` is the SPD matrix. Target contains the diagonal.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::structured::make_spd_matrix;
///
/// let ds = make_spd_matrix(20, Some(42)).expect("ok");
/// assert_eq!(ds.n_samples(), 20);
/// assert_eq!(ds.n_features(), 20);
/// ```
pub fn make_spd_matrix(n_dim: usize, random_state: Option<u64>) -> Result<Dataset> {
    if n_dim == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_dim must be > 0".to_string(),
        ));
    }

    let mut rng = create_rng(random_state);
    let normal = scirs2_core::random::Normal::new(0.0, 1.0).map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create normal dist: {e}"))
    })?;

    // Generate random matrix B
    let mut b_mat = Array2::zeros((n_dim, n_dim));
    for i in 0..n_dim {
        for j in 0..n_dim {
            b_mat[[i, j]] = normal.sample(&mut rng);
        }
    }

    // Compute A = B * B^T
    let mut matrix = Array2::zeros((n_dim, n_dim));
    for i in 0..n_dim {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..n_dim {
                sum += b_mat[[i, k]] * b_mat[[j, k]];
            }
            matrix[[i, j]] = sum;
            matrix[[j, i]] = sum;
        }
    }

    // Add regularization for strong positive definiteness
    let alpha = 0.1;
    for i in 0..n_dim {
        matrix[[i, i]] += alpha;
    }

    let diag = Array1::from_vec((0..n_dim).map(|i| matrix[[i, i]]).collect());
    let feature_names: Vec<String> = (0..n_dim).map(|j| format!("col_{j}")).collect();

    let dataset = Dataset::new(matrix, Some(diag))
        .with_featurenames(feature_names)
        .with_description(format!(
            "Random SPD matrix ({n_dim}x{n_dim}), A = B*B^T + {alpha}*I"
        ))
        .with_metadata("generator", "make_spd_matrix")
        .with_metadata("n_dim", &n_dim.to_string());

    Ok(dataset)
}

/// Generate a sparse symmetric positive definite matrix
///
/// Similar to `make_spd_matrix` but the resulting matrix has a controlled
/// sparsity pattern. The matrix is constructed by generating a sparse
/// Cholesky factor L and computing A = L * L^T + alpha * I.
///
/// # Arguments
///
/// * `n_dim` - Matrix dimension
/// * `alpha` - Diagonal regularization strength (>= 0)
/// * `density` - Sparsity of the Cholesky factor, in (0, 1]
/// * `random_state` - Optional random seed
///
/// # Returns
///
/// A `Dataset` whose `data` is the sparse SPD matrix
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::structured::make_sparse_spd_matrix;
///
/// let ds = make_sparse_spd_matrix(30, 0.1, 0.3, Some(42)).expect("ok");
/// assert_eq!(ds.n_samples(), 30);
/// assert_eq!(ds.n_features(), 30);
/// ```
pub fn make_sparse_spd_matrix(
    n_dim: usize,
    alpha: f64,
    density: f64,
    random_state: Option<u64>,
) -> Result<Dataset> {
    if n_dim == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_dim must be > 0".to_string(),
        ));
    }
    if alpha < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "alpha must be >= 0".to_string(),
        ));
    }
    if density <= 0.0 || density > 1.0 {
        return Err(DatasetsError::InvalidFormat(
            "density must be in (0, 1]".to_string(),
        ));
    }

    let mut rng = create_rng(random_state);
    let normal = scirs2_core::random::Normal::new(0.0, 1.0).map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create normal dist: {e}"))
    })?;
    let uniform = scirs2_core::random::Uniform::new(0.0, 1.0).map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create uniform dist: {e}"))
    })?;

    // Build sparse lower-triangular Cholesky factor
    let mut lower = Array2::zeros((n_dim, n_dim));
    for i in 0..n_dim {
        let v: f64 = normal.sample(&mut rng);
        lower[[i, i]] = 1.0 + v.abs(); // Positive diagonal
    }
    for i in 1..n_dim {
        for j in 0..i {
            if uniform.sample(&mut rng) < density {
                lower[[i, j]] = normal.sample(&mut rng);
            }
        }
    }

    // Compute A = L * L^T
    let mut matrix = Array2::zeros((n_dim, n_dim));
    for i in 0..n_dim {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..=j {
                sum += lower[[i, k]] * lower[[j, k]];
            }
            matrix[[i, j]] = sum;
            matrix[[j, i]] = sum;
        }
    }

    // Add regularization
    for i in 0..n_dim {
        matrix[[i, i]] += alpha;
    }

    // Count non-zeros
    let eps = 1e-14;
    let nnz: usize = matrix.iter().filter(|&&v: &&f64| v.abs() > eps).count();

    let diag = Array1::from_vec((0..n_dim).map(|i| matrix[[i, i]]).collect());
    let feature_names: Vec<String> = (0..n_dim).map(|j| format!("col_{j}")).collect();

    let dataset = Dataset::new(matrix, Some(diag))
        .with_featurenames(feature_names)
        .with_description(format!(
            "Sparse SPD matrix ({n_dim}x{n_dim}, factor density={density})"
        ))
        .with_metadata("generator", "make_sparse_spd_matrix")
        .with_metadata("n_dim", &n_dim.to_string())
        .with_metadata("alpha", &alpha.to_string())
        .with_metadata("density", &density.to_string())
        .with_metadata("nnz", &nnz.to_string());

    Ok(dataset)
}

/// Generate a sparse coded signal for dictionary learning
///
/// Creates a signal matrix Y = D * X where D is a dictionary matrix and X
/// is a sparse code matrix. Useful for testing dictionary learning and
/// sparse coding algorithms (e.g., OMP, Lasso).
///
/// # Arguments
///
/// * `n_samples` - Number of signal samples (columns of Y)
/// * `n_features` - Dimensionality of the signals (rows of Y / rows of D)
/// * `n_components` - Number of dictionary atoms (columns of D / rows of X)
/// * `n_nonzero_coefs` - Number of non-zero coefficients per signal in X
/// * `random_state` - Optional random seed
///
/// # Returns
///
/// A `Dataset` whose `data` field is the signal matrix Y (n_samples x n_features),
/// and whose `target` field stores the flattened dictionary D (n_features * n_components).
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::structured::make_sparse_coded_signal;
///
/// let ds = make_sparse_coded_signal(50, 20, 10, 3, Some(42)).expect("ok");
/// assert_eq!(ds.n_samples(), 50);
/// assert_eq!(ds.n_features(), 20);
/// ```
pub fn make_sparse_coded_signal(
    n_samples: usize,
    n_features: usize,
    n_components: usize,
    n_nonzero_coefs: usize,
    random_state: Option<u64>,
) -> Result<Dataset> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_samples must be > 0".to_string(),
        ));
    }
    if n_features == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_features must be > 0".to_string(),
        ));
    }
    if n_components == 0 {
        return Err(DatasetsError::InvalidFormat(
            "n_components must be > 0".to_string(),
        ));
    }
    if n_nonzero_coefs == 0 || n_nonzero_coefs > n_components {
        return Err(DatasetsError::InvalidFormat(format!(
            "n_nonzero_coefs ({n_nonzero_coefs}) must be in [1, n_components ({n_components})]"
        )));
    }

    let mut rng = create_rng(random_state);
    let normal = scirs2_core::random::Normal::new(0.0, 1.0).map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create normal dist: {e}"))
    })?;

    // Generate random dictionary D (n_features x n_components)
    let mut dictionary = Array2::zeros((n_features, n_components));
    for j in 0..n_components {
        // Generate random atom
        for i in 0..n_features {
            dictionary[[i, j]] = normal.sample(&mut rng);
        }
        // Normalize each atom to unit L2 norm
        let norm: f64 = (0..n_features)
            .map(|i| dictionary[[i, j]] * dictionary[[i, j]])
            .sum::<f64>()
            .sqrt();
        if norm > 1e-15 {
            for i in 0..n_features {
                dictionary[[i, j]] /= norm;
            }
        }
    }

    // Generate sparse code matrix X (n_components x n_samples)
    let mut code = Array2::zeros((n_components, n_samples));
    for s in 0..n_samples {
        // Randomly select n_nonzero_coefs indices
        let mut indices: Vec<usize> = (0..n_components).collect();
        // Fisher-Yates partial shuffle to select n_nonzero_coefs items
        for k in 0..n_nonzero_coefs {
            let swap_idx = rng.random_range(k..n_components);
            indices.swap(k, swap_idx);
        }
        // Set non-zero coefficients
        for k in 0..n_nonzero_coefs {
            code[[indices[k], s]] = normal.sample(&mut rng);
        }
    }

    // Compute Y = D * X  →  Y^T = X^T * D^T, so data[s, f] = sum_k code[k,s] * dict[f,k]
    let mut data = Array2::zeros((n_samples, n_features));
    for s in 0..n_samples {
        for f in 0..n_features {
            let mut val = 0.0;
            for k in 0..n_components {
                val += code[[k, s]] * dictionary[[f, k]];
            }
            data[[s, f]] = val;
        }
    }

    // Flatten dictionary into target for storage
    let mut dict_flat = Vec::with_capacity(n_features * n_components);
    for i in 0..n_features {
        for j in 0..n_components {
            dict_flat.push(dictionary[[i, j]]);
        }
    }
    let target = Array1::from_vec(dict_flat);

    let feature_names: Vec<String> = (0..n_features).map(|j| format!("signal_{j}")).collect();

    let dataset = Dataset::new(data, Some(target))
        .with_featurenames(feature_names)
        .with_description(format!(
            "Sparse coded signal: {n_samples} samples, {n_features} features, \
             {n_components} dictionary atoms, {n_nonzero_coefs} non-zero coefficients per sample. \
             Target stores the flattened dictionary ({n_features}x{n_components})."
        ))
        .with_metadata("generator", "make_sparse_coded_signal")
        .with_metadata("n_components", &n_components.to_string())
        .with_metadata("n_nonzero_coefs", &n_nonzero_coefs.to_string());

    Ok(dataset)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // make_biclusters tests
    // =========================================================================

    #[test]
    fn test_biclusters_basic() {
        let ds = make_biclusters((100, 50), 3, 0.5, false, Some(42)).expect("ok");
        assert_eq!(ds.n_samples(), 100);
        assert_eq!(ds.n_features(), 50);
        assert!(ds.target.is_some());
    }

    #[test]
    fn test_biclusters_structure_no_noise() {
        let ds = make_biclusters((90, 60), 3, 0.0, false, Some(42)).expect("ok");
        // With no noise and no shuffle, biclusters form block diagonal
        // First 30 rows, first 20 cols should have elevated signal
        let mut block_sum = 0.0;
        for i in 0..30 {
            for j in 0..20 {
                block_sum += ds.data[[i, j]];
            }
        }
        let block_mean: f64 = block_sum / (30.0 * 20.0);
        // Off-block should be zero (no noise)
        let mut off_block_sum = 0.0;
        for i in 0..30 {
            for j in 20..40 {
                off_block_sum += ds.data[[i, j]];
            }
        }
        let off_block_mean: f64 = off_block_sum / (30.0 * 20.0);

        assert!(
            block_mean > 4.0,
            "Block mean should be elevated, got {block_mean}"
        );
        assert!(
            off_block_mean.abs() < 0.01,
            "Off-block mean should be ~0, got {off_block_mean}"
        );
    }

    #[test]
    fn test_biclusters_labels() {
        let ds = make_biclusters((60, 30), 3, 0.0, false, Some(42)).expect("ok");
        let target = ds.target.as_ref().expect("target");
        // First 20 rows → label 0, next 20 → label 1, last 20 → label 2
        for i in 0..20 {
            assert_eq!(target[i], 0.0);
        }
        for i in 20..40 {
            assert_eq!(target[i], 1.0);
        }
        for i in 40..60 {
            assert_eq!(target[i], 2.0);
        }
    }

    #[test]
    fn test_biclusters_shuffle() {
        let ds = make_biclusters((60, 30), 3, 0.0, true, Some(42)).expect("ok");
        let target = ds.target.as_ref().expect("target");
        // After shuffle, labels should not be in order
        let mut is_sorted = true;
        for i in 1..60 {
            if target[i] < target[i - 1] {
                is_sorted = false;
                break;
            }
        }
        assert!(!is_sorted, "Labels should be shuffled");
    }

    #[test]
    fn test_biclusters_validation() {
        assert!(make_biclusters((0, 10), 2, 0.0, false, None).is_err());
        assert!(make_biclusters((10, 0), 2, 0.0, false, None).is_err());
        assert!(make_biclusters((10, 10), 0, 0.0, false, None).is_err());
        assert!(make_biclusters((10, 10), 2, -1.0, false, None).is_err());
    }

    // =========================================================================
    // make_checkerboard tests
    // =========================================================================

    #[test]
    fn test_checkerboard_basic() {
        let ds = make_checkerboard((100, 80), (4, 5), 0.0, Some(42)).expect("ok");
        assert_eq!(ds.n_samples(), 100);
        assert_eq!(ds.n_features(), 80);
    }

    #[test]
    fn test_checkerboard_pattern_noiseless() {
        let ds = make_checkerboard((40, 40), (2, 2), 0.0, Some(42)).expect("ok");
        // Top-left block (even+even): should be +5
        assert!(
            (ds.data[[0, 0]] - 5.0).abs() < 1e-10,
            "Top-left should be +5, got {}",
            ds.data[[0, 0]]
        );
        // Top-right block (even+odd): should be -5
        assert!(
            (ds.data[[0, 20]] - (-5.0)).abs() < 1e-10,
            "Top-right should be -5, got {}",
            ds.data[[0, 20]]
        );
        // Bottom-left block (odd+even): should be -5
        assert!(
            (ds.data[[20, 0]] - (-5.0)).abs() < 1e-10,
            "Bottom-left should be -5, got {}",
            ds.data[[20, 0]]
        );
        // Bottom-right block (odd+odd): should be +5
        assert!(
            (ds.data[[20, 20]] - 5.0).abs() < 1e-10,
            "Bottom-right should be +5, got {}",
            ds.data[[20, 20]]
        );
    }

    #[test]
    fn test_checkerboard_validation() {
        assert!(make_checkerboard((0, 10), (2, 2), 0.0, None).is_err());
        assert!(make_checkerboard((10, 10), (0, 2), 0.0, None).is_err());
        assert!(make_checkerboard((10, 10), (2, 2), -1.0, None).is_err());
    }

    // =========================================================================
    // make_spd_matrix tests
    // =========================================================================

    #[test]
    fn test_spd_matrix_basic() {
        let ds = make_spd_matrix(20, Some(42)).expect("ok");
        assert_eq!(ds.n_samples(), 20);
        assert_eq!(ds.n_features(), 20);
    }

    #[test]
    fn test_spd_matrix_symmetric() {
        let n = 15;
        let ds = make_spd_matrix(n, Some(42)).expect("ok");
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (ds.data[[i, j]] - ds.data[[j, i]]).abs() < 1e-10,
                    "Not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_spd_matrix_positive_definite() {
        let n = 10;
        let ds = make_spd_matrix(n, Some(42)).expect("ok");
        let mut rng = StdRng::seed_from_u64(999);
        let normal = scirs2_core::random::Normal::new(0.0, 1.0).expect("ok");

        for _ in 0..20 {
            let x: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();
            let mut xtax = 0.0;
            for i in 0..n {
                for j in 0..n {
                    xtax += x[i] * ds.data[[i, j]] * x[j];
                }
            }
            assert!(xtax > 0.0, "x^T A x should be positive, got {xtax}");
        }
    }

    #[test]
    fn test_spd_matrix_validation() {
        assert!(make_spd_matrix(0, None).is_err());
    }

    // =========================================================================
    // make_sparse_spd_matrix tests
    // =========================================================================

    #[test]
    fn test_sparse_spd_matrix_basic() {
        let ds = make_sparse_spd_matrix(30, 0.1, 0.3, Some(42)).expect("ok");
        assert_eq!(ds.n_samples(), 30);
        assert_eq!(ds.n_features(), 30);
    }

    #[test]
    fn test_sparse_spd_matrix_symmetric() {
        let n = 20;
        let ds = make_sparse_spd_matrix(n, 0.1, 0.2, Some(42)).expect("ok");
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (ds.data[[i, j]] - ds.data[[j, i]]).abs() < 1e-10,
                    "Not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_sparse_spd_matrix_positive_definite() {
        let n = 10;
        let ds = make_sparse_spd_matrix(n, 0.5, 0.3, Some(42)).expect("ok");
        let mut rng = StdRng::seed_from_u64(999);
        let normal = scirs2_core::random::Normal::new(0.0, 1.0).expect("ok");

        for _ in 0..20 {
            let x: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();
            let mut xtax = 0.0;
            for i in 0..n {
                for j in 0..n {
                    xtax += x[i] * ds.data[[i, j]] * x[j];
                }
            }
            assert!(xtax > 0.0, "x^T A x should be positive, got {xtax}");
        }
    }

    #[test]
    fn test_sparse_spd_matrix_validation() {
        assert!(make_sparse_spd_matrix(0, 0.1, 0.3, None).is_err());
        assert!(make_sparse_spd_matrix(10, -0.1, 0.3, None).is_err());
        assert!(make_sparse_spd_matrix(10, 0.1, 0.0, None).is_err());
        assert!(make_sparse_spd_matrix(10, 0.1, 1.5, None).is_err());
    }

    // =========================================================================
    // make_sparse_coded_signal tests
    // =========================================================================

    #[test]
    fn test_sparse_coded_signal_basic() {
        let ds = make_sparse_coded_signal(50, 20, 10, 3, Some(42)).expect("ok");
        assert_eq!(ds.n_samples(), 50);
        assert_eq!(ds.n_features(), 20);
        assert!(ds.target.is_some());
        // Target should store the flattened dictionary (20 * 10 = 200)
        let target = ds.target.as_ref().expect("target");
        assert_eq!(target.len(), 20 * 10);
    }

    #[test]
    fn test_sparse_coded_signal_dictionary_normalized() {
        let n_features = 15;
        let n_components = 8;
        let ds = make_sparse_coded_signal(30, n_features, n_components, 2, Some(42)).expect("ok");
        let target = ds.target.as_ref().expect("target");

        // Check that each dictionary atom has unit norm
        for j in 0..n_components {
            let mut norm_sq = 0.0;
            for i in 0..n_features {
                let val = target[i * n_components + j];
                norm_sq += val * val;
            }
            assert!(
                (norm_sq.sqrt() - 1.0).abs() < 1e-10,
                "Dictionary atom {j} should have unit norm, got {}",
                norm_sq.sqrt()
            );
        }
    }

    #[test]
    fn test_sparse_coded_signal_reproducibility() {
        let ds1 = make_sparse_coded_signal(20, 10, 5, 2, Some(42)).expect("ok");
        let ds2 = make_sparse_coded_signal(20, 10, 5, 2, Some(42)).expect("ok");
        for i in 0..20 {
            for j in 0..10 {
                assert!(
                    (ds1.data[[i, j]] - ds2.data[[i, j]]).abs() < 1e-12,
                    "Reproducibility failed at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_sparse_coded_signal_validation() {
        assert!(make_sparse_coded_signal(0, 10, 5, 2, None).is_err());
        assert!(make_sparse_coded_signal(10, 0, 5, 2, None).is_err());
        assert!(make_sparse_coded_signal(10, 10, 0, 2, None).is_err());
        assert!(make_sparse_coded_signal(10, 10, 5, 0, None).is_err());
        assert!(make_sparse_coded_signal(10, 10, 5, 6, None).is_err());
    }
}
