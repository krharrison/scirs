//! Sparse matrix dataset generators
//!
//! This module provides synthetic sparse matrix generators for testing
//! iterative solvers, eigenvalue methods, and graph algorithms. Includes:
//!
//! - **Sparse SPD**: symmetric positive definite matrices with controlled density
//! - **Sparse banded**: banded (diagonal-dominant) matrices
//! - **Sparse Laplacian**: graph Laplacian matrices from random graphs
//!
//! All generators return a `Dataset` whose `data` field is a dense
//! representation of the sparse matrix. The `target` field typically
//! contains the diagonal or eigenvalue-relevant information.

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

/// Generate a sparse symmetric positive definite (SPD) matrix
///
/// Constructs a sparse SPD matrix by generating a sparse lower-triangular
/// matrix L with the given density, then computing A = L * L^T + diag_shift * I
/// to guarantee positive definiteness. The resulting matrix is dense in the
/// `Dataset.data` field but has an underlying sparse structure.
///
/// # Arguments
///
/// * `n` - Matrix dimension (must be > 0)
/// * `density` - Fraction of non-zero off-diagonal entries in L, in (0, 1]
/// * `randomseed` - Optional random seed for reproducibility
///
/// # Returns
///
/// A `Dataset` where:
/// - `data` has shape (n, n), the SPD matrix
/// - `target` contains the diagonal entries of the matrix
///
/// # Notes
///
/// The density parameter controls sparsity of the *factor* L. The product
/// A = L * L^T may have higher density due to fill-in.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::sparse::make_sparse_spd;
///
/// let ds = make_sparse_spd(50, 0.1, Some(42)).expect("should succeed");
/// assert_eq!(ds.n_samples(), 50);
/// assert_eq!(ds.n_features(), 50);
/// ```
pub fn make_sparse_spd(n: usize, density: f64, randomseed: Option<u64>) -> Result<Dataset> {
    if n == 0 {
        return Err(DatasetsError::InvalidFormat("n must be > 0".to_string()));
    }
    if density <= 0.0 || density > 1.0 {
        return Err(DatasetsError::InvalidFormat(
            "density must be in (0, 1]".to_string(),
        ));
    }

    let mut rng = create_rng(randomseed);

    let uniform_01 = scirs2_core::random::Uniform::new(0.0, 1.0).map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create uniform dist: {e}"))
    })?;
    let normal = scirs2_core::random::Normal::new(0.0, 1.0).map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create normal dist: {e}"))
    })?;

    // Build sparse lower-triangular factor L
    let mut lower = Array2::zeros((n, n));

    // Diagonal entries of L: random positive values
    for i in 0..n {
        let sample_val: f64 = normal.sample(&mut rng);
        lower[[i, i]] = 1.0 + sample_val.abs();
    }

    // Off-diagonal entries with given density
    for i in 1..n {
        for j in 0..i {
            if uniform_01.sample(&mut rng) < density {
                lower[[i, j]] = normal.sample(&mut rng);
            }
        }
    }

    // Compute A = L * L^T
    let mut matrix = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            // Only need to sum over k up to min(i,j) since L is lower-triangular
            for k in 0..=j {
                sum += lower[[i, k]] * lower[[j, k]];
            }
            matrix[[i, j]] = sum;
            matrix[[j, i]] = sum;
        }
    }

    // Add a diagonal shift to ensure strong positive definiteness
    let diag_shift = 0.1;
    for i in 0..n {
        matrix[[i, i]] += diag_shift;
    }

    // Extract diagonal for target
    let diag = Array1::from_vec((0..n).map(|i| matrix[[i, i]]).collect());

    // Count non-zeros (entries with absolute value > epsilon)
    let eps = 1e-14;
    let nnz: usize = matrix.iter().filter(|&&v: &&f64| v.abs() > eps).count();

    let feature_names: Vec<String> = (0..n).map(|i| format!("col_{i}")).collect();

    let dataset = Dataset::new(matrix, Some(diag))
        .with_featurenames(feature_names)
        .with_description(format!(
            "Sparse SPD matrix ({n}x{n}, factor density={density})"
        ))
        .with_metadata("generator", "make_sparse_spd")
        .with_metadata("n", &n.to_string())
        .with_metadata("density", &density.to_string())
        .with_metadata("nnz", &nnz.to_string());

    Ok(dataset)
}

/// Generate a sparse banded matrix
///
/// Creates a diagonally dominant banded matrix where entries are non-zero
/// only within `bandwidth` diagonals of the main diagonal. The diagonal
/// elements are set to be larger than the sum of the absolute values of
/// the off-diagonal entries in the same row, ensuring non-singularity.
///
/// # Arguments
///
/// * `n` - Matrix dimension (must be > 0)
/// * `bandwidth` - Half-bandwidth: non-zero entries exist in diagonals
///   -bandwidth..=bandwidth. Must be >= 0 and < n.
/// * `randomseed` - Optional random seed for reproducibility
///
/// # Returns
///
/// A `Dataset` where:
/// - `data` has shape (n, n), the banded matrix
/// - `target` contains the diagonal entries
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::sparse::make_sparse_banded;
///
/// let ds = make_sparse_banded(100, 3, Some(42)).expect("should succeed");
/// assert_eq!(ds.n_samples(), 100);
/// assert_eq!(ds.n_features(), 100);
/// ```
pub fn make_sparse_banded(n: usize, bandwidth: usize, randomseed: Option<u64>) -> Result<Dataset> {
    if n == 0 {
        return Err(DatasetsError::InvalidFormat("n must be > 0".to_string()));
    }
    if bandwidth >= n {
        return Err(DatasetsError::InvalidFormat(format!(
            "bandwidth ({bandwidth}) must be < n ({n})"
        )));
    }

    let mut rng = create_rng(randomseed);

    let normal = scirs2_core::random::Normal::new(0.0, 1.0).map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create normal dist: {e}"))
    })?;

    let mut matrix = Array2::zeros((n, n));

    // Fill the band with random values
    for i in 0..n {
        for j in 0..n {
            let diff = i.abs_diff(j);
            if diff <= bandwidth && i != j {
                matrix[[i, j]] = normal.sample(&mut rng);
            }
        }
    }

    // Make it symmetric: A = (M + M^T) / 2
    let mut sym_matrix = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            sym_matrix[[i, j]] = (matrix[[i, j]] + matrix[[j, i]]) / 2.0;
        }
    }

    // Set diagonal to ensure diagonal dominance
    // d_ii = 1 + sum(|a_ij|, j != i)
    for i in 0..n {
        let off_diag_sum: f64 = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let v: f64 = sym_matrix[[i, j]];
                v.abs()
            })
            .sum();
        sym_matrix[[i, i]] = 1.0 + off_diag_sum;
    }

    let diag = Array1::from_vec((0..n).map(|i| sym_matrix[[i, i]]).collect());

    let eps = 1e-14;
    let nnz: usize = sym_matrix.iter().filter(|&&v: &&f64| v.abs() > eps).count();

    let feature_names: Vec<String> = (0..n).map(|i| format!("col_{i}")).collect();

    let dataset = Dataset::new(sym_matrix, Some(diag))
        .with_featurenames(feature_names)
        .with_description(format!(
            "Sparse banded matrix ({n}x{n}, bandwidth={bandwidth})"
        ))
        .with_metadata("generator", "make_sparse_banded")
        .with_metadata("n", &n.to_string())
        .with_metadata("bandwidth", &bandwidth.to_string())
        .with_metadata("nnz", &nnz.to_string());

    Ok(dataset)
}

/// Generate a graph Laplacian matrix
///
/// Constructs the Laplacian L = D - A of an Erdos-Renyi random graph
/// G(n, p_edge), where A is the adjacency matrix and D is the diagonal
/// degree matrix. The Laplacian is always symmetric positive semi-definite,
/// with smallest eigenvalue 0 (for connected graphs, the second-smallest
/// eigenvalue -- the algebraic connectivity -- is positive).
///
/// The edge probability `p_edge` is computed as `min(1.0, 3.0 * ln(n) / n)`
/// to ensure the graph is almost surely connected for large n, unless
/// a different value is desired.
///
/// # Arguments
///
/// * `n` - Number of nodes / matrix dimension (must be > 0)
/// * `randomseed` - Optional random seed for reproducibility
///
/// # Returns
///
/// A `Dataset` where:
/// - `data` has shape (n, n), the Laplacian matrix
/// - `target` contains the node degrees
///
/// # Properties of the Laplacian
///
/// - Symmetric and positive semi-definite
/// - Row sums and column sums are zero
/// - Diagonal entry `L[i,i] = degree(i)`
/// - Off-diagonal entry `L[i,j] = -A[i,j]`
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::generators::sparse::make_sparse_laplacian;
///
/// let ds = make_sparse_laplacian(50, Some(42)).expect("should succeed");
/// assert_eq!(ds.n_samples(), 50);
/// assert_eq!(ds.n_features(), 50);
/// ```
pub fn make_sparse_laplacian(n: usize, randomseed: Option<u64>) -> Result<Dataset> {
    if n == 0 {
        return Err(DatasetsError::InvalidFormat("n must be > 0".to_string()));
    }

    let mut rng = create_rng(randomseed);

    // Use p_edge that ensures connectivity whp: p > ln(n)/n
    // We use 3*ln(n)/n for safety margin, clamped to [0,1]
    let p_edge = if n == 1 {
        0.0
    } else {
        let p = 3.0 * (n as f64).ln() / n as f64;
        p.min(1.0)
    };

    let uniform = scirs2_core::random::Uniform::new(0.0, 1.0).map_err(|e| {
        DatasetsError::ComputationError(format!("Failed to create uniform dist: {e}"))
    })?;

    // Build adjacency matrix
    let mut adj = Array2::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            if uniform.sample(&mut rng) < p_edge {
                adj[[i, j]] = 1.0;
                adj[[j, i]] = 1.0;
            }
        }
    }

    // Compute degrees
    let mut degrees = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            degrees[i] += adj[[i, j]];
        }
    }

    // Build Laplacian: L = D - A
    let mut laplacian = Array2::zeros((n, n));
    for i in 0..n {
        laplacian[[i, i]] = degrees[i];
        for j in 0..n {
            if i != j {
                laplacian[[i, j]] = -adj[[i, j]];
            }
        }
    }

    let target = Array1::from_vec(degrees.clone());

    let n_edges: usize = {
        let total_degree: f64 = degrees.iter().sum();
        (total_degree / 2.0).round() as usize
    };

    let feature_names: Vec<String> = (0..n).map(|i| format!("node_{i}")).collect();

    let dataset = Dataset::new(laplacian, Some(target))
        .with_featurenames(feature_names)
        .with_description(format!("Graph Laplacian ({n}x{n}, p_edge={p_edge:.4})"))
        .with_metadata("generator", "make_sparse_laplacian")
        .with_metadata("n", &n.to_string())
        .with_metadata("p_edge", &p_edge.to_string())
        .with_metadata("n_edges", &n_edges.to_string());

    Ok(dataset)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // make_sparse_spd tests
    // =========================================================================

    #[test]
    fn test_sparse_spd_shape() {
        let ds = make_sparse_spd(50, 0.1, Some(42)).expect("should succeed");
        assert_eq!(ds.n_samples(), 50);
        assert_eq!(ds.n_features(), 50);
        assert!(ds.target.is_some());
        let target = ds.target.as_ref().expect("target exists");
        assert_eq!(target.len(), 50);
    }

    #[test]
    fn test_sparse_spd_symmetric() {
        let n = 30;
        let ds = make_sparse_spd(n, 0.2, Some(42)).expect("should succeed");
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (ds.data[[i, j]] - ds.data[[j, i]]).abs() < 1e-10,
                    "Not symmetric at ({i},{j}): {} vs {}",
                    ds.data[[i, j]],
                    ds.data[[j, i]]
                );
            }
        }
    }

    #[test]
    fn test_sparse_spd_positive_diagonal() {
        let n = 40;
        let ds = make_sparse_spd(n, 0.15, Some(42)).expect("should succeed");
        for i in 0..n {
            assert!(
                ds.data[[i, i]] > 0.0,
                "Diagonal element at [{i},{i}] should be positive, got {}",
                ds.data[[i, i]]
            );
        }
    }

    #[test]
    fn test_sparse_spd_positive_definite() {
        // A simple check: all diagonal entries should be positive (necessary for SPD)
        // and the matrix should equal L*L^T + shift*I which is guaranteed PD
        let n = 20;
        let ds = make_sparse_spd(n, 0.3, Some(42)).expect("should succeed");

        // Check Gershgorin: for a diagonally dominant matrix, all eigenvalues are positive
        // Our construction guarantees PD via Cholesky factorization, but let's verify
        // the matrix passes a basic sanity check: x^T A x > 0 for several random x
        let mut rng = StdRng::seed_from_u64(999);
        let normal = scirs2_core::random::Normal::new(0.0, 1.0).expect("normal dist");

        for trial in 0..10 {
            let x: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();
            let mut xtax = 0.0;
            for i in 0..n {
                for j in 0..n {
                    xtax += x[i] * ds.data[[i, j]] * x[j];
                }
            }
            assert!(
                xtax > 0.0,
                "x^T A x should be positive (trial {trial}), got {xtax}"
            );
        }
    }

    #[test]
    fn test_sparse_spd_reproducibility() {
        let ds1 = make_sparse_spd(25, 0.2, Some(77)).expect("should succeed");
        let ds2 = make_sparse_spd(25, 0.2, Some(77)).expect("should succeed");
        for i in 0..25 {
            for j in 0..25 {
                assert!(
                    (ds1.data[[i, j]] - ds2.data[[i, j]]).abs() < 1e-15,
                    "Matrix differs at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_sparse_spd_validation() {
        assert!(make_sparse_spd(0, 0.1, None).is_err());
        assert!(make_sparse_spd(10, 0.0, None).is_err());
        assert!(make_sparse_spd(10, -0.1, None).is_err());
        assert!(make_sparse_spd(10, 1.1, None).is_err());
    }

    // =========================================================================
    // make_sparse_banded tests
    // =========================================================================

    #[test]
    fn test_sparse_banded_shape() {
        let ds = make_sparse_banded(100, 3, Some(42)).expect("should succeed");
        assert_eq!(ds.n_samples(), 100);
        assert_eq!(ds.n_features(), 100);
        assert!(ds.target.is_some());
    }

    #[test]
    fn test_sparse_banded_band_structure() {
        let n = 50;
        let bw = 2;
        let ds = make_sparse_banded(n, bw, Some(42)).expect("should succeed");

        // Off-diagonal entries outside the band should be zero
        let eps = 1e-14;
        for i in 0..n {
            for j in 0..n {
                let diff = if i >= j { i - j } else { j - i };
                if diff > bw {
                    assert!(
                        ds.data[[i, j]].abs() < eps,
                        "Entry ({i},{j}) outside band should be 0, got {}",
                        ds.data[[i, j]]
                    );
                }
            }
        }
    }

    #[test]
    fn test_sparse_banded_symmetric() {
        let n = 40;
        let ds = make_sparse_banded(n, 3, Some(42)).expect("should succeed");
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
    fn test_sparse_banded_diagonally_dominant() {
        let n = 30;
        let ds = make_sparse_banded(n, 4, Some(42)).expect("should succeed");

        for i in 0..n {
            let diag_val = ds.data[[i, i]];
            let off_diag_sum: f64 = (0..n)
                .filter(|&j| j != i)
                .map(|j| ds.data[[i, j]].abs())
                .sum();
            assert!(
                diag_val >= 1.0 + off_diag_sum - 1e-10,
                "Row {i} not diagonally dominant: diag={diag_val}, off_sum={off_diag_sum}"
            );
        }
    }

    #[test]
    fn test_sparse_banded_reproducibility() {
        let ds1 = make_sparse_banded(30, 2, Some(55)).expect("should succeed");
        let ds2 = make_sparse_banded(30, 2, Some(55)).expect("should succeed");
        for i in 0..30 {
            for j in 0..30 {
                assert!(
                    (ds1.data[[i, j]] - ds2.data[[i, j]]).abs() < 1e-15,
                    "Matrix differs at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_sparse_banded_zero_bandwidth() {
        // bandwidth=0 means only diagonal
        let n = 10;
        let ds = make_sparse_banded(n, 0, Some(42)).expect("should succeed");

        let eps = 1e-14;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    assert!(
                        ds.data[[i, j]].abs() < eps,
                        "Off-diagonal ({i},{j}) should be 0 for bw=0"
                    );
                } else {
                    assert!(
                        ds.data[[i, i]] > 0.0,
                        "Diagonal ({i},{i}) should be positive"
                    );
                }
            }
        }
    }

    #[test]
    fn test_sparse_banded_validation() {
        assert!(make_sparse_banded(0, 1, None).is_err());
        assert!(make_sparse_banded(5, 5, None).is_err()); // bandwidth >= n
        assert!(make_sparse_banded(5, 10, None).is_err()); // bandwidth >= n
    }

    // =========================================================================
    // make_sparse_laplacian tests
    // =========================================================================

    #[test]
    fn test_sparse_laplacian_shape() {
        let ds = make_sparse_laplacian(50, Some(42)).expect("should succeed");
        assert_eq!(ds.n_samples(), 50);
        assert_eq!(ds.n_features(), 50);
        assert!(ds.target.is_some());
    }

    #[test]
    fn test_sparse_laplacian_symmetric() {
        let n = 30;
        let ds = make_sparse_laplacian(n, Some(42)).expect("should succeed");
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (ds.data[[i, j]] - ds.data[[j, i]]).abs() < 1e-10,
                    "Laplacian not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_sparse_laplacian_row_sums_zero() {
        let n = 40;
        let ds = make_sparse_laplacian(n, Some(42)).expect("should succeed");

        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| ds.data[[i, j]]).sum();
            assert!(
                row_sum.abs() < 1e-10,
                "Row {i} sum should be 0, got {row_sum}"
            );
        }
    }

    #[test]
    fn test_sparse_laplacian_positive_semidefinite() {
        // Check x^T L x >= 0 for random vectors
        let n = 20;
        let ds = make_sparse_laplacian(n, Some(42)).expect("should succeed");

        let mut rng = StdRng::seed_from_u64(999);
        let normal = scirs2_core::random::Normal::new(0.0, 1.0).expect("normal dist");

        for trial in 0..10 {
            let x: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();
            let mut xtlx = 0.0;
            for i in 0..n {
                for j in 0..n {
                    xtlx += x[i] * ds.data[[i, j]] * x[j];
                }
            }
            assert!(
                xtlx >= -1e-10,
                "x^T L x should be >= 0 (trial {trial}), got {xtlx}"
            );
        }
    }

    #[test]
    fn test_sparse_laplacian_off_diagonal_nonpositive() {
        let n = 30;
        let ds = make_sparse_laplacian(n, Some(42)).expect("should succeed");

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    assert!(
                        ds.data[[i, j]] <= 1e-10,
                        "Off-diagonal L[{i},{j}] should be <= 0, got {}",
                        ds.data[[i, j]]
                    );
                }
            }
        }
    }

    #[test]
    fn test_sparse_laplacian_diagonal_equals_degree() {
        let n = 25;
        let ds = make_sparse_laplacian(n, Some(42)).expect("should succeed");
        let target = ds.target.as_ref().expect("target exists");

        for i in 0..n {
            assert!(
                (ds.data[[i, i]] - target[i]).abs() < 1e-10,
                "Diagonal L[{i},{i}]={} should equal degree {}",
                ds.data[[i, i]],
                target[i]
            );
        }
    }

    #[test]
    fn test_sparse_laplacian_reproducibility() {
        let ds1 = make_sparse_laplacian(30, Some(88)).expect("should succeed");
        let ds2 = make_sparse_laplacian(30, Some(88)).expect("should succeed");
        for i in 0..30 {
            for j in 0..30 {
                assert!(
                    (ds1.data[[i, j]] - ds2.data[[i, j]]).abs() < 1e-15,
                    "Laplacian differs at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_sparse_laplacian_single_node() {
        // n=1: no edges, L = [0]
        let ds = make_sparse_laplacian(1, Some(42)).expect("should succeed");
        assert_eq!(ds.n_samples(), 1);
        assert!(ds.data[[0, 0]].abs() < 1e-15);
    }

    #[test]
    fn test_sparse_laplacian_validation() {
        assert!(make_sparse_laplacian(0, None).is_err());
    }
}
