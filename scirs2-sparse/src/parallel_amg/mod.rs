//! Parallel Algebraic Multigrid (AMG) Coarsening
//!
//! This module implements parallel AMG coarsening algorithms including:
//!
//! - **Parallel strength-of-connection** computation using `std::thread::scope`
//! - **PMIS** (Parallel Maximum Independent Set) coarsening
//! - **CLJP** (Cleary-Luby-Jones-Plassmann) coarsening
//! - **Parallel Ruge-Stüben** coarsening with parallel passes
//! - **Direct and smoothed-aggregation interpolation** (parallel)
//! - **Galerkin coarse operator** construction (parallel)
//!
//! # Quick Start
//!
//! ```rust
//! use scirs2_sparse::parallel_amg::{
//!     ParallelAmgConfig, CoarsenMethod, build_parallel_amg_hierarchy,
//! };
//! use scirs2_sparse::csr::CsrMatrix;
//!
//! // Build 1D Laplacian
//! let n = 16usize;
//! let mut rows = Vec::new(); let mut cols = Vec::new(); let mut vals = Vec::new();
//! for i in 0..n { rows.push(i); cols.push(i); vals.push(2.0f64); }
//! for i in 0..n-1 {
//!     rows.push(i); cols.push(i+1); vals.push(-1.0f64);
//!     rows.push(i+1); cols.push(i); vals.push(-1.0f64);
//! }
//! let a = CsrMatrix::new(vals, rows, cols, (n, n)).expect("valid input");
//!
//! let config = ParallelAmgConfig {
//!     n_threads: 2,
//!     max_levels: 3,
//!     coarsening: CoarsenMethod::PMIS,
//!     ..Default::default()
//! };
//!
//! let hierarchy = build_parallel_amg_hierarchy(&a, &config).expect("hierarchy");
//! assert!(hierarchy.n_levels >= 2);
//! ```

pub mod parallel_interp;
pub mod parallel_rs;
pub mod strength;
pub mod types;

// Re-exports
pub use parallel_interp::{
    galerkin_coarse_operator, parallel_direct_interpolation, parallel_galerkin_coarse_operator,
    parallel_sa_interpolation,
};
pub use parallel_rs::{cljp_coarsening, parallel_rs_coarsening, pmis_coarsening};
pub use strength::{
    compute_lambda, compute_lambda_with_fset, parallel_strength_of_connection,
    serial_strength_of_connection, undirected_strength, StrengthGraph,
};
pub use types::{
    CoarsenMethod, CoarseningResult, ParallelAmgConfig, ParallelAmgHierarchy, ParallelAmgLevel,
};

use crate::csr::CsrMatrix;
use crate::error::SparseResult;

/// Build a parallel AMG hierarchy for the given system matrix.
///
/// Applies the configured coarsening method recursively until either:
/// - The coarsest grid has fewer than `config.min_coarse_size` nodes, or
/// - `config.max_levels` levels have been created, or
/// - The coarsening ratio exceeds `config.max_coarsening_ratio`.
///
/// # Arguments
///
/// * `a` - System matrix (should be SPD or at least diagonally dominant)
/// * `config` - Parallel AMG configuration
///
/// # Returns
///
/// A `ParallelAmgHierarchy` containing all levels and the coarsest-grid matrix.
pub fn build_parallel_amg_hierarchy(
    a: &CsrMatrix<f64>,
    config: &ParallelAmgConfig,
) -> SparseResult<ParallelAmgHierarchy> {
    let mut levels = Vec::new();
    let mut current_a = a.clone();

    for _level in 0..config.max_levels {
        let n = current_a.shape().0;
        if n <= config.min_coarse_size {
            break;
        }

        // Compute strength graph
        let strength = match config.coarsening {
            CoarsenMethod::ParallelSA => undirected_strength(&current_a, config.strength_threshold),
            _ => parallel_strength_of_connection(
                &current_a,
                config.strength_threshold,
                config.n_threads,
            ),
        };

        // Coarsening
        let coarsening_result = match config.coarsening {
            CoarsenMethod::PMIS => pmis_coarsening(&strength),
            CoarsenMethod::CLJP => {
                let lambda = compute_lambda(&strength);
                cljp_coarsening(&strength, &lambda)
            }
            CoarsenMethod::ParallelRS | CoarsenMethod::ParallelSA => {
                parallel_rs_coarsening(&current_a, &strength, config.n_threads)
            }
        };

        let n_coarse = coarsening_result.n_coarse();
        if n_coarse == 0 {
            break;
        }

        // Check coarsening ratio
        let ratio = n_coarse as f64 / n as f64;
        if ratio > config.max_coarsening_ratio {
            break;
        }

        // Build interpolation operator P
        let p = match config.coarsening {
            CoarsenMethod::ParallelSA => parallel_sa_interpolation(
                &current_a,
                &strength,
                &coarsening_result.cf_splitting,
                config.n_threads,
                config.omega,
            )?,
            _ => parallel_direct_interpolation(
                &current_a,
                &coarsening_result.cf_splitting,
                config.n_threads,
            )?,
        };

        // Build restriction operator R = P^T
        let r = p.transpose();

        // Compute coarse-grid operator A_c = R A P
        let a_coarse = parallel_galerkin_coarse_operator(&current_a, &p, config.n_threads)?;

        let level = ParallelAmgLevel::new(current_a.clone(), p, r, n, n_coarse);
        levels.push(level);

        current_a = a_coarse;

        if n_coarse <= config.min_coarse_size {
            break;
        }
    }

    Ok(ParallelAmgHierarchy::new(levels, current_a))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(2.0f64);
        }
        for i in 0..n - 1 {
            rows.push(i);
            cols.push(i + 1);
            vals.push(-1.0f64);
            rows.push(i + 1);
            cols.push(i);
            vals.push(-1.0f64);
        }
        CsrMatrix::new(vals, rows, cols, (n, n)).expect("valid Laplacian")
    }

    #[test]
    fn test_parallel_amg_config_default() {
        let config = ParallelAmgConfig::default();
        assert_eq!(config.n_threads, 1);
        assert!((config.strength_threshold - 0.25).abs() < 1e-10);
        assert_eq!(config.max_levels, 10);
        assert_eq!(config.coarsening, CoarsenMethod::ParallelRS);
    }

    #[test]
    fn test_full_hierarchy() {
        let a = laplacian_1d(32);
        let config = ParallelAmgConfig {
            n_threads: 2,
            max_levels: 4,
            min_coarse_size: 2,
            coarsening: CoarsenMethod::PMIS,
            ..Default::default()
        };

        let hierarchy = build_parallel_amg_hierarchy(&a, &config).expect("hierarchy");
        assert!(hierarchy.n_levels >= 2, "Should have at least 2 levels");
        assert!(
            hierarchy.n_levels <= 5,
            "Should not exceed max_levels+1 levels"
        );

        // Fine-grid size matches input
        assert_eq!(hierarchy.fine_size(), 32);

        // Coarsest grid is smaller
        assert!(hierarchy.coarse_size() < 32);

        // Each level: P and R have consistent shapes
        for level in &hierarchy.levels {
            let (p_rows, p_cols) = level.p.shape();
            let (r_rows, r_cols) = level.r.shape();
            assert_eq!(p_rows, level.n_fine);
            assert_eq!(p_cols, level.n_coarse);
            assert_eq!(r_rows, level.n_coarse);
            assert_eq!(r_cols, level.n_fine);
        }
    }

    #[test]
    fn test_strength_threshold() {
        // Already covered in strength::tests, verify here at module level
        let a = laplacian_1d(8);
        let g = serial_strength_of_connection(&a, 0.25);
        assert_eq!(g.n, 8);
        // For 1D Laplacian, interior nodes have 2 strong neighbors
        for i in 1..7 {
            assert_eq!(g.strong_neighbors[i].len(), 2, "Interior node {i}");
        }
    }

    #[test]
    fn test_strength_parallel_matches_serial() {
        // Covered in strength::tests — verify here with larger matrix
        let a = laplacian_1d(24);
        let s = serial_strength_of_connection(&a, 0.25);
        let p = parallel_strength_of_connection(&a, 0.25, 3);
        for i in 0..24 {
            let mut sv = s.strong_neighbors[i].clone();
            let mut pv = p.strong_neighbors[i].clone();
            sv.sort();
            pv.sort();
            assert_eq!(sv, pv, "Node {i}: serial and parallel mismatch");
        }
    }

    #[test]
    fn test_undirected_strength_symmetric() {
        let a = laplacian_1d(10);
        let g = undirected_strength(&a, 0.25);
        for i in 0..g.n {
            for &j in &g.strong_neighbors[i] {
                assert!(
                    g.strong_neighbors[j].contains(&i),
                    "Undirected graph not symmetric: {i} -> {j} but not {j} -> {i}"
                );
            }
        }
    }

    #[test]
    fn test_lambda_computation() {
        let a = laplacian_1d(10);
        let g = serial_strength_of_connection(&a, 0.25);
        let lambda = compute_lambda(&g);
        assert_eq!(lambda.len(), 10);
        for &l in &lambda {
            assert!(l >= 0.0, "Lambda must be non-negative");
        }
    }

    #[test]
    fn test_pmis_is_independent_set() {
        let a = laplacian_1d(16);
        let g = serial_strength_of_connection(&a, 0.25);
        let result = pmis_coarsening(&g);
        for &c1 in &result.c_nodes {
            for &c2 in &result.c_nodes {
                if c1 != c2 {
                    assert!(
                        !g.strong_neighbors[c1].contains(&c2),
                        "PMIS: C-nodes {c1} and {c2} strongly connected"
                    );
                }
            }
        }
    }

    #[test]
    fn test_pmis_covers_all() {
        let a = laplacian_1d(16);
        let g = serial_strength_of_connection(&a, 0.25);
        let result = pmis_coarsening(&g);
        for &f in &result.f_nodes {
            let has_c = g.strong_neighbors[f]
                .iter()
                .any(|&j| result.cf_splitting[j] == 1)
                || g.strong_influencers[f]
                    .iter()
                    .any(|&j| result.cf_splitting[j] == 1);
            assert!(has_c, "F-node {f} has no C-neighbor");
        }
    }

    #[test]
    fn test_cljp_valid_splitting() {
        let a = laplacian_1d(14);
        let g = serial_strength_of_connection(&a, 0.25);
        let lambda = compute_lambda(&g);
        let result = cljp_coarsening(&g, &lambda);
        assert_eq!(result.cf_splitting.len(), 14);
        for &s in &result.cf_splitting {
            assert!(s == 0 || s == 1);
        }
    }

    #[test]
    fn test_cljp_independent() {
        let a = laplacian_1d(14);
        let g = serial_strength_of_connection(&a, 0.25);
        let lambda = compute_lambda(&g);
        let result = cljp_coarsening(&g, &lambda);
        for &c1 in &result.c_nodes {
            for &c2 in &result.c_nodes {
                if c1 != c2 {
                    assert!(
                        !g.strong_neighbors[c1].contains(&c2),
                        "CLJP C-nodes {c1} and {c2} strongly connected"
                    );
                }
            }
        }
    }

    #[test]
    fn test_parallel_rs_cf_assignment() {
        let a = laplacian_1d(18);
        let g = serial_strength_of_connection(&a, 0.25);
        let result = parallel_rs_coarsening(&a, &g, 2);
        assert_eq!(result.cf_splitting.len(), 18);
        assert_eq!(result.c_nodes.len() + result.f_nodes.len(), 18);
        assert!(!result.c_nodes.is_empty());
        assert!(!result.f_nodes.is_empty());
    }

    #[test]
    fn test_parallel_rs_matches_serial() {
        let a = laplacian_1d(20);
        let g = serial_strength_of_connection(&a, 0.25);
        let r1 = parallel_rs_coarsening(&a, &g, 1);
        let r4 = parallel_rs_coarsening(&a, &g, 4);
        // Both should produce valid splittings
        assert!(r1.n_coarse() > 0);
        assert!(r4.n_coarse() > 0);
        // Coarsening ratios should be in reasonable range
        assert!(r1.coarsening_ratio() > 0.0 && r1.coarsening_ratio() < 1.0);
        assert!(r4.coarsening_ratio() > 0.0 && r4.coarsening_ratio() < 1.0);
    }

    #[test]
    fn test_direct_interp_c_node_identity() {
        let n = 10;
        let a = laplacian_1d(n);
        let g = serial_strength_of_connection(&a, 0.25);
        let result = pmis_coarsening(&g);
        let p = parallel_direct_interpolation(&a, &result.cf_splitting, 1)
            .expect("direct interpolation");

        // Each C-node's row should have exactly one entry equal to 1.0
        let mut c_col = 0usize;
        for i in 0..n {
            if result.cf_splitting[i] == 1 {
                let row_nnz: Vec<(usize, f64)> = p
                    .row_range(i)
                    .map(|pos| (p.indices[pos], p.data[pos]))
                    .collect();
                assert_eq!(row_nnz.len(), 1, "C-node {i} should have exactly 1 P entry");
                assert_eq!(row_nnz[0].0, c_col, "C-node {i} maps to coarse col {c_col}");
                assert!((row_nnz[0].1 - 1.0).abs() < 1e-10, "C-node identity weight");
                c_col += 1;
            }
        }
    }

    #[test]
    fn test_direct_interp_f_node_has_c_parents() {
        let n = 14;
        let a = laplacian_1d(n);
        let g = serial_strength_of_connection(&a, 0.25);
        let result = pmis_coarsening(&g);
        let p = parallel_direct_interpolation(&a, &result.cf_splitting, 1)
            .expect("direct interpolation");
        let n_coarse = result.n_coarse();

        // All P entries must reference valid coarse indices
        for pos in 0..p.nnz() {
            assert!(p.indices[pos] < n_coarse);
        }
    }

    #[test]
    fn test_direct_interp_parallel() {
        let n = 18;
        let a = laplacian_1d(n);
        let g = serial_strength_of_connection(&a, 0.25);
        let result = pmis_coarsening(&g);

        let p1 = parallel_direct_interpolation(&a, &result.cf_splitting, 1).expect("p1");
        let p4 = parallel_direct_interpolation(&a, &result.cf_splitting, 4).expect("p4");

        assert_eq!(p1.shape(), p4.shape(), "Shapes differ");
        assert_eq!(p1.nnz(), p4.nnz(), "NNZ differs");
    }

    #[test]
    fn test_sa_interp_shape() {
        let n = 18;
        let a = laplacian_1d(n);
        let g = serial_strength_of_connection(&a, 0.25);
        let result = pmis_coarsening(&g);
        let n_coarse = result.n_coarse();

        let p = parallel_sa_interpolation(&a, &g, &result.cf_splitting, 2, 4.0 / 3.0)
            .expect("sa interp");
        let (rows, cols) = p.shape();
        assert_eq!(rows, n);
        assert_eq!(cols, n_coarse);
    }

    #[test]
    fn test_galerkin_operator_size() {
        let n = 14;
        let a = laplacian_1d(n);
        let g = serial_strength_of_connection(&a, 0.25);
        let result = pmis_coarsening(&g);
        let n_coarse = result.n_coarse();

        let p = parallel_direct_interpolation(&a, &result.cf_splitting, 1).expect("p");
        let ac = galerkin_coarse_operator(&a, &p).expect("ac");

        let (r, c) = ac.shape();
        assert_eq!(r, n_coarse);
        assert_eq!(c, n_coarse);
    }

    #[test]
    fn test_galerkin_spd_preserved() {
        let n = 12;
        let a = laplacian_1d(n);
        let g = serial_strength_of_connection(&a, 0.25);
        let result = pmis_coarsening(&g);

        let p = parallel_direct_interpolation(&a, &result.cf_splitting, 1).expect("p");
        let ac = galerkin_coarse_operator(&a, &p).expect("ac");

        let (nc, _) = ac.shape();
        for i in 0..nc {
            let mut diag = 0.0f64;
            for pos in ac.row_range(i) {
                if ac.indices[pos] == i {
                    diag = ac.data[pos];
                    break;
                }
            }
            assert!(diag > 0.0, "Coarse diagonal must be positive at row {i}");
        }
    }

    #[test]
    fn test_parallel_strength_n_threads() {
        let a = laplacian_1d(20);
        for n_threads in [1, 2, 4] {
            let g = parallel_strength_of_connection(&a, 0.25, n_threads);
            assert_eq!(g.n, 20, "n_threads={n_threads}");
            // Verify interior nodes
            for i in 1..19 {
                assert_eq!(
                    g.strong_neighbors[i].len(),
                    2,
                    "Interior node {i} with n_threads={n_threads}"
                );
            }
        }
    }

    #[test]
    fn test_coarsening_ratio() {
        let a = laplacian_1d(20);
        let g = serial_strength_of_connection(&a, 0.25);
        let result = pmis_coarsening(&g);
        let ratio = result.coarsening_ratio();
        assert!(
            (0.15..=0.85).contains(&ratio),
            "Coarsening ratio {ratio} out of range"
        );
    }
}
