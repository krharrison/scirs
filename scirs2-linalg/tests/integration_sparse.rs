//! Integration tests: scirs2-linalg + scirs2-sparse
//!
//! Covers:
//! - Dense-sparse matrix multiplication
//! - Sparse eigenvalue comparison against dense eigenvalue solver
//! - Round-trip conversion between dense and sparse representations
//! - Sparse matrix-vector products

use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_linalg::{eigh, sparse_dense};
use scirs2_sparse::{
    linalg::{eigsh, LanczosOptions},
    CsrMatrix, SymCsrMatrix,
};

// ---------------------------------------------------------------------------
// Helper: build a small symmetric tridiagonal CSR matrix
// ---------------------------------------------------------------------------

/// Build an n×n symmetric tridiagonal matrix with `diag` on the diagonal
/// and `off` on the off-diagonals, returning it as CsrMatrix<f64>.
fn symmetric_tridiagonal_csr(n: usize, diag: f64, off: f64) -> CsrMatrix<f64> {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    for i in 0..n {
        if i > 0 {
            rows.push(i);
            cols.push(i - 1);
            vals.push(off);
        }
        rows.push(i);
        cols.push(i);
        vals.push(diag);
        if i + 1 < n {
            rows.push(i);
            cols.push(i + 1);
            vals.push(off);
        }
    }

    CsrMatrix::from_triplets(n, n, rows, cols, vals)
        .expect("Failed to build symmetric tridiagonal CsrMatrix")
}

// ---------------------------------------------------------------------------
// 1. Dense matrix to sparse representation round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_dense_to_sparse_round_trip() {
    let dense: Array2<f64> = array![[2.0, 0.0, -1.0], [0.0, 4.0, 0.0], [-1.0, 0.0, 3.0]];

    // Convert to sparse via SparseMatrixView
    let sparse_view = sparse_dense::sparse_from_ndarray(&dense.view(), 1e-10_f64)
        .expect("Dense to sparse conversion failed");

    // Convert back to dense
    let reconstructed = sparse_view.to_dense();

    for i in 0..3 {
        for j in 0..3 {
            assert_abs_diff_eq!(reconstructed[[i, j]], dense[[i, j]], epsilon = 1e-12,);
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Sparse-dense matrix-vector product (CsrMatrix dot)
// ---------------------------------------------------------------------------

#[test]
fn test_csr_matrix_vector_product() {
    // A = [[2, -1], [-1, 3]]
    // x = [1, 2]
    // A @ x = [2*1 + (-1)*2, (-1)*1 + 3*2] = [0, 5]
    let csr = CsrMatrix::from_triplets(
        2,                              // nrows
        2,                              // ncols
        vec![0, 0, 1, 1],               // row indices
        vec![0, 1, 0, 1],               // col indices
        vec![2.0_f64, -1.0, -1.0, 3.0], // values
    )
    .expect("CsrMatrix construction failed");

    let x = vec![1.0_f64, 2.0];
    let result = csr.dot(&x).expect("CSR matvec failed");

    assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(result[1], 5.0, epsilon = 1e-12);
}

// ---------------------------------------------------------------------------
// 3. Sparse-dense matrix multiplication (SparseMatrixView * dense)
// ---------------------------------------------------------------------------

#[test]
fn test_sparse_dense_matrix_multiplication() {
    // Sparse A (3x2), dense B (2x3)
    let a_dense: Array2<f64> = array![[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]];
    let b_dense: Array2<f64> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    let a_sparse = sparse_dense::sparse_from_ndarray(&a_dense.view(), 1e-10_f64)
        .expect("A: dense to sparse conversion failed");

    let c = sparse_dense::sparse_dense_matmul(&a_sparse, &b_dense.view())
        .expect("sparse_dense_matmul failed");

    // A @ B = [[1, 2, 3], [8, 10, 12], [3, 6, 9]]
    let expected: Array2<f64> = array![[1.0, 2.0, 3.0], [8.0, 10.0, 12.0], [3.0, 6.0, 9.0]];

    for i in 0..3 {
        for j in 0..3 {
            assert_abs_diff_eq!(c[[i, j]], expected[[i, j]], epsilon = 1e-12);
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Sparse matrix-vector product via sparse_dense_matvec
// ---------------------------------------------------------------------------

#[test]
fn test_sparse_dense_matvec() {
    let a_dense: Array2<f64> = array![[3.0, 0.0, 1.0], [0.0, 2.0, 0.0], [1.0, 0.0, 4.0]];
    let x: Array1<f64> = Array1::from_vec(vec![1.0, 2.0, 3.0]);

    let a_sparse = sparse_dense::sparse_from_ndarray(&a_dense.view(), 1e-10_f64)
        .expect("dense to sparse failed");

    let result = sparse_dense::sparse_dense_matvec(&a_sparse, &x.view())
        .expect("sparse_dense_matvec failed");

    // A @ x = [3+0+3, 0+4+0, 1+0+12] = [6, 4, 13]
    assert_abs_diff_eq!(result[0], 6.0, epsilon = 1e-12);
    assert_abs_diff_eq!(result[1], 4.0, epsilon = 1e-12);
    assert_abs_diff_eq!(result[2], 13.0, epsilon = 1e-12);
}

// ---------------------------------------------------------------------------
// 5. Dense eigenvalue solver (linalg::eigh) on a known 2x2 matrix
// ---------------------------------------------------------------------------

#[test]
fn test_dense_eigh_2x2_known_eigenvalues() {
    // [[3, 1], [1, 3]] has eigenvalues 2 and 4
    let a: Array2<f64> = array![[3.0, 1.0], [1.0, 3.0]];
    let (vals, vecs) = eigh(&a.view(), None).expect("Dense eigh failed");

    // Sort eigenvalues
    let mut ev: Vec<f64> = vals.to_vec();
    ev.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    assert_abs_diff_eq!(ev[0], 2.0, epsilon = 1e-8);
    assert_abs_diff_eq!(ev[1], 4.0, epsilon = 1e-8);

    // Check A @ v = lambda * v for each eigenvector
    for k in 0..2 {
        let eigval = vals[k];
        let eigvec: Array1<f64> = vecs.column(k).to_owned();
        let av: Array1<f64> = a.dot(&eigvec);
        let lv: Array1<f64> = eigvec.mapv(|v| eigval * v);

        for (ai, li) in av.iter().zip(lv.iter()) {
            assert_abs_diff_eq!(ai, li, epsilon = 1e-8);
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Sparse eigsh on a symmetric tridiagonal matrix: compare vs analytical
// ---------------------------------------------------------------------------

#[test]
fn test_sparse_eigsh_tridiagonal_vs_analytical() {
    // For a 5x5 tridiagonal with diag=2, off=-1 (discrete Laplacian),
    // the analytical eigenvalues are:
    // lambda_k = 2 - 2*cos(k*pi/(n+1))  for k=1..n  (n=5)
    let n = 5_usize;
    let diag = 2.0_f64;
    let off = -1.0_f64;

    let csr = symmetric_tridiagonal_csr(n, diag, off);
    let sym_csr =
        SymCsrMatrix::from_csr(&csr).expect("Failed to build SymCsrMatrix from CsrMatrix");

    // Compute 3 smallest eigenvalues
    let opts = LanczosOptions {
        numeigenvalues: 3,
        max_iter: 500,
        tol: 1e-8,
        ..Default::default()
    };
    let result = eigsh(&sym_csr, Some(3), Some("SM"), Some(opts)).expect("Sparse eigsh failed");

    let mut computed: Vec<f64> = result.eigenvalues.to_vec();
    computed.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Analytical eigenvalues for k=1,2,3
    let pi = std::f64::consts::PI;
    for k in 1..=3 {
        let lambda_k = 2.0 - 2.0 * (k as f64 * pi / (n as f64 + 1.0)).cos();
        assert_abs_diff_eq!(computed[k - 1], lambda_k, epsilon = 1e-4,);
    }
}

// ---------------------------------------------------------------------------
// 7. Dense vs sparse eigenvalues should agree on a small matrix
// ---------------------------------------------------------------------------

#[test]
fn test_dense_vs_sparse_eigenvalues_agree() {
    // 4x4 symmetric positive definite matrix
    let n = 4_usize;
    let diag = 4.0_f64;
    let off = -1.0_f64;

    // Build dense matrix
    let mut dense = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        dense[[i, i]] = diag;
        if i + 1 < n {
            dense[[i, i + 1]] = off;
            dense[[i + 1, i]] = off;
        }
    }

    // Dense eigenvalues
    let (dense_vals, _) = eigh(&dense.view(), None).expect("Dense eigh failed");
    let mut dense_sorted: Vec<f64> = dense_vals.to_vec();
    dense_sorted.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Sparse eigenvalues
    let csr = symmetric_tridiagonal_csr(n, diag, off);
    let sym_csr = SymCsrMatrix::from_csr(&csr).expect("SymCsrMatrix from_csr failed");

    let opts = LanczosOptions {
        numeigenvalues: n,
        max_iter: 1000,
        tol: 1e-8,
        ..Default::default()
    };
    let sparse_result = eigsh(&sym_csr, Some(n), Some("LA"), Some(opts))
        .expect("Sparse eigsh (all eigenvalues) failed");

    let mut sparse_sorted: Vec<f64> = sparse_result.eigenvalues.to_vec();
    sparse_sorted.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Both should agree to reasonable tolerance
    for (ds, ss) in dense_sorted.iter().zip(sparse_sorted.iter()) {
        assert_abs_diff_eq!(ds, ss, epsilon = 1e-4);
    }
}

// ---------------------------------------------------------------------------
// 8. NNZ (non-zero count) correctness in sparse representation
// ---------------------------------------------------------------------------

#[test]
fn test_sparse_nnz_count() {
    let dense: Array2<f64> = array![
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 5.0],
        [2.0, 0.0, 5.0, 6.0]
    ];

    let sparse_view = sparse_dense::sparse_from_ndarray(&dense.view(), 1e-10_f64)
        .expect("Dense to sparse conversion failed");

    // Count non-zeros manually
    let expected_nnz = dense.iter().filter(|&&v| v.abs() > 1e-10).count();
    assert_eq!(
        sparse_view.nnz(),
        expected_nnz,
        "NNZ mismatch: sparse={}, expected={}",
        sparse_view.nnz(),
        expected_nnz
    );
}

// ---------------------------------------------------------------------------
// 9. Sparse transpose is consistent with dense transpose
// ---------------------------------------------------------------------------

#[test]
fn test_sparse_transpose_consistency() {
    let dense: Array2<f64> = array![[1.0, 2.0, 0.0], [0.0, -3.0, 4.0]];

    let sparse_view = sparse_dense::sparse_from_ndarray(&dense.view(), 1e-10_f64)
        .expect("Dense to sparse conversion failed");

    let sparse_t = sparse_dense::sparse_transpose(&sparse_view).expect("Sparse transpose failed");

    let dense_t = dense.t();
    let reconstructed = sparse_t.to_dense();

    for i in 0..3 {
        for j in 0..2 {
            assert_abs_diff_eq!(reconstructed[[i, j]], dense_t[[i, j]], epsilon = 1e-12);
        }
    }
}
