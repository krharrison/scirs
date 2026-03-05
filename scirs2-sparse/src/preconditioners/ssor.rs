//! SSOR (Symmetric Successive Over-Relaxation) preconditioner
//!
//! The SSOR preconditioner approximates A⁻¹ via
//!
//!   M = (D/ω + L)⁻¹ · (D/ω) · (D/ω + U)⁻¹
//!
//! where D = diag(A), L = strictly lower-triangular part of A,
//! U = strictly upper-triangular part of A, and ω ∈ (0, 2) is the
//! relaxation parameter.
//!
//! Application proceeds in two sweeps:
//!
//! 1. **Forward sweep**: solve `(D/ω + L) y = r`
//!    →  `y_i = ω (r_i − Σ_{j<i} A_{ij} y_j) / A_{ii}`
//!
//! 2. **Backward sweep**: solve `(D/ω + U) x = (D/ω) y`
//!    →  `x_i = ω (D_i/ω · y_i − Σ_{j>i} A_{ij} x_j) / A_{ii}`
//!
//! When ω = 1 this reduces to the symmetric Gauss-Seidel preconditioner.
//!
//! # References
//!
//! - Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed.
//!   SIAM.  §10.1.2.
//! - Barrett, R. et al. (1994). *Templates for the Solution of Linear Systems*.
//!   SIAM.

use crate::error::{SparseError, SparseResult};

/// SSOR (Symmetric Successive Over-Relaxation) preconditioner.
pub struct SSORPrecond {
    /// Relaxation parameter ω ∈ (0, 2).
    pub omega: f64,
    /// Diagonal entries of A.
    pub diag: Vec<f64>,
    /// Lower-triangular part of A (strictly, i.e., without diagonal).
    /// Stored as (indptr, indices, data) in CSR.
    pub l: (Vec<usize>, Vec<usize>, Vec<f64>),
    /// Upper-triangular part of A (strictly, i.e., without diagonal).
    /// Stored as (indptr, indices, data) in CSR.
    pub u: (Vec<usize>, Vec<usize>, Vec<f64>),
    n: usize,
}

impl SSORPrecond {
    /// Build the SSOR preconditioner from a square CSR matrix.
    ///
    /// # Arguments
    ///
    /// * `indptr`, `indices`, `data` – CSR representation of A.
    /// * `n`     – Matrix dimension.
    /// * `omega` – Relaxation parameter ω; must satisfy 0 < ω < 2.
    ///
    /// Returns an error if any diagonal entry is zero or if ω is out of range.
    pub fn new(
        indptr: &[usize],
        indices: &[usize],
        data: &[f64],
        n: usize,
        omega: f64,
    ) -> SparseResult<Self> {
        if indptr.len() != n + 1 {
            return Err(SparseError::InconsistentData {
                reason: format!("indptr length {} != n+1={}", indptr.len(), n + 1),
            });
        }
        if omega <= 0.0 || omega >= 2.0 {
            return Err(SparseError::ValueError(format!(
                "SSOR omega={omega} must satisfy 0 < omega < 2"
            )));
        }

        let mut diag = vec![0.0f64; n];
        let mut l_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let mut u_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];

        for i in 0..n {
            let mut found_diag = false;
            for pos in indptr[i]..indptr[i + 1] {
                let j = indices[pos];
                let v = data[pos];
                if j == i {
                    diag[i] = v;
                    found_diag = true;
                } else if j < i {
                    l_rows[i].push((j, v));
                } else {
                    u_rows[i].push((j, v));
                }
            }
            if !found_diag || diag[i].abs() < 1e-300 {
                return Err(SparseError::SingularMatrix(format!(
                    "SSOR: zero or missing diagonal at row {i}"
                )));
            }
        }

        let (l_indptr, l_indices, l_data) = rows_to_csr(&l_rows, n);
        let (u_indptr, u_indices, u_data) = rows_to_csr(&u_rows, n);

        Ok(Self {
            omega,
            diag,
            l: (l_indptr, l_indices, l_data),
            u: (u_indptr, u_indices, u_data),
            n,
        })
    }

    /// Apply the SSOR preconditioner: compute x = M⁻¹ r.
    ///
    /// The two-sweep algorithm:
    ///
    /// 1. Forward: `(D/ω + L) y = r`
    ///    →  `y_i = ω (r_i − Σ_{j<i} A_{ij} y_j) / A_{ii}`
    ///
    /// 2. Backward: `(D/ω + U) x = (D/ω) y`
    ///    →  `x_i = ω (A_{ii}/ω · y_i − Σ_{j>i} A_{ij} x_j) / A_{ii}`
    ///    simplifies to:
    ///    →  `x_i = y_i − ω Σ_{j>i} A_{ij} x_j / A_{ii}`
    pub fn apply(&self, r: &[f64]) -> Vec<f64> {
        let n = self.n;
        let omega = self.omega;
        let (l_ip, l_idx, l_dat) = &self.l;
        let (u_ip, u_idx, u_dat) = &self.u;

        // --- Forward sweep ---
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let mut acc = r[i];
            for pos in l_ip[i]..l_ip[i + 1] {
                acc -= l_dat[pos] * y[l_idx[pos]];
            }
            // y_i = omega * acc / diag_i
            y[i] = omega * acc / self.diag[i];
        }

        // --- Backward sweep ---
        // x_i = y_i - omega * Σ_{j>i} A_{ij} x_j / A_{ii}
        let mut x = vec![0.0f64; n];
        for ii in 0..n {
            let i = n - 1 - ii;
            let mut acc = y[i];
            for pos in u_ip[i]..u_ip[i + 1] {
                acc -= omega * u_dat[pos] * x[u_idx[pos]] / self.diag[i];
            }
            x[i] = acc;
        }

        x
    }

    /// Return the matrix dimension.
    pub fn size(&self) -> usize {
        self.n
    }

    /// Return the relaxation parameter ω.
    pub fn omega(&self) -> f64 {
        self.omega
    }
}

fn rows_to_csr(rows: &[Vec<(usize, f64)>], n: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut indptr = vec![0usize; n + 1];
    for (i, row) in rows.iter().enumerate() {
        indptr[i + 1] = indptr[i] + row.len();
    }
    let nnz = indptr[n];
    let mut col_indices = vec![0usize; nnz];
    let mut values = vec![0.0f64; nnz];
    for (i, row) in rows.iter().enumerate() {
        let start = indptr[i];
        for (k, &(col, val)) in row.iter().enumerate() {
            col_indices[start + k] = col;
            values[start + k] = val;
        }
    }
    (indptr, col_indices, values)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_csr() -> (Vec<usize>, Vec<usize>, Vec<f64>, usize) {
        let n = 4usize;
        let indptr = vec![0, 2, 5, 8, 10];
        let indices = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let data = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        (indptr, indices, data, n)
    }

    fn matvec(ip: &[usize], idx: &[usize], dat: &[f64], x: &[f64], n: usize) -> Vec<f64> {
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            for pos in ip[i]..ip[i + 1] {
                y[i] += dat[pos] * x[idx[pos]];
            }
        }
        y
    }

    #[test]
    fn test_ssor_omega1_equals_sgs() {
        // With omega=1, SSOR = symmetric Gauss-Seidel.
        let (ip, idx, dat, n) = build_test_csr();
        let ssor = SSORPrecond::new(&ip, &idx, &dat, n, 1.0).expect("SSOR omega=1 failed");

        assert_eq!(ssor.size(), n);
        assert_eq!(ssor.omega(), 1.0);

        let b = vec![1.0, 2.0, 3.0, 4.0];
        let x = ssor.apply(&b);
        assert_eq!(x.len(), n);
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ssor_valid_omega() {
        let (ip, idx, dat, n) = build_test_csr();
        for &omega in &[0.5, 1.0, 1.5, 1.9] {
            let ssor = SSORPrecond::new(&ip, &idx, &dat, n, omega)
                .unwrap_or_else(|e| panic!("SSOR omega={omega} failed: {e}"));
            let b = vec![1.0; n];
            let x = ssor.apply(&b);
            assert!(x.iter().all(|v| v.is_finite()), "omega={omega} gave non-finite");
        }
    }

    #[test]
    fn test_ssor_invalid_omega() {
        let (ip, idx, dat, n) = build_test_csr();
        assert!(SSORPrecond::new(&ip, &idx, &dat, n, 0.0).is_err());
        assert!(SSORPrecond::new(&ip, &idx, &dat, n, 2.0).is_err());
        assert!(SSORPrecond::new(&ip, &idx, &dat, n, -0.5).is_err());
    }

    #[test]
    fn test_ssor_reduces_residual() {
        let (ip, idx, dat, n) = build_test_csr();
        let ssor = SSORPrecond::new(&ip, &idx, &dat, n, 1.0).expect("SSOR");

        let b = vec![3.0, 2.0, 2.0, 3.0];
        let x = ssor.apply(&b);

        // Compute A * (M^{-1} b) and check that the residual is reduced.
        let ax = matvec(&ip, &idx, &dat, &x, n);
        let res: f64 = b.iter().zip(ax.iter()).map(|(bi, ai)| (bi - ai).powi(2)).sum::<f64>().sqrt();
        let b_norm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();

        assert!(
            res < b_norm,
            "SSOR should reduce residual: res={res}, b_norm={b_norm}"
        );
    }

    #[test]
    fn test_ssor_zero_diagonal_error() {
        let n = 3;
        let indptr = vec![0, 1, 2, 3];
        let indices = vec![0, 1, 2];
        let data = vec![1.0, 0.0, 1.0]; // zero diagonal at row 1
        let result = SSORPrecond::new(&indptr, &indices, &data, n, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_ssor_diagonal_matrix() {
        // Pure diagonal matrix: SSOR should invert exactly.
        let n = 3;
        let indptr = vec![0, 1, 2, 3];
        let indices = vec![0, 1, 2];
        let data = vec![2.0, 3.0, 4.0];
        let ssor = SSORPrecond::new(&indptr, &indices, &data, n, 1.0).expect("SSOR diag");
        let b = vec![4.0, 9.0, 16.0];
        let x = ssor.apply(&b);
        // For diagonal A with omega=1: forward gives y_i = b_i / d_i,
        // backward gives x_i = y_i.  So x = b ./ diag.
        assert!((x[0] - 2.0).abs() < 1e-10, "x[0] = {}", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-10, "x[1] = {}", x[1]);
        assert!((x[2] - 4.0).abs() < 1e-10, "x[2] = {}", x[2]);
    }
}
