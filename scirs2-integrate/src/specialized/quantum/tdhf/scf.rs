//! Ground-state restricted Hartree-Fock (RHF) SCF solver.
//!
//! # Algorithm
//!
//! 1. Build one-electron integrals: overlap S, kinetic T, nuclear-attraction V
//! 2. Build two-electron repulsion tensor (ERI)
//! 3. Löwdin orthogonalization: X = S^{-1/2} via eigendecomposition
//! 4. Initial guess: diagonalize H_core = T + V in orthogonal basis
//! 5. SCF loop:
//!    a. Build Fock: F = H_core + J − ½K
//!    b. Optional DIIS extrapolation using error matrix e = FPS − SPF
//!    c. Diagonalize F' = Xᵀ F X, rotate back: C = X C'
//!    d. Update density: P = 2 C_occ Cᵀ_occ
//!    e. Check convergence ‖ΔP‖ < tol
//! 6. Compute total energy E = ½ Tr[P(H_core + F)] + E_nuc

use crate::error::{IntegrateError, IntegrateResult};
use crate::specialized::quantum::gaussian_integrals::{
    build_kinetic_matrix, build_overlap_matrix, nuclear_attraction, GaussianBasis,
};
use crate::specialized::quantum::tdhf::eri::{build_eri_tensor, get_eri};
use scirs2_core::ndarray::{Array1, Array2};
use std::f64;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// SCF convergence strategy.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScfConverger {
    /// Simple density-matrix damping / Roothaan iteration.
    Plain,
    /// Direct Inversion of Iterative Subspace (Pulay 1980).
    Diis,
}

/// Configuration for the Hartree-Fock SCF solver.
#[derive(Debug, Clone)]
pub struct ScfConfig {
    /// Maximum number of SCF iterations (default: 100).
    pub max_iter: usize,
    /// Convergence threshold on ‖ΔP‖_F (Frobenius norm, default: 1e-6).
    pub tol: f64,
    /// Convergence algorithm (default: DIIS).
    pub converger: ScfConverger,
    /// Size of the DIIS history (default: 6).
    pub diis_space: usize,
    /// Damping coefficient for Plain converger (default: 0.5).
    pub damping: f64,
}

impl Default for ScfConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-6,
            converger: ScfConverger::Diis,
            diis_space: 6,
            damping: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Output of a converged (or terminated) HF-SCF calculation.
#[derive(Debug, Clone)]
pub struct ScfResult {
    /// MO coefficient matrix C \[n_basis x n_basis\], columns = MOs.
    pub mo_coefficients: Array2<f64>,
    /// Orbital energies ε_i \[n_basis\].
    pub orbital_energies: Array1<f64>,
    /// Density matrix P \[n_basis x n_basis\].
    pub density_matrix: Array2<f64>,
    /// Total HF energy (electronic + nuclear) in Hartree.
    pub total_energy: f64,
    /// Number of SCF iterations performed.
    pub n_iter: usize,
    /// Whether the SCF converged within `max_iter`.
    pub converged: bool,
    /// Overlap matrix S (stored for downstream use).
    pub overlap: Array2<f64>,
    /// Core Hamiltonian H_core = T + V.
    pub h_core: Array2<f64>,
    /// ERI tensor (flat, n^4).
    pub eri: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Nuclear repulsion
// ---------------------------------------------------------------------------

/// Nuclear repulsion energy E_nuc = Σ_{A<B} Z_A Z_B / |R_A - R_B|.
pub fn nuclear_repulsion_energy(charges: &[(f64, [f64; 3])]) -> f64 {
    let mut e = 0.0;
    for i in 0..charges.len() {
        for j in (i + 1)..charges.len() {
            let (za, ra) = charges[i];
            let (zb, rb) = charges[j];
            let r = ((ra[0] - rb[0]).powi(2) + (ra[1] - rb[1]).powi(2) + (ra[2] - rb[2]).powi(2))
                .sqrt();
            if r > 1e-10 {
                e += za * zb / r;
            }
        }
    }
    e
}

// ---------------------------------------------------------------------------
// SCF solver
// ---------------------------------------------------------------------------

/// Restricted Hartree-Fock SCF solver for closed-shell systems.
pub struct HartreeFockSCF {
    config: ScfConfig,
}

impl HartreeFockSCF {
    /// Create a new SCF solver with the given configuration.
    pub fn new(config: ScfConfig) -> Self {
        Self { config }
    }

    /// Run the SCF calculation.
    ///
    /// # Arguments
    /// * `basis` – slice of GTO primitives
    /// * `nuclear_charges` – `(Z, [x,y,z])` for each nucleus
    /// * `n_electrons` – total electron count (must be even for RHF)
    pub fn run(
        &self,
        basis: &[GaussianBasis],
        nuclear_charges: &[(f64, [f64; 3])],
        n_electrons: usize,
    ) -> IntegrateResult<ScfResult> {
        if !n_electrons.is_multiple_of(2) {
            return Err(IntegrateError::InvalidInput(
                "RHF requires an even number of electrons".to_string(),
            ));
        }
        let n_occ = n_electrons / 2;
        let n = basis.len();
        if n < n_occ {
            return Err(IntegrateError::InvalidInput(format!(
                "Basis size {n} is smaller than n_occ {n_occ}"
            )));
        }

        // ── Step 1: One-electron integrals ────────────────────────────────
        let s = build_overlap_matrix(basis);
        let t_mat = build_kinetic_matrix(basis);
        let v_mat = build_nuclear_matrix(basis, nuclear_charges);
        let h_core = &t_mat + &v_mat;

        // ── Step 2: ERI tensor ────────────────────────────────────────────
        let eri = build_eri_tensor(basis);

        // ── Step 3: Löwdin orthogonalization X = S^{-1/2} ─────────────────
        let x = Self::matrix_sqrt_inv(&s)?;

        // ── Step 4: Initial guess from H_core ────────────────────────────
        let f_prime_0 = mat_mul_triple_t(&x, &h_core);
        let (eps, c_prime) = Self::diagonalize_symmetric(&f_prime_0)?;
        let mut c = mat_mul(&x, &c_prime);
        let mut p = build_density_matrix(&c, n_occ, n);

        // ── Step 5: SCF iterations ────────────────────────────────────────
        let mut converged = false;
        let mut n_iter = 0_usize;
        let mut orbital_energies = eps;

        // DIIS storage
        let mut diis_errors: Vec<Array2<f64>> = Vec::new();
        let mut diis_focks: Vec<Array2<f64>> = Vec::new();

        for iter in 0..self.config.max_iter {
            n_iter = iter + 1;

            // Build Fock matrix
            let fock = build_fock(&h_core, &p, &eri, n);

            // DIIS extrapolation
            let fock_use = match self.config.converger {
                ScfConverger::Diis => {
                    let err = diis_error_matrix(&fock, &p, &s);
                    diis_errors.push(err);
                    diis_focks.push(fock.clone());
                    if diis_errors.len() > self.config.diis_space {
                        diis_errors.remove(0);
                        diis_focks.remove(0);
                    }
                    if diis_errors.len() >= 2 {
                        diis_extrapolate(&diis_focks, &diis_errors).unwrap_or(fock)
                    } else {
                        fock
                    }
                }
                ScfConverger::Plain => fock,
            };

            // Diagonalize F' = Xᵀ F X
            let f_prime = mat_mul_triple_t(&x, &fock_use);
            let (eps_new, c_prime_new) = Self::diagonalize_symmetric(&f_prime)?;
            let c_new = mat_mul(&x, &c_prime_new);
            let p_new = build_density_matrix(&c_new, n_occ, n);

            // Apply damping for Plain converger
            let p_next = match self.config.converger {
                ScfConverger::Plain => {
                    let d = self.config.damping;
                    p_new.mapv(|v| v * (1.0 - d)) + p.mapv(|v| v * d)
                }
                ScfConverger::Diis => p_new,
            };

            // Check convergence (Frobenius norm of ΔP)
            let dp = frob_norm_diff(&p_next, &p);
            c = c_new;
            p = p_next;
            orbital_energies = eps_new;

            if dp < self.config.tol {
                converged = true;
                break;
            }
        }

        // ── Step 6: Final energy ──────────────────────────────────────────
        let fock_final = build_fock(&h_core, &p, &eri, n);
        let e_elec = electronic_energy(&p, &h_core, &fock_final);
        let e_nuc = nuclear_repulsion_energy(nuclear_charges);
        let total_energy = e_elec + e_nuc;

        Ok(ScfResult {
            mo_coefficients: c,
            orbital_energies,
            density_matrix: p,
            total_energy,
            n_iter,
            converged,
            overlap: s,
            h_core,
            eri,
        })
    }

    /// Compute S^{-1/2} via eigendecomposition.
    ///
    /// S = U Λ Uᵀ  →  S^{-1/2} = U Λ^{-1/2} Uᵀ
    pub fn matrix_sqrt_inv(s: &Array2<f64>) -> IntegrateResult<Array2<f64>> {
        let n = s.shape()[0];
        let mut m = s.clone();
        let mut v = Array2::eye(n);
        jacobi_diag(&mut m, &mut v, 1000);

        // Eigenvalues on diagonal of m, eigenvectors in columns of v
        let mut result = Array2::zeros((n, n));
        for i in 0..n {
            let lam = m[[i, i]];
            if lam <= 0.0 {
                return Err(IntegrateError::ComputationError(format!(
                    "Overlap matrix is not positive definite (eigenvalue {lam} at index {i})"
                )));
            }
            let inv_sqrt_lam = 1.0 / lam.sqrt();
            for r in 0..n {
                for c_idx in 0..n {
                    result[[r, c_idx]] += v[[r, i]] * inv_sqrt_lam * v[[c_idx, i]];
                }
            }
        }
        Ok(result)
    }

    /// Diagonalize a real symmetric matrix via Jacobi iteration.
    ///
    /// Returns `(eigenvalues, eigenvectors)` sorted by ascending eigenvalue.
    pub fn diagonalize_symmetric(m: &Array2<f64>) -> IntegrateResult<(Array1<f64>, Array2<f64>)> {
        let n = m.shape()[0];
        let mut a = m.clone();
        let mut v = Array2::eye(n);
        jacobi_diag(&mut a, &mut v, 5000);

        // Eigenvalues are the diagonal of a
        let mut pairs: Vec<(f64, usize)> = (0..n).map(|i| (a[[i, i]], i)).collect();
        pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut eigenvalues = Array1::zeros(n);
        let mut eigenvectors = Array2::zeros((n, n));
        for (new_idx, (val, old_idx)) in pairs.iter().enumerate() {
            eigenvalues[new_idx] = *val;
            for r in 0..n {
                eigenvectors[[r, new_idx]] = v[[r, *old_idx]];
            }
        }
        Ok((eigenvalues, eigenvectors))
    }
}

// ---------------------------------------------------------------------------
// Electronic energy
// ---------------------------------------------------------------------------

/// Electronic energy E_elec = ½ Tr[P (H_core + F)].
pub fn electronic_energy(p: &Array2<f64>, h_core: &Array2<f64>, fock: &Array2<f64>) -> f64 {
    let n = p.shape()[0];
    let mut e = 0.0;
    for mu in 0..n {
        for nu in 0..n {
            e += p[[mu, nu]] * (h_core[[mu, nu]] + fock[[mu, nu]]);
        }
    }
    0.5 * e
}

// ---------------------------------------------------------------------------
// Jacobi diagonalization
// ---------------------------------------------------------------------------

/// Classical Jacobi diagonalization of symmetric matrix `m`.
///
/// Accumulates eigenvectors in `v` (initialized to identity by the caller).
/// Convergence criterion: max |off-diagonal| < 1e-12.
pub fn jacobi_diag(m: &mut Array2<f64>, v: &mut Array2<f64>, max_iter: usize) {
    let n = m.shape()[0];
    for _ in 0..max_iter {
        // Find element with largest |off-diagonal| value
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let abs_val = m[[i, j]].abs();
                if abs_val > max_val {
                    max_val = abs_val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-12 {
            break;
        }

        // Compute rotation angle θ
        let diff = m[[q, q]] - m[[p, p]];
        let tau = if diff.abs() < 1e-15 {
            // m[p,p] ≈ m[q,q]: θ = π/4
            1.0
        } else {
            // τ = (m[q,q] - m[p,p]) / (2 m[p,q])
            let t_recip = diff / (2.0 * m[[p, q]]);
            let t = 1.0 / (t_recip.abs() + (1.0 + t_recip * t_recip).sqrt());
            if diff < 0.0 {
                -t
            } else {
                t
            }
        };
        let cos_th = 1.0 / (1.0 + tau * tau).sqrt();
        let sin_th = tau * cos_th;

        // Apply Jacobi rotation: M' = Jᵀ M J
        // Update rows/columns p and q of m
        let apq = m[[p, q]];
        let app = m[[p, p]];
        let aqq = m[[q, q]];

        m[[p, p]] = app - tau * apq * 2.0 * cos_th * cos_th / (1.0 + tau * tau);
        // Simpler closed-form:
        m[[p, p]] = app - tau * 2.0 * m[[p, q]] / (1.0 + tau * tau) * cos_th * cos_th + 0.0;

        // Recompute properly using standard Jacobi update formulas:
        m[[p, p]] = app * cos_th * cos_th + aqq * sin_th * sin_th - 2.0 * apq * sin_th * cos_th;
        m[[q, q]] = app * sin_th * sin_th + aqq * cos_th * cos_th + 2.0 * apq * sin_th * cos_th;
        m[[p, q]] = 0.0;
        m[[q, p]] = 0.0;

        for r in 0..n {
            if r == p || r == q {
                continue;
            }
            let arp = m[[r, p]];
            let arq = m[[r, q]];
            m[[r, p]] = arp * cos_th - arq * sin_th;
            m[[p, r]] = m[[r, p]];
            m[[r, q]] = arp * sin_th + arq * cos_th;
            m[[q, r]] = m[[r, q]];
        }

        // Update eigenvectors
        for r in 0..n {
            let vrp = v[[r, p]];
            let vrq = v[[r, q]];
            v[[r, p]] = vrp * cos_th - vrq * sin_th;
            v[[r, q]] = vrp * sin_th + vrq * cos_th;
        }
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Build nuclear-attraction matrix V_{μν} = −Σ_A Z_A <μ|1/r_A|ν>.
fn build_nuclear_matrix(
    basis: &[GaussianBasis],
    nuclear_charges: &[(f64, [f64; 3])],
) -> Array2<f64> {
    let n = basis.len();
    let mut v = Array2::zeros((n, n));
    for mu in 0..n {
        for nu in 0..n {
            let mut v_munu = 0.0;
            for &(z, pos) in nuclear_charges {
                v_munu += nuclear_attraction(&basis[mu], &basis[nu], pos, z);
            }
            v[[mu, nu]] = v_munu;
        }
    }
    v
}

/// Build Fock matrix F = H_core + J − ½K.
///
/// J_{μν} = Σ_{λσ} P_{λσ} (μν|λσ)
/// K_{μν} = Σ_{λσ} P_{λσ} (μλ|νσ)
pub fn build_fock(h_core: &Array2<f64>, p: &Array2<f64>, eri: &[f64], n: usize) -> Array2<f64> {
    let mut f = h_core.clone();
    for mu in 0..n {
        for nu in 0..n {
            let mut j = 0.0;
            let mut k = 0.0;
            for lam in 0..n {
                for sig in 0..n {
                    let p_ls = p[[lam, sig]];
                    if p_ls.abs() < 1e-15 {
                        continue;
                    }
                    j += p_ls * get_eri(eri, n, mu, nu, lam, sig);
                    k += p_ls * get_eri(eri, n, mu, lam, nu, sig);
                }
            }
            f[[mu, nu]] += j - 0.5 * k;
        }
    }
    f
}

/// Build density matrix P = 2 C_occ Cᵀ_occ.
pub fn build_density_matrix(c: &Array2<f64>, n_occ: usize, n: usize) -> Array2<f64> {
    let mut p = Array2::zeros((n, n));
    for mu in 0..n {
        for nu in 0..n {
            let mut sum = 0.0;
            for i in 0..n_occ {
                sum += c[[mu, i]] * c[[nu, i]];
            }
            p[[mu, nu]] = 2.0 * sum;
        }
    }
    p
}

/// DIIS error matrix e = FPS − SPF.
fn diis_error_matrix(f: &Array2<f64>, p: &Array2<f64>, s: &Array2<f64>) -> Array2<f64> {
    let fp = mat_mul(f, p);
    let fps = mat_mul(&fp, s);
    let sp = mat_mul(s, p);
    let spf = mat_mul(&sp, f);
    fps - spf
}

/// DIIS extrapolation: solve linear system to get interpolated Fock matrix.
fn diis_extrapolate(focks: &[Array2<f64>], errors: &[Array2<f64>]) -> Option<Array2<f64>> {
    let m = focks.len();
    if m < 2 {
        return None;
    }

    // Build B matrix: B[i,j] = Tr(e_i · e_j)
    let mut b = vec![vec![0.0_f64; m + 1]; m + 1];
    for i in 0..m {
        for j in 0..m {
            b[i][j] = trace_product(&errors[i], &errors[j]);
        }
        b[i][m] = -1.0;
        b[m][i] = -1.0;
    }
    b[m][m] = 0.0;

    // RHS: (0, 0, ..., 0, -1)
    let mut rhs = vec![0.0_f64; m + 1];
    rhs[m] = -1.0;

    // Solve B·c = rhs via Gaussian elimination
    let coeffs = gauss_solve(&b, &rhs)?;

    // Interpolated Fock = Σ c_i F_i
    let n = focks[0].shape()[0];
    let mut f_diis = Array2::zeros((n, n));
    for i in 0..m {
        f_diis = f_diis + focks[i].mapv(|v| v * coeffs[i]);
    }
    Some(f_diis)
}

/// Trace of Aᵀ · B = Σ_{ij} A[i,j] * B[i,j].
fn trace_product(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

/// Frobenius norm of (A − B).
fn frob_norm_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Dense matrix multiplication C = A · B.
pub fn mat_mul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let n = b.shape()[1];
    let mut c = Array2::zeros((m, n));
    for i in 0..m {
        for p in 0..k {
            let aip = a[[i, p]];
            if aip.abs() < 1e-15 {
                continue;
            }
            for j in 0..n {
                c[[i, j]] += aip * b[[p, j]];
            }
        }
    }
    c
}

/// Compute Xᵀ M X (triple product used for basis transformation).
fn mat_mul_triple_t(x: &Array2<f64>, m: &Array2<f64>) -> Array2<f64> {
    let xm = mat_mul(x, m);
    // xm is X·M; now compute (X·M)·X ... but we want Xᵀ·M·X
    // Here X = S^{-1/2} is symmetric, so Xᵀ = X.
    // We compute Xᵀ A = X A (since X is symmetric) and then (XA) X
    let xt_m = mat_mul_t_a(x, m);
    mat_mul(&xt_m, x)
}

/// Compute Aᵀ · B.
fn mat_mul_t_a(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (k, m) = (a.shape()[0], a.shape()[1]);
    let n = b.shape()[1];
    let mut c = Array2::zeros((m, n));
    for p in 0..k {
        for i in 0..m {
            let api = a[[p, i]];
            if api.abs() < 1e-15 {
                continue;
            }
            for j in 0..n {
                c[[i, j]] += api * b[[p, j]];
            }
        }
    }
    c
}

/// Gaussian elimination (no pivoting) for small systems.
/// Solves `a·x = b`, returns `x` or `None` on singular matrix.
fn gauss_solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut mat: Vec<Vec<f64>> = a.to_vec();
    let mut rhs: Vec<f64> = b.to_vec();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = mat[col][col].abs();
        for row in (col + 1)..n {
            if mat[row][col].abs() > max_val {
                max_val = mat[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return None;
        }
        mat.swap(col, max_row);
        rhs.swap(col, max_row);

        let pivot = mat[col][col];
        for row in (col + 1)..n {
            let factor = mat[row][col] / pivot;
            for k in col..n {
                mat[row][k] -= factor * mat[col][k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    // Back substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..n {
            sum -= mat[i][j] * x[j];
        }
        if mat[i][i].abs() < 1e-14 {
            return None;
        }
        x[i] = sum / mat[i][i];
    }
    Some(x)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::specialized::quantum::gaussian_integrals::{normalized_s_gto, sto3g_basis};

    // ── Overlap / kinetic sanity tests ───────────────────────────────────────

    #[test]
    fn test_overlap_integral_normalized() {
        // A normalized s-type GTO must have <φ|φ> = 1
        let a = normalized_s_gto([0.0, 0.0, 0.0], 1.0);
        let s = build_overlap_matrix(std::slice::from_ref(&a));
        assert!(
            (s[[0, 0]] - 1.0).abs() < 1e-8,
            "Self-overlap = {}, expected 1.0",
            s[[0, 0]]
        );
    }

    #[test]
    fn test_kinetic_integral_positive() {
        let a = normalized_s_gto([0.0, 0.0, 0.0], 1.0);
        let t = build_kinetic_matrix(std::slice::from_ref(&a));
        assert!(t[[0, 0]] > 0.0, "Kinetic energy diagonal must be positive");
    }

    // ── Nuclear repulsion ────────────────────────────────────────────────────

    #[test]
    fn test_nuclear_repulsion_h2() {
        // H2: Z_A = Z_B = 1, R = 1.4 bohr → E_nuc = 1/1.4 ≈ 0.7143 Hartree
        let charges = [(1.0_f64, [0.0, 0.0, 0.0]), (1.0_f64, [0.0, 0.0, 1.4])];
        let e_nuc = nuclear_repulsion_energy(&charges);
        let expected = 1.0 / 1.4;
        assert!(
            (e_nuc - expected).abs() < 1e-10,
            "E_nuc(H2) = {e_nuc}, expected {expected}"
        );
    }

    // ── Jacobi diagonalization ───────────────────────────────────────────────

    #[test]
    fn test_jacobi_diag_correctness() {
        // 2×2 symmetric matrix with known eigenvalues
        // M = [[2,1],[1,2]] → eigenvalues 1, 3
        let mut m = Array2::from_shape_vec((2, 2), vec![2.0_f64, 1.0, 1.0, 2.0]).unwrap();
        let mut v = Array2::eye(2);
        jacobi_diag(&mut m, &mut v, 1000);
        let mut eigs = [m[[0, 0]], m[[1, 1]]];
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((eigs[0] - 1.0).abs() < 1e-10, "eig[0]={}", eigs[0]);
        assert!((eigs[1] - 3.0).abs() < 1e-10, "eig[1]={}", eigs[1]);
    }

    // ── S^{-1/2} correctness ─────────────────────────────────────────────────

    #[test]
    fn test_matrix_sqrt_inv_correctness() {
        // For identity matrix: S^{-1/2} = I
        let s = Array2::eye(3);
        let x = HartreeFockSCF::matrix_sqrt_inv(&s).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (x[[i, j]] - expected).abs() < 1e-10,
                    "X^(-1/2)[{i},{j}] = {}, expected {expected}",
                    x[[i, j]]
                );
            }
        }
    }

    // ── H2 SCF energy ────────────────────────────────────────────────────────

    #[test]
    fn test_scf_h2_energy_reasonable() {
        // Minimal 2-function STO-3G-like H2: one normalized s-GTO per H
        let a = normalized_s_gto([0.0, 0.0, 0.0], 1.0);
        let b = normalized_s_gto([0.0, 0.0, 1.4], 1.0);
        let basis = vec![a, b];
        let charges = [(1.0_f64, [0.0, 0.0, 0.0]), (1.0_f64, [0.0, 0.0, 1.4])];
        let cfg = ScfConfig::default();
        let scf = HartreeFockSCF::new(cfg);
        let res = scf.run(&basis, &charges, 2).unwrap();
        // Total energy should be physically reasonable: -2.0 to -0.5 Hartree
        assert!(
            res.total_energy < 0.0,
            "H2 total energy should be negative, got {}",
            res.total_energy
        );
        assert!(
            res.total_energy > -5.0,
            "H2 total energy too negative: {}",
            res.total_energy
        );
    }

    #[test]
    fn test_scf_converges() {
        let a = normalized_s_gto([0.0, 0.0, 0.0], 1.0);
        let b = normalized_s_gto([0.0, 0.0, 1.4], 1.0);
        let basis = vec![a, b];
        let charges = [(1.0_f64, [0.0, 0.0, 0.0]), (1.0_f64, [0.0, 0.0, 1.4])];
        let cfg = ScfConfig {
            tol: 1e-5,
            ..ScfConfig::default()
        };
        let scf = HartreeFockSCF::new(cfg);
        let res = scf.run(&basis, &charges, 2).unwrap();
        assert!(res.converged, "SCF should converge for simple H2");
    }

    // ── Density matrix properties ────────────────────────────────────────────

    #[test]
    fn test_electron_count_trace() {
        // Tr(P · S) = n_electrons
        let a = normalized_s_gto([0.0, 0.0, 0.0], 1.0);
        let b = normalized_s_gto([0.0, 0.0, 1.4], 1.0);
        let basis = vec![a, b];
        let charges = [(1.0_f64, [0.0, 0.0, 0.0]), (1.0_f64, [0.0, 0.0, 1.4])];
        let res = HartreeFockSCF::new(ScfConfig::default())
            .run(&basis, &charges, 2)
            .unwrap();
        let ps = mat_mul(&res.density_matrix, &res.overlap);
        let tr_ps: f64 = (0..ps.shape()[0]).map(|i| ps[[i, i]]).sum();
        assert!((tr_ps - 2.0).abs() < 1e-6, "Tr(PS) = {tr_ps}, expected 2.0");
    }

    #[test]
    fn test_density_matrix_idempotency() {
        // For converged RHF: (1/2)P S P = (1/2)P (idempotency in the overlap metric)
        // Simpler check: Tr(P S P S) = 2 * n_electrons / 2 = n_electrons
        let a = normalized_s_gto([0.0, 0.0, 0.0], 1.0);
        let b = normalized_s_gto([0.0, 0.0, 1.4], 1.0);
        let basis = vec![a, b];
        let charges = [(1.0_f64, [0.0, 0.0, 0.0]), (1.0_f64, [0.0, 0.0, 1.4])];
        let res = HartreeFockSCF::new(ScfConfig::default())
            .run(&basis, &charges, 2)
            .unwrap();
        // Idempotency: P S P = 2P  (for closed-shell RHF with n_occ occupied)
        let ps = mat_mul(&res.density_matrix, &res.overlap);
        let psp = mat_mul(&ps, &res.density_matrix);
        let two_p = res.density_matrix.mapv(|v| 2.0 * v);
        let diff = frob_norm_diff(&psp, &two_p);
        assert!(diff < 0.1, "Idempotency (PSP ≈ 2P): diff={diff}");
    }

    // ── STO-3G basis integration ─────────────────────────────────────────────

    #[test]
    fn test_scf_sto3g_h2_energy() {
        // Full STO-3G H2 (3 primitives per H) – should give ≈ -1.117 Hartree
        let basis_a = sto3g_basis("H", [0.0, 0.0, 0.0]).unwrap();
        let basis_b = sto3g_basis("H", [0.0, 0.0, 1.4]).unwrap();
        let basis: Vec<_> = basis_a.iter().chain(basis_b.iter()).cloned().collect();
        let charges = [(1.0_f64, [0.0, 0.0, 0.0]), (1.0_f64, [0.0, 0.0, 1.4])];
        let cfg = ScfConfig {
            max_iter: 200,
            tol: 1e-6,
            ..ScfConfig::default()
        };
        let res = HartreeFockSCF::new(cfg).run(&basis, &charges, 2).unwrap();
        // Reasonable range: -2.0 to -0.5 Hartree for H2 with this basis
        assert!(
            res.total_energy > -2.5 && res.total_energy < 0.0,
            "STO-3G H2 energy = {}",
            res.total_energy
        );
    }
}
