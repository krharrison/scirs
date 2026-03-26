//! Linear-response TDHF: Casida equations (Tamm-Dancoff approximation).
//!
//! # Casida / Tamm-Dancoff equations
//!
//! In the Tamm-Dancoff approximation (TDA) the excitation energies ω_I and
//! transition amplitudes X_I satisfy:
//!
//!   A · X_I = ω_I · X_I
//!
//! where the singles-singles coupling matrix is:
//!
//!   A_{ia,jb} = (ε_a − ε_i) δ_{ij} δ_{ab} + (ia|jb) − (ij|ab)
//!
//! Indices: i,j run over occupied MOs; a,b run over virtual MOs.
//! The MO-basis ERIs are obtained by transforming the AO-basis ERI tensor with
//! the SCF MO coefficients:
//!
//!   (ia|jb) = Σ_{μνλσ} C_{μi} C_{νa} (μν|λσ) C_{λj} C_{σb}
//!
//! # Oscillator strengths
//!
//!   f_I = (2/3) ω_I Σ_{α∈{x,y,z}} |⟨0|r_α|I⟩|²
//!
//! where the transition dipole from state I via occupied-virtual transition ia:
//!   ⟨0|r_z|I⟩ = Σ_{ia} X_{ia,I} ⟨i|z|a⟩
//! and ⟨i|z|a⟩ = Σ_{μν} C_{μi} ⟨μ|z|ν⟩ C_{νa}.
//!
//! The dipole integrals use the centroid approximation for simplicity.

use crate::error::{IntegrateError, IntegrateResult};
use crate::specialized::quantum::gaussian_integrals::build_overlap_matrix;
use crate::specialized::quantum::gaussian_integrals::GaussianBasis;
use crate::specialized::quantum::tdhf::eri::get_eri;
use crate::specialized::quantum::tdhf::scf::{jacobi_diag, HartreeFockSCF, ScfResult};
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Casida/TDA linear-response solver.
#[derive(Debug, Clone)]
pub struct CasidaConfig {
    /// Number of lowest excitation roots to compute (default: 5).
    pub n_roots: usize,
}

impl Default for CasidaConfig {
    fn default() -> Self {
        Self { n_roots: 5 }
    }
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Output of the Casida / TDA linear-response calculation.
#[derive(Debug, Clone)]
pub struct CasidaResult {
    /// Vertical excitation energies ω_I in Hartree (ascending order).
    pub excitation_energies: Vec<f64>,
    /// Dipole oscillator strengths f_I (dimensionless, length gauge).
    pub oscillator_strengths: Vec<f64>,
    /// Transition dipole moments μ_I^{xyz} in atomic units.
    pub transition_dipoles: Vec<[f64; 3]>,
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

/// Casida linear-response solver (Tamm-Dancoff approximation).
pub struct CasidaSolver {
    config: CasidaConfig,
}

impl CasidaSolver {
    /// Create a solver with the given configuration.
    pub fn new(config: CasidaConfig) -> Self {
        Self { config }
    }

    /// Solve the TDA Casida equations for the lowest `n_roots` excitations.
    ///
    /// # Arguments
    /// * `scf_result`  – converged HF-SCF result
    /// * `basis`       – GTO basis functions
    /// * `eri`         – AO-basis ERI tensor (flat, n^4)
    /// * `n_electrons` – total electron count
    pub fn solve(
        &self,
        scf_result: &ScfResult,
        basis: &[GaussianBasis],
        eri: &[f64],
        n_electrons: usize,
    ) -> IntegrateResult<CasidaResult> {
        let n_basis = basis.len();
        let n_occ = n_electrons / 2;
        let n_virt = n_basis.saturating_sub(n_occ);

        if n_virt == 0 {
            return Err(IntegrateError::InvalidInput(
                "No virtual orbitals available for Casida calculation".to_string(),
            ));
        }

        let n_singles = n_occ * n_virt;
        if n_singles == 0 {
            return Err(IntegrateError::InvalidInput(
                "Zero single-excitation space".to_string(),
            ));
        }

        // ── Build MO-basis dipole integrals (centroid approximation) ────────
        let overlap = build_overlap_matrix(basis);
        let dipole_ao = build_dipole_ao(basis, &overlap);

        // ── Build A matrix (TDA) ─────────────────────────────────────────────
        let a_mat = Self::build_a_matrix(scf_result, eri, n_basis, n_occ)?;

        // ── Diagonalize A ────────────────────────────────────────────────────
        let (eigs, vecs) = HartreeFockSCF::diagonalize_symmetric(&a_mat)?;

        // ── Filter positive excitations, take lowest n_roots ────────────────
        // n_physical_roots: at most n_singles physical excitations available
        let n_physical_roots = n_singles;
        // We always output exactly config.n_roots entries, padding with zeros if necessary
        let n_roots_out = self.config.n_roots;
        let n_roots_calc = n_roots_out.min(n_physical_roots);

        let mut excitation_energies = Vec::with_capacity(n_roots_out);
        let mut oscillator_strengths = Vec::with_capacity(n_roots_out);
        let mut transition_dipoles = Vec::with_capacity(n_roots_out);

        // Collect and sort positive excitation energies
        let mut valid_roots: Vec<(f64, usize)> = eigs
            .iter()
            .enumerate()
            .filter(|(_, &e)| e > 0.0)
            .map(|(i, &e)| (e, i))
            .collect();
        valid_roots.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for (omega, root_idx) in valid_roots.iter().take(n_roots_calc) {
            excitation_energies.push(*omega);

            // Transition amplitudes X_{ia,I}
            let x_vec: Vec<f64> = (0..n_singles).map(|s| vecs[[s, *root_idx]]).collect();

            // Compute transition dipoles in MO basis
            let tdip =
                compute_transition_dipole(&x_vec, n_occ, n_virt, n_basis, &dipole_ao, scf_result);

            // Oscillator strength: f = (2/3) ω Σ_α |μ_α|²
            let mu_sq = tdip[0].powi(2) + tdip[1].powi(2) + tdip[2].powi(2);
            let osc = (2.0 / 3.0) * omega * mu_sq;

            oscillator_strengths.push(osc.max(0.0));
            transition_dipoles.push(tdip);
        }

        // Pad with zeros to always return exactly n_roots_out entries
        while excitation_energies.len() < n_roots_out {
            excitation_energies.push(0.0);
            oscillator_strengths.push(0.0);
            transition_dipoles.push([0.0; 3]);
        }

        Ok(CasidaResult {
            excitation_energies,
            oscillator_strengths,
            transition_dipoles,
        })
    }

    /// Build the Tamm-Dancoff A matrix in the MO basis.
    ///
    /// A_{ia,jb} = (ε_a − ε_i) δ_{ij} δ_{ab} + (ia|jb) − (ij|ab)
    ///
    /// MO-ERI: (ia|jb) = Σ_{μνλσ} C_{μi} C_{νa} (μν|λσ) C_{λj} C_{σb}
    pub fn build_a_matrix(
        scf: &ScfResult,
        eri: &[f64],
        n_basis: usize,
        n_occ: usize,
    ) -> IntegrateResult<Array2<f64>> {
        let n_virt = n_basis.saturating_sub(n_occ);
        let n_singles = n_occ * n_virt;
        let c = &scf.mo_coefficients;
        let eps = &scf.orbital_energies;

        // Pre-compute two-center MO integrals (i,a,j,b) via four-index transform.
        // For small basis (n ≤ 6) a direct transform is feasible.
        //
        // Step 1: half-transform (μν|λσ) → (iν|λσ) and (aν|λσ)
        // Step 2: full transform
        //
        // For simplicity we do the full 4-index direct transformation.
        // Cost: O(n_occ * n_virt * n_basis^4) which is fine for small systems.

        let mut a = Array2::zeros((n_singles, n_singles));

        // Map compound index ia → flat index
        let ia_idx = |i: usize, a: usize| i * n_virt + a;

        for i in 0..n_occ {
            for a_idx in 0..n_virt {
                let a_mo = n_occ + a_idx; // virtual MO index
                let row = ia_idx(i, a_idx);

                // Diagonal: orbital energy difference
                a[[row, row]] = eps[a_mo] - eps[i];

                for j in 0..n_occ {
                    for b_idx in 0..n_virt {
                        let b_mo = n_occ + b_idx;
                        let col = ia_idx(j, b_idx);

                        // (ia|jb) and (ij|ab) via 4-index AO→MO transform
                        let iajb = mo_eri(eri, n_basis, c, i, a_mo, j, b_mo);
                        let ijab = mo_eri(eri, n_basis, c, i, j, a_mo, b_mo);

                        a[[row, col]] += iajb - ijab;
                    }
                }
            }
        }

        Ok(a)
    }
}

// ---------------------------------------------------------------------------
// MO-basis ERI (4-index transform)
// ---------------------------------------------------------------------------

/// Transform AO-basis ERI to MO basis for a single quartet (p,q,r,s) of MOs.
///
/// (pq|rs)_MO = Σ_{μνλσ} C_{μp} C_{νq} (μν|λσ) C_{λr} C_{σs}
fn mo_eri(
    eri: &[f64],
    n_basis: usize,
    c: &Array2<f64>,
    p: usize,
    q: usize,
    r: usize,
    s: usize,
) -> f64 {
    let mut val = 0.0;
    for mu in 0..n_basis {
        let cmu_p = c[[mu, p]];
        if cmu_p.abs() < 1e-14 {
            continue;
        }
        for nu in 0..n_basis {
            let cnu_q = c[[nu, q]];
            if cnu_q.abs() < 1e-14 {
                continue;
            }
            let cmu_cnu = cmu_p * cnu_q;
            for lam in 0..n_basis {
                let clam_r = c[[lam, r]];
                if clam_r.abs() < 1e-14 {
                    continue;
                }
                for sig in 0..n_basis {
                    let csig_s = c[[sig, s]];
                    if csig_s.abs() < 1e-14 {
                        continue;
                    }
                    val += cmu_cnu * get_eri(eri, n_basis, mu, nu, lam, sig) * clam_r * csig_s;
                }
            }
        }
    }
    val
}

// ---------------------------------------------------------------------------
// Dipole integrals (AO basis)
// ---------------------------------------------------------------------------

/// Build the Cartesian (x,y,z) dipole integral matrices in the AO basis.
///
/// Uses centroid approximation: ⟨μ|r_α|ν⟩ ≈ (A_α + B_α)/2 · S_{μν}.
fn build_dipole_ao(basis: &[GaussianBasis], overlap: &Array2<f64>) -> [Array2<f64>; 3] {
    let n = basis.len();
    let mut rmat: [Array2<f64>; 3] = [
        Array2::zeros((n, n)),
        Array2::zeros((n, n)),
        Array2::zeros((n, n)),
    ];
    for mu in 0..n {
        for nu in 0..n {
            for alpha in 0..3 {
                let centroid = 0.5 * (basis[mu].center[alpha] + basis[nu].center[alpha]);
                rmat[alpha][[mu, nu]] = centroid * overlap[[mu, nu]];
            }
        }
    }
    rmat
}

/// Compute the transition dipole ⟨0|r_α|I⟩ for excitation I.
///
/// ⟨0|r_α|I⟩ = Σ_{ia} X_{ia,I} ⟨i|r_α|a⟩_MO
/// ⟨i|r_α|a⟩_MO = Σ_{μν} C_{μi} r^AO_{α,μν} C_{νa}
fn compute_transition_dipole(
    x_vec: &[f64],
    n_occ: usize,
    n_virt: usize,
    n_basis: usize,
    dipole_ao: &[Array2<f64>; 3],
    scf: &ScfResult,
) -> [f64; 3] {
    let c = &scf.mo_coefficients;
    let mut tdip = [0.0_f64; 3];

    for alpha in 0..3 {
        let mut mu_alpha = 0.0;
        for i in 0..n_occ {
            for a_idx in 0..n_virt {
                let a_mo = n_occ + a_idx;
                let x_ia = x_vec[i * n_virt + a_idx];
                if x_ia.abs() < 1e-15 {
                    continue;
                }
                // ⟨i|r_α|a⟩_MO = Σ_{μν} C_{μi} r^AO_μν C_{νa}
                let mut dp = 0.0;
                for mu in 0..n_basis {
                    for nu in 0..n_basis {
                        dp += c[[mu, i]] * dipole_ao[alpha][[mu, nu]] * c[[nu, a_mo]];
                    }
                }
                mu_alpha += x_ia * dp;
            }
        }
        tdip[alpha] = mu_alpha;
    }
    tdip
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::specialized::quantum::gaussian_integrals::normalized_s_gto;
    use crate::specialized::quantum::tdhf::eri::build_eri_tensor;
    use crate::specialized::quantum::tdhf::scf::{HartreeFockSCF, ScfConfig};

    fn minimal_h2() -> (Vec<GaussianBasis>, Vec<(f64, [f64; 3])>) {
        let basis = vec![
            normalized_s_gto([0.0, 0.0, 0.0], 1.0),
            normalized_s_gto([0.0, 0.0, 1.4], 1.0),
        ];
        let charges = vec![
            (1.0_f64, [0.0_f64, 0.0, 0.0]),
            (1.0_f64, [0.0_f64, 0.0, 1.4]),
        ];
        (basis, charges)
    }

    #[test]
    fn test_casida_positive_excitation_energies() {
        let (basis, charges) = minimal_h2();
        let scf = HartreeFockSCF::new(ScfConfig::default())
            .run(&basis, &charges, 2)
            .unwrap();
        let eri = build_eri_tensor(&basis);
        let cfg = CasidaConfig { n_roots: 1 };
        let result = CasidaSolver::new(cfg).solve(&scf, &basis, &eri, 2).unwrap();
        for &e in &result.excitation_energies {
            assert!(
                e >= 0.0,
                "Excitation energy should be non-negative, got {e}"
            );
        }
    }

    #[test]
    fn test_casida_n_roots_respected() {
        // Use a slightly larger basis: 3 s-type GTOs, 2 electrons → n_occ=1, n_virt=2, n_singles=2
        let basis = vec![
            normalized_s_gto([0.0, 0.0, 0.0], 1.5),
            normalized_s_gto([0.0, 0.0, 1.4], 0.8),
            normalized_s_gto([0.0, 0.0, 2.8], 1.2),
        ];
        let charges = vec![
            (1.0_f64, [0.0_f64, 0.0, 0.0]),
            (1.0_f64, [0.0_f64, 0.0, 1.4]),
        ];
        let scf = HartreeFockSCF::new(ScfConfig::default())
            .run(&basis, &charges, 2)
            .unwrap();
        let eri = build_eri_tensor(&basis);
        // n_singles = n_occ * n_virt = 1 * 2 = 2
        // Requesting 3 roots → will pad to 3
        let cfg = CasidaConfig { n_roots: 3 };
        let result = CasidaSolver::new(cfg).solve(&scf, &basis, &eri, 2).unwrap();
        // Should return exactly n_roots entries (padded with zeros if fewer positive roots)
        assert_eq!(
            result.excitation_energies.len(),
            3,
            "Expected 3 roots, got {}",
            result.excitation_energies.len()
        );
        assert_eq!(result.oscillator_strengths.len(), 3);
        assert_eq!(result.transition_dipoles.len(), 3);
    }

    #[test]
    fn test_casida_oscillator_strength_nonneg() {
        let (basis, charges) = minimal_h2();
        let scf = HartreeFockSCF::new(ScfConfig::default())
            .run(&basis, &charges, 2)
            .unwrap();
        let eri = build_eri_tensor(&basis);
        let cfg = CasidaConfig { n_roots: 2 };
        let result = CasidaSolver::new(cfg).solve(&scf, &basis, &eri, 2).unwrap();
        for &f in &result.oscillator_strengths {
            assert!(
                f >= 0.0,
                "Oscillator strength must be non-negative, got {f}"
            );
        }
    }

    #[test]
    fn test_casida_a_matrix_diagonal_positive() {
        // For a converged SCF the diagonal (ε_a - ε_i) should be positive
        let (basis, charges) = minimal_h2();
        let scf = HartreeFockSCF::new(ScfConfig::default())
            .run(&basis, &charges, 2)
            .unwrap();
        let eri = build_eri_tensor(&basis);
        let n_basis = basis.len();
        let n_occ = 1;
        let a = CasidaSolver::build_a_matrix(&scf, &eri, n_basis, n_occ).unwrap();
        // 1×1 A matrix for minimal H2
        let n_singles = n_occ * (n_basis - n_occ);
        for i in 0..n_singles {
            assert!(a[[i, i]] >= -1e-8, "A diagonal[{i}] = {}", a[[i, i]]);
        }
    }
}
