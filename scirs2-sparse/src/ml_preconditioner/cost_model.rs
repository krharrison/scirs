//! Cost model for preconditioner setup and application.
//!
//! Provides asymptotic FLOP estimates for setup, per-iteration cost, and
//! expected number of Krylov iterations, enabling cost-aware ranking of
//! preconditioner candidates.

use super::types::{CostEstimate, MatrixFeatures, PreconditionerType};

/// Estimate the iteration count as a function of condition number.
///
/// Uses the CG-like bound: iterations ≈ sqrt(κ) / 2, clamped to [1, 10000].
fn iter_from_cond(kappa: f64) -> usize {
    let raw = (kappa.sqrt() / 2.0).ceil() as usize;
    raw.clamp(1, 10_000)
}

/// Estimate the total cost (setup + iterations × per_iteration) for a
/// given preconditioner type applied to a matrix with the supplied features.
pub fn estimate_cost(precond_type: PreconditionerType, features: &MatrixFeatures) -> CostEstimate {
    let n = features.n as f64;
    let nnz = features.nnz as f64;
    let kappa = features.cond_estimate;

    let (setup_cost, per_iteration_cost, estimated_iterations) = match precond_type {
        PreconditionerType::Jacobi => {
            // Setup: extract diagonal O(n)
            // Per iter: element-wise divide O(n)
            // Iterations: full CG count (no acceleration)
            let iters = iter_from_cond(kappa);
            (n, n, iters)
        }
        PreconditionerType::SSOR => {
            // Setup: compute omega + diagonal O(nnz)
            // Per iter: forward+backward sweep O(nnz)
            // Iterations: roughly half of unpreconditioned
            let iters = iter_from_cond(kappa) / 2;
            (nnz, nnz, iters.max(1))
        }
        PreconditionerType::ILU0 => {
            // Setup: incomplete LU factorization O(nnz)
            // Per iter: triangular solves O(nnz)
            // Iterations: roughly a third of unpreconditioned
            let iters = iter_from_cond(kappa) / 3;
            (nnz, nnz, iters.max(1))
        }
        PreconditionerType::IC0 => {
            // Setup: incomplete Cholesky O(nnz)
            // Per iter: triangular solve O(nnz)
            // Iterations: roughly a third of unpreconditioned
            let iters = iter_from_cond(kappa) / 3;
            (nnz, nnz, iters.max(1))
        }
        PreconditionerType::AMG => {
            // Setup: hierarchy construction O(nnz * log(n))
            // Per iter: V-cycle O(nnz)
            // Iterations: nearly constant, typically 5-20
            let log_n = (n + 1.0).ln();
            let setup = nnz * log_n;
            let iters = 10usize; // AMG yields near-constant iterations
            (setup, nnz, iters)
        }
        PreconditionerType::SPAI => {
            // Setup: compute sparse approximate inverse O(nnz * max_row^2)
            // Per iter: SpMV O(nnz)
            let max_row = features.max_row_nnz as f64;
            let setup = nnz * max_row * max_row;
            let iters = iter_from_cond(kappa) / 3;
            (setup, nnz, iters.max(1))
        }
        PreconditionerType::Polynomial => {
            // Setup: compute coefficients O(n)
            // Per iter: polynomial evaluation via SpMVs O(k * nnz), k ~ 5
            let k = 5.0;
            let iters = iter_from_cond(kappa) / 2;
            (n, k * nnz, iters.max(1))
        }
        PreconditionerType::None => {
            // No preconditioner: zero setup, zero overhead per iter
            let iters = iter_from_cond(kappa);
            (0.0, 0.0, iters)
        }
        #[allow(unreachable_patterns)]
        _ => {
            // Unknown/future variant: fallback to ILU0-like estimate
            let iters = iter_from_cond(kappa) / 3;
            (nnz, nnz, iters.max(1))
        }
    };

    let total_cost = setup_cost + (estimated_iterations as f64) * per_iteration_cost;

    CostEstimate {
        setup_cost,
        per_iteration_cost,
        estimated_iterations,
        total_cost,
    }
}

/// Rank candidate preconditioner types by estimated total cost (ascending).
///
/// Returns a vector of `(PreconditionerType, CostEstimate)` sorted from
/// cheapest to most expensive.
pub fn rank_by_cost(
    features: &MatrixFeatures,
    candidates: &[PreconditionerType],
) -> Vec<(PreconditionerType, CostEstimate)> {
    let mut ranked: Vec<(PreconditionerType, CostEstimate)> = candidates
        .iter()
        .map(|&pt| (pt, estimate_cost(pt, features)))
        .collect();
    ranked.sort_by(|a, b| {
        a.1.total_cost
            .partial_cmp(&b.1.total_cost)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_features() -> MatrixFeatures {
        MatrixFeatures {
            n: 1000,
            nnz: 5000,
            density: 0.005,
            max_row_nnz: 7,
            mean_row_nnz: 5.0,
            bandwidth: 50,
            bandwidth_ratio: 0.05,
            cond_estimate: 10000.0,
            spectral_radius: 20.0,
            diag_dominance: 1.5,
            symmetry_measure: 0.9,
            has_positive_diagonal: true,
        }
    }

    #[test]
    fn test_jacobi_cost() {
        let f = sample_features();
        let c = estimate_cost(PreconditionerType::Jacobi, &f);
        assert!(c.setup_cost > 0.0);
        assert!(c.per_iteration_cost > 0.0);
        assert!(c.estimated_iterations > 0);
        assert!(c.total_cost > c.setup_cost);
    }

    #[test]
    fn test_none_zero_overhead() {
        let f = sample_features();
        let c = estimate_cost(PreconditionerType::None, &f);
        assert!((c.setup_cost - 0.0).abs() < 1e-10);
        assert!((c.per_iteration_cost - 0.0).abs() < 1e-10);
        // Total cost is zero since per_iter * iters = 0
        assert!((c.total_cost - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_amg_fewer_iterations() {
        let f = sample_features();
        let c_jacobi = estimate_cost(PreconditionerType::Jacobi, &f);
        let c_amg = estimate_cost(PreconditionerType::AMG, &f);
        assert!(c_amg.estimated_iterations < c_jacobi.estimated_iterations);
    }

    #[test]
    fn test_rank_by_cost_sorted() {
        let f = sample_features();
        let candidates = vec![
            PreconditionerType::Jacobi,
            PreconditionerType::AMG,
            PreconditionerType::ILU0,
            PreconditionerType::None,
        ];
        let ranked = rank_by_cost(&f, &candidates);
        assert_eq!(ranked.len(), 4);
        // Verify sorted ascending by total_cost
        for w in ranked.windows(2) {
            assert!(w[0].1.total_cost <= w[1].1.total_cost);
        }
    }

    #[test]
    fn test_iter_from_cond_bounds() {
        assert_eq!(iter_from_cond(1.0), 1);
        assert!(iter_from_cond(1e10) <= 10_000);
    }
}
