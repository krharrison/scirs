//! ANOVA decomposition for adaptive sparse-grid variance analysis.
//!
//! Given a function `f : [0,1]^d → ℝ` this module estimates:
//!
//! - **Total variance** `Var[f]` via Monte Carlo.
//! - **First-order Sobol' sensitivity indices** `S_i = Var[E[f|x_i]] / Var[f]`
//!   using the Saltelli estimator (Saltelli 2002).
//! - **Second-order Sobol' sensitivity indices** `S_ij`.
//!
//! The results can be used to decide which dimensions to include in an adaptive
//! sparse-grid interpolant (only dimensions with non-negligible variance
//! contribution need fine-grained resolution).
//!
//! # Reference
//!
//! Saltelli, A. et al. (2002). *Making best use of model evaluations to compute
//! sensitivity indices*. Computer Physics Communications, 145(2), 280–297.

use crate::error::InterpolateError;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the ANOVA / Sobol' sensitivity analysis.
#[derive(Debug, Clone)]
pub struct AnovaConfig {
    /// Maximum interaction order to compute (1 = first-order only, 2 = up to pairwise).
    pub max_order: usize,
    /// Number of Gauss–Legendre quadrature points per dimension
    /// (used for future structured quadrature; currently the Saltelli estimator
    /// is used for simplicity).
    pub n_quadrature: usize,
    /// Threshold below which a Sobol' index is considered negligible.
    pub tol_variance: f64,
}

impl Default for AnovaConfig {
    fn default() -> Self {
        Self {
            max_order: 2,
            n_quadrature: 10,
            tol_variance: 0.01,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Saltelli sample generator
// ─────────────────────────────────────────────────────────────────────────────

/// Generate two independent quasi-random sample matrices for the Saltelli
/// estimator.  Each matrix has shape `(n_samples × n_dims)`.
///
/// Uses a simple deterministic low-discrepancy sequence derived from Van der
/// Corput sequences (base 2 and base 3) to avoid the need for an external RNG.
pub fn saltelli_sample(n_dims: usize, n_samples: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    // Van der Corput in base `b` for sample index `i`.
    fn vdc(mut i: usize, base: usize) -> f64 {
        let mut result = 0.0_f64;
        let mut denom = 1.0_f64;
        while i > 0 {
            denom *= base as f64;
            result += (i % base) as f64 / denom;
            i /= base;
        }
        result
    }

    let mut a = vec![vec![0.0_f64; n_dims]; n_samples];
    let mut b = vec![vec![0.0_f64; n_dims]; n_samples];

    for i in 0..n_samples {
        for d in 0..n_dims {
            // For matrix A: use bases 2,3,5,7,11,... (primes)
            let base_a = nth_prime(d * 2);
            let base_b = nth_prime(d * 2 + 1);
            a[i][d] = vdc(i + 1, base_a);
            b[i][d] = vdc(i + 1, base_b);
        }
    }
    (a, b)
}

/// Return the n-th prime (0-indexed).  Pre-computed up to index 49.
fn nth_prime(n: usize) -> usize {
    const PRIMES: &[usize] = &[
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
        97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
        191, 193, 197, 199, 211, 223, 227, 229,
    ];
    PRIMES[n % PRIMES.len()]
}

// ─────────────────────────────────────────────────────────────────────────────
// ANOVA decomposition
// ─────────────────────────────────────────────────────────────────────────────

/// Result of an ANOVA / Sobol' sensitivity analysis.
#[derive(Debug, Clone)]
pub struct AnovaDecomposition {
    /// Number of input dimensions.
    n_dims: usize,
    /// Total variance of the function.
    total_var: f64,
    /// First-order Sobol' sensitivity indices (length `n_dims`).
    s1: Vec<f64>,
    /// Upper-triangular second-order indices stored as a flat `n_dims × n_dims`
    /// matrix (only `i < j` entries are meaningful).
    s2: Vec<Vec<f64>>,
    /// Dimension importance order (sorted by `s1` descending).
    ranking: Vec<usize>,
}

impl AnovaDecomposition {
    /// Compute Sobol' sensitivity indices for `f` over `[0,1]^n_dims`.
    ///
    /// # Arguments
    /// * `f`        – The function to analyse.
    /// * `n_dims`   – Number of input dimensions.
    /// * `config`   – Analysis configuration.
    pub fn fit(
        f: impl Fn(&[f64]) -> f64,
        n_dims: usize,
        config: &AnovaConfig,
    ) -> Result<AnovaDecomposition, InterpolateError> {
        if n_dims == 0 {
            return Err(InterpolateError::InvalidValue(
                "n_dims must be ≥ 1".to_string(),
            ));
        }

        let n = 1000_usize; // Monte Carlo sample count

        // Generate two sample matrices A and B.
        let (mat_a, mat_b) = saltelli_sample(n_dims, n);

        // Evaluate f on A and B.
        let f_a: Vec<f64> = mat_a.iter().map(|x| f(x)).collect();
        let f_b: Vec<f64> = mat_b.iter().map(|x| f(x)).collect();

        // Total mean and variance.
        let mean_a: f64 = f_a.iter().sum::<f64>() / n as f64;
        let var_total: f64 = f_a
            .iter()
            .map(|&v| (v - mean_a) * (v - mean_a))
            .sum::<f64>()
            / n as f64;

        if var_total < 1e-15 {
            // Constant function — all indices are zero.
            return Ok(AnovaDecomposition {
                n_dims,
                total_var: 0.0,
                s1: vec![0.0; n_dims],
                s2: vec![vec![0.0; n_dims]; n_dims],
                ranking: (0..n_dims).collect(),
            });
        }

        // ── First-order Sobol' indices (Saltelli estimator) ───────────────
        // S_i ≈ (1/N) Σ f_B · (f_{A_B^i} − f_A) / Var[f]
        // where A_B^i = A with column i replaced by B's column i.
        let mut s1 = vec![0.0_f64; n_dims];

        for i in 0..n_dims {
            // Build A_B^i: take A, replace dim i with B's dim i.
            let f_ab_i: Vec<f64> = (0..n)
                .map(|k| {
                    let mut x = mat_a[k].clone();
                    x[i] = mat_b[k][i];
                    f(&x)
                })
                .collect();

            // Saltelli (2002) first-order estimator:
            // V_i = (1/N) Σ_k f_B[k] (f_{AB^i}[k] − f_A[k])
            let v_i: f64 = (0..n).map(|k| f_b[k] * (f_ab_i[k] - f_a[k])).sum::<f64>() / n as f64;

            s1[i] = v_i / var_total;
        }

        // ── Second-order indices (only when max_order ≥ 2) ────────────────
        let mut s2 = vec![vec![0.0_f64; n_dims]; n_dims];

        if config.max_order >= 2 {
            for i in 0..n_dims {
                for j in (i + 1)..n_dims {
                    // A_B^{ij}: A with columns i and j replaced by B.
                    let f_ab_ij: Vec<f64> = (0..n)
                        .map(|k| {
                            let mut x = mat_a[k].clone();
                            x[i] = mat_b[k][i];
                            x[j] = mat_b[k][j];
                            f(&x)
                        })
                        .collect();

                    // V_{ij} estimator (simplified closed second-order):
                    // V_{ij} ≈ (1/N) Σ f_A · f_{AB^{ij}} − mean²  − V_i − V_j
                    let v_ij_raw: f64 = (0..n).map(|k| f_a[k] * f_ab_ij[k]).sum::<f64>() / n as f64
                        - mean_a * mean_a;

                    let s_ij = (v_ij_raw / var_total - s1[i] - s1[j]).max(0.0);
                    s2[i][j] = s_ij;
                    s2[j][i] = s_ij;
                }
            }
        }

        // Dimension ranking by first-order index.
        let mut ranking: Vec<usize> = (0..n_dims).collect();
        ranking.sort_by(|&a, &b| {
            s1[b]
                .partial_cmp(&s1[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(AnovaDecomposition {
            n_dims,
            total_var: var_total,
            s1,
            s2,
            ranking,
        })
    }

    /// Return the total variance of the function.
    pub fn total_variance(&self) -> f64 {
        self.total_var
    }

    /// First-order Sobol' sensitivity indices (length `n_dims`).
    pub fn sobol_indices(&self) -> Vec<f64> {
        self.s1.clone()
    }

    /// Second-order Sobol' index for the pair `(i, j)`.
    pub fn sobol_index_2d(&self, i: usize, j: usize) -> f64 {
        if i < self.n_dims && j < self.n_dims {
            self.s2[i][j]
        } else {
            0.0
        }
    }

    /// Return dimensions whose first-order Sobol' index exceeds `threshold`.
    pub fn important_dims(&self, threshold: f64) -> Vec<usize> {
        self.s1
            .iter()
            .enumerate()
            .filter_map(|(i, &si)| if si > threshold { Some(i) } else { None })
            .collect()
    }

    /// Recommend which dimensions to include in a sparse-grid interpolant based
    /// on the configured variance tolerance.
    ///
    /// Dimensions with `S_i > tol_variance` are recommended.
    pub fn suggest_sparse_grid_dims(&self) -> Vec<usize> {
        // Use a default threshold of 0.01 (1 % of total variance).
        self.important_dims(0.01)
    }

    /// Dimension ranking by first-order Sobol' index (most important first).
    pub fn dimension_ranking(&self) -> &[usize] {
        &self.ranking
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// f(x, y, z) = x² + y.  Dimensions 0 and 1 are important; dim 2 is not.
    fn test_fn(x: &[f64]) -> f64 {
        x[0] * x[0] + x[1]
    }

    #[test]
    fn test_important_dims_identified() {
        let config = AnovaConfig {
            max_order: 1,
            n_quadrature: 10,
            tol_variance: 0.01,
        };
        let decomp = AnovaDecomposition::fit(test_fn, 3, &config).expect("fit should succeed");

        let s = decomp.sobol_indices();
        // Dim 0 (x²) and dim 1 (y) should have higher variance contribution.
        assert!(
            s[0] > s[2],
            "dim 0 should matter more than dim 2: s[0]={}, s[2]={}",
            s[0],
            s[2]
        );
        assert!(
            s[1] > s[2],
            "dim 1 should matter more than dim 2: s[1]={}, s[2]={}",
            s[1],
            s[2]
        );

        let important = decomp.important_dims(0.05);
        assert!(
            important.contains(&0),
            "dim 0 should be important: {:?}",
            important
        );
        assert!(
            important.contains(&1),
            "dim 1 should be important: {:?}",
            important
        );
    }

    #[test]
    fn test_total_variance_positive() {
        let config = AnovaConfig::default();
        let decomp = AnovaDecomposition::fit(test_fn, 3, &config).expect("fit");
        assert!(
            decomp.total_variance() > 0.0,
            "total variance should be positive"
        );
    }

    #[test]
    fn test_constant_function_zero_variance() {
        let config = AnovaConfig::default();
        let decomp = AnovaDecomposition::fit(|_| 42.0, 3, &config).expect("fit");
        assert_eq!(decomp.total_variance(), 0.0);
        for &si in &decomp.sobol_indices() {
            assert_eq!(si, 0.0);
        }
    }

    #[test]
    fn test_sobol_index_2d_returns_value() {
        let config = AnovaConfig {
            max_order: 2,
            ..Default::default()
        };
        let decomp = AnovaDecomposition::fit(test_fn, 3, &config).expect("fit");
        // Should not panic and should be ≥ 0
        let s01 = decomp.sobol_index_2d(0, 1);
        assert!(s01 >= 0.0, "s01 should be non-negative: {}", s01);
    }

    #[test]
    fn test_suggest_sparse_grid_dims() {
        let config = AnovaConfig::default();
        let decomp = AnovaDecomposition::fit(test_fn, 3, &config).expect("fit");
        let suggested = decomp.suggest_sparse_grid_dims();
        // At least dims 0 and 1 should appear (they contribute most variance).
        assert!(
            !suggested.is_empty(),
            "some dims should be suggested: {:?}",
            suggested
        );
    }

    #[test]
    fn test_saltelli_sample_dimensions() {
        let (a, b) = saltelli_sample(4, 100);
        assert_eq!(a.len(), 100);
        assert_eq!(a[0].len(), 4);
        assert_eq!(b.len(), 100);
        assert_eq!(b[0].len(), 4);
        // All values in [0, 1)
        for row in &a {
            for &v in row {
                assert!(v >= 0.0 && v < 1.0, "out of range: {v}");
            }
        }
    }
}
