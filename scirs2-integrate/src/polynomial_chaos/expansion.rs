//! PCE coefficient computation via projection and regression.
//!
//! The [`PolynomialChaosExpansion`] struct manages the full workflow:
//! 1. Generate multi-indices based on the truncation scheme
//! 2. Compute coefficients via projection (quadrature) or regression (least-squares)
//! 3. Evaluate the PCE surrogate at new points in random space

use crate::error::{IntegrateError, IntegrateResult};

use super::basis::{
    basis_norm_squared_nd, evaluate_basis_1d, evaluate_basis_nd, gauss_quadrature,
    generate_multi_indices,
};
use super::statistics;
use super::types::{CoefficientMethod, PCEConfig, PCEResult, PolynomialBasis, TruncationScheme};

/// Polynomial Chaos Expansion.
///
/// Represents a surrogate model Y ≈ sum_k c_k Psi_k(Xi)
/// where Xi are random variables and Psi_k are orthogonal polynomial basis functions.
#[derive(Debug, Clone)]
pub struct PolynomialChaosExpansion {
    /// PCE configuration.
    pub config: PCEConfig,
    /// Computed coefficients (None until `fit` is called).
    pub coefficients: Option<Vec<f64>>,
    /// Multi-indices defining each basis function.
    pub multi_indices: Vec<Vec<usize>>,
    /// Squared norms ||Psi_k||^2 for each basis function.
    pub basis_norms_squared: Vec<f64>,
}

impl PolynomialChaosExpansion {
    /// Create a new PCE with the given configuration.
    ///
    /// Generates multi-indices and computes basis norms but does not
    /// compute coefficients until `fit` is called.
    pub fn new(config: PCEConfig) -> IntegrateResult<Self> {
        if config.bases.is_empty() {
            return Err(IntegrateError::ValueError(
                "At least one basis is required".to_string(),
            ));
        }
        if config.max_degree == 0 {
            return Err(IntegrateError::ValueError(
                "Maximum degree must be >= 1".to_string(),
            ));
        }

        let dim = config.bases.len();
        let multi_indices = generate_multi_indices(dim, config.max_degree, &config.truncation);

        let basis_norms_squared: Vec<f64> = multi_indices
            .iter()
            .map(|alpha| basis_norm_squared_nd(&config.bases, alpha))
            .collect();

        Ok(Self {
            config,
            coefficients: None,
            multi_indices,
            basis_norms_squared,
        })
    }

    /// Number of PCE terms.
    pub fn n_terms(&self) -> usize {
        self.multi_indices.len()
    }

    /// Dimensionality (number of random input variables).
    pub fn dim(&self) -> usize {
        self.config.bases.len()
    }

    /// Fit PCE coefficients to a model function.
    ///
    /// The model function maps a random input vector xi to a scalar output.
    /// After fitting, the PCE can be used as a surrogate for the original model.
    pub fn fit<F>(&mut self, model: F) -> IntegrateResult<PCEResult>
    where
        F: Fn(&[f64]) -> IntegrateResult<f64>,
    {
        let coefficients = match &self.config.coefficient_method {
            CoefficientMethod::Projection { quadrature_order } => {
                self.fit_projection(&model, *quadrature_order)?
            }
            CoefficientMethod::Regression { n_samples, seed } => {
                self.fit_regression(&model, *n_samples, *seed)?
            }
        };

        self.coefficients = Some(coefficients.clone());

        let mean = statistics::pce_mean(&coefficients);
        let variance = statistics::pce_variance(&coefficients, &self.basis_norms_squared);

        let sobol = statistics::sobol_indices(
            &coefficients,
            &self.multi_indices,
            &self.basis_norms_squared,
        )
        .ok();
        let total_sobol = statistics::total_sobol_indices(
            &coefficients,
            &self.multi_indices,
            &self.basis_norms_squared,
        )
        .ok();

        Ok(PCEResult {
            coefficients,
            multi_indices: self.multi_indices.clone(),
            basis_norms_squared: self.basis_norms_squared.clone(),
            mean,
            variance,
            sobol_indices: sobol,
            total_sobol_indices: total_sobol,
            n_terms: self.n_terms(),
        })
    }

    /// Projection method: c_k = E\[f(Xi) Psi_k(Xi)\] / ||Psi_k||^2.
    ///
    /// Uses tensor-product Gauss quadrature to compute the inner products.
    fn fit_projection<F>(&self, model: &F, quadrature_order: usize) -> IntegrateResult<Vec<f64>>
    where
        F: Fn(&[f64]) -> IntegrateResult<f64>,
    {
        let dim = self.dim();
        let n_terms = self.n_terms();

        // Get 1-D quadrature rules for each dimension
        let mut quad_rules: Vec<(Vec<f64>, Vec<f64>)> = Vec::with_capacity(dim);
        for basis in &self.config.bases {
            quad_rules.push(gauss_quadrature(basis, quadrature_order)?);
        }

        // Build tensor-product quadrature points
        let total_points: usize = quad_rules.iter().map(|(n, _)| n.len()).product();
        let mut quad_points = Vec::with_capacity(total_points);
        let mut quad_weights = Vec::with_capacity(total_points);

        build_tensor_product_quadrature(&quad_rules, dim, &mut quad_points, &mut quad_weights);

        // Evaluate model at all quadrature points
        let mut model_values = Vec::with_capacity(total_points);
        for point in &quad_points {
            model_values.push(model(point)?);
        }

        // Compute coefficients: c_k = sum_i w_i * f(x_i) * Psi_k(x_i) / ||Psi_k||^2
        let mut coefficients = vec![0.0_f64; n_terms];
        for (k, alpha) in self.multi_indices.iter().enumerate() {
            let mut numerator = 0.0_f64;
            for (i, point) in quad_points.iter().enumerate() {
                let psi_k = evaluate_basis_nd(&self.config.bases, alpha, point)?;
                numerator += quad_weights[i] * model_values[i] * psi_k;
            }
            let norm_sq = self.basis_norms_squared[k];
            if norm_sq.abs() < 1e-30 {
                coefficients[k] = 0.0;
            } else {
                coefficients[k] = numerator / norm_sq;
            }
        }

        Ok(coefficients)
    }

    /// Regression method: least-squares on random samples.
    ///
    /// Generates random samples from the appropriate distribution,
    /// builds the design matrix Psi\[i,k\] = Psi_k(xi_i),
    /// and solves min ||Psi c - y||^2 via normal equations.
    fn fit_regression<F>(&self, model: &F, n_samples: usize, seed: u64) -> IntegrateResult<Vec<f64>>
    where
        F: Fn(&[f64]) -> IntegrateResult<f64>,
    {
        let dim = self.dim();
        let n_terms = self.n_terms();

        if n_samples < n_terms {
            return Err(IntegrateError::ValueError(format!(
                "Need at least {} samples for {} PCE terms, got {}",
                n_terms, n_terms, n_samples
            )));
        }

        // Generate random samples using a simple LCG for reproducibility
        let samples = generate_random_samples(&self.config.bases, n_samples, dim, seed);

        // Evaluate model at all sample points
        let mut y = vec![0.0_f64; n_samples];
        for (i, sample) in samples.iter().enumerate() {
            y[i] = model(sample)?;
        }

        // Build design matrix Psi[i,k]
        let mut psi = vec![0.0_f64; n_samples * n_terms];
        for (i, sample) in samples.iter().enumerate() {
            for (k, alpha) in self.multi_indices.iter().enumerate() {
                psi[i * n_terms + k] = evaluate_basis_nd(&self.config.bases, alpha, sample)?;
            }
        }

        // Solve normal equations: (Psi^T Psi) c = Psi^T y
        solve_least_squares(&psi, &y, n_samples, n_terms)
    }

    /// Evaluate the PCE surrogate at a point in random space.
    ///
    /// Returns Y(xi) = sum_k c_k Psi_k(xi).
    pub fn evaluate(&self, xi: &[f64]) -> IntegrateResult<f64> {
        let coeffs = self.coefficients.as_ref().ok_or_else(|| {
            IntegrateError::ComputationError("PCE not fitted yet; call fit() first".to_string())
        })?;

        if xi.len() != self.dim() {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Expected {} dimensions, got {}",
                self.dim(),
                xi.len()
            )));
        }

        let mut result = 0.0_f64;
        for (k, alpha) in self.multi_indices.iter().enumerate() {
            let psi_k = evaluate_basis_nd(&self.config.bases, alpha, xi)?;
            result += coeffs[k] * psi_k;
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Helper: tensor-product quadrature construction
// ---------------------------------------------------------------------------

/// Build tensor-product quadrature points and weights from 1-D rules.
fn build_tensor_product_quadrature(
    rules: &[(Vec<f64>, Vec<f64>)],
    dim: usize,
    points: &mut Vec<Vec<f64>>,
    weights: &mut Vec<f64>,
) {
    if dim == 0 {
        points.push(vec![]);
        weights.push(1.0);
        return;
    }

    // Start with first dimension
    let (ref nodes0, ref weights0) = rules[0];
    let mut current_points: Vec<Vec<f64>> = nodes0.iter().map(|&x| vec![x]).collect();
    let mut current_weights: Vec<f64> = weights0.clone();

    // Expand for each additional dimension
    for d in 1..dim {
        let (ref nodes_d, ref weights_d) = rules[d];
        let mut new_points = Vec::with_capacity(current_points.len() * nodes_d.len());
        let mut new_weights = Vec::with_capacity(current_weights.len() * weights_d.len());

        for (i, pt) in current_points.iter().enumerate() {
            for (j, &xd) in nodes_d.iter().enumerate() {
                let mut new_pt = pt.clone();
                new_pt.push(xd);
                new_points.push(new_pt);
                new_weights.push(current_weights[i] * weights_d[j]);
            }
        }
        current_points = new_points;
        current_weights = new_weights;
    }

    *points = current_points;
    *weights = current_weights;
}

// ---------------------------------------------------------------------------
// Helper: random sample generation
// ---------------------------------------------------------------------------

/// Generate random samples from appropriate distributions using a simple LCG.
fn generate_random_samples(
    bases: &[PolynomialBasis],
    n_samples: usize,
    dim: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let mut rng_state = seed;
    let mut samples = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let mut sample = Vec::with_capacity(dim);
        for basis in bases {
            let u = lcg_uniform(&mut rng_state);
            let value = match basis {
                PolynomialBasis::Hermite => {
                    // Box-Muller transform for Gaussian
                    let u2 = lcg_uniform(&mut rng_state);
                    let u1_clamped = u.max(1e-15).min(1.0 - 1e-15);
                    (-2.0 * u1_clamped.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                }
                PolynomialBasis::Legendre => {
                    // Uniform on [-1, 1]
                    2.0 * u - 1.0
                }
                PolynomialBasis::Laguerre => {
                    // Exponential: -ln(1-u)
                    let u_clamped = u.max(1e-15).min(1.0 - 1e-15);
                    -(1.0 - u_clamped).ln()
                }
                PolynomialBasis::Jacobi { .. } => {
                    // Approximate: use uniform on [-1,1] (acceptable for regression)
                    2.0 * u - 1.0
                }
            };
            sample.push(value);
        }
        samples.push(sample);
    }
    samples
}

/// Linear congruential generator returning a value in \[0, 1).
fn lcg_uniform(state: &mut u64) -> f64 {
    // Numerical Recipes LCG
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

// ---------------------------------------------------------------------------
// Helper: least-squares solver via normal equations
// ---------------------------------------------------------------------------

/// Solve min ||A x - b||^2 via normal equations: (A^T A) x = A^T b.
///
/// Uses Cholesky decomposition of the n_cols x n_cols matrix A^T A.
fn solve_least_squares(
    a: &[f64],
    b: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> IntegrateResult<Vec<f64>> {
    // Compute A^T A (n_cols x n_cols, symmetric)
    let mut ata = vec![0.0_f64; n_cols * n_cols];
    for i in 0..n_cols {
        for j in i..n_cols {
            let mut sum = 0.0_f64;
            for k in 0..n_rows {
                sum += a[k * n_cols + i] * a[k * n_cols + j];
            }
            ata[i * n_cols + j] = sum;
            ata[j * n_cols + i] = sum;
        }
    }

    // Compute A^T b (n_cols)
    let mut atb = vec![0.0_f64; n_cols];
    for i in 0..n_cols {
        let mut sum = 0.0_f64;
        for k in 0..n_rows {
            sum += a[k * n_cols + i] * b[k];
        }
        atb[i] = sum;
    }

    // Add small regularization for numerical stability
    for i in 0..n_cols {
        ata[i * n_cols + i] += 1e-12;
    }

    // Cholesky decomposition: A^T A = L L^T
    let mut l = vec![0.0_f64; n_cols * n_cols];
    for j in 0..n_cols {
        let mut sum = 0.0_f64;
        for k in 0..j {
            sum += l[j * n_cols + k] * l[j * n_cols + k];
        }
        let diag = ata[j * n_cols + j] - sum;
        if diag <= 0.0 {
            return Err(IntegrateError::LinearSolveError(
                "Normal equations matrix is not positive definite".to_string(),
            ));
        }
        l[j * n_cols + j] = diag.sqrt();

        for i in (j + 1)..n_cols {
            let mut sum = 0.0_f64;
            for k in 0..j {
                sum += l[i * n_cols + k] * l[j * n_cols + k];
            }
            l[i * n_cols + j] = (ata[i * n_cols + j] - sum) / l[j * n_cols + j];
        }
    }

    // Forward substitution: L y = A^T b
    let mut y = vec![0.0_f64; n_cols];
    for i in 0..n_cols {
        let mut sum = 0.0_f64;
        for k in 0..i {
            sum += l[i * n_cols + k] * y[k];
        }
        y[i] = (atb[i] - sum) / l[i * n_cols + i];
    }

    // Back substitution: L^T x = y
    let mut x = vec![0.0_f64; n_cols];
    for i in (0..n_cols).rev() {
        let mut sum = 0.0_f64;
        for k in (i + 1)..n_cols {
            sum += l[k * n_cols + i] * x[k];
        }
        x[i] = (y[i] - sum) / l[i * n_cols + i];
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pce_legendre_quadratic() {
        // f(xi) = xi^2 on Uniform[-1,1]
        // Legendre expansion: xi^2 = (1/3) P_0 + (2/3) P_2
        // Mean = 1/3, Variance = integral(xi^4) - (1/3)^2 = 1/5 - 1/9 = 4/45
        let config = PCEConfig {
            bases: vec![PolynomialBasis::Legendre],
            max_degree: 4,
            truncation: TruncationScheme::TotalDegree,
            coefficient_method: CoefficientMethod::Projection {
                quadrature_order: 5,
            },
        };
        let mut pce = PolynomialChaosExpansion::new(config).expect("PCE creation failed");
        let result = pce.fit(|xi| Ok(xi[0] * xi[0])).expect("PCE fit failed");

        // Mean should be 1/3
        assert!(
            (result.mean - 1.0 / 3.0).abs() < 1e-10,
            "Mean: got {}, expected {}",
            result.mean,
            1.0 / 3.0
        );

        // Variance should be 4/45
        assert!(
            (result.variance - 4.0 / 45.0).abs() < 1e-10,
            "Variance: got {}, expected {}",
            result.variance,
            4.0 / 45.0
        );
    }

    #[test]
    fn test_pce_hermite_cubic() {
        // f(xi) = xi^3 on Gaussian
        // Hermite expansion: xi^3 = H_3 + 3*H_1 (probabilist's)
        // So c_1 = 3, c_3 = 1, all others = 0
        let config = PCEConfig {
            bases: vec![PolynomialBasis::Hermite],
            max_degree: 4,
            truncation: TruncationScheme::TotalDegree,
            coefficient_method: CoefficientMethod::Projection {
                quadrature_order: 5,
            },
        };
        let mut pce = PolynomialChaosExpansion::new(config).expect("PCE creation failed");
        let result = pce.fit(|xi| Ok(xi[0].powi(3))).expect("PCE fit failed");

        // c_0 = 0 (mean of xi^3 for standard normal)
        assert!(
            result.coefficients[0].abs() < 1e-10,
            "c_0: got {}",
            result.coefficients[0]
        );
        // c_1 = 3 (coefficient of H_1 = xi)
        assert!(
            (result.coefficients[1] - 3.0).abs() < 1e-10,
            "c_1: got {}, expected 3",
            result.coefficients[1]
        );
        // c_3 = 1 (coefficient of H_3)
        assert!(
            (result.coefficients[3] - 1.0).abs() < 1e-10,
            "c_3: got {}, expected 1",
            result.coefficients[3]
        );
    }

    #[test]
    fn test_pce_evaluate_roundtrip() {
        // Fit a simple polynomial and verify evaluation matches
        let config = PCEConfig {
            bases: vec![PolynomialBasis::Legendre],
            max_degree: 3,
            truncation: TruncationScheme::TotalDegree,
            coefficient_method: CoefficientMethod::Projection {
                quadrature_order: 5,
            },
        };
        let mut pce = PolynomialChaosExpansion::new(config).expect("PCE creation failed");
        let f = |xi: &[f64]| -> IntegrateResult<f64> { Ok(2.0 * xi[0] + 0.5 * xi[0] * xi[0]) };
        let _result = pce.fit(f).expect("PCE fit failed");

        // Evaluate at a test point and compare
        let test_points = [0.0, 0.5, -0.5, 0.9];
        for &x in &test_points {
            let pce_val = pce.evaluate(&[x]).expect("evaluation failed");
            let exact = 2.0 * x + 0.5 * x * x;
            assert!(
                (pce_val - exact).abs() < 1e-10,
                "At x={x}: got {pce_val}, expected {exact}"
            );
        }
    }

    #[test]
    fn test_projection_vs_regression() {
        // Both methods should agree on a smooth polynomial function
        let config_proj = PCEConfig {
            bases: vec![PolynomialBasis::Legendre],
            max_degree: 3,
            truncation: TruncationScheme::TotalDegree,
            coefficient_method: CoefficientMethod::Projection {
                quadrature_order: 5,
            },
        };
        let config_reg = PCEConfig {
            bases: vec![PolynomialBasis::Legendre],
            max_degree: 3,
            truncation: TruncationScheme::TotalDegree,
            coefficient_method: CoefficientMethod::Regression {
                n_samples: 500,
                seed: 42,
            },
        };

        let f = |xi: &[f64]| -> IntegrateResult<f64> { Ok(xi[0] * xi[0] + 0.5 * xi[0]) };

        let mut pce_proj = PolynomialChaosExpansion::new(config_proj).expect("PCE creation failed");
        let result_proj = pce_proj.fit(f).expect("projection fit failed");

        let mut pce_reg = PolynomialChaosExpansion::new(config_reg).expect("PCE creation failed");
        let result_reg = pce_reg.fit(f).expect("regression fit failed");

        // Means should agree within tolerance
        assert!(
            (result_proj.mean - result_reg.mean).abs() < 0.05,
            "Means differ: proj={}, reg={}",
            result_proj.mean,
            result_reg.mean
        );

        // Variances should agree within tolerance
        assert!(
            (result_proj.variance - result_reg.variance).abs() < 0.05,
            "Variances differ: proj={}, reg={}",
            result_proj.variance,
            result_reg.variance
        );
    }
}
