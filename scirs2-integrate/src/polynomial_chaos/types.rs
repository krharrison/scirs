//! Core types for Polynomial Chaos Expansion

/// Polynomial basis following the Wiener-Askey scheme.
///
/// Each variant corresponds to a specific probability distribution:
/// - Hermite <-> Gaussian
/// - Legendre <-> Uniform on \[-1, 1\]
/// - Laguerre <-> Exponential on \[0, inf)
/// - Jacobi <-> Beta on \[-1, 1\]
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PolynomialBasis {
    /// Hermite polynomials (probabilist's convention) for Gaussian distribution.
    Hermite,
    /// Legendre polynomials for Uniform distribution on \[-1, 1\].
    Legendre,
    /// Laguerre polynomials for Exponential distribution on \[0, inf).
    Laguerre,
    /// Jacobi polynomials P_n^{(alpha, beta)} for Beta distribution.
    Jacobi {
        /// Alpha parameter (must be > -1).
        alpha: f64,
        /// Beta parameter (must be > -1).
        beta: f64,
    },
}

/// Multi-index truncation scheme for selecting which polynomial terms to include.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum TruncationScheme {
    /// Total degree: |alpha| <= p, where |alpha| = sum of all indices.
    TotalDegree,
    /// Hyperbolic truncation: (sum alpha_i^q)^{1/q} <= p.
    /// Favors lower-order interaction terms when q < 1.
    Hyperbolic {
        /// Quasi-norm parameter (typically 0 < q <= 1).
        q: f64,
    },
    /// Full tensor product: each alpha_i <= p independently.
    Tensor,
}

/// Method for computing PCE coefficients.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum CoefficientMethod {
    /// Galerkin projection using quadrature: c_k = E\[f * Psi_k\] / E\[Psi_k^2\].
    Projection {
        /// Number of quadrature points per dimension.
        quadrature_order: usize,
    },
    /// Least-squares regression on random samples.
    Regression {
        /// Number of random samples to generate.
        n_samples: usize,
        /// Random seed for reproducibility.
        seed: u64,
    },
}

/// Configuration for Polynomial Chaos Expansion.
#[derive(Debug, Clone)]
pub struct PCEConfig {
    /// One basis per random input dimension (Wiener-Askey scheme).
    pub bases: Vec<PolynomialBasis>,
    /// Maximum polynomial degree p.
    pub max_degree: usize,
    /// Multi-index truncation scheme.
    pub truncation: TruncationScheme,
    /// Method for computing PCE coefficients.
    pub coefficient_method: CoefficientMethod,
}

/// Result of a PCE fit.
#[derive(Debug, Clone)]
pub struct PCEResult {
    /// PCE coefficients c_k for each basis function Psi_k.
    pub coefficients: Vec<f64>,
    /// Multi-indices alpha_k defining each basis function.
    pub multi_indices: Vec<Vec<usize>>,
    /// Squared norms ||Psi_k||^2 for each basis function.
    pub basis_norms_squared: Vec<f64>,
    /// Mean E\[Y\] = c_0.
    pub mean: f64,
    /// Variance Var\[Y\] = sum_{k>=1} c_k^2 * ||Psi_k||^2.
    pub variance: f64,
    /// First-order Sobol sensitivity indices (if computed).
    pub sobol_indices: Option<Vec<f64>>,
    /// Total Sobol sensitivity indices (if computed).
    pub total_sobol_indices: Option<Vec<f64>>,
    /// Number of PCE terms.
    pub n_terms: usize,
}
