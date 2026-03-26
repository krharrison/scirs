//! Types for functional data analysis and functional regression.
//!
//! This module provides core data structures for representing functional data
//! (curves observed on a common grid), basis function specifications, and
//! results from functional PCA and regression.

use scirs2_core::ndarray::{Array1, Array2};

/// Functional data: a collection of curves observed on a common grid.
///
/// Each observation is a function sampled at the same grid points.
/// For example, temperature curves measured hourly over multiple days,
/// or growth curves of different subjects measured at the same ages.
#[derive(Debug, Clone)]
pub struct FunctionalData {
    /// Common evaluation grid (sorted, length T)
    pub grid: Vec<f64>,
    /// Observed curves: `observations[i]` is the i-th curve, each of length T
    pub observations: Vec<Vec<f64>>,
}

impl FunctionalData {
    /// Create new functional data, validating dimensions.
    ///
    /// # Errors
    /// Returns an error if:
    /// - `grid` is empty
    /// - `observations` is empty
    /// - Any observation has a different length than `grid`
    pub fn new(grid: Vec<f64>, observations: Vec<Vec<f64>>) -> crate::error::StatsResult<Self> {
        if grid.is_empty() {
            return Err(crate::error::StatsError::InvalidInput(
                "Grid must not be empty".to_string(),
            ));
        }
        if observations.is_empty() {
            return Err(crate::error::StatsError::InvalidInput(
                "Observations must not be empty".to_string(),
            ));
        }
        let t = grid.len();
        for (i, obs) in observations.iter().enumerate() {
            if obs.len() != t {
                return Err(crate::error::StatsError::DimensionMismatch(format!(
                    "Observation {} has length {}, expected {} (grid length)",
                    i,
                    obs.len(),
                    t
                )));
            }
        }
        Ok(Self { grid, observations })
    }

    /// Number of curves (observations).
    pub fn n_curves(&self) -> usize {
        self.observations.len()
    }

    /// Number of grid points.
    pub fn n_grid(&self) -> usize {
        self.grid.len()
    }
}

/// Type of basis functions for representing functional data.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum BasisType {
    /// B-spline basis with specified number of basis functions and polynomial degree.
    BSpline {
        /// Number of basis functions
        n_basis: usize,
        /// Polynomial degree (typically 3 for cubic)
        degree: usize,
    },
    /// Fourier basis (sin/cos pairs) with specified number of basis functions.
    /// `n_basis` should be odd: 1 constant + pairs of (sin, cos).
    Fourier {
        /// Number of basis functions (should be odd for complete pairs)
        n_basis: usize,
    },
    /// Polynomial basis: 1, t, t^2, ..., t^degree
    Polynomial {
        /// Maximum polynomial degree
        degree: usize,
    },
}

/// Configuration for functional data analysis.
#[derive(Debug, Clone)]
pub struct FunctionalConfig {
    /// Basis function type
    pub basis: BasisType,
    /// Smoothing parameter (lambda). If None, selected by GCV.
    pub smoothing_param: Option<f64>,
    /// Number of principal components to retain
    pub n_components: usize,
}

impl Default for FunctionalConfig {
    fn default() -> Self {
        Self {
            basis: BasisType::BSpline {
                n_basis: 15,
                degree: 3,
            },
            smoothing_param: None,
            n_components: 3,
        }
    }
}

/// Result of functional PCA.
#[derive(Debug, Clone)]
pub struct FPCAResult {
    /// Eigenvalues in descending order (length = n_components)
    pub eigenvalues: Array1<f64>,
    /// Eigenfunctions evaluated on the grid: rows = components, cols = grid points
    pub eigenfunctions: Array2<f64>,
    /// Scores: `scores[[i, k]]` = score of curve i on component k
    pub scores: Array2<f64>,
    /// Fraction of variance explained by each component
    pub variance_explained: Array1<f64>,
    /// The grid on which eigenfunctions are evaluated
    pub grid: Vec<f64>,
}

/// Result of scalar-on-function regression.
#[derive(Debug, Clone)]
pub struct SoFResult {
    /// Estimated coefficient function beta(t) evaluated on the grid
    pub beta: Array1<f64>,
    /// Intercept
    pub intercept: f64,
    /// Basis coefficients of beta
    pub beta_coefficients: Array1<f64>,
    /// The basis type used
    pub basis: BasisType,
    /// The grid
    pub grid: Vec<f64>,
    /// Smoothing parameter used
    pub lambda: f64,
    /// Fitted values
    pub fitted_values: Array1<f64>,
    /// R-squared
    pub r_squared: f64,
}

/// Result of function-on-function regression.
#[derive(Debug, Clone)]
pub struct FoFResult {
    /// Estimated bivariate coefficient function beta(s,t)
    /// Shape: (n_grid_s, n_grid_t)
    pub beta_surface: Array2<f64>,
    /// Basis coefficients (vectorized)
    pub beta_coefficients: Array1<f64>,
    /// Predictor basis type
    pub predictor_basis: BasisType,
    /// Response basis type
    pub response_basis: BasisType,
    /// Predictor grid
    pub predictor_grid: Vec<f64>,
    /// Response grid
    pub response_grid: Vec<f64>,
    /// Smoothing parameter used
    pub lambda: f64,
    /// Fitted curves: each row is a fitted response curve
    pub fitted_curves: Array2<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_functional_data_valid() {
        let grid = vec![0.0, 0.5, 1.0];
        let obs = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let data = FunctionalData::new(grid, obs).expect("Should succeed");
        assert_eq!(data.n_curves(), 2);
        assert_eq!(data.n_grid(), 3);
    }

    #[test]
    fn test_functional_data_empty_grid() {
        let result = FunctionalData::new(vec![], vec![vec![1.0]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_functional_data_empty_observations() {
        let result = FunctionalData::new(vec![0.0, 1.0], vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_functional_data_dimension_mismatch() {
        let result = FunctionalData::new(vec![0.0, 1.0], vec![vec![1.0, 2.0, 3.0]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_functional_config_default() {
        let config = FunctionalConfig::default();
        assert!(config.smoothing_param.is_none());
        assert_eq!(config.n_components, 3);
        match &config.basis {
            BasisType::BSpline { n_basis, degree } => {
                assert_eq!(*n_basis, 15);
                assert_eq!(*degree, 3);
            }
            _ => panic!("Default should be BSpline"),
        }
    }
}
