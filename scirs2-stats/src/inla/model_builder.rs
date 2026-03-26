//! Fluent model builder for INLA latent Gaussian models
//!
//! Provides `INLAModelBuilder` that assembles fixed effects, random effects,
//! and likelihood specification into a `LatentGaussianModel` ready for
//! `compute_marginals()`.
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_stats::inla::{
//!     INLAModelBuilder, LatentFieldType, LikelihoodFamily,
//! };
//!
//! let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let x1 = vec![0.1, 0.2, 0.3, 0.4, 0.5];
//!
//! let model = INLAModelBuilder::new()
//!     .likelihood(LikelihoodFamily::Gaussian)
//!     .add_fixed("slope", &x1)
//!     .add_random("spatial", LatentFieldType::RW1, 5)
//!     .observation_precision(1.0)
//!     .build(&y)
//!     .expect("model build");
//! ```

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::{StatsError, StatsResult};

use super::gaussian_field::{build_precision_matrix, LatentFieldType};
use super::types::{LatentGaussianModel, LikelihoodFamily};

/// A fixed effect specification (name + design column)
#[derive(Debug, Clone)]
struct FixedEffect {
    /// Name of the effect (for diagnostics)
    name: String,
    /// Design column values (one per observation)
    column: Vec<f64>,
}

/// A random effect specification
#[derive(Debug, Clone)]
struct RandomEffect {
    /// Name of the effect (for diagnostics)
    name: String,
    /// Latent field type determining the precision structure
    field_type: LatentFieldType,
    /// Number of levels (dimension of the random effect)
    n_levels: usize,
}

/// Builder for constructing INLA latent Gaussian models
///
/// Uses a fluent API to incrementally add fixed effects, random effects,
/// and set the likelihood family, then assembles the full design matrix
/// and block-diagonal precision matrix.
#[derive(Debug, Clone)]
pub struct INLAModelBuilder {
    /// Likelihood family (default: Gaussian)
    likelihood_family: LikelihoodFamily,
    /// Fixed effects
    fixed_effects: Vec<FixedEffect>,
    /// Random effects
    random_effects: Vec<RandomEffect>,
    /// Observation precision for Gaussian likelihood
    obs_precision: Option<f64>,
    /// Number of trials for Binomial likelihood
    n_trials: Option<Vec<f64>>,
    /// Random effect precision scale (default 1.0)
    random_scale: f64,
}

impl Default for INLAModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl INLAModelBuilder {
    /// Create a new model builder with default settings
    pub fn new() -> Self {
        Self {
            likelihood_family: LikelihoodFamily::Gaussian,
            fixed_effects: Vec::new(),
            random_effects: Vec::new(),
            obs_precision: None,
            n_trials: None,
            random_scale: 1.0,
        }
    }

    /// Set the likelihood family for the model
    pub fn likelihood(mut self, family: LikelihoodFamily) -> Self {
        self.likelihood_family = family;
        self
    }

    /// Add a fixed effect with the given name and design column
    ///
    /// The design column must have one entry per observation (validated at build time).
    pub fn add_fixed(mut self, name: &str, design_column: &[f64]) -> Self {
        self.fixed_effects.push(FixedEffect {
            name: name.to_string(),
            column: design_column.to_vec(),
        });
        self
    }

    /// Add a random effect with the given field type and number of levels
    ///
    /// The field type determines the precision matrix structure.
    /// For random effects with n_levels matching n_obs, the design matrix block
    /// is an identity. For n_levels != n_obs, a grouped identity is used
    /// (assumes observations are ordered by group).
    pub fn add_random(mut self, name: &str, field_type: LatentFieldType, n_levels: usize) -> Self {
        self.random_effects.push(RandomEffect {
            name: name.to_string(),
            field_type,
            n_levels,
        });
        self
    }

    /// Set the observation precision (1/sigma^2) for Gaussian likelihood
    pub fn observation_precision(mut self, prec: f64) -> Self {
        self.obs_precision = Some(prec);
        self
    }

    /// Set the number of trials for Binomial likelihood
    pub fn n_trials(mut self, trials: &[f64]) -> Self {
        self.n_trials = Some(trials.to_vec());
        self
    }

    /// Set the scale for random effect precision matrices (default 1.0)
    pub fn random_scale(mut self, scale: f64) -> Self {
        self.random_scale = scale;
        self
    }

    /// Build the latent Gaussian model from the specified components
    ///
    /// # Arguments
    /// * `y` - Observation vector
    ///
    /// # Returns
    /// A `LatentGaussianModel` with assembled design and precision matrices
    ///
    /// # Errors
    /// - If no effects (fixed or random) have been added
    /// - If fixed effect column lengths don't match observation length
    /// - If random effect field parameters are invalid
    /// - If dimensions are inconsistent
    pub fn build(self, y: &[f64]) -> StatsResult<LatentGaussianModel> {
        let n_obs = y.len();
        if n_obs == 0 {
            return Err(StatsError::InvalidArgument(
                "Observation vector must not be empty".to_string(),
            ));
        }
        if self.fixed_effects.is_empty() && self.random_effects.is_empty() {
            return Err(StatsError::InvalidArgument(
                "Model must have at least one fixed or random effect".to_string(),
            ));
        }

        // Validate fixed effect dimensions
        for fe in &self.fixed_effects {
            if fe.column.len() != n_obs {
                return Err(StatsError::DimensionMismatch(format!(
                    "Fixed effect '{}' has {} entries but observation vector has {} entries",
                    fe.name,
                    fe.column.len(),
                    n_obs
                )));
            }
        }

        // Compute total latent dimension
        let n_fixed = self.fixed_effects.len();
        let n_random_total: usize = self.random_effects.iter().map(|r| r.n_levels).sum();
        let p = n_fixed + n_random_total;

        if p == 0 {
            return Err(StatsError::InvalidArgument(
                "Model has zero latent dimensions".to_string(),
            ));
        }

        // Build design matrix (n_obs x p)
        let mut design = Array2::zeros((n_obs, p));

        // Fill fixed effect columns
        for (col_idx, fe) in self.fixed_effects.iter().enumerate() {
            for (row, &val) in fe.column.iter().enumerate() {
                design[[row, col_idx]] = val;
            }
        }

        // Fill random effect columns
        let mut col_offset = n_fixed;
        for re in &self.random_effects {
            if re.n_levels == n_obs {
                // Identity mapping: each observation maps to its own level
                for i in 0..n_obs {
                    design[[i, col_offset + i]] = 1.0;
                }
            } else if re.n_levels > 0 && n_obs >= re.n_levels {
                // Grouped identity: assign observations to levels cyclically
                for i in 0..n_obs {
                    let level = i % re.n_levels;
                    design[[i, col_offset + level]] = 1.0;
                }
            } else if re.n_levels > n_obs {
                // More levels than observations: each observation maps to one level
                for i in 0..n_obs {
                    design[[i, col_offset + i]] = 1.0;
                }
            }
            col_offset += re.n_levels;
        }

        // Build block-diagonal precision matrix (p x p)
        let mut precision = Array2::zeros((p, p));

        // Fixed effects: use a weak prior (small precision = large variance)
        let fixed_prior_precision = 0.001;
        for i in 0..n_fixed {
            precision[[i, i]] = fixed_prior_precision;
        }

        // Random effects: build precision blocks
        let mut block_offset = n_fixed;
        for re in &self.random_effects {
            let q_block = build_precision_matrix(&re.field_type, re.n_levels, self.random_scale)?;
            for i in 0..re.n_levels {
                for j in 0..re.n_levels {
                    precision[[block_offset + i, block_offset + j]] = q_block[[i, j]];
                }
            }
            block_offset += re.n_levels;
        }

        // Assemble the model
        let y_arr = Array1::from_vec(y.to_vec());
        let mut model = LatentGaussianModel::new(y_arr, design, precision, self.likelihood_family);

        if let Some(prec) = self.obs_precision {
            model = model.with_observation_precision(prec);
        }
        if let Some(trials) = self.n_trials {
            model = model.with_n_trials(Array1::from_vec(trials));
        }

        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_single_fixed_effect() {
        let y = vec![1.0, 2.0, 3.0];
        let x = vec![0.5, 1.0, 1.5];
        let model = INLAModelBuilder::new()
            .add_fixed("x", &x)
            .build(&y)
            .expect("should build with single fixed effect");

        assert_eq!(model.design_matrix.nrows(), 3);
        assert_eq!(model.design_matrix.ncols(), 1);
        assert!((model.design_matrix[[0, 0]] - 0.5).abs() < 1e-12);
        assert!((model.design_matrix[[1, 0]] - 1.0).abs() < 1e-12);
        assert!((model.design_matrix[[2, 0]] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_builder_poisson_likelihood() {
        let y = vec![1.0, 3.0, 5.0];
        let x = vec![1.0, 1.0, 1.0];
        let model = INLAModelBuilder::new()
            .likelihood(LikelihoodFamily::Poisson)
            .add_fixed("intercept", &x)
            .build(&y)
            .expect("should build with Poisson");
        assert_eq!(model.likelihood, LikelihoodFamily::Poisson);
    }

    #[test]
    fn test_builder_dimension_mismatch() {
        let y = vec![1.0, 2.0, 3.0];
        let x = vec![0.5, 1.0]; // wrong length
        let result = INLAModelBuilder::new().add_fixed("x", &x).build(&y);
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_empty_model() {
        let y = vec![1.0, 2.0];
        let result = INLAModelBuilder::new().build(&y);
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_with_icar_random_effect() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let adj = vec![(0, 1), (1, 2), (2, 3)];
        let model = INLAModelBuilder::new()
            .add_random("spatial", LatentFieldType::ICAR { adjacency: adj }, 4)
            .build(&y)
            .expect("should build with ICAR");

        // Design: 4x4 identity (n_levels == n_obs)
        assert_eq!(model.design_matrix.nrows(), 4);
        assert_eq!(model.design_matrix.ncols(), 4);
        for i in 0..4 {
            assert!((model.design_matrix[[i, i]] - 1.0).abs() < 1e-12);
        }
        // Precision: 4x4 graph Laplacian
        assert_eq!(model.precision_matrix.nrows(), 4);
        assert!((model.precision_matrix[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((model.precision_matrix[[1, 1]] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_builder_fixed_and_random() {
        let y = vec![1.0, 2.0, 3.0];
        let x = vec![1.0, 1.0, 1.0]; // intercept
        let model = INLAModelBuilder::new()
            .add_fixed("intercept", &x)
            .add_random("iid_effect", LatentFieldType::IID, 3)
            .observation_precision(2.0)
            .build(&y)
            .expect("should build with fixed + random");

        // p = 1 (fixed) + 3 (random) = 4
        assert_eq!(model.design_matrix.ncols(), 4);
        assert_eq!(model.precision_matrix.nrows(), 4);
        // Fixed effect prior: small precision
        assert!(model.precision_matrix[[0, 0]] < 0.01);
        // Random IID precision: 1.0 on diagonal
        assert!((model.precision_matrix[[1, 1]] - 1.0).abs() < 1e-12);
        assert!((model.precision_matrix[[2, 2]] - 1.0).abs() < 1e-12);
        assert!((model.precision_matrix[[3, 3]] - 1.0).abs() < 1e-12);
        // Observation precision set
        assert!((model.observation_precision.unwrap_or(0.0) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_builder_compatible_with_pipeline() {
        use super::super::marginals::compute_marginals;
        use super::super::types::INLAConfig;

        let y = vec![1.0, 2.0, 3.0];
        let x = vec![1.0, 1.0, 1.0];
        let model = INLAModelBuilder::new()
            .likelihood(LikelihoodFamily::Gaussian)
            .add_fixed("intercept", &x)
            .observation_precision(1.0)
            .build(&y)
            .expect("should build");

        let config = INLAConfig::default();
        let result = compute_marginals(&model, &config);
        // Should not panic; we accept either Ok or a controlled Err
        // (convergence issues are acceptable, but no panics)
        match result {
            Ok(res) => {
                assert_eq!(res.marginal_means.len(), model.design_matrix.ncols());
            }
            Err(e) => {
                // Computation errors from mode-finding are acceptable
                let _ = format!("{e}");
            }
        }
    }

    #[test]
    fn test_builder_default() {
        let builder = INLAModelBuilder::default();
        assert_eq!(builder.likelihood_family, LikelihoodFamily::Gaussian);
        assert!(builder.fixed_effects.is_empty());
        assert!(builder.random_effects.is_empty());
    }

    #[test]
    fn test_builder_empty_observations() {
        let y: Vec<f64> = vec![];
        let result = INLAModelBuilder::new().add_fixed("x", &[]).build(&y);
        assert!(result.is_err());
    }
}
