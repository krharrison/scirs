//! Feature selection utilities
//!
//! This module provides comprehensive methods for selecting relevant features
//! from datasets, reducing dimensionality and improving model performance.
//!
//! # Available Methods
//!
//! | Method | Description | Use Case |
//! |--------|-------------|----------|
//! | `VarianceThreshold` | Remove low-variance features | Unsupervised, constant removal |
//! | `MutualInfoSelector` | Mutual information selection | Classification or regression |
//! | `Chi2Selector` | Chi-squared test | Classification (non-negative features) |
//! | `FTestSelector` | ANOVA F-test | Classification or regression |
//! | `RecursiveFeatureElimination` | RFE with importance callback | Any importance measure |
//! | `SelectKBest` | Generic top-K by score | Any scoring function |
//! | `SelectByPValue` | Select by significance | Any scoring function |
//! | `SelectPercentile` | Select top percentile | Any scoring function |

/// Variance threshold feature selection
pub mod variance_threshold;

/// Mutual information based feature selection
pub mod mutual_info;

/// Chi-squared test for categorical feature selection
pub mod chi_squared;

/// ANOVA F-test for feature ranking
pub mod f_test;

/// Recursive Feature Elimination
pub mod recursive_elimination;

/// Select K best features by score
pub mod select_k_best;

// Re-exports
pub use chi_squared::{chi2_scores, Chi2Result, Chi2Selector};
pub use f_test::{f_classif, f_regression, FTestResult, FTestSelector};
pub use mutual_info::{
    mutual_info_classif, mutual_info_regression, MutualInfoMethod, MutualInfoSelector,
};
pub use recursive_elimination::{correlation_importance, RecursiveFeatureElimination};
pub use select_k_best::{SelectByPValue, SelectKBest, SelectPercentile};
pub use variance_threshold::VarianceThreshold;
