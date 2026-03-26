//! ML-guided preconditioner selection.
//!
//! This module provides automatic selection of sparse-matrix preconditioners
//! based on extracted matrix features and a classification pipeline (heuristic
//! rules or a trained random-forest model). A cost model estimates the total
//! solve cost to further refine the recommendation.
//!
//! # Quick start
//!
//! ```
//! use scirs2_sparse::ml_preconditioner::{select_preconditioner, SelectionConfig};
//!
//! // 3×3 tridiagonal SPD matrix in CSR format
//! let values  = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
//! let col_idx = vec![0, 1, 0, 1, 2, 1, 2];
//! let row_ptr = vec![0, 2, 5, 7];
//!
//! let result = select_preconditioner(&values, &row_ptr, &col_idx, 3, &SelectionConfig::default())
//!     .expect("selection");
//! println!("Recommended: {}", result.recommended);
//! ```

/// Types and configuration.
pub mod types;

/// Feature extraction from raw CSR data.
pub mod feature_extraction;

/// Classification models (heuristic + random forest).
pub mod classifier;

/// Cost estimation and ranking.
pub mod cost_model;

// Re-exports
pub use classifier::{
    select_preconditioner, DecisionStump, DecisionTree, HeuristicClassifier,
    PreconditionerClassifier, RandomForest,
};
pub use cost_model::{estimate_cost, rank_by_cost};
pub use feature_extraction::{extract_features, normalize_features};
pub use types::{
    CostEstimate, MatrixFeatures, PreconditionerType, SelectionConfig, SelectionResult,
};
