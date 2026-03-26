//! Types for Nield-Kuznetsov function computations.
//!
//! This module defines configuration, result, and classification types used
//! throughout the Nield-Kuznetsov function implementations.

use serde::{Deserialize, Serialize};

/// Configuration for Nield-Kuznetsov function computations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NKConfig {
    /// Convergence tolerance for quadrature (default: 1e-12)
    pub tol: f64,
    /// Maximum number of quadrature sub-intervals (default: 200)
    pub max_terms: usize,
    /// Number of asymptotic expansion terms (default: 15)
    pub asymptotic_terms: usize,
    /// Threshold for switching to asymptotic expansion (default: 8.0)
    pub asymptotic_threshold: f64,
    /// Enable confluent hypergeometric connection (default: true)
    pub use_hypergeometric: bool,
}

impl Default for NKConfig {
    fn default() -> Self {
        NKConfig {
            tol: 1e-12,
            max_terms: 200,
            asymptotic_terms: 15,
            asymptotic_threshold: 8.0,
            use_hypergeometric: true,
        }
    }
}

/// Result of a Nield-Kuznetsov function evaluation.
#[derive(Debug, Clone)]
pub struct NKResult {
    /// The computed function value
    pub value: f64,
    /// The derivative value (if computed)
    pub derivative: Option<f64>,
    /// The function index i in Nk_i
    pub index: usize,
    /// Evaluation point x
    pub x: f64,
    /// Estimated numerical error (if available)
    pub error_estimate: Option<f64>,
    /// Method used for computation
    pub method: NKMethod,
}

/// Method used for Nield-Kuznetsov computation.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NKMethod {
    /// Direct quadrature via variation of parameters
    Quadrature,
    /// Asymptotic expansion for large x
    AsymptoticLarge,
    /// Series expansion for small x
    SeriesSmall,
    /// Confluent hypergeometric function connection
    Hypergeometric,
}

/// Type of Nield-Kuznetsov function.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NKFunctionType {
    /// Nk_0: Standard Nield-Kuznetsov function (Brinkman equation)
    Nk0,
    /// Nk_1: First-order Nield-Kuznetsov function
    Nk1,
    /// Nk_2: Second-order Nield-Kuznetsov function
    Nk2,
    /// Nk_3: Third-order Nield-Kuznetsov function
    Nk3,
    /// General Nk_i for arbitrary order i
    NkGeneral(usize),
}

impl NKFunctionType {
    /// Get the function index.
    pub fn index(&self) -> usize {
        match self {
            NKFunctionType::Nk0 => 0,
            NKFunctionType::Nk1 => 1,
            NKFunctionType::Nk2 => 2,
            NKFunctionType::Nk3 => 3,
            NKFunctionType::NkGeneral(i) => *i,
        }
    }

    /// Create from index.
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => NKFunctionType::Nk0,
            1 => NKFunctionType::Nk1,
            2 => NKFunctionType::Nk2,
            3 => NKFunctionType::Nk3,
            _ => NKFunctionType::NkGeneral(i),
        }
    }
}

/// Physical model associated with Nield-Kuznetsov functions.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NKPhysicalModel {
    /// Brinkman equation for flow through porous media
    Brinkman,
    /// Darcy-Lapwood equation for convective instability
    DarcyLapwood,
    /// Acoustic gravity waves in stratified media
    AcousticGravity,
    /// General Airy-type integral model
    AiryIntegral,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nk_config_default() {
        let config = NKConfig::default();
        assert!((config.tol - 1e-12).abs() < f64::EPSILON);
        assert_eq!(config.max_terms, 200);
        assert_eq!(config.asymptotic_terms, 15);
        assert!((config.asymptotic_threshold - 8.0).abs() < f64::EPSILON);
        assert!(config.use_hypergeometric);
    }

    #[test]
    fn test_nk_function_type_index() {
        assert_eq!(NKFunctionType::Nk0.index(), 0);
        assert_eq!(NKFunctionType::Nk1.index(), 1);
        assert_eq!(NKFunctionType::Nk2.index(), 2);
        assert_eq!(NKFunctionType::Nk3.index(), 3);
        assert_eq!(NKFunctionType::NkGeneral(5).index(), 5);
    }

    #[test]
    fn test_nk_function_type_from_index() {
        assert_eq!(NKFunctionType::from_index(0), NKFunctionType::Nk0);
        assert_eq!(NKFunctionType::from_index(1), NKFunctionType::Nk1);
        assert_eq!(NKFunctionType::from_index(7), NKFunctionType::NkGeneral(7));
    }
}
