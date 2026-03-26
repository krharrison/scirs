//! Alignment module: Procrustes analysis and related methods.

pub mod procrustes;

pub use procrustes::{
    generalized_procrustes, orthogonal_procrustes, ProcrustesConfig, ProcrustesResult,
};
