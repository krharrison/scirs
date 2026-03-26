//! Types for Parallel Algebraic Multigrid (AMG) Coarsening
//!
//! This module defines configuration, result, and hierarchy types for
//! the parallel AMG coarsening algorithms.

use crate::csr::CsrMatrix;

/// Configuration for parallel AMG coarsening algorithms
#[derive(Debug, Clone)]
pub struct ParallelAmgConfig {
    /// Number of threads for parallel computations
    pub n_threads: usize,
    /// Strength-of-connection threshold (default 0.25)
    pub strength_threshold: f64,
    /// Coarsening method to use
    pub coarsening: CoarsenMethod,
    /// Maximum number of hierarchy levels
    pub max_levels: usize,
    /// Minimum coarse grid size (stop coarsening below this)
    pub min_coarse_size: usize,
    /// Coarsening ratio target (stop if ratio exceeds this)
    pub max_coarsening_ratio: f64,
    /// Jacobi smoothing weight for SA interpolation
    pub omega: f64,
}

impl Default for ParallelAmgConfig {
    fn default() -> Self {
        Self {
            n_threads: 1,
            strength_threshold: 0.25,
            coarsening: CoarsenMethod::ParallelRS,
            max_levels: 10,
            min_coarse_size: 4,
            max_coarsening_ratio: 0.85,
            omega: 4.0 / 3.0,
        }
    }
}

/// Method for parallel coarsening
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoarsenMethod {
    /// Parallel Ruge-Stüben coarsening with parallel pass-1
    ParallelRS,
    /// Smoothed aggregation coarsening
    ParallelSA,
    /// Parallel Maximum Independent Set coarsening
    PMIS,
    /// Cleary-Luby-Jones-Plassmann coarsening
    CLJP,
}

/// Result of a coarsening step: C/F splitting
#[derive(Debug, Clone)]
pub struct CoarseningResult {
    /// Indices of coarse (C) nodes
    pub c_nodes: Vec<usize>,
    /// Indices of fine (F) nodes
    pub f_nodes: Vec<usize>,
    /// C/F labeling array: 0 = F-node, 1 = C-node
    pub cf_splitting: Vec<u8>,
}

impl CoarseningResult {
    /// Create a CoarseningResult from a cf_splitting array
    pub fn from_splitting(cf_splitting: Vec<u8>) -> Self {
        let n = cf_splitting.len();
        let mut c_nodes = Vec::new();
        let mut f_nodes = Vec::new();
        for i in 0..n {
            if cf_splitting[i] == 1 {
                c_nodes.push(i);
            } else {
                f_nodes.push(i);
            }
        }
        Self {
            c_nodes,
            f_nodes,
            cf_splitting,
        }
    }

    /// Number of nodes
    pub fn n(&self) -> usize {
        self.cf_splitting.len()
    }

    /// Number of coarse nodes
    pub fn n_coarse(&self) -> usize {
        self.c_nodes.len()
    }

    /// Number of fine nodes
    pub fn n_fine(&self) -> usize {
        self.f_nodes.len()
    }

    /// Coarsening ratio (n_coarse / n_total)
    pub fn coarsening_ratio(&self) -> f64 {
        let n = self.n();
        if n == 0 {
            return 0.0;
        }
        self.n_coarse() as f64 / n as f64
    }
}

/// A single level in the parallel AMG hierarchy
#[derive(Debug, Clone)]
pub struct ParallelAmgLevel {
    /// Fine-grid system matrix A
    pub a: CsrMatrix<f64>,
    /// Prolongation (interpolation) operator P: coarse → fine
    pub p: CsrMatrix<f64>,
    /// Restriction operator R: fine → coarse (= P^T for SA)
    pub r: CsrMatrix<f64>,
    /// Number of fine-grid unknowns
    pub n_fine: usize,
    /// Number of coarse-grid unknowns
    pub n_coarse: usize,
}

impl ParallelAmgLevel {
    /// Create a new level
    pub fn new(
        a: CsrMatrix<f64>,
        p: CsrMatrix<f64>,
        r: CsrMatrix<f64>,
        n_fine: usize,
        n_coarse: usize,
    ) -> Self {
        Self {
            a,
            p,
            r,
            n_fine,
            n_coarse,
        }
    }
}

/// Complete parallel AMG hierarchy
#[derive(Debug, Clone)]
pub struct ParallelAmgHierarchy {
    /// Hierarchy levels (from finest to coarsest)
    pub levels: Vec<ParallelAmgLevel>,
    /// Number of levels in hierarchy
    pub n_levels: usize,
    /// Coarsest-grid system matrix
    pub coarsest_a: CsrMatrix<f64>,
}

impl ParallelAmgHierarchy {
    /// Create a new hierarchy
    pub fn new(levels: Vec<ParallelAmgLevel>, coarsest_a: CsrMatrix<f64>) -> Self {
        let n_levels = levels.len() + 1; // levels + coarsest
        Self {
            levels,
            n_levels,
            coarsest_a,
        }
    }

    /// Size of the fine grid
    pub fn fine_size(&self) -> usize {
        if self.levels.is_empty() {
            self.coarsest_a.shape().0
        } else {
            self.levels[0].n_fine
        }
    }

    /// Size of the coarsest grid
    pub fn coarse_size(&self) -> usize {
        self.coarsest_a.shape().0
    }
}
