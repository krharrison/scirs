//! Types for the Fast Multipole Method (FMM) for 2D N-body problems.

/// A 2D charged particle.
#[derive(Debug, Clone, Copy)]
pub struct Particle {
    /// 2D position (x, y).
    pub position: [f64; 2],
    /// Scalar charge.
    pub charge: f64,
}

/// Configuration for the FMM solver.
#[derive(Debug, Clone)]
pub struct FmmConfig {
    /// Maximum tree depth (number of refinement levels).
    pub max_level: usize,
    /// Maximum particles per leaf cell before subdivision.
    pub n_crit: usize,
    /// Order of multipole / local expansions (number of complex coefficients).
    pub p_order: usize,
}

impl Default for FmmConfig {
    fn default() -> Self {
        Self {
            max_level: 6,
            n_crit: 8,
            p_order: 6,
        }
    }
}

/// A single FMM cell / box.
#[derive(Debug, Clone)]
pub struct FmmCell {
    /// Centre of the box.
    pub center: [f64; 2],
    /// Half the side length of the box.
    pub half_size: f64,
    /// Multipole expansion coefficients (complex: [re0, im0, re1, im1, …]).
    pub multipole: Vec<f64>,
    /// Local expansion coefficients (complex: [re0, im0, re1, im1, …]).
    pub local: Vec<f64>,
}

impl FmmCell {
    /// Create a new empty cell.
    pub fn new(center: [f64; 2], half_size: f64, p_order: usize) -> Self {
        Self {
            center,
            half_size,
            multipole: vec![0.0; 2 * p_order],
            local: vec![0.0; 2 * p_order],
        }
    }
}

/// Result of an FMM computation.
#[derive(Debug, Clone)]
pub struct FmmResult {
    /// Scalar potential at each particle position.
    pub potentials: Vec<f64>,
    /// Force (gradient of potential) at each particle position.
    pub forces: Vec<[f64; 2]>,
}
