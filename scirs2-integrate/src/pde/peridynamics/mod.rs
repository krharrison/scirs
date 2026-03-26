//! Peridynamics for Fracture Mechanics
//!
//! Peridynamics is a non-local continuum mechanics formulation that naturally
//! handles crack initiation and propagation without the need for special
//! crack-tracking algorithms. The formulation replaces spatial derivatives with
//! integrals over a finite horizon, making it applicable even where the
//! displacement field is discontinuous.
//!
//! ## Formulations
//!
//! * **Bond-based**: Original Silling (2000) formulation. Pairwise force between
//!   material points limited to Poisson's ratio ν = 1/3 (3D) or ν = 1/4 (2D).
//! * **State-based (ordinary)**: Silling et al. (2007) generalization supporting
//!   arbitrary elastic moduli via force states.
//!
//! ## References
//!
//! * Silling, S.A. (2000). Reformulation of elasticity theory for discontinuities
//!   and long-range forces. *Journal of the Mechanics and Physics of Solids*, 48(1), 175-209.
//! * Silling, S.A. et al. (2007). Peridynamic states and constitutive modeling.
//!   *Journal of Elasticity*, 88(2), 151-184.

pub mod bond_based;
pub mod integration;
pub mod neighbor_list;
pub mod state_based;

pub use bond_based::{BondBasedConfig, BondBasedSolver, IsotropicMaterial};
pub use integration::{BodyForce, DirichletBC, PdIntegrator};
pub use neighbor_list::{BucketGrid, NeighborList};
pub use state_based::{StateBasedConfig, StateBasedSolver};

/// Trait for peridynamic material models.
///
/// Implementors define the constitutive response of each bond.
pub trait PeridynamicMaterial: Send + Sync {
    /// Compute the bond force scalar given stretch `s` and reference bond vector `xi`.
    ///
    /// The force density vector is obtained by multiplying this scalar by the unit
    /// bond direction vector and the volume of the interacting particle.
    ///
    /// # Arguments
    ///
    /// * `stretch` - Dimensionless bond stretch: `s = (|y| - |xi|) / |xi|`
    /// * `xi` - Reference bond vector `xi = x_j - x_i` in the reference configuration
    fn bond_force(&self, stretch: f64, xi: [f64; 3]) -> f64;

    /// Return the critical stretch beyond which a bond is irreversibly broken.
    fn critical_stretch(&self) -> f64;
}

/// Complete peridynamic state for a particle system.
#[derive(Debug, Clone)]
pub struct PeridynamicState {
    /// Reference (initial) positions of all particles, shape \[n_particles\]\[3\].
    pub positions: Vec<[f64; 3]>,
    /// Current displacement vectors, shape \[n_particles\]\[3\].
    pub displacements: Vec<[f64; 3]>,
    /// Current velocity vectors, shape \[n_particles\]\[3\].
    pub velocities: Vec<[f64; 3]>,
    /// Damage index per particle: 0.0 = intact, 1.0 = fully broken.
    pub damage: Vec<f64>,
    /// Total number of particles.
    pub n_particles: usize,
}

impl PeridynamicState {
    /// Construct a state with zero displacements, velocities, and damage.
    pub fn new(positions: Vec<[f64; 3]>) -> Self {
        let n = positions.len();
        Self {
            displacements: vec![[0.0, 0.0, 0.0]; n],
            velocities: vec![[0.0, 0.0, 0.0]; n],
            damage: vec![0.0; n],
            n_particles: n,
            positions,
        }
    }

    /// Return the current (deformed) position of particle `i`.
    #[inline]
    pub fn current_position(&self, i: usize) -> [f64; 3] {
        [
            self.positions[i][0] + self.displacements[i][0],
            self.positions[i][1] + self.displacements[i][1],
            self.positions[i][2] + self.displacements[i][2],
        ]
    }
}

/// A bond connecting two particles in the peridynamic horizon.
#[derive(Debug, Clone, Copy)]
pub struct Bond {
    /// Index of the first particle.
    pub i: usize,
    /// Index of the second particle (j > i by convention).
    pub j: usize,
    /// Initial (reference) bond vector: `xi = x_j - x_i`.
    pub xi: [f64; 3],
    /// Whether this bond is still active (has not failed).
    pub active: bool,
}

impl Bond {
    /// Compute the Euclidean length of the reference bond vector.
    #[inline]
    pub fn reference_length(&self) -> f64 {
        (self.xi[0] * self.xi[0] + self.xi[1] * self.xi[1] + self.xi[2] * self.xi[2]).sqrt()
    }
}

/// Reason a bond has failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FailureMode {
    /// No bond exists between the pair (outside horizon).
    NoBond,
    /// Bond stretch exceeded the critical stretch threshold.
    StretchExceeded,
    /// Strain energy density exceeded the critical energy threshold.
    EnergyExceeded,
}
