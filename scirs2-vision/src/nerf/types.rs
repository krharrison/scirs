//! Core types for Neural Radiance Fields (NeRF) and Instant-NGP.
//!
//! Defines configuration structs, ray/sample types, and rendering result types
//! used across the NeRF implementation.

/// Configuration for a standard NeRF MLP model (Mildenhall et al. 2020).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct NerfConfig {
    /// Number of hidden layers in the geometry network.
    pub n_layers: usize,
    /// Width (number of units) of each hidden layer.
    pub hidden_dim: usize,
    /// Number of frequency bands for positional encoding of 3-D location.
    pub n_freq_pos: usize,
    /// Number of frequency bands for positional encoding of view direction.
    pub n_freq_dir: usize,
    /// Near clipping distance along the ray.
    pub near: f64,
    /// Far clipping distance along the ray.
    pub far: f64,
    /// Number of coarse stratified samples per ray.
    pub n_samples: usize,
    /// Number of additional importance samples per ray (hierarchical sampling).
    pub n_importance: usize,
}

impl Default for NerfConfig {
    fn default() -> Self {
        Self {
            n_layers: 8,
            hidden_dim: 256,
            n_freq_pos: 10,
            n_freq_dir: 4,
            near: 2.0,
            far: 6.0,
            n_samples: 64,
            n_importance: 128,
        }
    }
}

/// Configuration for Instant-NGP multi-resolution hash encoding (Müller et al. 2022).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct NgpConfig {
    /// Number of resolution levels in the hash grid hierarchy.
    pub n_levels: usize,
    /// Number of feature dimensions stored per hash entry per level.
    pub n_features_per_level: usize,
    /// log₂ of the hash table capacity at each level.
    pub log2_hashmap_size: usize,
    /// Grid resolution at the coarsest level.
    pub base_resolution: usize,
    /// Grid resolution at the finest level.
    pub finest_resolution: usize,
}

impl Default for NgpConfig {
    fn default() -> Self {
        Self {
            n_levels: 16,
            n_features_per_level: 2,
            log2_hashmap_size: 19,
            base_resolution: 16,
            finest_resolution: 512,
        }
    }
}

/// A camera ray defined by an origin point and a unit-length direction vector.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    /// World-space origin of the ray (camera position).
    pub origin: [f64; 3],
    /// Unit-length direction vector of the ray in world space.
    pub direction: [f64; 3],
}

impl Ray {
    /// Construct a new [`Ray`], normalising `direction` to unit length.
    ///
    /// Returns `None` when `direction` has zero (or near-zero) magnitude.
    pub fn new(origin: [f64; 3], direction: [f64; 3]) -> Option<Self> {
        let mag = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
            .sqrt();
        if mag < 1e-12 {
            return None;
        }
        Some(Self {
            origin,
            direction: [direction[0] / mag, direction[1] / mag, direction[2] / mag],
        })
    }

    /// Evaluate the ray at parameter `t`: `origin + t * direction`.
    #[inline]
    pub fn at(&self, t: f64) -> [f64; 3] {
        [
            self.origin[0] + t * self.direction[0],
            self.origin[1] + t * self.direction[1],
            self.origin[2] + t * self.direction[2],
        ]
    }
}

/// A single volumetric sample along a ray.
#[derive(Debug, Clone)]
pub struct SamplePoint {
    /// World-space 3-D position of the sample.
    pub position: [f64; 3],
    /// Distance along the ray at which this sample was taken.
    pub t: f64,
    /// Volume density σ predicted by the MLP (non-negative).
    pub density: f64,
    /// RGB radiance (each channel in [0, 1]) predicted by the MLP.
    pub color: [f64; 3],
}

impl SamplePoint {
    /// Create a new sample, clamping `density` to ≥ 0.
    pub fn new(position: [f64; 3], t: f64, density: f64, color: [f64; 3]) -> Self {
        Self {
            position,
            t,
            density: density.max(0.0),
            color: [
                color[0].clamp(0.0, 1.0),
                color[1].clamp(0.0, 1.0),
                color[2].clamp(0.0, 1.0),
            ],
        }
    }
}

/// Output of the discrete volume-rendering integral.
#[derive(Debug, Clone)]
pub struct VolumeRenderResult {
    /// Rendered RGB color for the ray (each channel in [0, 1]).
    pub color: [f64; 3],
    /// Expected depth — weighted sum of sample distances.
    pub depth: f64,
    /// Accumulated transmittance remaining after all samples.
    pub transmittance: f64,
    /// Per-sample alpha-compositing weights (Tᵢ · αᵢ).
    pub weights: Vec<f64>,
}
