//! Neural Radiance Fields (NeRF) and Instant-NGP.
//!
//! This module implements:
//!
//! - **Positional encoding** – Fourier feature encoding (Mildenhall et al. 2020).
//! - **NeRF MLP** – the standard 8-layer geometry + 2-layer radiance network.
//! - **Volume rendering** – discrete volume rendering integral, stratified
//!   sampling, and hierarchical (importance) sampling.
//! - **Hash encoding** – Instant-NGP multi-resolution hash grid (Müller et al. 2022)
//!   together with a tiny 2-layer MLP.
//!
//! ## References
//!
//! - Mildenhall, B. et al. (2020). "NeRF: Representing Scenes as Neural Radiance
//!   Fields for View Synthesis". ECCV 2020.
//! - Müller, T. et al. (2022). "Instant Neural Graphics Primitives with a
//!   Multiresolution Hash Encoding". SIGGRAPH 2022.

pub mod hash_encoding;
pub mod mlp_nerf;
pub mod positional_encoding;
pub mod types;
pub mod volume_rendering;

// ── Convenience re-exports ────────────────────────────────────────────────

pub use hash_encoding::{hash_coords, HashEncoder, InstantNgpMlp};
pub use mlp_nerf::NerfMlp;
pub use positional_encoding::{encode_direction, encode_position, positional_encode};
pub use types::{NerfConfig, NgpConfig, Ray, SamplePoint, VolumeRenderResult};
pub use volume_rendering::{generate_rays, importance_sample, stratified_sample, volume_render};
