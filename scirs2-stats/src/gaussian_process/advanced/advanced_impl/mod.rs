//! Implementation sub-modules for advanced Gaussian Process methods.

pub(crate) mod linalg;
pub mod classification;
pub mod deep;
pub mod hyperopt;
pub mod kernels;
pub mod multioutput;
pub mod sparse;

// Re-export all public items so callers can use advanced_impl::* or go through advanced.rs.
pub use classification::{ClassificationInference, ClassificationLikelihood, GPClassification};
pub use deep::{DeepGP, DeepGPLayerConfig};
pub use hyperopt::GPHyperparamOpt;
pub use kernels::{
    ARDKernel, AdditiveKernel, AdvancedKernel, ArcCosineKernel, NeuralTangentKernel,
    SpectralMixtureComponent, SpectralMixtureKernel,
};
pub use multioutput::MultiOutputGP;
pub use sparse::{SparseApproximation, SparseGP};
