//! Graph Signal Processing (GSP) module.
//!
//! This module provides tools for analysing and processing signals defined on
//! the vertices of a graph, using the spectral theory of graph Laplacians.
//!
//! ## Submodules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`gsp`] | Graph Fourier Transform, spectral filters (LP/HP/BP), diffusion wavelets, Tikhonov smoother |
//! | [`sampling`] | Optimal sampling set selection, bandlimited reconstruction, Gershgorin bounds, uncertainty principle |
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use scirs2_core::ndarray::{Array1, Array2};
//! use scirs2_graph::signal_processing::gsp::{GraphFourierTransform, IdealLowPass, GraphFilter};
//! use scirs2_graph::signal_processing::sampling::{GraphSampling, BandlimitedReconstruction};
//!
//! // Build a small path graph
//! let mut adj = Array2::<f64>::zeros((6, 6));
//! for i in 0..5_usize {
//!     adj[[i, i+1]] = 1.0;
//!     adj[[i+1, i]] = 1.0;
//! }
//!
//! // Graph Fourier Transform + low-pass filtering
//! let gft = GraphFourierTransform::from_adjacency(&adj).unwrap();
//! let signal = Array1::from_vec(vec![1.0, 0.5, 0.0, -0.5, -1.0, -0.5]);
//! let smoothed = IdealLowPass::new(2).apply(&gft, &signal).unwrap();
//!
//! // Optimal sampling and reconstruction
//! let sampler = GraphSampling::new(2);
//! let set = sampler.greedy_sampling_set(&gft).unwrap();
//! let samples = Array1::from_iter(set.iter().map(|&i| signal[i]));
//! let rec = BandlimitedReconstruction::new(2)
//!     .reconstruct(&gft, &set, &samples)
//!     .unwrap();
//! ```

pub mod gsp;
pub mod sampling;

// Re-export the most commonly used types
pub use gsp::{
    GraphBandpass, GraphFilter, GraphFourierTransform, GraphSignalSmoother, GraphWavelet,
    IdealHighPass, IdealLowPass,
};
pub use sampling::{
    BandlimitedReconstruction, GershgorinBound, GraphSampling, GraphUncertaintyPrinciple,
};
