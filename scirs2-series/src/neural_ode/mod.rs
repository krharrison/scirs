//! Neural ODE and Continuous-Time Time Series Models
//!
//! This module provides a suite of neural ODE-based architectures for
//! modelling **continuous-time** and **irregularly-sampled** time series:
//!
//! | Sub-module | Model | Key idea |
//! |------------|-------|----------|
//! | [`latent_ode`] | Latent Neural ODE | ODE-RNN encoder → latent ODE dynamics → decoder |
//! | [`neural_cde`] | Neural CDE | CDE driven by a cubic-Hermite control path + log-signature features |
//! | [`liquid_snn`] | Liquid Time-Constant (LTC) / CfC / Liquid State Network | Input-adaptive time constants, closed-form CDE approximation |
//! | [`ode_rnn`] | ODE-RNN & GRU-ODE | Standard RNN augmented with ODE between observations |
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use scirs2_series::neural_ode::{
//!     latent_ode::{LatentODE, LatentODEConfig},
//!     neural_cde::{NeuralCDE, NeuralCDEConfig},
//!     liquid_snn::{LiquidSNN, LiquidSNNConfig},
//!     ode_rnn::{OdeRnn, OdeRnnConfig, GruOde, GruOdeConfig},
//! };
//! use scirs2_core::ndarray::array;
//!
//! // --- Latent Neural ODE ---
//! let latent_model = LatentODE::<f64>::new(LatentODEConfig {
//!     obs_dim: 1,
//!     latent_dim: 8,
//!     ..Default::default()
//! }).expect("should succeed");
//! let times = vec![0.0_f64, 0.5, 1.0];
//! let obs = vec![array![1.0_f64], array![0.8], array![0.6]];
//! let future = vec![1.5_f64, 2.0];
//! let preds = latent_model.predict(&times, &obs, &future).expect("should succeed");
//!
//! // --- Neural CDE ---
//! let cde_model = NeuralCDE::<f64>::new(NeuralCDEConfig {
//!     input_channels: 1,
//!     hidden_dim: 8,
//!     output_dim: 1,
//!     ..Default::default()
//! }).expect("should succeed");
//! let obs2 = vec![array![0.0_f64], array![1.0], array![0.5]];
//! let pred = cde_model.forward(&times, &obs2).expect("should succeed");
//!
//! // --- Liquid SNN (LTC) ---
//! let snn = LiquidSNN::<f64>::new(LiquidSNNConfig {
//!     input_dim: 1,
//!     reservoir_size: 16,
//!     output_dim: 1,
//!     ..Default::default()
//! }).expect("should succeed");
//! let inputs = vec![array![0.5_f64], array![0.8], array![0.3]];
//! let outs = snn.forward(&inputs).expect("should succeed");
//!
//! // --- ODE-RNN ---
//! let ode_rnn = OdeRnn::<f64>::new(OdeRnnConfig {
//!     input_dim: 1,
//!     hidden_dim: 8,
//!     output_dim: 1,
//!     ..Default::default()
//! }).expect("should succeed");
//! let obs3 = vec![array![1.0_f64], array![0.9], array![0.7]];
//! let pred2 = ode_rnn.predict(&times, &obs3).expect("should succeed");
//! ```
//!
//! ## Mathematical background
//!
//! All four models build on the **Neural ODE** framework (Chen et al., NeurIPS 2018):
//! the hidden state trajectory is defined by a differential equation whose right-hand
//! side is parameterised by a neural network.  This gives:
//!
//! * **Irregular sampling** – handled naturally because we integrate between
//!   observed timestamps rather than assuming a uniform grid.
//! * **Memory efficiency** – adjoint-based backprop needs only `O(1)` memory
//!   (not implemented here; weights are forward-pass only).
//! * **Expressive continuous dynamics** – unlike discrete RNNs, the model can
//!   capture arbitrary continuous-time behaviours.
//!
//! ### References
//!
//! * Chen et al., "Neural Ordinary Differential Equations", NeurIPS 2018.
//! * Rubanova et al., "Latent ODEs for Irregularly-Sampled Time Series", NeurIPS 2019.
//! * Kidger et al., "Neural Controlled Differential Equations for Irregular Time Series", NeurIPS 2020.
//! * Hasani et al., "Liquid Time-Constant Networks", AAAI 2021.
//! * Hasani et al., "Closed-Form Continuous-Depth Models", NeurIPS 2022.
//! * de Brouwer et al., "GRU-ODE-Bayes: Continuous Modeling of Sporadically-Observed Time Series", NeurIPS 2019.

pub mod cnf;
pub mod ffjord;
pub mod latent_ode;
pub mod liquid_snn;
pub mod neural_cde;
pub mod ode_rnn;

// ---------------------------------------------------------------------------
// Convenience re-exports
// ---------------------------------------------------------------------------

pub use latent_ode::{GruCell as LatentGruCell, LatentDynamics, LatentODE, LatentODEConfig};
pub use liquid_snn::{CfCCell, LiquidSNN, LiquidSNNConfig, LtcCell, LtcState};
pub use neural_cde::{CubicHermiteSpline, LogSignatureFeatures, NeuralCDE, NeuralCDEConfig};
pub use ode_rnn::{GruCell, GruOde, GruOdeConfig, HiddenDynamics, OdeRnn, OdeRnnConfig};
