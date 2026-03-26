//! KAN layer and network implementations.
//!
//! A KAN layer replaces fixed activation functions with learnable univariate
//! functions on each edge. The output is:
//!
//! ```text
//! y_j = sum_{i=1}^{n_in}  phi_{i,j}(x_i)
//! ```
//!
//! where `phi_{i,j}` is a learned activation (B-spline or rational) on the
//! edge connecting input node `i` to output node `j`.

use scirs2_core::ndarray::{Array1, Array2};

use crate::NeuralError;

use super::{
    rational::{RationalActivation, RationalConfig},
    spline::{BSplineActivation, SplineConfig},
    KanResult,
};

// ---------------------------------------------------------------------------
// Activation type enum
// ---------------------------------------------------------------------------

/// Selects which type of learnable activation to use on each KAN edge.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum ActivationType {
    /// B-spline activations (default; piecewise polynomial, smooth).
    BSpline(SplineConfig),
    /// Rational activations (Padé-type; globally smooth).
    Rational(RationalConfig),
}

impl Default for ActivationType {
    fn default() -> Self {
        ActivationType::BSpline(SplineConfig::default())
    }
}

// ---------------------------------------------------------------------------
// KanLayerConfig
// ---------------------------------------------------------------------------

/// Configuration for a single KAN layer.
#[derive(Debug, Clone)]
pub struct KanLayerConfig {
    /// Number of input features.
    pub n_in: usize,
    /// Number of output features.
    pub n_out: usize,
    /// Type of learnable activation to place on each edge.
    pub activation_type: ActivationType,
}

impl Default for KanLayerConfig {
    fn default() -> Self {
        Self {
            n_in: 4,
            n_out: 4,
            activation_type: ActivationType::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// KanLayer
// ---------------------------------------------------------------------------

/// A single KAN layer with `n_in × n_out` learnable edge-activations.
///
/// Each activation `phi_{i,j}` acts on input dimension `i` and contributes to
/// output dimension `j`. The layer output is the column-wise sum:
/// `output[j] = sum_i phi_{i,j}(input[i])`.
pub struct KanLayer {
    config: KanLayerConfig,
    /// Flat storage: `spline_activations[i * n_out + j]` = `phi_{i,j}`.
    /// `None` when `ActivationType::Rational` is chosen.
    spline_activations: Option<Vec<BSplineActivation>>,
    /// Flat storage: `rational_activations[i * n_out + j]` = `phi_{i,j}`.
    /// `None` when `ActivationType::BSpline` is chosen.
    rational_activations: Option<Vec<RationalActivation>>,
}

impl KanLayer {
    /// Construct a new `KanLayer` from a [`KanLayerConfig`].
    ///
    /// Returns an error if any activation config is invalid or if `n_in`/`n_out`
    /// are zero.
    pub fn new(config: KanLayerConfig) -> KanResult<Self> {
        if config.n_in == 0 {
            return Err(NeuralError::InvalidArgument(
                "KanLayer: n_in must be > 0".to_string(),
            ));
        }
        if config.n_out == 0 {
            return Err(NeuralError::InvalidArgument(
                "KanLayer: n_out must be > 0".to_string(),
            ));
        }

        let n_edges = config.n_in * config.n_out;
        match &config.activation_type {
            ActivationType::BSpline(sc) => {
                let mut activations = Vec::with_capacity(n_edges);
                for _ in 0..n_edges {
                    activations.push(BSplineActivation::new(sc)?);
                }
                Ok(Self {
                    config,
                    spline_activations: Some(activations),
                    rational_activations: None,
                })
            }
            ActivationType::Rational(rc) => {
                let mut activations = Vec::with_capacity(n_edges);
                for _ in 0..n_edges {
                    activations.push(RationalActivation::new(rc)?);
                }
                Ok(Self {
                    config,
                    spline_activations: None,
                    rational_activations: Some(activations),
                })
            }
        }
    }

    /// Evaluate `phi_{i,j}(x)` for the activation at edge `(i, j)`.
    fn eval_activation(&self, i: usize, j: usize, x: f64) -> f64 {
        let idx = i * self.config.n_out + j;
        match (&self.spline_activations, &self.rational_activations) {
            (Some(splines), _) => splines[idx].evaluate(x),
            (_, Some(rationals)) => rationals[idx].evaluate(x),
            _ => 0.0, // unreachable in practice; both arms always set one
        }
    }

    /// Forward pass: `input` has shape `[n_in]`; returns `[n_out]`.
    pub fn forward(&self, input: &Array1<f64>) -> KanResult<Array1<f64>> {
        if input.len() != self.config.n_in {
            return Err(NeuralError::DimensionMismatch(format!(
                "KanLayer::forward expected n_in={} but got {}",
                self.config.n_in,
                input.len()
            )));
        }
        let mut output = Array1::zeros(self.config.n_out);
        for i in 0..self.config.n_in {
            let x_i = input[i];
            for j in 0..self.config.n_out {
                output[j] += self.eval_activation(i, j, x_i);
            }
        }
        Ok(output)
    }

    /// Batch forward pass: `input` has shape `[batch, n_in]`; returns `[batch, n_out]`.
    pub fn forward_batch(&self, input: &Array2<f64>) -> KanResult<Array2<f64>> {
        let (batch, n_in) = input.dim();
        if n_in != self.config.n_in {
            return Err(NeuralError::DimensionMismatch(format!(
                "KanLayer::forward_batch expected n_in={} but got {n_in}",
                self.config.n_in
            )));
        }
        let mut output = Array2::zeros((batch, self.config.n_out));
        for b in 0..batch {
            for i in 0..self.config.n_in {
                let x_i = input[(b, i)];
                for j in 0..self.config.n_out {
                    output[(b, j)] += self.eval_activation(i, j, x_i);
                }
            }
        }
        Ok(output)
    }

    /// Total number of learnable parameters across all edge activations.
    pub fn n_params(&self) -> usize {
        match (&self.spline_activations, &self.rational_activations) {
            (Some(splines), _) => splines.iter().map(|s| s.n_params()).sum(),
            (_, Some(rationals)) => rationals.iter().map(|r| r.n_params()).sum(),
            _ => 0,
        }
    }

    /// Number of input features.
    pub fn n_in(&self) -> usize {
        self.config.n_in
    }

    /// Number of output features.
    pub fn n_out(&self) -> usize {
        self.config.n_out
    }

    /// Prune low-importance edges by zeroing activations whose L1 norm is below
    /// `threshold`.
    ///
    /// Returns the number of pruned edges.
    pub fn prune_edges(&mut self, threshold: f64) -> usize {
        let mut pruned = 0;
        if let Some(ref mut splines) = self.spline_activations {
            for sp in splines.iter_mut() {
                let l1: f64 = sp.coefficients.iter().map(|c| c.abs()).sum();
                if l1 < threshold {
                    sp.coefficients.fill(0.0);
                    pruned += 1;
                }
            }
        }
        if let Some(ref mut rationals) = self.rational_activations {
            for ra in rationals.iter_mut() {
                let l1: f64 = ra
                    .p_coeffs
                    .iter()
                    .chain(ra.q_coeffs.iter())
                    .map(|c| c.abs())
                    .sum();
                if l1 < threshold {
                    ra.p_coeffs.fill(0.0);
                    ra.q_coeffs.fill(0.0);
                    pruned += 1;
                }
            }
        }
        pruned
    }

    /// Mutable access to spline activations (for parameter updates during training).
    pub fn spline_activations_mut(&mut self) -> Option<&mut Vec<BSplineActivation>> {
        self.spline_activations.as_mut()
    }

    /// Mutable access to rational activations (for parameter updates during training).
    pub fn rational_activations_mut(&mut self) -> Option<&mut Vec<RationalActivation>> {
        self.rational_activations.as_mut()
    }

    /// Read-only reference to the layer config.
    pub fn config(&self) -> &KanLayerConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// KanConfig
// ---------------------------------------------------------------------------

/// Configuration for a full KAN network (stack of KAN layers).
#[derive(Debug, Clone)]
pub struct KanConfig {
    /// Width of each layer including input and output.
    ///
    /// For example, `[2, 8, 4, 1]` means input dim 2, two hidden layers of
    /// widths 8 and 4, and scalar output.
    pub layer_widths: Vec<usize>,
    /// Activation type used for every layer.
    pub activation_type: ActivationType,
}

impl Default for KanConfig {
    fn default() -> Self {
        Self {
            layer_widths: vec![2, 8, 1],
            activation_type: ActivationType::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// KanNetwork
// ---------------------------------------------------------------------------

/// A full KAN network consisting of stacked [`KanLayer`]s.
///
/// # Example
///
/// ```rust,ignore
/// use scirs2_neural::layers::kan::{KanConfig, KanNetwork, ActivationType, SplineConfig};
///
/// let config = KanConfig {
///     layer_widths: vec![2, 8, 1],
///     activation_type: ActivationType::BSpline(SplineConfig::default()),
/// };
/// let net = KanNetwork::new(config).unwrap();
/// ```
pub struct KanNetwork {
    layers: Vec<KanLayer>,
    config: KanConfig,
}

impl KanNetwork {
    /// Build a new `KanNetwork` from a [`KanConfig`].
    ///
    /// Returns an error if `layer_widths` has fewer than 2 entries or if any
    /// per-layer construction fails.
    pub fn new(config: KanConfig) -> KanResult<Self> {
        if config.layer_widths.len() < 2 {
            return Err(NeuralError::InvalidArgument(
                "KanNetwork: layer_widths must have at least 2 entries (input and output)"
                    .to_string(),
            ));
        }
        for (i, &w) in config.layer_widths.iter().enumerate() {
            if w == 0 {
                return Err(NeuralError::InvalidArgument(format!(
                    "KanNetwork: layer_widths[{i}] must be > 0"
                )));
            }
        }

        let mut layers = Vec::with_capacity(config.layer_widths.len() - 1);
        for pair in config.layer_widths.windows(2) {
            let layer_cfg = KanLayerConfig {
                n_in: pair[0],
                n_out: pair[1],
                activation_type: config.activation_type.clone(),
            };
            layers.push(KanLayer::new(layer_cfg)?);
        }

        Ok(Self { layers, config })
    }

    /// Forward pass through all layers: `input` shape `[layer_widths[0]]`;
    /// returns `[layer_widths.last()]`.
    pub fn forward(&self, input: &Array1<f64>) -> KanResult<Array1<f64>> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    /// Batch forward pass: `input` shape `[batch, layer_widths[0]]`;
    /// returns `[batch, layer_widths.last()]`.
    pub fn forward_batch(&self, input: &Array2<f64>) -> KanResult<Array2<f64>> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward_batch(&x)?;
        }
        Ok(x)
    }

    /// Total number of learnable parameters across all layers.
    pub fn n_params(&self) -> usize {
        self.layers.iter().map(|l| l.n_params()).sum()
    }

    /// Number of KAN layers (= `layer_widths.len() - 1`).
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Prune low-importance edges across all layers.
    ///
    /// Returns the total number of pruned edges.
    pub fn prune_edges(&mut self, threshold: f64) -> usize {
        self.layers
            .iter_mut()
            .map(|l| l.prune_edges(threshold))
            .sum()
    }

    /// Read-only access to the underlying layers.
    pub fn layers(&self) -> &[KanLayer] {
        &self.layers
    }

    /// Mutable access to the underlying layers (for training).
    pub fn layers_mut(&mut self) -> &mut Vec<KanLayer> {
        &mut self.layers
    }

    /// Read-only reference to the network config.
    pub fn config(&self) -> &KanConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::kan::rational::RationalConfig;
    use crate::layers::kan::spline::SplineConfig;

    fn default_spline_layer(n_in: usize, n_out: usize) -> KanLayer {
        KanLayer::new(KanLayerConfig {
            n_in,
            n_out,
            activation_type: ActivationType::BSpline(SplineConfig::default()),
        })
        .expect("valid layer config")
    }

    // ------------------------------------------------------------------
    // KanLayer tests
    // ------------------------------------------------------------------

    /// `forward` output has the correct length.
    #[test]
    fn kan_layer_forward_shape() {
        let layer = default_spline_layer(3, 5);
        let input = Array1::zeros(3);
        let output = layer.forward(&input).expect("forward ok");
        assert_eq!(output.len(), 5, "Output length mismatch");
    }

    /// `forward_batch` output has the correct shape.
    #[test]
    fn kan_layer_batch_forward_shape() {
        let layer = default_spline_layer(4, 6);
        let input = Array2::zeros((8, 4));
        let output = layer.forward_batch(&input).expect("batch forward ok");
        assert_eq!(output.dim(), (8, 6), "Batch output shape mismatch");
    }

    /// With zero spline coefficients all outputs are zero.
    #[test]
    fn kan_layer_zero_coeffs_outputs_zero() {
        let layer = default_spline_layer(3, 4);
        let input = Array1::from_vec(vec![0.5, -0.3, 0.9]);
        let output = layer.forward(&input).expect("forward ok");
        for (j, &v) in output.iter().enumerate() {
            assert!(
                v.abs() < 1e-14,
                "Expected 0 at output[{j}] but got {v}"
            );
        }
    }

    /// Mismatched input dimension returns an error.
    #[test]
    fn kan_layer_dimension_mismatch() {
        let layer = default_spline_layer(3, 4);
        let bad_input = Array1::zeros(5);
        assert!(
            layer.forward(&bad_input).is_err(),
            "Should return error on dimension mismatch"
        );
    }

    /// Single-input single-output layer works.
    #[test]
    fn kan_layer_single_input_single_output() {
        let layer = default_spline_layer(1, 1);
        let input = Array1::from_vec(vec![0.1]);
        let output = layer.forward(&input).expect("forward ok");
        assert_eq!(output.len(), 1);
    }

    /// Rational-config layer computes forward correctly.
    #[test]
    fn kan_rational_layer() {
        let config = KanLayerConfig {
            n_in: 3,
            n_out: 2,
            activation_type: ActivationType::Rational(RationalConfig::default()),
        };
        let layer = KanLayer::new(config).expect("valid rational layer");
        let input = Array1::from_vec(vec![0.1, -0.5, 0.8]);
        let output = layer.forward(&input).expect("forward ok");
        assert_eq!(output.len(), 2);
        // Zero coefficients → output is 0
        for (j, &v) in output.iter().enumerate() {
            assert!(v.abs() < 1e-14, "Expected 0 at output[{j}] got {v}");
        }
    }

    /// `prune_edges` returns the count of zeroed edges and output changes.
    #[test]
    fn kan_prune_edges() {
        let mut layer = default_spline_layer(2, 3);
        // All coefficients are zero => L1 = 0 < 1.0 => all edges pruned
        let pruned = layer.prune_edges(1.0);
        assert_eq!(pruned, 6, "All 2×3=6 edges should be pruned");

        // Now set some coefficients to be large enough to survive pruning
        if let Some(splines) = layer.spline_activations_mut() {
            splines[0].coefficients[0] = 10.0;
        }
        let pruned2 = layer.prune_edges(1.0);
        // Edge 0 has L1 = 10 >= 1, so it survives; the rest (5) are pruned
        assert_eq!(pruned2, 5);
    }

    // ------------------------------------------------------------------
    // KanNetwork tests
    // ------------------------------------------------------------------

    fn default_network() -> KanNetwork {
        KanNetwork::new(KanConfig {
            layer_widths: vec![2, 4, 3, 1],
            activation_type: ActivationType::BSpline(SplineConfig::default()),
        })
        .expect("valid network config")
    }

    /// `forward` output has the correct shape for a multi-layer network.
    #[test]
    fn kan_network_forward() {
        let net = default_network();
        let input = Array1::from_vec(vec![0.3, -0.7]);
        let output = net.forward(&input).expect("network forward ok");
        assert_eq!(output.len(), 1, "Output should be scalar (width=1)");
    }

    /// `forward_batch` output has the correct shape.
    #[test]
    fn kan_network_batch_forward() {
        let net = default_network();
        let input = Array2::zeros((5, 2));
        let output = net.forward_batch(&input).expect("batch forward ok");
        assert_eq!(output.dim(), (5, 1), "Batch output shape mismatch");
    }

    /// `n_params` is non-zero and proportional to the total number of activations.
    #[test]
    fn kan_network_n_params() {
        let net = default_network();
        let n = net.n_params();
        // widths [2,4,3,1]: (2*4 + 4*3 + 3*1) * 8 basis = (8 + 12 + 3) * 8 = 184
        assert_eq!(n, 184, "n_params mismatch: got {n}");
    }

    /// Config with fewer than 2 widths returns an error.
    #[test]
    fn kan_invalid_config_single_width() {
        let result = KanNetwork::new(KanConfig {
            layer_widths: vec![4],
            activation_type: ActivationType::default(),
        });
        assert!(result.is_err(), "Should fail with single-element layer_widths");
    }

    /// `n_layers` returns the correct number of layers.
    #[test]
    fn kan_network_n_layers() {
        let net = default_network();
        // widths [2,4,3,1] → 3 layers
        assert_eq!(net.n_layers(), 3);
    }

    /// Full prune with threshold > 0 zeroes all edges (all coefficients start at 0).
    #[test]
    fn kan_network_prune_all_zero() {
        let mut net = default_network();
        let pruned = net.prune_edges(1.0);
        // (2*4 + 4*3 + 3*1) = 23 edges, all with L1=0 < 1.0 → all pruned
        assert_eq!(pruned, 23, "All 23 edges should be pruned; got {pruned}");
    }
}
