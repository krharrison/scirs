//! Model analysis utilities: parameter counting, FLOPs estimation,
//! memory footprint, and compression ratio computation.

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::Array2;

// ─────────────────────────────────────────────────────────────────────────────
// Parameter counting
// ─────────────────────────────────────────────────────────────────────────────

/// Count total scalar parameters across a collection of weight matrices.
///
/// Each `Array2<f32>` represents one weight tensor (bias vectors can be
/// wrapped in `Array2` with one row).
pub fn count_parameters(weights: &[Array2<f32>]) -> usize {
    weights.iter().map(|w| w.len()).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// FLOPs estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Multiply-accumulate (MAC) count for a dense (fully-connected) layer.
///
/// Each output element requires `input_size` multiplications and
/// `input_size - 1` additions ≈ `2 * input_size` FLOPs.
///
/// Returns `2 * input_size * output_size`.
pub fn count_flops_dense(input_size: usize, output_size: usize) -> usize {
    2 * input_size * output_size
}

/// Multiply-accumulate count for a 2-D convolutional layer.
///
/// For each output spatial location (out_h × out_w) and each output channel,
/// the inner product over `(in_channels, kernel_h, kernel_w)` is computed:
///
/// `FLOPs = 2 * out_h * out_w * in_channels * out_channels * kernel_h * kernel_w`
///
/// Output spatial dimensions assume *same* padding (out = input):
/// `out_h = input_h`,  `out_w = input_w`.
pub fn count_flops_conv(
    input_h: usize,
    input_w: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
) -> usize {
    2 * input_h * input_w * in_channels * out_channels * kernel_h * kernel_w
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory footprint
// ─────────────────────────────────────────────────────────────────────────────

/// Total memory in bytes for `f32` weight matrices (`4` bytes per element).
pub fn memory_footprint_bytes(weights: &[Array2<f32>]) -> usize {
    count_parameters(weights) * std::mem::size_of::<f32>()
}

// ─────────────────────────────────────────────────────────────────────────────
// Compression ratio
// ─────────────────────────────────────────────────────────────────────────────

/// Ratio of original parameter count to compressed parameter count.
///
/// A ratio of `2.0` means the compressed model is half the size of the original.
///
/// # Errors
/// Returns an error if either slice is empty or if the compressed count is zero.
pub fn compression_ratio(original: &[Array2<f32>], compressed: &[Array2<f32>]) -> Result<f64> {
    let n_orig = count_parameters(original);
    let n_comp = count_parameters(compressed);
    if n_orig == 0 {
        return Err(NeuralError::InvalidArchitecture(
            "compression_ratio: original model has 0 parameters".into(),
        ));
    }
    if n_comp == 0 {
        return Err(NeuralError::InvalidArchitecture(
            "compression_ratio: compressed model has 0 parameters".into(),
        ));
    }
    Ok(n_orig as f64 / n_comp as f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// Model summary
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics for a collection of weight matrices.
#[derive(Debug, Clone)]
pub struct ModelSummary {
    /// Total number of scalar parameters.
    pub total_parameters: usize,
    /// Fraction of parameters that are (near-)zero.
    pub sparsity: f64,
    /// Memory footprint in bytes (f32 representation).
    pub memory_bytes: usize,
    /// Number of weight matrices.
    pub num_layers: usize,
    /// Per-layer parameter counts.
    pub layer_params: Vec<usize>,
}

impl ModelSummary {
    /// Build a summary for a slice of weight matrices.
    pub fn from_weights(weights: &[Array2<f32>]) -> Self {
        let layer_params: Vec<usize> = weights.iter().map(|w| w.len()).collect();
        let total_parameters: usize = layer_params.iter().sum();
        let total_zeros: usize = weights
            .iter()
            .flat_map(|w| w.iter())
            .filter(|&&v| v.abs() < 1e-8)
            .count();
        let sparsity = if total_parameters == 0 {
            0.0
        } else {
            total_zeros as f64 / total_parameters as f64
        };
        Self {
            total_parameters,
            sparsity,
            memory_bytes: total_parameters * std::mem::size_of::<f32>(),
            num_layers: weights.len(),
            layer_params,
        }
    }
}

/// Display a human-readable summary (returns a formatted string).
pub fn format_summary(summary: &ModelSummary) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "Model Summary\n  Layers     : {}\n  Parameters : {}\n  Sparsity   : {:.2}%\n  Memory     : {} KB\n",
        summary.num_layers,
        summary.total_parameters,
        summary.sparsity * 100.0,
        summary.memory_bytes / 1024,
    ));
    for (i, &p) in summary.layer_params.iter().enumerate() {
        s.push_str(&format!("  Layer {:3}: {:8} params\n", i, p));
    }
    s
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_layer(nrows: usize, ncols: usize) -> Array2<f32> {
        Array2::from_elem((nrows, ncols), 1.0_f32)
    }

    #[test]
    fn test_count_parameters() {
        let w = vec![make_layer(3, 4), make_layer(4, 5)];
        assert_eq!(count_parameters(&w), 3 * 4 + 4 * 5);
    }

    #[test]
    fn test_count_parameters_empty() {
        assert_eq!(count_parameters(&[]), 0);
    }

    #[test]
    fn test_count_flops_dense() {
        // 2 * 784 * 256
        assert_eq!(count_flops_dense(784, 256), 401_408);
    }

    #[test]
    fn test_count_flops_dense_single_neuron() {
        assert_eq!(count_flops_dense(10, 1), 20);
    }

    #[test]
    fn test_count_flops_conv() {
        // 2 * 28 * 28 * 1 * 32 * 3 * 3
        let expected = 2 * 28 * 28 * 1 * 32 * 3 * 3;
        assert_eq!(count_flops_conv(28, 28, 1, 32, 3, 3), expected);
    }

    #[test]
    fn test_memory_footprint_bytes() {
        let w = vec![make_layer(10, 10)];
        // 100 f32 params × 4 bytes = 400
        assert_eq!(memory_footprint_bytes(&w), 400);
    }

    #[test]
    fn test_compression_ratio() {
        let original = vec![make_layer(100, 100)];
        let compressed = vec![make_layer(50, 50)];
        let ratio = compression_ratio(&original, &compressed).expect("ratio failed");
        assert!((ratio - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_compression_ratio_equal() {
        let w = vec![make_layer(10, 10)];
        let ratio = compression_ratio(&w, &w).expect("ratio failed");
        assert!((ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_compression_ratio_empty_original() {
        let result = compression_ratio(&[], &[make_layer(1, 1)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_compression_ratio_empty_compressed() {
        let result = compression_ratio(&[make_layer(1, 1)], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_model_summary() {
        let w = vec![make_layer(10, 5), make_layer(5, 2)];
        let summary = ModelSummary::from_weights(&w);
        assert_eq!(summary.total_parameters, 60);
        assert_eq!(summary.num_layers, 2);
        assert_eq!(summary.memory_bytes, 60 * 4);
        assert!((summary.sparsity - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_model_summary_sparse() {
        // Half the weights are zero.
        let flat: Vec<f32> = (0..20).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
        let w = Array2::from_shape_vec((4, 5), flat).expect("shape");
        let summary = ModelSummary::from_weights(&[w]);
        assert!((summary.sparsity - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_format_summary_contains_key_info() {
        let w = vec![make_layer(8, 8)];
        let summary = ModelSummary::from_weights(&w);
        let text = format_summary(&summary);
        assert!(text.contains("64"), "should mention param count 64");
        assert!(text.contains("Layer"), "should mention layers");
    }
}
