//! Weight initialization strategies for neural networks
//!
//! Provides a comprehensive set of weight initialization strategies following
//! best practices from the deep learning literature:
//!
//! - **Zeros / Ones / Constant**: Simple constant initializations
//! - **Uniform / Normal**: Random initialization from uniform or normal distributions
//! - **Xavier/Glorot** (uniform and normal): For sigmoid/tanh activations (Glorot & Bengio, 2010)
//! - **Kaiming/He** (uniform and normal): For ReLU-family activations (He et al., 2015)
//! - **LeCun**: For SELU activations (Klambauer et al., 2017)
//! - **Orthogonal**: QR-decomposition-based initialization (Saxe et al., 2014)
//! - **Sparse**: Random sparse initialization
//!
//! # Example
//!
//! ```rust
//! use scirs2_neural::utils::initializers::{Initializer, init_weights, compute_fan};
//! use scirs2_core::ndarray::{Array, IxDyn};
//!
//! let shape = IxDyn(&[256, 128]);
//! let (fan_in, fan_out) = compute_fan(&shape);
//! let mut rng = scirs2_core::random::rng();
//! let weights: Array<f64, IxDyn> = Initializer::XavierNormal
//!     .initialize(shape, fan_in, fan_out, &mut rng)
//!     .expect("initialization failed");
//! assert_eq!(weights.shape(), &[256, 128]);
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array, Dimension, IxDyn};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::Rng;
use std::fmt::Debug;

// ============================================================================
// Initializer enum
// ============================================================================

/// Initialization strategies for neural network weights
#[derive(Debug, Clone, Copy)]
pub enum Initializer {
    /// Zero initialization
    Zeros,
    /// One initialization
    Ones,
    /// Constant initialization
    Constant {
        /// The constant value
        value: f64,
    },
    /// Uniform random initialization
    Uniform {
        /// Minimum value
        min: f64,
        /// Maximum value
        max: f64,
    },
    /// Normal random initialization
    Normal {
        /// Mean
        mean: f64,
        /// Standard deviation
        std: f64,
    },
    /// Xavier/Glorot uniform initialization
    ///
    /// Samples from U(-limit, limit) where limit = sqrt(6 / (fan_in + fan_out))
    Xavier,
    /// Xavier/Glorot normal initialization
    ///
    /// Samples from N(0, std) where std = sqrt(2 / (fan_in + fan_out))
    XavierNormal,
    /// Kaiming/He normal initialization (for ReLU)
    ///
    /// Samples from N(0, std) where std = sqrt(2 / fan_in)
    He,
    /// Kaiming/He uniform initialization (for ReLU)
    ///
    /// Samples from U(-limit, limit) where limit = sqrt(6 / fan_in)
    HeUniform,
    /// LeCun normal initialization (for SELU)
    ///
    /// Samples from N(0, std) where std = sqrt(1 / fan_in)
    LeCun,
    /// LeCun uniform initialization (for SELU)
    ///
    /// Samples from U(-limit, limit) where limit = sqrt(3 / fan_in)
    LeCunUniform,
    /// Orthogonal initialization (Saxe et al., 2014)
    ///
    /// Generates an orthogonal or semi-orthogonal matrix via QR decomposition,
    /// then scales by the given gain.
    Orthogonal {
        /// Multiplicative gain factor
        gain: f64,
    },
    /// Sparse initialization: each column has a fraction of zeros
    Sparse {
        /// Fraction of elements set to zero in each column (0.0 to 1.0)
        sparsity: f64,
        /// Standard deviation for the non-zero elements
        std: f64,
    },
}

impl Initializer {
    /// Initialize weights according to the strategy
    ///
    /// # Arguments
    /// * `shape` - Shape of the weights array
    /// * `fan_in` - Number of input connections (for Xavier, He, LeCun)
    /// * `fan_out` - Number of output connections (for Xavier)
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// * Initialized weights array
    pub fn initialize<F: Float + Debug, R: Rng>(
        &self,
        shape: IxDyn,
        fan_in: usize,
        fan_out: usize,
        rng: &mut R,
    ) -> Result<Array<F, IxDyn>> {
        let size: usize = shape.as_array_view().iter().product();
        match self {
            Initializer::Zeros => Ok(Array::zeros(shape)),

            Initializer::Ones => {
                let ones: Vec<F> = (0..size).map(|_| F::one()).collect();
                make_array(shape, ones)
            }

            Initializer::Constant { value } => {
                let c = F::from(*value).ok_or_else(|| {
                    NeuralError::InvalidArchitecture("Failed to convert constant value".to_string())
                })?;
                let vals: Vec<F> = (0..size).map(|_| c).collect();
                make_array(shape, vals)
            }

            Initializer::Uniform { min, max } => {
                let values: Vec<F> = (0..size)
                    .map(|_| {
                        let val = rng.random_range(*min..*max);
                        F::from(val).ok_or_else(|| {
                            NeuralError::InvalidArchitecture(
                                "Failed to convert random value".to_string(),
                            )
                        })
                    })
                    .collect::<Result<Vec<F>>>()?;
                make_array(shape, values)
            }

            Initializer::Normal { mean, std } => {
                let values = generate_normal(size, *mean, *std, rng)?;
                make_array(shape, values)
            }

            Initializer::Xavier => {
                // Xavier uniform: U(-limit, limit) where limit = sqrt(6 / (fan_in + fan_out))
                let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
                let values = generate_uniform(size, -limit, limit, rng)?;
                make_array(shape, values)
            }

            Initializer::XavierNormal => {
                // Xavier normal: N(0, std) where std = sqrt(2 / (fan_in + fan_out))
                let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
                let values = generate_normal(size, 0.0, std, rng)?;
                make_array(shape, values)
            }

            Initializer::He => {
                // He/Kaiming normal: N(0, std) where std = sqrt(2 / fan_in)
                let std = (2.0 / fan_in.max(1) as f64).sqrt();
                let values = generate_normal(size, 0.0, std, rng)?;
                make_array(shape, values)
            }

            Initializer::HeUniform => {
                // He/Kaiming uniform: U(-limit, limit) where limit = sqrt(6 / fan_in)
                let limit = (6.0 / fan_in.max(1) as f64).sqrt();
                let values = generate_uniform(size, -limit, limit, rng)?;
                make_array(shape, values)
            }

            Initializer::LeCun => {
                // LeCun normal: N(0, std) where std = sqrt(1 / fan_in)
                let std = (1.0 / fan_in.max(1) as f64).sqrt();
                let values = generate_normal(size, 0.0, std, rng)?;
                make_array(shape, values)
            }

            Initializer::LeCunUniform => {
                // LeCun uniform: U(-limit, limit) where limit = sqrt(3 / fan_in)
                let limit = (3.0 / fan_in.max(1) as f64).sqrt();
                let values = generate_uniform(size, -limit, limit, rng)?;
                make_array(shape, values)
            }

            Initializer::Orthogonal { gain } => orthogonal_init(shape, *gain, rng),

            Initializer::Sparse { sparsity, std } => {
                sparse_init(shape, *sparsity, *std, fan_in, rng)
            }
        }
    }
}

// ============================================================================
// Convenience functions
// ============================================================================

/// Compute fan_in and fan_out from a weight tensor shape.
///
/// Follows PyTorch conventions:
/// - For 1D tensors (bias): fan_in = fan_out = shape[0]
/// - For 2D tensors (linear): fan_in = shape[0], fan_out = shape[1]
/// - For 3D+ tensors (conv): fan_in = shape[0] * prod(kernel_dims), fan_out = shape[1] * prod(kernel_dims)
pub fn compute_fan(shape: &IxDyn) -> (usize, usize) {
    match shape.ndim() {
        0 => (1, 1),
        1 => (shape[0], shape[0]),
        2 => (shape[0], shape[1]),
        _ => {
            // For conv layers: shape is typically [out_channels, in_channels, *kernel_size]
            let receptive_field: usize = shape.as_array_view().iter().skip(2).product();
            let fan_in = shape[1] * receptive_field.max(1);
            let fan_out = shape[0] * receptive_field.max(1);
            (fan_in, fan_out)
        }
    }
}

/// Initialize a weight tensor using the given strategy, automatically computing fan_in/fan_out.
///
/// # Arguments
/// * `strategy` - The initialization strategy to use
/// * `shape` - Shape of the tensor to initialize
///
/// # Returns
/// * Initialized weights array
pub fn init_weights<F: Float + Debug>(
    strategy: Initializer,
    shape: IxDyn,
) -> Result<Array<F, IxDyn>> {
    let (fan_in, fan_out) = compute_fan(&shape);
    let mut rng = scirs2_core::random::rng();
    strategy.initialize(shape, fan_in, fan_out, &mut rng)
}

/// Xavier/Glorot uniform initialization (convenience function)
///
/// # Arguments
/// * `shape` - Shape of the weights array
///
/// # Returns
/// * Initialized weights array
pub fn xavier_uniform<F: Float + Debug + NumAssign>(shape: IxDyn) -> Result<Array<F, IxDyn>> {
    init_weights(Initializer::Xavier, shape)
}

/// Xavier/Glorot normal initialization (convenience function)
pub fn xavier_normal<F: Float + Debug + NumAssign>(shape: IxDyn) -> Result<Array<F, IxDyn>> {
    init_weights(Initializer::XavierNormal, shape)
}

/// Kaiming/He normal initialization (convenience function)
pub fn kaiming_normal<F: Float + Debug + NumAssign>(shape: IxDyn) -> Result<Array<F, IxDyn>> {
    init_weights(Initializer::He, shape)
}

/// Kaiming/He uniform initialization (convenience function)
pub fn kaiming_uniform<F: Float + Debug + NumAssign>(shape: IxDyn) -> Result<Array<F, IxDyn>> {
    init_weights(Initializer::HeUniform, shape)
}

/// Orthogonal initialization with gain=1.0 (convenience function)
pub fn orthogonal<F: Float + Debug + NumAssign>(shape: IxDyn) -> Result<Array<F, IxDyn>> {
    init_weights(Initializer::Orthogonal { gain: 1.0 }, shape)
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Wrap a Vec into an Array with the given shape.
fn make_array<F: Float + Debug>(shape: IxDyn, values: Vec<F>) -> Result<Array<F, IxDyn>> {
    Array::from_shape_vec(shape, values)
        .map_err(|e| NeuralError::InvalidArchitecture(format!("Failed to create array: {e}")))
}

/// Generate `n` samples from N(mean, std) using the Box-Muller transform.
fn generate_normal<F: Float + Debug, R: Rng>(
    n: usize,
    mean: f64,
    std: f64,
    rng: &mut R,
) -> Result<Vec<F>> {
    let values: Vec<F> = (0..((n / 2) + 1))
        .flat_map(|_| {
            let u1: f64 = rng.random_range(1e-10..1.0); // avoid ln(0)
            let u2: f64 = rng.random_range(0.0..1.0);
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            let z0 = mean + std * r * theta.cos();
            let z1 = mean + std * r * theta.sin();
            vec![
                F::from(z0).unwrap_or_else(|| F::zero()),
                F::from(z1).unwrap_or_else(|| F::zero()),
            ]
        })
        .take(n)
        .collect();
    Ok(values)
}

/// Generate `n` samples from U(low, high).
fn generate_uniform<F: Float + Debug, R: Rng>(
    n: usize,
    low: f64,
    high: f64,
    rng: &mut R,
) -> Result<Vec<F>> {
    let values: Vec<F> = (0..n)
        .map(|_| {
            let val = rng.random_range(low..high);
            F::from(val).unwrap_or_else(|| F::zero())
        })
        .collect();
    Ok(values)
}

/// Orthogonal initialization via a pure-Rust QR decomposition (Gram-Schmidt).
///
/// For a (rows x cols) matrix with rows >= cols, this produces a semi-orthogonal matrix
/// where columns are orthonormal. For rows < cols, we transpose, orthogonalize, and
/// transpose back.
///
/// This follows the approach of Saxe et al. (2014) "Exact solutions to the nonlinear
/// dynamics of learning in deep linear neural networks".
fn orthogonal_init<F: Float + Debug, R: Rng>(
    shape: IxDyn,
    gain: f64,
    rng: &mut R,
) -> Result<Array<F, IxDyn>> {
    let ndim = shape.ndim();
    if ndim < 2 {
        return Err(NeuralError::InvalidArchitecture(
            "Orthogonal initialization requires at least 2D shape".to_string(),
        ));
    }

    // Flatten to 2D: (first_dim, product_of_rest)
    let rows = shape[0];
    let cols: usize = shape.as_array_view().iter().skip(1).product();

    // We need the larger dimension for the initial random matrix
    let (m, n) = if rows >= cols {
        (rows, cols)
    } else {
        (cols, rows)
    };

    // Generate random matrix
    let random_vals = generate_normal::<f64, R>(m * n, 0.0, 1.0, rng)?;
    let mut a: Vec<Vec<f64>> = Vec::with_capacity(m);
    for i in 0..m {
        a.push(random_vals[i * n..(i + 1) * n].to_vec());
    }

    // Modified Gram-Schmidt QR decomposition to get Q
    // We compute Q (m x n) with orthonormal columns
    let q = gram_schmidt_qr(&a, m, n)?;

    // If rows < cols, we transposed the problem, so we need to re-arrange
    let result_2d = if rows >= cols {
        // Q is (rows x cols), take first `rows` rows
        let mut flat = Vec::with_capacity(rows * cols);
        for row in q.iter().take(rows) {
            for &val in row.iter().take(cols) {
                flat.push(val * gain);
            }
        }
        flat
    } else {
        // Q is (cols x rows), we need its transpose: (rows x cols)
        // Each col_vec has `rows` elements; we collect row r across all cols.
        let mut flat = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for col_vec in q.iter().take(cols) {
                flat.push(col_vec[r] * gain);
            }
        }
        flat
    };

    // Convert to F
    let typed_vals: Vec<F> = result_2d
        .iter()
        .map(|&v| F::from(v).unwrap_or_else(|| F::zero()))
        .collect();

    make_array(shape, typed_vals)
}

/// Modified Gram-Schmidt QR decomposition.
///
/// Given an m x n matrix A (m >= n), returns Q (m x n) with orthonormal columns.
fn gram_schmidt_qr(a: &[Vec<f64>], m: usize, n: usize) -> Result<Vec<Vec<f64>>> {
    // Work column-major for Gram-Schmidt
    let mut columns: Vec<Vec<f64>> = (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect();

    for j in 0..n {
        // Normalize column j
        let norm = vector_norm(&columns[j]);
        if norm < 1e-15 {
            // Near-zero column: replace with a random unit vector component
            // This handles degenerate cases gracefully
            for elem in columns[j].iter_mut() {
                *elem = 0.0;
            }
            if j < m {
                columns[j][j] = 1.0;
            }
        } else {
            for elem in columns[j].iter_mut() {
                *elem /= norm;
            }
        }

        // Project out column j from all subsequent columns
        for k in (j + 1)..n {
            let dot = dot_product(&columns[j], &columns[k]);
            let col_j = columns[j].clone();
            for (elem_k, elem_j) in columns[k].iter_mut().zip(col_j.iter()) {
                *elem_k -= dot * elem_j;
            }
        }
    }

    // Convert back to row-major Q
    let mut q: Vec<Vec<f64>> = (0..m).map(|_| vec![0.0; n]).collect();
    for j in 0..n {
        for (i, row) in q.iter_mut().enumerate().take(m) {
            row[j] = columns[j][i];
        }
    }

    Ok(q)
}

fn vector_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Sparse initialization: each column has `sparsity` fraction of its elements set to zero.
fn sparse_init<F: Float + Debug, R: Rng>(
    shape: IxDyn,
    sparsity: f64,
    std: f64,
    _fan_in: usize,
    rng: &mut R,
) -> Result<Array<F, IxDyn>> {
    let ndim = shape.ndim();
    if ndim < 2 {
        return Err(NeuralError::InvalidArchitecture(
            "Sparse initialization requires at least 2D shape".to_string(),
        ));
    }

    let rows = shape[0];
    let cols: usize = shape.as_array_view().iter().skip(1).product();
    let total = rows * cols;

    // Generate all normal values first
    let normal_vals = generate_normal::<F, R>(total, 0.0, std, rng)?;
    let mut values = normal_vals;
    // Ensure we have exactly the right count
    values.truncate(total);
    while values.len() < total {
        values.push(F::zero());
    }

    // For each column, zero out a fraction of elements
    let num_zeros_per_col = ((rows as f64) * sparsity.clamp(0.0, 1.0)).round() as usize;
    for col in 0..cols {
        // Generate a random permutation of row indices, then zero out the first num_zeros
        let mut indices: Vec<usize> = (0..rows).collect();
        // Fisher-Yates shuffle
        for i in (1..rows).rev() {
            let j = rng.random_range(0..=(i as u64)) as usize;
            indices.swap(i, j);
        }
        for &row_idx in indices.iter().take(num_zeros_per_col) {
            values[row_idx * cols + col] = F::zero();
        }
    }

    make_array(shape, values)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::IxDyn;

    fn rng() -> impl Rng {
        scirs2_core::random::rng()
    }

    #[test]
    fn test_zeros_initialization() {
        let shape = IxDyn(&[3, 4]);
        let arr: Array<f64, IxDyn> = Initializer::Zeros
            .initialize(shape, 3, 4, &mut rng())
            .expect("zeros should work");
        assert!(arr.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_ones_initialization() {
        let shape = IxDyn(&[2, 3]);
        let arr: Array<f64, IxDyn> = Initializer::Ones
            .initialize(shape, 2, 3, &mut rng())
            .expect("ones should work");
        assert!(arr.iter().all(|&v| (v - 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_constant_initialization() {
        let shape = IxDyn(&[5]);
        let arr: Array<f64, IxDyn> = Initializer::Constant { value: 42.0 }
            .initialize(shape, 5, 5, &mut rng())
            .expect("constant should work");
        assert!(arr.iter().all(|&v| (v - 42.0).abs() < 1e-10));
    }

    #[test]
    fn test_xavier_uniform_range() {
        let shape = IxDyn(&[100, 200]);
        let (fan_in, fan_out) = compute_fan(&shape);
        let arr: Array<f64, IxDyn> = Initializer::Xavier
            .initialize(shape, fan_in, fan_out, &mut rng())
            .expect("xavier should work");

        let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
        for &v in arr.iter() {
            assert!(v >= -limit && v <= limit, "value {} out of range", v);
        }
    }

    #[test]
    fn test_xavier_normal_distribution() {
        let shape = IxDyn(&[1000, 500]);
        let (fan_in, fan_out) = compute_fan(&shape);
        let arr: Array<f64, IxDyn> = Initializer::XavierNormal
            .initialize(shape, fan_in, fan_out, &mut rng())
            .expect("xavier_normal should work");

        let expected_std = (2.0 / (fan_in + fan_out) as f64).sqrt();
        let mean: f64 = arr.iter().sum::<f64>() / arr.len() as f64;
        let var: f64 =
            arr.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (arr.len() as f64 - 1.0);
        let actual_std = var.sqrt();

        // Mean should be close to 0
        assert!(mean.abs() < 0.05, "Mean {} should be close to 0", mean);
        // Std should be close to expected
        assert!(
            (actual_std - expected_std).abs() / expected_std < 0.15,
            "Std {} should be close to {}",
            actual_std,
            expected_std
        );
    }

    #[test]
    fn test_he_normal_distribution() {
        let shape = IxDyn(&[1000, 500]);
        let (fan_in, fan_out) = compute_fan(&shape);
        let arr: Array<f64, IxDyn> = Initializer::He
            .initialize(shape, fan_in, fan_out, &mut rng())
            .expect("he should work");

        let expected_std = (2.0 / fan_in as f64).sqrt();
        let mean: f64 = arr.iter().sum::<f64>() / arr.len() as f64;
        let var: f64 =
            arr.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (arr.len() as f64 - 1.0);
        let actual_std = var.sqrt();

        assert!(mean.abs() < 0.05, "Mean {} should be close to 0", mean);
        assert!(
            (actual_std - expected_std).abs() / expected_std < 0.15,
            "Std {} should be close to {}",
            actual_std,
            expected_std
        );
    }

    #[test]
    fn test_he_uniform_range() {
        let shape = IxDyn(&[100, 200]);
        let (fan_in, fan_out) = compute_fan(&shape);
        let arr: Array<f64, IxDyn> = Initializer::HeUniform
            .initialize(shape, fan_in, fan_out, &mut rng())
            .expect("he_uniform should work");

        let limit = (6.0 / fan_in as f64).sqrt();
        for &v in arr.iter() {
            assert!(v >= -limit && v <= limit, "value {} out of range", v);
        }
    }

    #[test]
    fn test_lecun_normal_distribution() {
        let shape = IxDyn(&[1000, 500]);
        let (fan_in, fan_out) = compute_fan(&shape);
        let arr: Array<f64, IxDyn> = Initializer::LeCun
            .initialize(shape, fan_in, fan_out, &mut rng())
            .expect("lecun should work");

        let expected_std = (1.0 / fan_in as f64).sqrt();
        let mean: f64 = arr.iter().sum::<f64>() / arr.len() as f64;
        let var: f64 =
            arr.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (arr.len() as f64 - 1.0);
        let actual_std = var.sqrt();

        assert!(mean.abs() < 0.05, "Mean {} should be close to 0", mean);
        assert!(
            (actual_std - expected_std).abs() / expected_std < 0.15,
            "Std {} should be close to {}",
            actual_std,
            expected_std
        );
    }

    #[test]
    fn test_lecun_uniform_range() {
        let shape = IxDyn(&[100, 200]);
        let (fan_in, fan_out) = compute_fan(&shape);
        let arr: Array<f64, IxDyn> = Initializer::LeCunUniform
            .initialize(shape, fan_in, fan_out, &mut rng())
            .expect("lecun_uniform should work");

        let limit = (3.0 / fan_in as f64).sqrt();
        for &v in arr.iter() {
            assert!(v >= -limit && v <= limit, "value {} out of range", v);
        }
    }

    #[test]
    fn test_orthogonal_columns_are_orthonormal() {
        let shape = IxDyn(&[8, 4]);
        let arr: Array<f64, IxDyn> = Initializer::Orthogonal { gain: 1.0 }
            .initialize(shape.clone(), 8, 4, &mut rng())
            .expect("orthogonal should work");

        let rows = shape[0];
        let cols = shape[1];

        // Check that columns are approximately orthonormal
        for j1 in 0..cols {
            for j2 in j1..cols {
                let mut dot = 0.0;
                for i in 0..rows {
                    dot += arr[[i, j1]] * arr[[i, j2]];
                }
                if j1 == j2 {
                    assert!(
                        (dot - 1.0).abs() < 0.1,
                        "Column {} norm should be ~1.0, got {}",
                        j1,
                        dot
                    );
                } else {
                    assert!(
                        dot.abs() < 0.1,
                        "Columns {} and {} should be orthogonal, dot = {}",
                        j1,
                        j2,
                        dot
                    );
                }
            }
        }
    }

    #[test]
    fn test_orthogonal_with_gain() {
        let shape = IxDyn(&[4, 4]);
        let arr: Array<f64, IxDyn> = Initializer::Orthogonal { gain: 2.0 }
            .initialize(shape.clone(), 4, 4, &mut rng())
            .expect("orthogonal should work");

        // Column norms should be approximately `gain` = 2.0
        for j in 0..4 {
            let col_norm: f64 = (0..4).map(|i| arr[[i, j]].powi(2)).sum::<f64>().sqrt();
            assert!(
                (col_norm - 2.0).abs() < 0.2,
                "Column {} norm should be ~2.0, got {}",
                j,
                col_norm
            );
        }
    }

    #[test]
    fn test_orthogonal_wide_matrix() {
        // rows < cols case
        let shape = IxDyn(&[3, 8]);
        let arr: Array<f64, IxDyn> = Initializer::Orthogonal { gain: 1.0 }
            .initialize(shape.clone(), 3, 8, &mut rng())
            .expect("orthogonal wide should work");

        // Check rows are approximately orthonormal (for wide matrix, rows should be orthonormal)
        let rows = shape[0];
        let cols = shape[1];
        for r1 in 0..rows {
            for r2 in r1..rows {
                let mut dot = 0.0;
                for c in 0..cols {
                    dot += arr[[r1, c]] * arr[[r2, c]];
                }
                if r1 == r2 {
                    assert!(
                        (dot - 1.0).abs() < 0.2,
                        "Row {} norm should be ~1.0, got {}",
                        r1,
                        dot
                    );
                } else {
                    assert!(
                        dot.abs() < 0.2,
                        "Rows {} and {} should be orthogonal, dot = {}",
                        r1,
                        r2,
                        dot
                    );
                }
            }
        }
    }

    #[test]
    fn test_orthogonal_rejects_1d() {
        let shape = IxDyn(&[10]);
        let result: Result<Array<f64, IxDyn>> =
            Initializer::Orthogonal { gain: 1.0 }.initialize(shape, 10, 10, &mut rng());
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_initialization() {
        let shape = IxDyn(&[100, 50]);
        let arr: Array<f64, IxDyn> = Initializer::Sparse {
            sparsity: 0.5,
            std: 0.01,
        }
        .initialize(shape.clone(), 100, 50, &mut rng())
        .expect("sparse should work");

        // Count zeros
        let total = arr.len();
        let zeros = arr.iter().filter(|&&v| v == 0.0).count();
        let sparsity_ratio = zeros as f64 / total as f64;
        // Should be roughly 50% zeros (allowing statistical variation)
        assert!(
            (sparsity_ratio - 0.5).abs() < 0.15,
            "Sparsity ratio {} should be close to 0.5",
            sparsity_ratio
        );
    }

    #[test]
    fn test_compute_fan_2d() {
        let (fi, fo) = compute_fan(&IxDyn(&[128, 64]));
        assert_eq!(fi, 128);
        assert_eq!(fo, 64);
    }

    #[test]
    fn test_compute_fan_4d_conv() {
        // Shape: [out_channels=32, in_channels=16, kernel_h=3, kernel_w=3]
        let (fi, fo) = compute_fan(&IxDyn(&[32, 16, 3, 3]));
        assert_eq!(fi, 16 * 9); // in_channels * kernel_size
        assert_eq!(fo, 32 * 9); // out_channels * kernel_size
    }

    #[test]
    fn test_compute_fan_1d() {
        let (fi, fo) = compute_fan(&IxDyn(&[256]));
        assert_eq!(fi, 256);
        assert_eq!(fo, 256);
    }

    #[test]
    fn test_init_weights_convenience() {
        let shape = IxDyn(&[64, 32]);
        let arr: Array<f64, IxDyn> =
            init_weights(Initializer::He, shape).expect("init_weights should work");
        assert_eq!(arr.shape(), &[64, 32]);
    }

    #[test]
    fn test_convenience_functions() {
        let shape = IxDyn(&[32, 16]);
        let _ = xavier_uniform::<f64>(shape.clone()).expect("xavier_uniform should work");
        let _ = xavier_normal::<f64>(shape.clone()).expect("xavier_normal should work");
        let _ = kaiming_normal::<f64>(shape.clone()).expect("kaiming_normal should work");
        let _ = kaiming_uniform::<f64>(shape.clone()).expect("kaiming_uniform should work");
        let _ = orthogonal::<f64>(shape).expect("orthogonal should work");
    }

    #[test]
    fn test_f32_initialization() {
        let shape = IxDyn(&[10, 5]);
        let arr: Array<f32, IxDyn> =
            init_weights(Initializer::XavierNormal, shape).expect("f32 init should work");
        assert_eq!(arr.shape(), &[10, 5]);
        assert!(arr.iter().all(|v| v.is_finite()));
    }
}
