//! Mixed-precision matrix multiplication (GEMM) with F16/BF16 inputs
//!
//! Implements general matrix multiplication where inputs are stored in
//! half-precision (F16 or BF16) but accumulation happens in higher
//! precision (f32 or f64) for numerical stability.
//!
//! Three accumulation strategies are available:
//! - **F32**: standard f32 accumulation (fastest)
//! - **F64**: f64 accumulation (most precise)
//! - **Kahan**: Kahan compensated summation in f32 (good precision/speed trade-off)

use scirs2_core::ndarray::Array2;

use crate::error::{LinalgError, LinalgResult};

use super::types::{BF16, F16};

// ============================================================================
// Configuration
// ============================================================================

/// Accumulation strategy for mixed-precision GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum AccumulationType {
    /// Accumulate in f32 (default, fastest)
    F32,
    /// Accumulate in f64 (highest precision)
    F64,
    /// Kahan compensated summation in f32 (good precision/speed trade-off)
    Kahan,
}

impl Default for AccumulationType {
    fn default() -> Self {
        AccumulationType::F32
    }
}

/// Configuration for mixed-precision GEMM operations.
#[derive(Clone, Debug)]
pub struct MixedPrecisionConfig {
    /// Accumulation strategy
    pub accumulation: AccumulationType,
    /// Optional loss scaling factor (used during gradient computation
    /// in mixed-precision training). When `Some(s)`, the output is
    /// multiplied by `s`.
    pub loss_scaling: Option<f64>,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            accumulation: AccumulationType::F32,
            loss_scaling: None,
        }
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Kahan (compensated) dot product of two F16 slices, returning f32.
fn kahan_dot_f16(a: &[F16], b: &[F16]) -> f32 {
    let mut sum: f32 = 0.0;
    let mut comp: f32 = 0.0; // compensation for lost low-order bits
    for (av, bv) in a.iter().zip(b.iter()) {
        let product = av.to_f32() * bv.to_f32();
        let y = product - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Kahan (compensated) dot product of two BF16 slices, returning f32.
fn kahan_dot_bf16(a: &[BF16], b: &[BF16]) -> f32 {
    let mut sum: f32 = 0.0;
    let mut comp: f32 = 0.0;
    for (av, bv) in a.iter().zip(b.iter()) {
        let product = av.to_f32() * bv.to_f32();
        let y = product - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Kahan dot product of F16 slice and f32 slice, returning f32.
fn kahan_dot_mixed(a: &[F16], b: &[f32]) -> f32 {
    let mut sum: f32 = 0.0;
    let mut comp: f32 = 0.0;
    for (av, &bv) in a.iter().zip(b.iter()) {
        let product = av.to_f32() * bv;
        let y = product - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Validate matrix dimensions for GEMM: C[m,n] = A[m,k] * B[k,n].
fn validate_gemm_dims(
    a_rows: usize,
    a_cols: usize,
    b_rows: usize,
    _b_cols: usize,
) -> LinalgResult<()> {
    if a_cols != b_rows {
        return Err(LinalgError::DimensionError(format!(
            "GEMM dimension mismatch: A is {}x{}, B is {}x{} (inner dimensions must match)",
            a_rows, a_cols, b_rows, _b_cols
        )));
    }
    Ok(())
}

/// Apply loss scaling to the output matrix if configured.
fn apply_loss_scaling(c: &mut Array2<f32>, config: &MixedPrecisionConfig) {
    if let Some(scale) = config.loss_scaling {
        let s = scale as f32;
        c.mapv_inplace(|v| v * s);
    }
}

// ============================================================================
// F16 GEMM
// ============================================================================

/// Compute C_f32 = A_f16 x B_f16 with the specified accumulation strategy.
///
/// Both input matrices are in half-precision (F16), but accumulation and
/// the output are in f32 (or f64 internally, depending on config).
///
/// # Errors
/// Returns [`LinalgError::DimensionError`] if inner dimensions do not match.
pub fn gemm_f16(
    a: &Array2<F16>,
    b: &Array2<F16>,
    config: &MixedPrecisionConfig,
) -> LinalgResult<Array2<f32>> {
    let (m, k) = (a.nrows(), a.ncols());
    let (k2, n) = (b.nrows(), b.ncols());
    validate_gemm_dims(m, k, k2, n)?;

    let mut c = Array2::<f32>::zeros((m, n));

    match config.accumulation {
        AccumulationType::F32 => {
            for i in 0..m {
                for j in 0..n {
                    let mut acc: f32 = 0.0;
                    for p in 0..k {
                        acc += a[[i, p]].to_f32() * b[[p, j]].to_f32();
                    }
                    c[[i, j]] = acc;
                }
            }
        }
        AccumulationType::F64 => {
            for i in 0..m {
                for j in 0..n {
                    let mut acc: f64 = 0.0;
                    for p in 0..k {
                        acc += a[[i, p]].to_f64() * b[[p, j]].to_f64();
                    }
                    c[[i, j]] = acc as f32;
                }
            }
        }
        AccumulationType::Kahan => {
            for i in 0..m {
                for j in 0..n {
                    let mut sum: f32 = 0.0;
                    let mut comp: f32 = 0.0;
                    for p in 0..k {
                        let product = a[[i, p]].to_f32() * b[[p, j]].to_f32();
                        let y = product - comp;
                        let t = sum + y;
                        comp = (t - sum) - y;
                        sum = t;
                    }
                    c[[i, j]] = sum;
                }
            }
        }
    }

    apply_loss_scaling(&mut c, config);
    Ok(c)
}

// ============================================================================
// BF16 GEMM
// ============================================================================

/// Compute C_f32 = A_bf16 x B_bf16 with the specified accumulation strategy.
///
/// # Errors
/// Returns [`LinalgError::DimensionError`] if inner dimensions do not match.
pub fn gemm_bf16(
    a: &Array2<BF16>,
    b: &Array2<BF16>,
    config: &MixedPrecisionConfig,
) -> LinalgResult<Array2<f32>> {
    let (m, k) = (a.nrows(), a.ncols());
    let (k2, n) = (b.nrows(), b.ncols());
    validate_gemm_dims(m, k, k2, n)?;

    let mut c = Array2::<f32>::zeros((m, n));

    match config.accumulation {
        AccumulationType::F32 => {
            for i in 0..m {
                for j in 0..n {
                    let mut acc: f32 = 0.0;
                    for p in 0..k {
                        acc += a[[i, p]].to_f32() * b[[p, j]].to_f32();
                    }
                    c[[i, j]] = acc;
                }
            }
        }
        AccumulationType::F64 => {
            for i in 0..m {
                for j in 0..n {
                    let mut acc: f64 = 0.0;
                    for p in 0..k {
                        acc += a[[i, p]].to_f64() * b[[p, j]].to_f64();
                    }
                    c[[i, j]] = acc as f32;
                }
            }
        }
        AccumulationType::Kahan => {
            for i in 0..m {
                for j in 0..n {
                    let mut sum: f32 = 0.0;
                    let mut comp: f32 = 0.0;
                    for p in 0..k {
                        let product = a[[i, p]].to_f32() * b[[p, j]].to_f32();
                        let y = product - comp;
                        let t = sum + y;
                        comp = (t - sum) - y;
                        sum = t;
                    }
                    c[[i, j]] = sum;
                }
            }
        }
    }

    apply_loss_scaling(&mut c, config);
    Ok(c)
}

// ============================================================================
// Mixed-type GEMM (F16 x f32)
// ============================================================================

/// Compute C_f32 = A_f16 x B_f32 (mixed input types).
///
/// Useful when weights are stored in F16 but activations remain in f32.
///
/// # Errors
/// Returns [`LinalgError::DimensionError`] if inner dimensions do not match.
pub fn gemm_mixed(
    a: &Array2<F16>,
    b: &Array2<f32>,
    config: &MixedPrecisionConfig,
) -> LinalgResult<Array2<f32>> {
    let (m, k) = (a.nrows(), a.ncols());
    let (k2, n) = (b.nrows(), b.ncols());
    validate_gemm_dims(m, k, k2, n)?;

    let mut c = Array2::<f32>::zeros((m, n));

    match config.accumulation {
        AccumulationType::F32 => {
            for i in 0..m {
                for j in 0..n {
                    let mut acc: f32 = 0.0;
                    for p in 0..k {
                        acc += a[[i, p]].to_f32() * b[[p, j]];
                    }
                    c[[i, j]] = acc;
                }
            }
        }
        AccumulationType::F64 => {
            for i in 0..m {
                for j in 0..n {
                    let mut acc: f64 = 0.0;
                    for p in 0..k {
                        acc += a[[i, p]].to_f64() * (b[[p, j]] as f64);
                    }
                    c[[i, j]] = acc as f32;
                }
            }
        }
        AccumulationType::Kahan => {
            for i in 0..m {
                for j in 0..n {
                    let mut sum: f32 = 0.0;
                    let mut comp: f32 = 0.0;
                    for p in 0..k {
                        let product = a[[i, p]].to_f32() * b[[p, j]];
                        let y = product - comp;
                        let t = sum + y;
                        comp = (t - sum) - y;
                        sum = t;
                    }
                    c[[i, j]] = sum;
                }
            }
        }
    }

    apply_loss_scaling(&mut c, config);
    Ok(c)
}

// ============================================================================
// Convenience: row/column extraction for contiguous Kahan dot
// ============================================================================

/// Compute the dot product of a row of A (F16) with a column of B (F16)
/// using Kahan summation. This is a convenience for the GEMM inner loop
/// when data is available in contiguous slices.
pub fn kahan_gemm_element_f16(a_row: &[F16], b_col: &[F16]) -> f32 {
    kahan_dot_f16(a_row, b_col)
}

/// Compute the dot product of a row of A (BF16) with a column of B (BF16)
/// using Kahan summation.
pub fn kahan_gemm_element_bf16(a_row: &[BF16], b_col: &[BF16]) -> f32 {
    kahan_dot_bf16(a_row, b_col)
}

/// Compute the dot product of a row of A (F16) with a column of B (f32)
/// using Kahan summation.
pub fn kahan_gemm_element_mixed(a_row: &[F16], b_col: &[f32]) -> f32 {
    kahan_dot_mixed(a_row, b_col)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_f16_matrix(data: &[f32], rows: usize, cols: usize) -> Array2<F16> {
        let v: Vec<F16> = data.iter().map(|&x| F16::from_f32(x)).collect();
        Array2::from_shape_vec((rows, cols), v).expect("valid shape")
    }

    fn make_bf16_matrix(data: &[f32], rows: usize, cols: usize) -> Array2<BF16> {
        let v: Vec<BF16> = data.iter().map(|&x| BF16::from_f32(x)).collect();
        Array2::from_shape_vec((rows, cols), v).expect("valid shape")
    }

    #[test]
    fn test_gemm_f16_identity() {
        // A * I = A
        let a = make_f16_matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let eye = make_f16_matrix(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let config = MixedPrecisionConfig::default();
        let c = gemm_f16(&a, &eye, &config).expect("gemm ok");
        assert!((c[[0, 0]] - 1.0).abs() < 0.01);
        assert!((c[[0, 1]] - 2.0).abs() < 0.01);
        assert!((c[[1, 0]] - 3.0).abs() < 0.01);
        assert!((c[[1, 1]] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_gemm_f16_basic() {
        // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = make_f16_matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = make_f16_matrix(&[5.0, 6.0, 7.0, 8.0], 2, 2);
        let config = MixedPrecisionConfig::default();
        let c = gemm_f16(&a, &b, &config).expect("gemm ok");
        assert!((c[[0, 0]] - 19.0).abs() < 0.1);
        assert!((c[[0, 1]] - 22.0).abs() < 0.1);
        assert!((c[[1, 0]] - 43.0).abs() < 0.1);
        assert!((c[[1, 1]] - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_gemm_f16_vs_f64_reference() {
        // Compare F16 GEMM to f64 GEMM
        let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let a = make_f16_matrix(&a_data, 2, 3);
        let b = make_f16_matrix(&b_data, 3, 2);

        // f64 reference
        let a64 =
            Array2::from_shape_vec((2, 3), a_data.iter().map(|&x| x as f64).collect()).expect("ok");
        let b64 =
            Array2::from_shape_vec((3, 2), b_data.iter().map(|&x| x as f64).collect()).expect("ok");
        let ref_c = a64.dot(&b64);

        let config = MixedPrecisionConfig {
            accumulation: AccumulationType::F64,
            loss_scaling: None,
        };
        let c = gemm_f16(&a, &b, &config).expect("gemm ok");

        for i in 0..2 {
            for j in 0..2 {
                let err = (c[[i, j]] as f64 - ref_c[[i, j]]).abs();
                assert!(
                    err < 1.0,
                    "f16 GEMM result [{i},{j}] = {}, reference = {}, error = {err}",
                    c[[i, j]],
                    ref_c[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_gemm_f16_dimension_mismatch() {
        let a = make_f16_matrix(&[1.0, 2.0, 3.0], 1, 3);
        let b = make_f16_matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let config = MixedPrecisionConfig::default();
        assert!(gemm_f16(&a, &b, &config).is_err());
    }

    #[test]
    fn test_gemm_bf16_basic() {
        let a = make_bf16_matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = make_bf16_matrix(&[5.0, 6.0, 7.0, 8.0], 2, 2);
        let config = MixedPrecisionConfig::default();
        let c = gemm_bf16(&a, &b, &config).expect("gemm ok");
        assert!((c[[0, 0]] - 19.0).abs() < 1.0);
        assert!((c[[0, 1]] - 22.0).abs() < 1.0);
        assert!((c[[1, 0]] - 43.0).abs() < 1.0);
        assert!((c[[1, 1]] - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_gemm_mixed_basic() {
        let a = make_f16_matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Array2::from_shape_vec((2, 2), vec![5.0f32, 6.0, 7.0, 8.0]).expect("ok");
        let config = MixedPrecisionConfig::default();
        let c = gemm_mixed(&a, &b, &config).expect("gemm ok");
        assert!((c[[0, 0]] - 19.0).abs() < 0.1);
        assert!((c[[1, 1]] - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_gemm_with_loss_scaling() {
        let a = make_f16_matrix(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        let b = make_f16_matrix(&[2.0, 0.0, 0.0, 3.0], 2, 2);
        let config = MixedPrecisionConfig {
            accumulation: AccumulationType::F32,
            loss_scaling: Some(10.0),
        };
        let c = gemm_f16(&a, &b, &config).expect("gemm ok");
        assert!((c[[0, 0]] - 20.0).abs() < 0.1);
        assert!((c[[1, 1]] - 30.0).abs() < 0.1);
    }

    #[test]
    fn test_kahan_dot_f16_accuracy() {
        // Sum many small values: Kahan should be more accurate than naive
        let n = 1000;
        let vals: Vec<F16> = (0..n).map(|_| F16::from_f32(0.001)).collect();
        let ones: Vec<F16> = (0..n).map(|_| F16::ONE).collect();

        let kahan_result = kahan_dot_f16(&vals, &ones);
        // Expected: 0.001 * 1000 = 1.0
        // Due to f16 precision, 0.001 is approximately F16(0.0009765625)
        let expected = F16::from_f32(0.001).to_f32() * (n as f32);
        assert!(
            (kahan_result - expected).abs() < 0.05,
            "Kahan dot: expected ~{expected}, got {kahan_result}"
        );

        // Compare with naive sum
        let naive: f32 = vals
            .iter()
            .zip(ones.iter())
            .map(|(a, b)| a.to_f32() * b.to_f32())
            .sum();
        // Kahan should be at least as good as naive (or better)
        let kahan_err = (kahan_result - expected).abs();
        let naive_err = (naive - expected).abs();
        assert!(
            kahan_err <= naive_err + 1e-6,
            "Kahan ({kahan_err}) should be at least as accurate as naive ({naive_err})"
        );
    }

    #[test]
    fn test_gemm_kahan_accumulation() {
        let a = make_f16_matrix(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = make_f16_matrix(&[5.0, 6.0, 7.0, 8.0], 2, 2);
        let config = MixedPrecisionConfig {
            accumulation: AccumulationType::Kahan,
            loss_scaling: None,
        };
        let c = gemm_f16(&a, &b, &config).expect("gemm ok");
        assert!((c[[0, 0]] - 19.0).abs() < 0.1);
        assert!((c[[1, 1]] - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_gemm_non_square() {
        // [2x3] * [3x4] = [2x4]
        let a = make_f16_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = make_f16_matrix(
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            3,
            4,
        );
        let config = MixedPrecisionConfig::default();
        let c = gemm_f16(&a, &b, &config).expect("gemm ok");
        assert_eq!(c.shape(), &[2, 4]);
        // A * [I3 | 0] should give first 3 cols = A, last col = 0
        assert!((c[[0, 0]] - 1.0).abs() < 0.01);
        assert!((c[[0, 1]] - 2.0).abs() < 0.01);
        assert!((c[[0, 2]] - 3.0).abs() < 0.01);
        assert!(c[[0, 3]].abs() < 0.01);
    }
}
