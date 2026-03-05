//! GPU-accelerated matrix multiplication operations.
//!
//! Provides single and batched matrix multiplication using GPU backends
//! from scirs2-core, with automatic CPU fallback when GPU is unavailable.

use scirs2_core::gpu::{GpuBackend, GpuContext, GpuError};
use scirs2_core::ndarray::{Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};

use super::types::BatchMatmulResult;

/// Minimum matrix dimension to benefit from GPU acceleration.
/// Below this threshold, CPU execution is typically faster due to
/// transfer overhead.
const GPU_MATMUL_THRESHOLD: usize = 64;

/// Perform GPU-accelerated matrix multiplication: C = A * B
///
/// Transfers matrices to GPU, performs the multiplication, and transfers
/// the result back to host memory. Falls back to CPU implementation for
/// small matrices or when GPU is unavailable.
///
/// # Arguments
///
/// * `ctx` - GPU context (or None for CPU fallback)
/// * `a` - Left matrix (m x k)
/// * `b` - Right matrix (k x n)
///
/// # Returns
///
/// Result matrix C (m x n)
///
/// # Errors
///
/// Returns `LinalgError` if matrices have incompatible shapes or if
/// GPU execution fails without a viable fallback.
pub fn gpu_matmul<F>(
    ctx: Option<&GpuContext>,
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (m, k_a) = a.dim();
    let (k_b, n) = b.dim();

    if k_a != k_b {
        return Err(LinalgError::DimensionError(format!(
            "Matrix multiplication dimension mismatch: A is {}x{} but B is {}x{}",
            m, k_a, k_b, n
        )));
    }
    let k = k_a;

    // Check if GPU acceleration is worthwhile
    let use_gpu = ctx.is_some()
        && m >= GPU_MATMUL_THRESHOLD
        && n >= GPU_MATMUL_THRESHOLD
        && k >= GPU_MATMUL_THRESHOLD;

    if use_gpu {
        if let Some(gpu_ctx) = ctx {
            match gpu_matmul_impl(gpu_ctx, a, b, m, k, n) {
                Ok(result) => return Ok(result),
                Err(_) => {
                    // Fall through to CPU fallback
                }
            }
        }
    }

    // CPU fallback: standard matrix multiplication
    cpu_matmul(a, b)
}

/// Perform GPU-accelerated batched matrix multiplication.
///
/// Each pair (a_i, b_i) is multiplied independently. If possible, all
/// multiplications are dispatched to the GPU in a single batch for
/// maximum throughput.
///
/// # Arguments
///
/// * `ctx` - GPU context (or None for CPU fallback)
/// * `a_batch` - Batch of left matrices
/// * `b_batch` - Batch of right matrices (must have same length as `a_batch`)
///
/// # Returns
///
/// `BatchMatmulResult` containing one result matrix per input pair.
///
/// # Errors
///
/// Returns `LinalgError` if batch sizes differ or any pair has incompatible shapes.
pub fn gpu_batched_matmul<F>(
    ctx: Option<&GpuContext>,
    a_batch: &[Array2<F>],
    b_batch: &[Array2<F>],
) -> LinalgResult<BatchMatmulResult<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    if a_batch.len() != b_batch.len() {
        return Err(LinalgError::DimensionError(format!(
            "Batch size mismatch: {} left matrices vs {} right matrices",
            a_batch.len(),
            b_batch.len()
        )));
    }

    if a_batch.is_empty() {
        return Ok(BatchMatmulResult {
            results: Vec::new(),
        });
    }

    // Validate all pairs
    for (i, (a, b)) in a_batch.iter().zip(b_batch.iter()).enumerate() {
        let (_, k_a) = a.dim();
        let (k_b, _) = b.dim();
        if k_a != k_b {
            return Err(LinalgError::DimensionError(format!(
                "Batch element {}: dimension mismatch A({}x{}) * B({}x{})",
                i,
                a.nrows(),
                k_a,
                k_b,
                b.ncols()
            )));
        }
    }

    // Try GPU batch execution
    if let Some(gpu_ctx) = ctx {
        match gpu_batched_matmul_impl(gpu_ctx, a_batch, b_batch) {
            Ok(results) => return Ok(results),
            Err(_) => {
                // Fall through to CPU
            }
        }
    }

    // CPU fallback: multiply each pair individually
    let mut results = Vec::with_capacity(a_batch.len());
    for (a, b) in a_batch.iter().zip(b_batch.iter()) {
        results.push(cpu_matmul(&a.view(), &b.view())?);
    }
    Ok(BatchMatmulResult { results })
}

// ---------------------------------------------------------------------------
// Internal GPU implementations
// ---------------------------------------------------------------------------

/// GPU matrix multiplication using scirs2-core GpuContext.
fn gpu_matmul_impl<F>(
    ctx: &GpuContext,
    a: &ArrayView2<F>,
    b: &ArrayView2<F>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Array2<F>, LinalgError>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    // Convert to f64 for GPU computation (most GPU kernels work in f64)
    let a_flat: Vec<f64> = a.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
    let b_flat: Vec<f64> = b.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();

    // Create GPU buffers and transfer data
    let gpu_a = ctx.create_buffer_from_slice(&a_flat);
    let gpu_b = ctx.create_buffer_from_slice(&b_flat);

    // Perform GEMM on GPU
    let gpu_c = ctx
        .gemm(&gpu_a, &gpu_b, m, k, n)
        .map_err(|e| LinalgError::ComputationError(format!("GPU GEMM failed: {}", e)))?;

    // Transfer result back to host
    let c_flat = gpu_c.to_vec();

    // Convert back to F and reshape
    let c_data: Vec<F> = c_flat
        .iter()
        .map(|&v| F::from(v).unwrap_or_else(F::zero))
        .collect();

    Array2::from_shape_vec((m, n), c_data)
        .map_err(|e| LinalgError::ShapeError(format!("Failed to reshape GPU result: {}", e)))
}

/// GPU batched matrix multiplication implementation.
fn gpu_batched_matmul_impl<F>(
    ctx: &GpuContext,
    a_batch: &[Array2<F>],
    b_batch: &[Array2<F>],
) -> Result<BatchMatmulResult<F>, LinalgError>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let mut results = Vec::with_capacity(a_batch.len());

    for (a, b) in a_batch.iter().zip(b_batch.iter()) {
        let (m, k) = a.dim();
        let (_, n) = b.dim();

        let a_flat: Vec<f64> = a.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
        let b_flat: Vec<f64> = b.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();

        let gpu_a = ctx.create_buffer_from_slice(&a_flat);
        let gpu_b = ctx.create_buffer_from_slice(&b_flat);

        let gpu_c = ctx.gemm(&gpu_a, &gpu_b, m, k, n).map_err(|e| {
            LinalgError::ComputationError(format!("GPU batched GEMM failed: {}", e))
        })?;

        let c_flat = gpu_c.to_vec();
        let c_data: Vec<F> = c_flat
            .iter()
            .map(|&v| F::from(v).unwrap_or_else(F::zero))
            .collect();

        let result = Array2::from_shape_vec((m, n), c_data).map_err(|e| {
            LinalgError::ShapeError(format!("Failed to reshape batch result: {}", e))
        })?;
        results.push(result);
    }

    Ok(BatchMatmulResult { results })
}

// ---------------------------------------------------------------------------
// CPU fallback
// ---------------------------------------------------------------------------

/// CPU matrix multiplication fallback.
fn cpu_matmul<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static,
{
    let (m, k) = a.dim();
    let (_, n) = b.dim();

    let mut c = Array2::<F>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut sum = F::zero();
            for p in 0..k {
                sum += a[[i, p]] * b[[p, j]];
            }
            c[[i, j]] = sum;
        }
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_cpu_matmul_basic() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b = array![[5.0_f64, 6.0], [7.0, 8.0]];
        let c = cpu_matmul(&a.view(), &b.view()).expect("matmul failed");
        assert!((c[[0, 0]] - 19.0).abs() < 1e-10);
        assert!((c[[0, 1]] - 22.0).abs() < 1e-10);
        assert!((c[[1, 0]] - 43.0).abs() < 1e-10);
        assert!((c[[1, 1]] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_gpu_matmul_dimension_mismatch() {
        let a = array![[1.0_f64, 2.0, 3.0]];
        let b = array![[1.0_f64], [2.0]];
        let result = gpu_matmul(None, &a.view(), &b.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_gpu_matmul_cpu_fallback() {
        // Without GPU context, should fall back to CPU
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let b = array![[5.0_f64, 6.0], [7.0, 8.0]];
        let c = gpu_matmul(None, &a.view(), &b.view()).expect("fallback failed");
        assert!((c[[0, 0]] - 5.0).abs() < 1e-10);
        assert!((c[[1, 1]] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_gpu_matmul_rectangular() {
        let a = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]]; // 2x3
        let b = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]]; // 3x2
        let c = gpu_matmul(None, &a.view(), &b.view()).expect("rectangular matmul failed");
        assert_eq!(c.dim(), (2, 2));
        assert!((c[[0, 0]] - 22.0).abs() < 1e-10);
        assert!((c[[0, 1]] - 28.0).abs() < 1e-10);
        assert!((c[[1, 0]] - 49.0).abs() < 1e-10);
        assert!((c[[1, 1]] - 64.0).abs() < 1e-10);
    }

    #[test]
    fn test_gpu_batched_matmul_empty() {
        let result: LinalgResult<BatchMatmulResult<f64>> = gpu_batched_matmul(None, &[], &[]);
        let batch = result.expect("empty batch failed");
        assert!(batch.results.is_empty());
    }

    #[test]
    fn test_gpu_batched_matmul_size_mismatch() {
        let a = vec![array![[1.0_f64]]];
        let b: Vec<Array2<f64>> = vec![];
        let result = gpu_batched_matmul(None, &a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_gpu_batched_matmul_cpu_fallback() {
        let a1 = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let b1 = array![[5.0_f64, 6.0], [7.0, 8.0]];
        let a2 = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let b2 = array![[9.0_f64, 10.0], [11.0, 12.0]];

        let batch = gpu_batched_matmul(None, &[a1, a2], &[b1, b2]).expect("batch failed");
        assert_eq!(batch.results.len(), 2);
        // First pair
        assert!((batch.results[0][[0, 0]] - 19.0).abs() < 1e-10);
        // Second pair (identity * b2 = b2)
        assert!((batch.results[1][[0, 0]] - 9.0).abs() < 1e-10);
        assert!((batch.results[1][[1, 1]] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_gpu_matmul_f32() {
        let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
        let b = array![[5.0_f32, 6.0], [7.0, 8.0]];
        let c = gpu_matmul(None, &a.view(), &b.view()).expect("f32 matmul failed");
        assert!((c[[0, 0]] - 19.0).abs() < 1e-4);
        assert!((c[[1, 1]] - 50.0).abs() < 1e-4);
    }

    #[test]
    fn test_gpu_matmul_with_cpu_context() {
        // Create a CPU-backend GPU context to test the GPU path logic
        let ctx = GpuContext::new(GpuBackend::Cpu);
        match ctx {
            Ok(gpu_ctx) => {
                let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
                let b = array![[5.0_f64, 6.0], [7.0, 8.0]];
                // Small matrices => will use CPU fallback due to threshold
                let c = gpu_matmul(Some(&gpu_ctx), &a.view(), &b.view())
                    .expect("matmul with ctx failed");
                assert!((c[[0, 0]] - 19.0).abs() < 1e-10);
            }
            Err(_) => {
                // If we cannot create a CPU context, skip this test
            }
        }
    }
}
