//! WASM SIMD128 Accelerated Operations
//!
//! Provides vectorized numerical operations using the WASM `simd128` target feature
//! when available, with automatic scalar fallback for non-WASM builds and environments
//! without SIMD support.
//!
//! All public functions are exported via `#[wasm_bindgen]` so they can be called
//! directly from JavaScript/TypeScript.
//!
//! ## Memory layout conventions
//!
//! - Vectors and matrices are passed as **flat `f32` slices** (row-major for matrices).
//! - Results are returned as `Vec<f32>` which wasm-bindgen maps to `Float32Array`.
//!
//! ## SIMD strategy
//!
//! The crate is compiled once.  SIMD intrinsics are gated on
//! `#[cfg(target_feature = "simd128")]` which is set by the compiler when the
//! `-C target-feature=+simd128` flag is passed (e.g. via `wasm-pack build
//! --target web -- -C target-feature=+simd128`).  On other targets (native
//! tests, browsers without SIMD) the module falls through to the pure-scalar
//! implementations automatically.

use crate::error::WasmError;
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// WASM SIMD128 intrinsics – only compiled when the feature is active
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use core::arch::wasm32::*;

// ---------------------------------------------------------------------------
// Public WASM-bindgen API
// ---------------------------------------------------------------------------

/// Compute the dot product of two f32 vectors.
///
/// Uses WASM SIMD128 lanes (4 × f32 per step) when compiled with
/// `target_feature = "simd128"`, otherwise falls back to a scalar loop.
///
/// # Errors
///
/// Returns a JS error if the two slices have different lengths.
#[wasm_bindgen]
pub fn simd_dot_product_f32(a: &[f32], b: &[f32]) -> Result<f32, JsValue> {
    if a.len() != b.len() {
        return Err(WasmError::InvalidParameter(format!(
            "simd_dot_product_f32: length mismatch: {} vs {}",
            a.len(),
            b.len()
        ))
        .into());
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return Ok(dot_product_simd(a, b));
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        return Ok(dot_product_scalar(a, b));
    }
}

/// SIMD-accelerated matrix multiplication for square f32 matrices.
///
/// Supports 4×4 and 8×8 matrices. For other sizes the function falls back to
/// a pure-scalar O(n³) implementation.
///
/// # Arguments
///
/// * `a` – row-major flat representation of an n×n matrix (length n²)
/// * `b` – row-major flat representation of an n×n matrix (length n²)
/// * `n` – matrix side length (must satisfy `a.len() == n*n`)
///
/// # Returns
///
/// Row-major flat f32 `Vec` of the n×n result.
///
/// # Errors
///
/// Returns a JS error if the lengths are inconsistent with n.
#[wasm_bindgen]
pub fn simd_matrix_multiply_f32(a: &[f32], b: &[f32], n: usize) -> Result<Vec<f32>, JsValue> {
    if n == 0 {
        return Err(WasmError::InvalidParameter("n must be > 0".to_string()).into());
    }
    let expected = n.checked_mul(n).ok_or_else(|| {
        WasmError::InvalidParameter("n² overflows usize".to_string())
    })?;

    if a.len() != expected || b.len() != expected {
        return Err(WasmError::InvalidParameter(format!(
            "simd_matrix_multiply_f32: expected {}×{} = {} elements, got a={} b={}",
            n,
            n,
            expected,
            a.len(),
            b.len()
        ))
        .into());
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        if n == 4 {
            return Ok(matmul4x4_simd(a, b));
        }
        if n == 8 {
            return Ok(matmul8x8_simd(a, b));
        }
    }

    Ok(matmul_scalar(a, b, n))
}

/// Compute softmax of a f32 vector in-place, returning the result.
///
/// Uses WASM SIMD128 where available (max + subtract + exp + sum + divide).
///
/// # Errors
///
/// Returns a JS error if the input is empty.
#[wasm_bindgen]
pub fn simd_softmax_f32(input: &[f32]) -> Result<Vec<f32>, JsValue> {
    if input.is_empty() {
        return Err(WasmError::InvalidParameter(
            "simd_softmax_f32: input must not be empty".to_string(),
        )
        .into());
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return Ok(softmax_simd(input));
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        return Ok(softmax_scalar(input));
    }
}

/// Apply the ReLU activation (max(0, x)) element-wise to a f32 slice.
///
/// Uses WASM SIMD128 `f32x4_max` lanes where available.
#[wasm_bindgen]
pub fn simd_relu_f32(input: &[f32]) -> Vec<f32> {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        relu_simd(input)
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        relu_scalar(input)
    }
}

/// Apply the sigmoid activation (1 / (1 + exp(-x))) element-wise.
///
/// Falls back to scalar on non-SIMD targets because the exponential has no
/// SIMD128 intrinsic; the scalar path is still vectorised by LLVM.
#[wasm_bindgen]
pub fn simd_sigmoid_f32(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| 1.0_f32 / (1.0_f32 + (-x).exp())).collect()
}

/// Compute element-wise sum of two f32 slices.
///
/// # Errors
///
/// Returns a JS error if the slices differ in length.
#[wasm_bindgen]
pub fn simd_add_f32(a: &[f32], b: &[f32]) -> Result<Vec<f32>, JsValue> {
    if a.len() != b.len() {
        return Err(WasmError::InvalidParameter(format!(
            "simd_add_f32: length mismatch: {} vs {}",
            a.len(),
            b.len()
        ))
        .into());
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return Ok(add_simd(a, b));
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        return Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect());
    }
}

/// Compute element-wise multiply of two f32 slices.
///
/// # Errors
///
/// Returns a JS error if the slices differ in length.
#[wasm_bindgen]
pub fn simd_mul_f32(a: &[f32], b: &[f32]) -> Result<Vec<f32>, JsValue> {
    if a.len() != b.len() {
        return Err(WasmError::InvalidParameter(format!(
            "simd_mul_f32: length mismatch: {} vs {}",
            a.len(),
            b.len()
        ))
        .into());
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        return Ok(mul_simd(a, b));
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        return Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect());
    }
}

/// Compute L2 (Euclidean) norm of a f32 vector.
#[wasm_bindgen]
pub fn simd_l2_norm_f32(input: &[f32]) -> f32 {
    let sq_sum = {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            dot_product_simd(input, input)
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            dot_product_scalar(input, input)
        }
    };
    sq_sum.sqrt()
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

/// Return `true` if this binary was compiled with WASM SIMD128 support.
#[wasm_bindgen]
pub fn simd_ops_available() -> bool {
    #[cfg(target_feature = "simd128")]
    { true }
    #[cfg(not(target_feature = "simd128"))]
    { false }
}

// ---------------------------------------------------------------------------
// ---- Internal scalar implementations (used as fallbacks) -----------------
// ---------------------------------------------------------------------------

#[inline(always)]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    // Manual 4-way unroll to hint LLVM auto-vectorisation.
    let n = a.len();
    let chunks = n / 4;
    let mut acc0 = 0.0_f32;
    let mut acc1 = 0.0_f32;
    let mut acc2 = 0.0_f32;
    let mut acc3 = 0.0_f32;

    for i in 0..chunks {
        let base = i * 4;
        acc0 += a[base]     * b[base];
        acc1 += a[base + 1] * b[base + 1];
        acc2 += a[base + 2] * b[base + 2];
        acc3 += a[base + 3] * b[base + 3];
    }

    let mut remainder = acc0 + acc1 + acc2 + acc3;
    for i in (chunks * 4)..n {
        remainder += a[i] * b[i];
    }
    remainder
}

fn matmul_scalar(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; n * n];
    for row in 0..n {
        for k in 0..n {
            let aik = a[row * n + k];
            for col in 0..n {
                c[row * n + col] += aik * b[k * n + col];
            }
        }
    }
    c
}

fn softmax_scalar(input: &[f32]) -> Vec<f32> {
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        let n = exps.len() as f32;
        exps.iter_mut().for_each(|v| *v = 1.0 / n);
    } else {
        exps.iter_mut().for_each(|v| *v /= sum);
    }
    exps
}

fn relu_scalar(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| x.max(0.0_f32)).collect()
}

// ---------------------------------------------------------------------------
// ---- WASM SIMD128 implementations ----------------------------------------
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 4;

    // Safety: We use raw pointers only to load aligned SIMD lanes.
    // The slices are valid for `n` elements and we only access within bounds.
    let mut acc = f32x4_splat(0.0);

    for i in 0..chunks {
        let base = i * 4;
        // Safety: base..base+4 is within the slice length (chunks = n/4).
        let va = unsafe {
            v128_load(a.as_ptr().add(base) as *const v128)
        };
        let vb = unsafe {
            v128_load(b.as_ptr().add(base) as *const v128)
        };
        acc = f32x4_add(acc, f32x4_mul(va, vb));
    }

    // Horizontal sum of the 4 accumulator lanes.
    let mut sum = f32x4_extract_lane::<0>(acc)
        + f32x4_extract_lane::<1>(acc)
        + f32x4_extract_lane::<2>(acc)
        + f32x4_extract_lane::<3>(acc);

    // Scalar tail.
    for i in (chunks * 4)..n {
        sum += a[i] * b[i];
    }
    sum
}

/// 4×4 SIMD matrix multiply.  Each row of A is broadcast and multiplied by
/// the corresponding row of B, accumulated using f32x4 lanes.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn matmul4x4_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0_f32; 16];

    for row in 0..4_usize {
        // Load the output row as an accumulator (zero-initialised).
        let mut c_row = f32x4_splat(0.0);

        for k in 0..4_usize {
            // Broadcast a[row][k] into all four lanes.
            let aik = f32x4_splat(a[row * 4 + k]);
            // Load the k-th row of B.
            let b_row = unsafe {
                v128_load(b.as_ptr().add(k * 4) as *const v128)
            };
            c_row = f32x4_add(c_row, f32x4_mul(aik, b_row));
        }

        // Store result row.
        unsafe {
            v128_store(c.as_mut_ptr().add(row * 4) as *mut v128, c_row);
        }
    }
    c
}

/// 8×8 SIMD matrix multiply – two f32x4 registers per row.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn matmul8x8_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0_f32; 64];

    for row in 0..8_usize {
        let mut c_lo = f32x4_splat(0.0); // cols 0–3
        let mut c_hi = f32x4_splat(0.0); // cols 4–7

        for k in 0..8_usize {
            let aik = f32x4_splat(a[row * 8 + k]);

            let b_lo = unsafe {
                v128_load(b.as_ptr().add(k * 8) as *const v128)
            };
            let b_hi = unsafe {
                v128_load(b.as_ptr().add(k * 8 + 4) as *const v128)
            };

            c_lo = f32x4_add(c_lo, f32x4_mul(aik, b_lo));
            c_hi = f32x4_add(c_hi, f32x4_mul(aik, b_hi));
        }

        unsafe {
            v128_store(c.as_mut_ptr().add(row * 8)     as *mut v128, c_lo);
            v128_store(c.as_mut_ptr().add(row * 8 + 4) as *mut v128, c_hi);
        }
    }
    c
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn softmax_simd(input: &[f32]) -> Vec<f32> {
    // Step 1: compute maximum via scalar pass (horizontal max in SIMD is involved).
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let max_splat = f32x4_splat(max_val);

    let n = input.len();
    let chunks = n / 4;

    // Step 2: compute shifted exp() and accumulate sum.
    // NOTE: WASM SIMD128 has no native exp() – we call scalar exp() per element
    // after SIMD subtract, then store back.
    let mut exps = vec![0.0_f32; n];
    let mut sum = 0.0_f32;

    for i in 0..chunks {
        let base = i * 4;
        let v = unsafe { v128_load(input.as_ptr().add(base) as *const v128) };
        let shifted = f32x4_sub(v, max_splat);

        // Extract, apply scalar exp, accumulate.
        let e0 = f32x4_extract_lane::<0>(shifted).exp();
        let e1 = f32x4_extract_lane::<1>(shifted).exp();
        let e2 = f32x4_extract_lane::<2>(shifted).exp();
        let e3 = f32x4_extract_lane::<3>(shifted).exp();

        exps[base]     = e0;
        exps[base + 1] = e1;
        exps[base + 2] = e2;
        exps[base + 3] = e3;
        sum += e0 + e1 + e2 + e3;
    }

    for i in (chunks * 4)..n {
        let e = (input[i] - max_val).exp();
        exps[i] = e;
        sum += e;
    }

    // Step 3: divide by sum using SIMD.
    if sum == 0.0 {
        let inv = 1.0_f32 / n as f32;
        return vec![inv; n];
    }

    let inv_sum = f32x4_splat(1.0_f32 / sum);

    for i in 0..chunks {
        let base = i * 4;
        let v = unsafe { v128_load(exps.as_ptr().add(base) as *const v128) };
        let normed = f32x4_mul(v, inv_sum);
        unsafe { v128_store(exps.as_mut_ptr().add(base) as *mut v128, normed) };
    }
    for i in (chunks * 4)..n {
        exps[i] /= sum;
    }

    exps
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn relu_simd(input: &[f32]) -> Vec<f32> {
    let n = input.len();
    let chunks = n / 4;
    let zero = f32x4_splat(0.0);
    let mut out = vec![0.0_f32; n];

    for i in 0..chunks {
        let base = i * 4;
        let v = unsafe { v128_load(input.as_ptr().add(base) as *const v128) };
        let r = f32x4_max(v, zero);
        unsafe { v128_store(out.as_mut_ptr().add(base) as *mut v128, r) };
    }
    for i in (chunks * 4)..n {
        out[i] = input[i].max(0.0);
    }
    out
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn add_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len();
    let chunks = n / 4;
    let mut out = vec![0.0_f32; n];

    for i in 0..chunks {
        let base = i * 4;
        let va = unsafe { v128_load(a.as_ptr().add(base) as *const v128) };
        let vb = unsafe { v128_load(b.as_ptr().add(base) as *const v128) };
        let r = f32x4_add(va, vb);
        unsafe { v128_store(out.as_mut_ptr().add(base) as *mut v128, r) };
    }
    for i in (chunks * 4)..n {
        out[i] = a[i] + b[i];
    }
    out
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn mul_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len();
    let chunks = n / 4;
    let mut out = vec![0.0_f32; n];

    for i in 0..chunks {
        let base = i * 4;
        let va = unsafe { v128_load(a.as_ptr().add(base) as *const v128) };
        let vb = unsafe { v128_load(b.as_ptr().add(base) as *const v128) };
        let r = f32x4_mul(va, vb);
        unsafe { v128_store(out.as_mut_ptr().add(base) as *mut v128, r) };
    }
    for i in (chunks * 4)..n {
        out[i] = a[i] * b[i];
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_basic() {
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [1.0_f32, 1.0, 1.0, 1.0];
        let result = simd_dot_product_f32(&a, &b).expect("dot product ok");
        assert!((result - 10.0).abs() < 1e-6, "expected 10, got {result}");
    }

    #[test]
    fn test_dot_product_non_multiple_of_4() {
        let a = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0_f32, 2.0, 2.0, 2.0, 2.0];
        let result = simd_dot_product_f32(&a, &b).expect("dot product ok");
        assert!((result - 30.0).abs() < 1e-5);
    }

    #[test]
    fn test_dot_product_length_mismatch() {
        let result = simd_dot_product_f32(&[1.0], &[1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_4x4_identity() {
        let identity: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let a: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let result = simd_matrix_multiply_f32(&a, &identity, 4).expect("matmul ok");
        for (r, e) in result.iter().zip(a.iter()) {
            assert!((r - e).abs() < 1e-5, "expected {e}, got {r}");
        }
    }

    #[test]
    fn test_matmul_bad_size() {
        let a = vec![1.0_f32; 9]; // 3x3
        let b = vec![1.0_f32; 9];
        // n=3 → scalar path, should succeed
        let result = simd_matrix_multiply_f32(&a, &b, 3);
        assert!(result.is_ok());
    }

    #[test]
    fn test_matmul_zero_n() {
        let result = simd_matrix_multiply_f32(&[], &[], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = simd_softmax_f32(&input).expect("softmax ok");
        let total: f32 = result.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "sum = {total}");
        for &v in &result {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_softmax_empty() {
        assert!(simd_softmax_f32(&[]).is_err());
    }

    #[test]
    fn test_relu_basic() {
        let input = [-3.0_f32, -1.0, 0.0, 1.0, 3.0];
        let result = simd_relu_f32(&input);
        let expected = [0.0_f32, 0.0, 0.0, 1.0, 3.0];
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-7, "expected {e}, got {r}");
        }
    }

    #[test]
    fn test_add_f32() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        let result = simd_add_f32(&a, &b).expect("add ok");
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_mul_f32() {
        let a = [2.0_f32, 3.0, 4.0];
        let b = [5.0_f32, 6.0, 7.0];
        let result = simd_mul_f32(&a, &b).expect("mul ok");
        assert_eq!(result, vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_l2_norm() {
        let v = [3.0_f32, 4.0]; // 3-4-5 triangle
        assert!((simd_l2_norm_f32(&v) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_sigmoid() {
        let v = [0.0_f32];
        let result = simd_sigmoid_f32(&v);
        assert!((result[0] - 0.5).abs() < 1e-6);
    }
}
