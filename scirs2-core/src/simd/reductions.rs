//! Statistical reduction operations with SIMD acceleration
//!
//! This module provides optimized implementations of common statistical
//! reduction operations like sum, mean, variance, and standard deviation.

use ndarray::{Array1, ArrayView1};

pub fn simd_sum_f32(input: &ArrayView1<f32>) -> f32 {
    let len = input.len();
    if len == 0 {
        return 0.0;
    }

    // Fall back to scalar sum if the array is not contiguous (e.g. column slices)
    if input.as_slice().is_none() {
        return input.iter().copied().sum();
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut sum_vec = _mm256_setzero_ps();
                let mut i = 0;

                // Process 8 f32s at a time with AVX2
                while i + 8 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 8];
                    let vec = _mm256_loadu_ps(slice.as_ptr());
                    sum_vec = _mm256_add_ps(sum_vec, vec);
                    i += 8;
                }

                // Horizontal sum of AVX2 register
                let high = _mm256_extractf128_ps(sum_vec, 1);
                let low = _mm256_castps256_ps128(sum_vec);
                let sum128 = _mm_add_ps(low, high);

                // Sum the 4 elements in the 128-bit register
                let shuf = _mm_shuffle_ps(sum128, sum128, 0b1110);
                let sum_partial = _mm_add_ps(sum128, shuf);
                let shuf2 = _mm_shuffle_ps(sum_partial, sum_partial, 0b0001);
                let final_sum = _mm_add_ps(sum_partial, shuf2);

                let mut result = _mm_cvtss_f32(final_sum);

                // Handle remaining elements
                for j in i..len {
                    result += input[j];
                }

                result
            }
        } else if is_x86_feature_detected!("sse") {
            unsafe {
                let mut sum_vec = _mm_setzero_ps();
                let mut i = 0;

                // Process 4 f32s at a time with SSE
                while i + 4 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 4];
                    let vec = _mm_loadu_ps(slice.as_ptr());
                    sum_vec = _mm_add_ps(sum_vec, vec);
                    i += 4;
                }

                // Horizontal sum of SSE register
                let shuf = _mm_shuffle_ps(sum_vec, sum_vec, 0b1110);
                let sum_partial = _mm_add_ps(sum_vec, shuf);
                let shuf2 = _mm_shuffle_ps(sum_partial, sum_partial, 0b0001);
                let final_sum = _mm_add_ps(sum_partial, shuf2);

                let mut result = _mm_cvtss_f32(final_sum);

                // Handle remaining elements
                for j in i..len {
                    result += input[j];
                }

                result
            }
        } else {
            // Scalar fallback with loop unrolling
            let mut sum0 = 0.0f32;
            let mut sum1 = 0.0f32;
            let mut sum2 = 0.0f32;
            let mut sum3 = 0.0f32;
            let mut i = 0;

            while i + 4 <= len {
                sum0 += input[i];
                sum1 += input[i + 1];
                sum2 += input[i + 2];
                sum3 += input[i + 3];
                i += 4;
            }

            let mut result = sum0 + sum1 + sum2 + sum3;
            for j in i..len {
                result += input[j];
            }
            result
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                let mut sum_vec = vdupq_n_f32(0.0);
                let mut i = 0;

                // Process 4 f32s at a time with NEON
                while i + 4 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 4];
                    let vec = vld1q_f32(slice.as_ptr());
                    sum_vec = vaddq_f32(sum_vec, vec);
                    i += 4;
                }

                // Horizontal sum of NEON register
                let sum_pair = vpadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
                let final_sum = vpadd_f32(sum_pair, sum_pair);
                let mut result = vget_lane_f32(final_sum, 0);

                // Handle remaining elements
                for j in i..len {
                    result += input[j];
                }

                result
            }
        } else {
            input.sum()
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        input.sum()
    }
}

/// SIMD-accelerated sum reduction for f64 arrays
///
/// Uses AVX2/SSE2/NEON for vectorized summation with horizontal reduction.
#[allow(dead_code)]
pub fn simd_sum_f64(input: &ArrayView1<f64>) -> f64 {
    let len = input.len();
    if len == 0 {
        return 0.0;
    }

    // Fall back to scalar sum if the array is not contiguous (e.g. column slices)
    if input.as_slice().is_none() {
        return input.iter().copied().sum();
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut sum_vec = _mm256_setzero_pd();
                let mut i = 0;

                // Process 4 f64s at a time with AVX2
                while i + 4 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 4];
                    let vec = _mm256_loadu_pd(slice.as_ptr());
                    sum_vec = _mm256_add_pd(sum_vec, vec);
                    i += 4;
                }

                // Horizontal sum of AVX2 register (4 f64s)
                let high = _mm256_extractf128_pd(sum_vec, 1);
                let low = _mm256_castpd256_pd128(sum_vec);
                let sum128 = _mm_add_pd(low, high);

                // Sum the 2 f64s in the 128-bit register
                let high64 = _mm_unpackhi_pd(sum128, sum128);
                let final_sum = _mm_add_sd(sum128, high64);

                let mut result = _mm_cvtsd_f64(final_sum);

                // Handle remaining elements
                for j in i..len {
                    result += input[j];
                }

                result
            }
        } else if is_x86_feature_detected!("sse2") {
            unsafe {
                let mut sum_vec = _mm_setzero_pd();
                let mut i = 0;

                // Process 2 f64s at a time with SSE2
                while i + 2 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 2];
                    let vec = _mm_loadu_pd(slice.as_ptr());
                    sum_vec = _mm_add_pd(sum_vec, vec);
                    i += 2;
                }

                // Horizontal sum of SSE2 register
                let high64 = _mm_unpackhi_pd(sum_vec, sum_vec);
                let final_sum = _mm_add_sd(sum_vec, high64);

                let mut result = _mm_cvtsd_f64(final_sum);

                // Handle remaining elements
                for j in i..len {
                    result += input[j];
                }

                result
            }
        } else {
            // Scalar fallback with loop unrolling
            let mut sum0 = 0.0f64;
            let mut sum1 = 0.0f64;
            let mut sum2 = 0.0f64;
            let mut sum3 = 0.0f64;
            let mut i = 0;

            while i + 4 <= len {
                sum0 += input[i];
                sum1 += input[i + 1];
                sum2 += input[i + 2];
                sum3 += input[i + 3];
                i += 4;
            }

            let mut result = sum0 + sum1 + sum2 + sum3;
            for j in i..len {
                result += input[j];
            }
            result
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                let mut sum_vec = vdupq_n_f64(0.0);
                let mut i = 0;

                // Process 2 f64s at a time with NEON
                while i + 2 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 2];
                    let vec = vld1q_f64(slice.as_ptr());
                    sum_vec = vaddq_f64(sum_vec, vec);
                    i += 2;
                }

                // Horizontal sum of NEON register (2 f64s)
                let low = vgetq_lane_f64(sum_vec, 0);
                let high = vgetq_lane_f64(sum_vec, 1);
                let mut result = low + high;

                // Handle remaining elements
                for j in i..len {
                    result += input[j];
                }

                result
            }
        } else {
            input.sum()
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        input.sum()
    }
}

/// SIMD-accelerated mean calculation for f32 arrays
///
/// Uses SIMD sum followed by division for efficient mean computation.
#[allow(dead_code)]
pub fn simd_mean_f32(input: &ArrayView1<f32>) -> f32 {
    let len = input.len();
    if len == 0 {
        return f32::NAN;
    }
    simd_sum_f32(input) / (len as f32)
}

/// SIMD-accelerated mean calculation for f64 arrays
///
/// Uses SIMD sum followed by division for efficient mean computation.
#[allow(dead_code)]
pub fn simd_mean_f64(input: &ArrayView1<f64>) -> f64 {
    let len = input.len();
    if len == 0 {
        return f64::NAN;
    }
    simd_sum_f64(input) / (len as f64)
}

/// SIMD-accelerated variance calculation for f32 arrays
///
/// Uses two-pass algorithm: first compute mean, then sum of squared differences.
#[allow(dead_code)]
pub fn simd_variance_f32(input: &ArrayView1<f32>) -> f32 {
    let len = input.len();
    if len < 2 {
        return f32::NAN;
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        let mean = input.iter().copied().sum::<f32>() / len as f32;
        return input
            .iter()
            .copied()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f32>()
            / (len - 1) as f32;
    }

    let mean = simd_mean_f32(input);

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mean_vec = _mm256_set1_ps(mean);
                let mut sum_sq_vec = _mm256_setzero_ps();
                let mut i = 0;

                // Process 8 f32s at a time with AVX2
                while i + 8 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 8];
                    let vec = _mm256_loadu_ps(slice.as_ptr());
                    let diff = _mm256_sub_ps(vec, mean_vec);
                    let sq = _mm256_mul_ps(diff, diff);
                    sum_sq_vec = _mm256_add_ps(sum_sq_vec, sq);
                    i += 8;
                }

                // Horizontal sum
                let high = _mm256_extractf128_ps(sum_sq_vec, 1);
                let low = _mm256_castps256_ps128(sum_sq_vec);
                let sum128 = _mm_add_ps(low, high);
                let shuf = _mm_shuffle_ps(sum128, sum128, 0b1110);
                let sum_partial = _mm_add_ps(sum128, shuf);
                let shuf2 = _mm_shuffle_ps(sum_partial, sum_partial, 0b0001);
                let final_sum = _mm_add_ps(sum_partial, shuf2);
                let mut sum_sq = _mm_cvtss_f32(final_sum);

                // Handle remaining elements
                for j in i..len {
                    let diff = input[j] - mean;
                    sum_sq += diff * diff;
                }

                sum_sq / (len as f32 - 1.0)
            }
        } else {
            // Scalar fallback
            let mut sum_sq = 0.0f32;
            for &x in input.iter() {
                let diff = x - mean;
                sum_sq += diff * diff;
            }
            sum_sq / (len as f32 - 1.0)
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                let mean_vec = vdupq_n_f32(mean);
                let mut sum_sq_vec = vdupq_n_f32(0.0);
                let mut i = 0;

                // Process 4 f32s at a time with NEON
                while i + 4 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 4];
                    let vec = vld1q_f32(slice.as_ptr());
                    let diff = vsubq_f32(vec, mean_vec);
                    let sq = vmulq_f32(diff, diff);
                    sum_sq_vec = vaddq_f32(sum_sq_vec, sq);
                    i += 4;
                }

                // Horizontal sum
                let sum_pair = vpadd_f32(vget_low_f32(sum_sq_vec), vget_high_f32(sum_sq_vec));
                let final_sum = vpadd_f32(sum_pair, sum_pair);
                let mut sum_sq = vget_lane_f32(final_sum, 0);

                // Handle remaining elements
                for j in i..len {
                    let diff = input[j] - mean;
                    sum_sq += diff * diff;
                }

                sum_sq / (len as f32 - 1.0)
            }
        } else {
            // Scalar fallback
            let mut sum_sq = 0.0f32;
            for &x in input.iter() {
                let diff = x - mean;
                sum_sq += diff * diff;
            }
            sum_sq / (len as f32 - 1.0)
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut sum_sq = 0.0f32;
        for &x in input.iter() {
            let diff = x - mean;
            sum_sq += diff * diff;
        }
        sum_sq / (len as f32 - 1.0)
    }
}

/// SIMD-accelerated variance calculation for f64 arrays
///
/// Uses two-pass algorithm: first compute mean, then sum of squared differences.
#[allow(dead_code)]
pub fn simd_variance_f64(input: &ArrayView1<f64>) -> f64 {
    let len = input.len();
    if len < 2 {
        return f64::NAN;
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        let mean = input.iter().copied().sum::<f64>() / len as f64;
        return input
            .iter()
            .copied()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f64>()
            / (len - 1) as f64;
    }

    let mean = simd_mean_f64(input);

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mean_vec = _mm256_set1_pd(mean);
                let mut sum_sq_vec = _mm256_setzero_pd();
                let mut i = 0;

                // Process 4 f64s at a time with AVX2
                while i + 4 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 4];
                    let vec = _mm256_loadu_pd(slice.as_ptr());
                    let diff = _mm256_sub_pd(vec, mean_vec);
                    let sq = _mm256_mul_pd(diff, diff);
                    sum_sq_vec = _mm256_add_pd(sum_sq_vec, sq);
                    i += 4;
                }

                // Horizontal sum
                let high = _mm256_extractf128_pd(sum_sq_vec, 1);
                let low = _mm256_castpd256_pd128(sum_sq_vec);
                let sum128 = _mm_add_pd(low, high);
                let high64 = _mm_unpackhi_pd(sum128, sum128);
                let final_sum = _mm_add_sd(sum128, high64);
                let mut sum_sq = _mm_cvtsd_f64(final_sum);

                // Handle remaining elements
                for j in i..len {
                    let diff = input[j] - mean;
                    sum_sq += diff * diff;
                }

                sum_sq / (len as f64 - 1.0)
            }
        } else {
            // Scalar fallback
            let mut sum_sq = 0.0f64;
            for &x in input.iter() {
                let diff = x - mean;
                sum_sq += diff * diff;
            }
            sum_sq / (len as f64 - 1.0)
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                let mean_vec = vdupq_n_f64(mean);
                let mut sum_sq_vec = vdupq_n_f64(0.0);
                let mut i = 0;

                // Process 2 f64s at a time with NEON
                while i + 2 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 2];
                    let vec = vld1q_f64(slice.as_ptr());
                    let diff = vsubq_f64(vec, mean_vec);
                    let sq = vmulq_f64(diff, diff);
                    sum_sq_vec = vaddq_f64(sum_sq_vec, sq);
                    i += 2;
                }

                // Horizontal sum
                let low = vgetq_lane_f64(sum_sq_vec, 0);
                let high = vgetq_lane_f64(sum_sq_vec, 1);
                let mut sum_sq = low + high;

                // Handle remaining elements
                for j in i..len {
                    let diff = input[j] - mean;
                    sum_sq += diff * diff;
                }

                sum_sq / (len as f64 - 1.0)
            }
        } else {
            // Scalar fallback
            let mut sum_sq = 0.0f64;
            for &x in input.iter() {
                let diff = x - mean;
                sum_sq += diff * diff;
            }
            sum_sq / (len as f64 - 1.0)
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut sum_sq = 0.0f64;
        for &x in input.iter() {
            let diff = x - mean;
            sum_sq += diff * diff;
        }
        sum_sq / (len as f64 - 1.0)
    }
}

/// SIMD-accelerated standard deviation for f32 arrays
#[allow(dead_code)]
pub fn simd_std_f32(input: &ArrayView1<f32>) -> f32 {
    simd_variance_f32(input).sqrt()
}

/// SIMD-accelerated standard deviation for f64 arrays
#[allow(dead_code)]
pub fn simd_std_f64(input: &ArrayView1<f64>) -> f64 {
    simd_variance_f64(input).sqrt()
}
pub fn simd_min_f32(input: &ArrayView1<f32>) -> f32 {
    let len = input.len();
    if len == 0 {
        return f32::INFINITY;
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.iter().copied().fold(f32::INFINITY, f32::min);
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                // Initialize with positive infinity
                let mut min_vec = _mm256_set1_ps(f32::INFINITY);
                let mut i = 0;

                // Process 8 f32s at a time
                while i + 8 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 8];
                    let vec = _mm256_loadu_ps(slice.as_ptr());
                    min_vec = _mm256_min_ps(min_vec, vec);
                    i += 8;
                }

                // Horizontal reduction: reduce 8 lanes to 1
                // Extract high and low 128-bit lanes
                let low = _mm256_castps256_ps128(min_vec);
                let high = _mm256_extractf128_ps(min_vec, 1);
                let min128 = _mm_min_ps(low, high);

                // Reduce 4 lanes to 2
                let shuffled = _mm_movehl_ps(min128, min128);
                let min64 = _mm_min_ps(min128, shuffled);

                // Reduce 2 lanes to 1
                let shuffled2 = _mm_shuffle_ps(min64, min64, 0x55);
                let min_scalar = _mm_min_ss(min64, shuffled2);
                let mut result = _mm_cvtss_f32(min_scalar);

                // Handle remaining elements
                for j in i..len {
                    let val = input[j];
                    if val < result {
                        result = val;
                    }
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse") {
            unsafe {
                let mut min_vec = _mm_set1_ps(f32::INFINITY);
                let mut i = 0;

                // Process 4 f32s at a time
                while i + 4 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 4];
                    let vec = _mm_loadu_ps(slice.as_ptr());
                    min_vec = _mm_min_ps(min_vec, vec);
                    i += 4;
                }

                // Horizontal reduction
                let shuffled = _mm_movehl_ps(min_vec, min_vec);
                let min64 = _mm_min_ps(min_vec, shuffled);
                let shuffled2 = _mm_shuffle_ps(min64, min64, 0x55);
                let min_scalar = _mm_min_ss(min64, shuffled2);
                let mut result = _mm_cvtss_f32(min_scalar);

                // Handle remaining elements
                for j in i..len {
                    let val = input[j];
                    if val < result {
                        result = val;
                    }
                }

                return result;
            }
        } else {
            // Scalar fallback
            let mut result = f32::INFINITY;
            for &x in input.iter() {
                if x < result {
                    result = x;
                }
            }
            return result;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let mut min_vec = vdupq_n_f32(f32::INFINITY);
            let mut i = 0;

            // Process 4 f32s at a time
            while i + 4 <= len {
                let slice = &input.as_slice().expect("Operation failed")[i..i + 4];
                let vec = vld1q_f32(slice.as_ptr());
                min_vec = vminq_f32(min_vec, vec);
                i += 4;
            }

            // Horizontal reduction using pairwise min
            let min64 = vpmin_f32(vget_low_f32(min_vec), vget_high_f32(min_vec));
            let min_pair = vpmin_f32(min64, min64);
            let mut result = vget_lane_f32(min_pair, 0);

            // Handle remaining elements
            for j in i..len {
                let val = input[j];
                if val < result {
                    result = val;
                }
            }

            return result;
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut result = f32::INFINITY;
        for &x in input.iter() {
            if x < result {
                result = x;
            }
        }
        result
    }
}

/// SIMD-accelerated minimum value for f64 arrays
///
/// Finds the minimum value in the array using SIMD horizontal reduction.
/// Returns f64::INFINITY for empty arrays.
#[allow(dead_code)]
pub fn simd_min_f64(input: &ArrayView1<f64>) -> f64 {
    let len = input.len();
    if len == 0 {
        return f64::INFINITY;
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.iter().copied().fold(f64::INFINITY, f64::min);
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                // Initialize with positive infinity
                let mut min_vec = _mm256_set1_pd(f64::INFINITY);
                let mut i = 0;

                // Process 4 f64s at a time
                while i + 4 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 4];
                    let vec = _mm256_loadu_pd(slice.as_ptr());
                    min_vec = _mm256_min_pd(min_vec, vec);
                    i += 4;
                }

                // Horizontal reduction: reduce 4 lanes to 1
                let low = _mm256_castpd256_pd128(min_vec);
                let high = _mm256_extractf128_pd(min_vec, 1);
                let min128 = _mm_min_pd(low, high);

                // Reduce 2 lanes to 1
                let high_lane = _mm_unpackhi_pd(min128, min128);
                let min_scalar = _mm_min_sd(min128, high_lane);
                let mut result = _mm_cvtsd_f64(min_scalar);

                // Handle remaining elements
                for j in i..len {
                    let val = input[j];
                    if val < result {
                        result = val;
                    }
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse2") {
            unsafe {
                let mut min_vec = _mm_set1_pd(f64::INFINITY);
                let mut i = 0;

                // Process 2 f64s at a time
                while i + 2 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 2];
                    let vec = _mm_loadu_pd(slice.as_ptr());
                    min_vec = _mm_min_pd(min_vec, vec);
                    i += 2;
                }

                // Horizontal reduction
                let high_lane = _mm_unpackhi_pd(min_vec, min_vec);
                let min_scalar = _mm_min_sd(min_vec, high_lane);
                let mut result = _mm_cvtsd_f64(min_scalar);

                // Handle remaining elements
                for j in i..len {
                    let val = input[j];
                    if val < result {
                        result = val;
                    }
                }

                return result;
            }
        } else {
            // Scalar fallback
            let mut result = f64::INFINITY;
            for &x in input.iter() {
                if x < result {
                    result = x;
                }
            }
            return result;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let mut min_vec = vdupq_n_f64(f64::INFINITY);
            let mut i = 0;

            // Process 2 f64s at a time
            while i + 2 <= len {
                let slice = &input.as_slice().expect("Operation failed")[i..i + 2];
                let vec = vld1q_f64(slice.as_ptr());
                min_vec = vminq_f64(min_vec, vec);
                i += 2;
            }

            // Horizontal reduction - extract lanes and compare
            let low = vgetq_lane_f64(min_vec, 0);
            let high = vgetq_lane_f64(min_vec, 1);
            let mut result = if low < high { low } else { high };

            // Handle remaining elements
            for j in i..len {
                let val = input[j];
                if val < result {
                    result = val;
                }
            }

            return result;
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut result = f64::INFINITY;
        for &x in input.iter() {
            if x < result {
                result = x;
            }
        }
        result
    }
}

/// SIMD-accelerated maximum value for f32 arrays
///
/// Finds the maximum value in the array using SIMD horizontal reduction.
/// Returns f32::NEG_INFINITY for empty arrays.
#[allow(dead_code)]
pub fn simd_max_f32(input: &ArrayView1<f32>) -> f32 {
    let len = input.len();
    if len == 0 {
        return f32::NEG_INFINITY;
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                // Initialize with negative infinity
                let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
                let mut i = 0;

                // Process 8 f32s at a time
                while i + 8 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 8];
                    let vec = _mm256_loadu_ps(slice.as_ptr());
                    max_vec = _mm256_max_ps(max_vec, vec);
                    i += 8;
                }

                // Horizontal reduction: reduce 8 lanes to 1
                let low = _mm256_castps256_ps128(max_vec);
                let high = _mm256_extractf128_ps(max_vec, 1);
                let max128 = _mm_max_ps(low, high);

                // Reduce 4 lanes to 2
                let shuffled = _mm_movehl_ps(max128, max128);
                let max64 = _mm_max_ps(max128, shuffled);

                // Reduce 2 lanes to 1
                let shuffled2 = _mm_shuffle_ps(max64, max64, 0x55);
                let max_scalar = _mm_max_ss(max64, shuffled2);
                let mut result = _mm_cvtss_f32(max_scalar);

                // Handle remaining elements
                for j in i..len {
                    let val = input[j];
                    if val > result {
                        result = val;
                    }
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse") {
            unsafe {
                let mut max_vec = _mm_set1_ps(f32::NEG_INFINITY);
                let mut i = 0;

                // Process 4 f32s at a time
                while i + 4 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 4];
                    let vec = _mm_loadu_ps(slice.as_ptr());
                    max_vec = _mm_max_ps(max_vec, vec);
                    i += 4;
                }

                // Horizontal reduction
                let shuffled = _mm_movehl_ps(max_vec, max_vec);
                let max64 = _mm_max_ps(max_vec, shuffled);
                let shuffled2 = _mm_shuffle_ps(max64, max64, 0x55);
                let max_scalar = _mm_max_ss(max64, shuffled2);
                let mut result = _mm_cvtss_f32(max_scalar);

                // Handle remaining elements
                for j in i..len {
                    let val = input[j];
                    if val > result {
                        result = val;
                    }
                }

                return result;
            }
        } else {
            // Scalar fallback
            let mut result = f32::NEG_INFINITY;
            for &x in input.iter() {
                if x > result {
                    result = x;
                }
            }
            return result;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);
            let mut i = 0;

            // Process 4 f32s at a time
            while i + 4 <= len {
                let slice = &input.as_slice().expect("Operation failed")[i..i + 4];
                let vec = vld1q_f32(slice.as_ptr());
                max_vec = vmaxq_f32(max_vec, vec);
                i += 4;
            }

            // Horizontal reduction using pairwise max
            let max64 = vpmax_f32(vget_low_f32(max_vec), vget_high_f32(max_vec));
            let max_pair = vpmax_f32(max64, max64);
            let mut result = vget_lane_f32(max_pair, 0);

            // Handle remaining elements
            for j in i..len {
                let val = input[j];
                if val > result {
                    result = val;
                }
            }

            return result;
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut result = f32::NEG_INFINITY;
        for &x in input.iter() {
            if x > result {
                result = x;
            }
        }
        result
    }
}

/// SIMD-accelerated maximum value for f64 arrays
///
/// Finds the maximum value in the array using SIMD horizontal reduction.
/// Returns f64::NEG_INFINITY for empty arrays.
#[allow(dead_code)]
pub fn simd_max_f64(input: &ArrayView1<f64>) -> f64 {
    let len = input.len();
    if len == 0 {
        return f64::NEG_INFINITY;
    }

    // Fall back to scalar for non-contiguous arrays (e.g. column slices)
    if input.as_slice().is_none() {
        return input.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    }

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                // Initialize with negative infinity
                let mut max_vec = _mm256_set1_pd(f64::NEG_INFINITY);
                let mut i = 0;

                // Process 4 f64s at a time
                while i + 4 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 4];
                    let vec = _mm256_loadu_pd(slice.as_ptr());
                    max_vec = _mm256_max_pd(max_vec, vec);
                    i += 4;
                }

                // Horizontal reduction: reduce 4 lanes to 1
                let low = _mm256_castpd256_pd128(max_vec);
                let high = _mm256_extractf128_pd(max_vec, 1);
                let max128 = _mm_max_pd(low, high);

                // Reduce 2 lanes to 1
                let high_lane = _mm_unpackhi_pd(max128, max128);
                let max_scalar = _mm_max_sd(max128, high_lane);
                let mut result = _mm_cvtsd_f64(max_scalar);

                // Handle remaining elements
                for j in i..len {
                    let val = input[j];
                    if val > result {
                        result = val;
                    }
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse2") {
            unsafe {
                let mut max_vec = _mm_set1_pd(f64::NEG_INFINITY);
                let mut i = 0;

                // Process 2 f64s at a time
                while i + 2 <= len {
                    let slice = &input.as_slice().expect("Operation failed")[i..i + 2];
                    let vec = _mm_loadu_pd(slice.as_ptr());
                    max_vec = _mm_max_pd(max_vec, vec);
                    i += 2;
                }

                // Horizontal reduction
                let high_lane = _mm_unpackhi_pd(max_vec, max_vec);
                let max_scalar = _mm_max_sd(max_vec, high_lane);
                let mut result = _mm_cvtsd_f64(max_scalar);

                // Handle remaining elements
                for j in i..len {
                    let val = input[j];
                    if val > result {
                        result = val;
                    }
                }

                return result;
            }
        } else {
            // Scalar fallback
            let mut result = f64::NEG_INFINITY;
            for &x in input.iter() {
                if x > result {
                    result = x;
                }
            }
            return result;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let mut max_vec = vdupq_n_f64(f64::NEG_INFINITY);
            let mut i = 0;

            // Process 2 f64s at a time
            while i + 2 <= len {
                let slice = &input.as_slice().expect("Operation failed")[i..i + 2];
                let vec = vld1q_f64(slice.as_ptr());
                max_vec = vmaxq_f64(max_vec, vec);
                i += 2;
            }

            // Horizontal reduction - extract lanes and compare
            let low = vgetq_lane_f64(max_vec, 0);
            let high = vgetq_lane_f64(max_vec, 1);
            let mut result = if low > high { low } else { high };

            // Handle remaining elements
            for j in i..len {
                let val = input[j];
                if val > result {
                    result = val;
                }
            }

            return result;
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut result = f64::NEG_INFINITY;
        for &x in input.iter() {
            if x > result {
                result = x;
            }
        }
        result
    }
}
