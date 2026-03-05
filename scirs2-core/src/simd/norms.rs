//! Vector norm operations with SIMD acceleration
//!
//! This module provides optimized implementations of vector norms
//! including L1 (Manhattan), L2 (Euclidean), and L-infinity (maximum) norms.

use ndarray::{Array1, ArrayView1};

/// Get a contiguous slice from an ArrayView1.
/// If the view is already contiguous, returns the slice directly.
/// Otherwise, copies elements to the provided buffer and returns a slice of it.
#[inline]
fn contiguous_f32<'a>(input: &'a ArrayView1<f32>, buf: &'a mut Vec<f32>) -> &'a [f32] {
    match input.as_slice() {
        Some(s) => s,
        None => {
            buf.clear();
            buf.extend(input.iter().copied());
            buf.as_slice()
        }
    }
}

/// Get a contiguous slice from an ArrayView1 of f64.
/// If the view is already contiguous, returns the slice directly.
/// Otherwise, copies elements to the provided buffer and returns a slice of it.
#[inline]
fn contiguous_f64<'a>(input: &'a ArrayView1<f64>, buf: &'a mut Vec<f64>) -> &'a [f64] {
    match input.as_slice() {
        Some(s) => s,
        None => {
            buf.clear();
            buf.extend(input.iter().copied());
            buf.as_slice()
        }
    }
}

pub fn simd_norm_l1_f32(input: &ArrayView1<f32>) -> f32 {
    let len = input.len();
    if len == 0 {
        return 0.0;
    }

    let mut buf = Vec::new();
    let input_slice = contiguous_f32(input, &mut buf);

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
                let mut sum_vec = _mm256_setzero_ps();
                let mut i = 0;

                while i + 8 <= len {
                    let vec = _mm256_loadu_ps(input_slice.as_ptr().add(i));
                    let abs_vec = _mm256_and_ps(vec, abs_mask);
                    sum_vec = _mm256_add_ps(sum_vec, abs_vec);
                    i += 8;
                }

                let low = _mm256_castps256_ps128(sum_vec);
                let high = _mm256_extractf128_ps(sum_vec, 1);
                let sum128 = _mm_add_ps(low, high);
                let shuffled = _mm_movehl_ps(sum128, sum128);
                let sum64 = _mm_add_ps(sum128, shuffled);
                let shuffled2 = _mm_shuffle_ps(sum64, sum64, 0x55);
                let sum_scalar = _mm_add_ss(sum64, shuffled2);
                let mut result = _mm_cvtss_f32(sum_scalar);

                for j in i..len {
                    result += input_slice[j].abs();
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse") {
            unsafe {
                let abs_mask = _mm_set1_ps(f32::from_bits(0x7FFF_FFFF));
                let mut sum_vec = _mm_setzero_ps();
                let mut i = 0;

                while i + 4 <= len {
                    let vec = _mm_loadu_ps(input_slice.as_ptr().add(i));
                    let abs_vec = _mm_and_ps(vec, abs_mask);
                    sum_vec = _mm_add_ps(sum_vec, abs_vec);
                    i += 4;
                }

                let shuffled = _mm_movehl_ps(sum_vec, sum_vec);
                let sum64 = _mm_add_ps(sum_vec, shuffled);
                let shuffled2 = _mm_shuffle_ps(sum64, sum64, 0x55);
                let sum_scalar = _mm_add_ss(sum64, shuffled2);
                let mut result = _mm_cvtss_f32(sum_scalar);

                for j in i..len {
                    result += input_slice[j].abs();
                }

                return result;
            }
        } else {
            let mut result = 0.0f32;
            for &x in input_slice.iter() {
                result += x.abs();
            }
            return result;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let mut sum_vec = vdupq_n_f32(0.0);
            let mut i = 0;

            while i + 4 <= len {
                let vec = vld1q_f32(input_slice.as_ptr().add(i));
                let abs_vec = vabsq_f32(vec);
                sum_vec = vaddq_f32(sum_vec, abs_vec);
                i += 4;
            }

            let sum64 = vpadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
            let sum_pair = vpadd_f32(sum64, sum64);
            let mut result = vget_lane_f32(sum_pair, 0);

            for j in i..len {
                result += input_slice[j].abs();
            }

            return result;
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut result = 0.0f32;
        for &x in input_slice.iter() {
            result += x.abs();
        }
        result
    }
}

/// SIMD-accelerated L1 norm (sum of absolute values) for f64 arrays
///
/// Computes ||x||_1 = |x_1| + |x_2| + ... + |x_n|
#[allow(dead_code)]
pub fn simd_norm_l1_f64(input: &ArrayView1<f64>) -> f64 {
    let len = input.len();
    if len == 0 {
        return 0.0;
    }

    let mut buf = Vec::new();
    let input_slice = contiguous_f64(input, &mut buf);

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let abs_mask = _mm256_set1_pd(f64::from_bits(0x7FFF_FFFF_FFFF_FFFF));
                let mut sum_vec = _mm256_setzero_pd();
                let mut i = 0;

                while i + 4 <= len {
                    let vec = _mm256_loadu_pd(input_slice.as_ptr().add(i));
                    let abs_vec = _mm256_and_pd(vec, abs_mask);
                    sum_vec = _mm256_add_pd(sum_vec, abs_vec);
                    i += 4;
                }

                let low = _mm256_castpd256_pd128(sum_vec);
                let high = _mm256_extractf128_pd(sum_vec, 1);
                let sum128 = _mm_add_pd(low, high);
                let high_lane = _mm_unpackhi_pd(sum128, sum128);
                let sum_scalar = _mm_add_sd(sum128, high_lane);
                let mut result = _mm_cvtsd_f64(sum_scalar);

                for j in i..len {
                    result += input_slice[j].abs();
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse2") {
            unsafe {
                let abs_mask = _mm_set1_pd(f64::from_bits(0x7FFF_FFFF_FFFF_FFFF));
                let mut sum_vec = _mm_setzero_pd();
                let mut i = 0;

                while i + 2 <= len {
                    let vec = _mm_loadu_pd(input_slice.as_ptr().add(i));
                    let abs_vec = _mm_and_pd(vec, abs_mask);
                    sum_vec = _mm_add_pd(sum_vec, abs_vec);
                    i += 2;
                }

                let high_lane = _mm_unpackhi_pd(sum_vec, sum_vec);
                let sum_scalar = _mm_add_sd(sum_vec, high_lane);
                let mut result = _mm_cvtsd_f64(sum_scalar);

                for j in i..len {
                    result += input_slice[j].abs();
                }

                return result;
            }
        } else {
            let mut result = 0.0f64;
            for &x in input_slice.iter() {
                result += x.abs();
            }
            return result;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let mut sum_vec = vdupq_n_f64(0.0);
            let mut i = 0;

            while i + 2 <= len {
                let vec = vld1q_f64(input_slice.as_ptr().add(i));
                let abs_vec = vabsq_f64(vec);
                sum_vec = vaddq_f64(sum_vec, abs_vec);
                i += 2;
            }

            let low = vgetq_lane_f64(sum_vec, 0);
            let high = vgetq_lane_f64(sum_vec, 1);
            let mut result = low + high;

            for j in i..len {
                result += input_slice[j].abs();
            }

            return result;
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut result = 0.0f64;
        for &x in input_slice.iter() {
            result += x.abs();
        }
        result
    }
}

/// SIMD-accelerated L2 norm (Euclidean norm) for f32 arrays
///
/// Computes ||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
#[allow(dead_code)]
pub fn simd_norm_l2_f32(input: &ArrayView1<f32>) -> f32 {
    let len = input.len();
    if len == 0 {
        return 0.0;
    }

    let mut buf = Vec::new();
    let input_slice = contiguous_f32(input, &mut buf);

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut sum_sq_vec = _mm256_setzero_ps();
                let mut i = 0;

                while i + 8 <= len {
                    let vec = _mm256_loadu_ps(input_slice.as_ptr().add(i));
                    let sq_vec = _mm256_mul_ps(vec, vec);
                    sum_sq_vec = _mm256_add_ps(sum_sq_vec, sq_vec);
                    i += 8;
                }

                let low = _mm256_castps256_ps128(sum_sq_vec);
                let high = _mm256_extractf128_ps(sum_sq_vec, 1);
                let sum128 = _mm_add_ps(low, high);
                let shuffled = _mm_movehl_ps(sum128, sum128);
                let sum64 = _mm_add_ps(sum128, shuffled);
                let shuffled2 = _mm_shuffle_ps(sum64, sum64, 0x55);
                let sum_scalar = _mm_add_ss(sum64, shuffled2);
                let mut sum_sq = _mm_cvtss_f32(sum_scalar);

                for j in i..len {
                    let val = input_slice[j];
                    sum_sq += val * val;
                }

                return sum_sq.sqrt();
            }
        } else if is_x86_feature_detected!("sse") {
            unsafe {
                let mut sum_sq_vec = _mm_setzero_ps();
                let mut i = 0;

                while i + 4 <= len {
                    let vec = _mm_loadu_ps(input_slice.as_ptr().add(i));
                    let sq_vec = _mm_mul_ps(vec, vec);
                    sum_sq_vec = _mm_add_ps(sum_sq_vec, sq_vec);
                    i += 4;
                }

                let shuffled = _mm_movehl_ps(sum_sq_vec, sum_sq_vec);
                let sum64 = _mm_add_ps(sum_sq_vec, shuffled);
                let shuffled2 = _mm_shuffle_ps(sum64, sum64, 0x55);
                let sum_scalar = _mm_add_ss(sum64, shuffled2);
                let mut sum_sq = _mm_cvtss_f32(sum_scalar);

                for j in i..len {
                    let val = input_slice[j];
                    sum_sq += val * val;
                }

                return sum_sq.sqrt();
            }
        } else {
            let mut sum_sq = 0.0f32;
            for &x in input_slice.iter() {
                sum_sq += x * x;
            }
            return sum_sq.sqrt();
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let mut sum_sq_vec = vdupq_n_f32(0.0);
            let mut i = 0;

            while i + 4 <= len {
                let vec = vld1q_f32(input_slice.as_ptr().add(i));
                let sq_vec = vmulq_f32(vec, vec);
                sum_sq_vec = vaddq_f32(sum_sq_vec, sq_vec);
                i += 4;
            }

            let sum64 = vpadd_f32(vget_low_f32(sum_sq_vec), vget_high_f32(sum_sq_vec));
            let sum_pair = vpadd_f32(sum64, sum64);
            let mut sum_sq = vget_lane_f32(sum_pair, 0);

            for j in i..len {
                let val = input_slice[j];
                sum_sq += val * val;
            }

            return sum_sq.sqrt();
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut sum_sq = 0.0f32;
        for &x in input_slice.iter() {
            sum_sq += x * x;
        }
        sum_sq.sqrt()
    }
}

/// SIMD-accelerated L2 norm (Euclidean norm) for f64 arrays
///
/// Computes ||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
#[allow(dead_code)]
pub fn simd_norm_l2_f64(input: &ArrayView1<f64>) -> f64 {
    let len = input.len();
    if len == 0 {
        return 0.0;
    }

    let mut buf = Vec::new();
    let input_slice = contiguous_f64(input, &mut buf);

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut sum_sq_vec = _mm256_setzero_pd();
                let mut i = 0;

                while i + 4 <= len {
                    let vec = _mm256_loadu_pd(input_slice.as_ptr().add(i));
                    let sq_vec = _mm256_mul_pd(vec, vec);
                    sum_sq_vec = _mm256_add_pd(sum_sq_vec, sq_vec);
                    i += 4;
                }

                let low = _mm256_castpd256_pd128(sum_sq_vec);
                let high = _mm256_extractf128_pd(sum_sq_vec, 1);
                let sum128 = _mm_add_pd(low, high);
                let high_lane = _mm_unpackhi_pd(sum128, sum128);
                let sum_scalar = _mm_add_sd(sum128, high_lane);
                let mut sum_sq = _mm_cvtsd_f64(sum_scalar);

                for j in i..len {
                    let val = input_slice[j];
                    sum_sq += val * val;
                }

                return sum_sq.sqrt();
            }
        } else if is_x86_feature_detected!("sse2") {
            unsafe {
                let mut sum_sq_vec = _mm_setzero_pd();
                let mut i = 0;

                while i + 2 <= len {
                    let vec = _mm_loadu_pd(input_slice.as_ptr().add(i));
                    let sq_vec = _mm_mul_pd(vec, vec);
                    sum_sq_vec = _mm_add_pd(sum_sq_vec, sq_vec);
                    i += 2;
                }

                let high_lane = _mm_unpackhi_pd(sum_sq_vec, sum_sq_vec);
                let sum_scalar = _mm_add_sd(sum_sq_vec, high_lane);
                let mut sum_sq = _mm_cvtsd_f64(sum_scalar);

                for j in i..len {
                    let val = input_slice[j];
                    sum_sq += val * val;
                }

                return sum_sq.sqrt();
            }
        } else {
            let mut sum_sq = 0.0f64;
            for &x in input_slice.iter() {
                sum_sq += x * x;
            }
            return sum_sq.sqrt();
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let mut sum_sq_vec = vdupq_n_f64(0.0);
            let mut i = 0;

            while i + 2 <= len {
                let vec = vld1q_f64(input_slice.as_ptr().add(i));
                let sq_vec = vmulq_f64(vec, vec);
                sum_sq_vec = vaddq_f64(sum_sq_vec, sq_vec);
                i += 2;
            }

            let low = vgetq_lane_f64(sum_sq_vec, 0);
            let high = vgetq_lane_f64(sum_sq_vec, 1);
            let mut sum_sq = low + high;

            for j in i..len {
                let val = input_slice[j];
                sum_sq += val * val;
            }

            return sum_sq.sqrt();
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut sum_sq = 0.0f64;
        for &x in input_slice.iter() {
            sum_sq += x * x;
        }
        sum_sq.sqrt()
    }
}

/// SIMD-accelerated L-infinity norm (maximum absolute value) for f32 arrays
///
/// Computes ||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
#[allow(dead_code)]
pub fn simd_norm_linf_f32(input: &ArrayView1<f32>) -> f32 {
    let len = input.len();
    if len == 0 {
        return 0.0;
    }

    let mut buf = Vec::new();
    let input_slice = contiguous_f32(input, &mut buf);

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
                let mut max_vec = _mm256_setzero_ps();
                let mut i = 0;

                while i + 8 <= len {
                    let vec = _mm256_loadu_ps(input_slice.as_ptr().add(i));
                    let abs_vec = _mm256_and_ps(vec, abs_mask);
                    max_vec = _mm256_max_ps(max_vec, abs_vec);
                    i += 8;
                }

                let low = _mm256_castps256_ps128(max_vec);
                let high = _mm256_extractf128_ps(max_vec, 1);
                let max128 = _mm_max_ps(low, high);
                let shuffled = _mm_movehl_ps(max128, max128);
                let max64 = _mm_max_ps(max128, shuffled);
                let shuffled2 = _mm_shuffle_ps(max64, max64, 0x55);
                let max_scalar = _mm_max_ss(max64, shuffled2);
                let mut result = _mm_cvtss_f32(max_scalar);

                for j in i..len {
                    let abs_val = input_slice[j].abs();
                    if abs_val > result {
                        result = abs_val;
                    }
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse") {
            unsafe {
                let abs_mask = _mm_set1_ps(f32::from_bits(0x7FFF_FFFF));
                let mut max_vec = _mm_setzero_ps();
                let mut i = 0;

                while i + 4 <= len {
                    let vec = _mm_loadu_ps(input_slice.as_ptr().add(i));
                    let abs_vec = _mm_and_ps(vec, abs_mask);
                    max_vec = _mm_max_ps(max_vec, abs_vec);
                    i += 4;
                }

                let shuffled = _mm_movehl_ps(max_vec, max_vec);
                let max64 = _mm_max_ps(max_vec, shuffled);
                let shuffled2 = _mm_shuffle_ps(max64, max64, 0x55);
                let max_scalar = _mm_max_ss(max64, shuffled2);
                let mut result = _mm_cvtss_f32(max_scalar);

                for j in i..len {
                    let abs_val = input_slice[j].abs();
                    if abs_val > result {
                        result = abs_val;
                    }
                }

                return result;
            }
        } else {
            let mut result = 0.0f32;
            for &x in input_slice.iter() {
                let abs_val = x.abs();
                if abs_val > result {
                    result = abs_val;
                }
            }
            return result;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let mut max_vec = vdupq_n_f32(0.0);
            let mut i = 0;

            while i + 4 <= len {
                let vec = vld1q_f32(input_slice.as_ptr().add(i));
                let abs_vec = vabsq_f32(vec);
                max_vec = vmaxq_f32(max_vec, abs_vec);
                i += 4;
            }

            let max64 = vpmax_f32(vget_low_f32(max_vec), vget_high_f32(max_vec));
            let max_pair = vpmax_f32(max64, max64);
            let mut result = vget_lane_f32(max_pair, 0);

            for j in i..len {
                let abs_val = input_slice[j].abs();
                if abs_val > result {
                    result = abs_val;
                }
            }

            return result;
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut result = 0.0f32;
        for &x in input_slice.iter() {
            let abs_val = x.abs();
            if abs_val > result {
                result = abs_val;
            }
        }
        result
    }
}

/// SIMD-accelerated L-infinity norm (maximum absolute value) for f64 arrays
///
/// Computes ||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
#[allow(dead_code)]
pub fn simd_norm_linf_f64(input: &ArrayView1<f64>) -> f64 {
    let len = input.len();
    if len == 0 {
        return 0.0;
    }

    let mut buf = Vec::new();
    let input_slice = contiguous_f64(input, &mut buf);

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                let abs_mask = _mm256_set1_pd(f64::from_bits(0x7FFF_FFFF_FFFF_FFFF));
                let mut max_vec = _mm256_setzero_pd();
                let mut i = 0;

                while i + 4 <= len {
                    let vec = _mm256_loadu_pd(input_slice.as_ptr().add(i));
                    let abs_vec = _mm256_and_pd(vec, abs_mask);
                    max_vec = _mm256_max_pd(max_vec, abs_vec);
                    i += 4;
                }

                let low = _mm256_castpd256_pd128(max_vec);
                let high = _mm256_extractf128_pd(max_vec, 1);
                let max128 = _mm_max_pd(low, high);
                let high_lane = _mm_unpackhi_pd(max128, max128);
                let max_scalar = _mm_max_sd(max128, high_lane);
                let mut result = _mm_cvtsd_f64(max_scalar);

                for j in i..len {
                    let abs_val = input_slice[j].abs();
                    if abs_val > result {
                        result = abs_val;
                    }
                }

                return result;
            }
        } else if is_x86_feature_detected!("sse2") {
            unsafe {
                let abs_mask = _mm_set1_pd(f64::from_bits(0x7FFF_FFFF_FFFF_FFFF));
                let mut max_vec = _mm_setzero_pd();
                let mut i = 0;

                while i + 2 <= len {
                    let vec = _mm_loadu_pd(input_slice.as_ptr().add(i));
                    let abs_vec = _mm_and_pd(vec, abs_mask);
                    max_vec = _mm_max_pd(max_vec, abs_vec);
                    i += 2;
                }

                let high_lane = _mm_unpackhi_pd(max_vec, max_vec);
                let max_scalar = _mm_max_sd(max_vec, high_lane);
                let mut result = _mm_cvtsd_f64(max_scalar);

                for j in i..len {
                    let abs_val = input_slice[j].abs();
                    if abs_val > result {
                        result = abs_val;
                    }
                }

                return result;
            }
        } else {
            let mut result = 0.0f64;
            for &x in input_slice.iter() {
                let abs_val = x.abs();
                if abs_val > result {
                    result = abs_val;
                }
            }
            return result;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        unsafe {
            let mut max_vec = vdupq_n_f64(0.0);
            let mut i = 0;

            while i + 2 <= len {
                let vec = vld1q_f64(input_slice.as_ptr().add(i));
                let abs_vec = vabsq_f64(vec);
                max_vec = vmaxq_f64(max_vec, abs_vec);
                i += 2;
            }

            let low = vgetq_lane_f64(max_vec, 0);
            let high = vgetq_lane_f64(max_vec, 1);
            let mut result = if low > high { low } else { high };

            for j in i..len {
                let abs_val = input_slice[j].abs();
                if abs_val > result {
                    result = abs_val;
                }
            }

            return result;
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let mut result = 0.0f64;
        for &x in input_slice.iter() {
            let abs_val = x.abs();
            if abs_val > result {
                result = abs_val;
            }
        }
        result
    }
}
