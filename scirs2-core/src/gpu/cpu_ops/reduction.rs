//! Reduction and broadcast CPU fallback implementations.
//!
//! Provides sum, mean, max, min (total and axis-wise), broadcast,
//! and scale operations used by the [`super::GpuContext`] CPU fallback.

use super::super::GpuError;
use super::traits::NumericOps;

// =============================================================================
// Reduction operations
// =============================================================================

/// CPU fallback for sum reduction
pub(super) fn sum_all_cpu<T: NumericOps>(data: &[T]) -> T {
    let mut acc = T::zero();
    for &val in data {
        acc = acc.add(val);
    }
    acc
}

/// CPU fallback for mean reduction
pub(super) fn mean_all_cpu<T: NumericOps>(data: &[T]) -> T {
    if data.is_empty() {
        return T::zero();
    }
    let sum = sum_all_cpu(data);
    let count = T::from_f64(data.len() as f64);
    // For float types, this divides. For int types, it truncates.
    T::from_f64(sum.to_f64() / count.to_f64())
}

/// CPU fallback for max reduction
pub(super) fn max_all_cpu<T: NumericOps>(data: &[T]) -> T {
    if data.is_empty() {
        return T::zero();
    }
    let mut max_val = data[0];
    for &val in &data[1..] {
        if max_val.partial_lt(val) {
            max_val = val;
        }
    }
    max_val
}

/// CPU fallback for min reduction
pub(super) fn min_all_cpu<T: NumericOps>(data: &[T]) -> T {
    if data.is_empty() {
        return T::zero();
    }
    let mut min_val = data[0];
    for &val in &data[1..] {
        if val.partial_lt(min_val) {
            min_val = val;
        }
    }
    min_val
}

/// CPU fallback for sum along an axis.
/// `data` is a tensor with shape `shape`, flattened in row-major order.
/// We sum along `axis` and produce output shape with shape[axis] = 1.
pub(super) fn sum_axis_cpu<T: NumericOps>(data: &[T], shape: &[usize], axis: usize) -> Vec<T> {
    let ndim = shape.len();
    let total: usize = shape.iter().product();

    // Compute output shape (same as input but axis dimension = 1)
    let mut output_shape = shape.to_vec();
    output_shape[axis] = 1;
    let output_total: usize = output_shape.iter().product();
    let mut output = vec![T::zero(); output_total];

    if total == 0 {
        return output;
    }

    // Compute strides for the input shape (row-major)
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Compute strides for the output shape
    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * output_shape[i + 1];
    }

    // For each element in the input, find its output index (axis coord = 0)
    // and accumulate
    for flat_idx in 0..total {
        // Convert flat index to multi-dimensional index
        let mut remaining = flat_idx;
        let mut out_flat_idx = 0;
        for d in 0..ndim {
            let coord = remaining / strides[d];
            remaining %= strides[d];
            if d != axis {
                out_flat_idx += coord * out_strides[d];
            }
            // For the axis dimension, out coord is always 0 (already in out_flat_idx init)
        }
        output[out_flat_idx] = output[out_flat_idx].add(data[flat_idx]);
    }

    output
}

/// CPU fallback for mean along an axis.
pub(super) fn mean_axis_cpu<T: NumericOps>(data: &[T], shape: &[usize], axis: usize) -> Vec<T> {
    let sum_result = sum_axis_cpu(data, shape, axis);
    let axis_size = shape[axis] as f64;
    if axis_size == 0.0 {
        return sum_result;
    }
    sum_result
        .iter()
        .map(|&v| T::from_f64(v.to_f64() / axis_size))
        .collect()
}

/// CPU fallback for max along an axis.
pub(super) fn max_axis_cpu<T: NumericOps>(data: &[T], shape: &[usize], axis: usize) -> Vec<T> {
    let ndim = shape.len();
    let total: usize = shape.iter().product();

    let mut output_shape = shape.to_vec();
    output_shape[axis] = 1;
    let output_total: usize = output_shape.iter().product();
    let mut output = vec![T::min_value(); output_total];

    if total == 0 {
        return vec![T::zero(); output_total];
    }

    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * output_shape[i + 1];
    }

    for flat_idx in 0..total {
        let mut remaining = flat_idx;
        let mut out_flat_idx = 0;
        for d in 0..ndim {
            let coord = remaining / strides[d];
            remaining %= strides[d];
            if d != axis {
                out_flat_idx += coord * out_strides[d];
            }
        }
        if output[out_flat_idx].partial_lt(data[flat_idx]) {
            output[out_flat_idx] = data[flat_idx];
        }
    }

    output
}

/// CPU fallback for min along an axis.
pub(super) fn min_axis_cpu<T: NumericOps>(data: &[T], shape: &[usize], axis: usize) -> Vec<T> {
    let ndim = shape.len();
    let total: usize = shape.iter().product();

    let mut output_shape = shape.to_vec();
    output_shape[axis] = 1;
    let output_total: usize = output_shape.iter().product();
    let mut output = vec![T::max_value(); output_total];

    if total == 0 {
        return vec![T::zero(); output_total];
    }

    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * output_shape[i + 1];
    }

    for flat_idx in 0..total {
        let mut remaining = flat_idx;
        let mut out_flat_idx = 0;
        for d in 0..ndim {
            let coord = remaining / strides[d];
            remaining %= strides[d];
            if d != axis {
                out_flat_idx += coord * out_strides[d];
            }
        }
        if data[flat_idx].partial_lt(output[out_flat_idx]) {
            output[out_flat_idx] = data[flat_idx];
        }
    }

    output
}

// =============================================================================
// Broadcast
// =============================================================================

/// CPU fallback for broadcasting.
/// Implements NumPy-style broadcasting rules.
pub(super) fn broadcast_cpu<T: NumericOps>(
    data: &[T],
    from_shape: &[usize],
    to_shape: &[usize],
) -> Result<Vec<T>, GpuError> {
    // Validate broadcast compatibility
    // Shapes are aligned from the trailing dimensions.
    // Each dimension must be either equal or 1 in the source.
    let from_ndim = from_shape.len();
    let to_ndim = to_shape.len();

    if from_ndim > to_ndim {
        return Err(GpuError::InvalidParameter(format!(
            "Cannot broadcast shape {:?} to shape {:?}: source has more dimensions",
            from_shape, to_shape
        )));
    }

    // Pad from_shape with leading 1s to match to_shape length
    let mut padded_from = vec![1usize; to_ndim];
    let offset = to_ndim - from_ndim;
    padded_from[offset..(from_ndim + offset)].copy_from_slice(&from_shape[..from_ndim]);

    // Validate compatibility
    for (i, (&f, &t)) in padded_from.iter().zip(to_shape.iter()).enumerate() {
        if f != 1 && f != t {
            return Err(GpuError::InvalidParameter(format!(
                "Cannot broadcast shape {:?} to shape {:?}: dimension {} is {} but target is {}",
                from_shape, to_shape, i, f, t
            )));
        }
    }

    let output_total: usize = to_shape.iter().product();
    let mut output = Vec::with_capacity(output_total);

    // Compute strides for the padded source shape
    let mut from_strides = vec![1usize; to_ndim];
    for i in (0..to_ndim.saturating_sub(1)).rev() {
        from_strides[i] = from_strides[i + 1] * padded_from[i + 1];
    }

    // Compute strides for the target shape
    let mut to_strides = vec![1usize; to_ndim];
    for i in (0..to_ndim.saturating_sub(1)).rev() {
        to_strides[i] = to_strides[i + 1] * to_shape[i + 1];
    }

    for flat_idx in 0..output_total {
        // Convert flat index to multi-dimensional coords in target shape
        let mut remaining = flat_idx;
        let mut src_flat_idx = 0;

        for d in 0..to_ndim {
            let coord = remaining / to_strides[d];
            remaining %= to_strides[d];

            // If source dimension is 1, clamp coord to 0 (broadcast)
            let src_coord = if padded_from[d] == 1 { 0 } else { coord };
            src_flat_idx += src_coord * from_strides[d];
        }

        output.push(data[src_flat_idx]);
    }

    Ok(output)
}

// =============================================================================
// Scale
// =============================================================================

/// CPU fallback for scalar multiplication
pub(super) fn scale_cpu<T: NumericOps>(data: &[T], scalar: T) -> Vec<T> {
    data.iter().map(|&x| x.mul(scalar)).collect()
}
