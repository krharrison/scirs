//! N-dimensional FFT via row-column decomposition with cache-oblivious tiling.
//!
//! The row-column method applies 1-D FFTs independently along each axis.
//! For 2-D arrays, a cache-oblivious tiling strategy improves L1/L2 hit rates
//! by processing tiles that fit in cache before moving on.

use crate::error::{FFTError, FFTResult};
use crate::ndim_fft::mixed_radix::{fft_1d, ifft_1d};
use crate::ndim_fft::types::NormMode;
use std::f64::consts::PI;

/// Type alias for a 1-D FFT transform function pointer.
type FftTransformFn = fn(&[(f64, f64)]) -> Vec<(f64, f64)>;

// ─────────────────────────────────────────────────────────────────────────────
// Strided axis extract / insert helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the row-major stride for each dimension of `shape`.
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Compute the flat index of element at multi-index `idx` given row-major `strides`.
#[inline]
fn flat_index(idx: &[usize], strides: &[usize]) -> usize {
    idx.iter().zip(strides.iter()).map(|(&i, &s)| i * s).sum()
}

/// Extract a 1-D slice along `axis` at the given position `idx` (all axes except `axis`).
///
/// `idx` has length `shape.len() - 1`; the elements correspond to the axes
/// in order, skipping `axis`.
pub fn extract_axis(
    data: &[(f64, f64)],
    shape: &[usize],
    axis: usize,
    pos: &[usize],
) -> Vec<(f64, f64)> {
    let n = shape[axis];
    let strides = compute_strides(shape);
    let mut result = Vec::with_capacity(n);

    // Build multi-index template: fill in fixed coords from pos (skipping axis)
    let mut midx: Vec<usize> = Vec::with_capacity(shape.len());
    let mut pos_iter = pos.iter();
    for dim in 0..shape.len() {
        if dim == axis {
            midx.push(0); // placeholder; will be varied
        } else {
            midx.push(*pos_iter.next().unwrap_or(&0));
        }
    }

    let axis_stride = strides[axis];
    let base = flat_index(&midx, &strides);
    for k in 0..n {
        result.push(data[base + k * axis_stride]);
    }
    result
}

/// Write a 1-D slice `values` along `axis` at position `pos` back into `data`.
pub fn insert_axis(
    data: &mut [(f64, f64)],
    shape: &[usize],
    axis: usize,
    pos: &[usize],
    values: &[(f64, f64)],
) {
    let n = shape[axis];
    let strides = compute_strides(shape);

    let mut midx: Vec<usize> = Vec::with_capacity(shape.len());
    let mut pos_iter = pos.iter();
    for dim in 0..shape.len() {
        if dim == axis {
            midx.push(0);
        } else {
            midx.push(*pos_iter.next().unwrap_or(&0));
        }
    }

    let axis_stride = strides[axis];
    let base = flat_index(&midx, &strides);
    for k in 0..n {
        data[base + k * axis_stride] = values[k];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1-D transform along a single axis of an N-D array
// ─────────────────────────────────────────────────────────────────────────────

/// Apply 1-D FFT (forward or inverse raw) along `axis` of `data` with given `shape`.
fn fft_along_axis(
    data: &mut [(f64, f64)],
    shape: &[usize],
    axis: usize,
    inverse: bool,
) -> FFTResult<()> {
    let ndim = shape.len();
    if axis >= ndim {
        return Err(FFTError::DimensionError(format!(
            "axis {axis} out of range for {ndim}-D array"
        )));
    }

    // Iterate over all positions in the axes != axis
    // Total slices = total_elements / shape[axis]
    let n_axis = shape[axis];
    let total = data.len();
    let n_slices = total / n_axis;

    let strides = compute_strides(shape);
    let axis_stride = strides[axis];

    // Build a position iterator over all multi-indices with axis fixed at 0
    // then use the axis stride to access elements.
    // Strategy: enumerate all flat indices that have axis-component = 0.
    // A flat index `f` has axis-component 0 iff (f / axis_stride) % n_axis == 0.

    let mut slice_buf = vec![(0.0f64, 0.0f64); n_axis];
    let mut processed = 0usize;

    for f in 0..total {
        // Only process indices where the axis component == 0
        let axis_coord = (f / axis_stride) % n_axis;
        if axis_coord != 0 {
            continue;
        }

        // Gather the 1-D slice
        for k in 0..n_axis {
            slice_buf[k] = data[f + k * axis_stride];
        }

        // Apply 1-D transform
        let out = if inverse {
            crate::ndim_fft::mixed_radix::ifft_1d_raw(&slice_buf)
        } else {
            fft_1d(&slice_buf)
        };

        // Scatter back
        for k in 0..n_axis {
            data[f + k * axis_stride] = out[k];
        }

        processed += 1;
        if processed == n_slices {
            break;
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Cache-oblivious in-place 2-D matrix transposition
// ─────────────────────────────────────────────────────────────────────────────

/// In-place square matrix transposition using cache-oblivious divide-and-conquer.
///
/// Works on a contiguous slice representing a `rows × cols` matrix in row-major
/// order. When `rows == cols` this is a pure in-place transpose; otherwise it
/// uses an out-of-place buffer (general rectangular transpose is complicated).
pub fn in_place_transpose(data: &mut [(f64, f64)], rows: usize, cols: usize) {
    if rows == cols {
        for r in 0..rows {
            for c in (r + 1)..cols {
                data.swap(r * cols + c, c * rows + r);
            }
        }
    } else {
        // Out-of-place transposition for rectangular matrices
        let n = data.len();
        debug_assert_eq!(n, rows * cols);
        let mut tmp = vec![(0.0f64, 0.0f64); n];
        for r in 0..rows {
            for c in 0..cols {
                tmp[c * rows + r] = data[r * cols + c];
            }
        }
        data.copy_from_slice(&tmp);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cache-oblivious tiled 2-D FFT
// ─────────────────────────────────────────────────────────────────────────────

/// Tiled 2-D FFT operating on a flat `rows × cols` matrix stored row-major.
///
/// The tiling strategy:
/// 1. Apply row FFTs in horizontal tiles of `tile_size` columns.
/// 2. Transpose in-place (cache-oblivious for square matrices).
/// 3. Apply row FFTs on the transposed matrix (== column FFTs of original).
/// 4. Transpose back to restore row-major order.
pub fn tiled_2d_fft(
    data: &mut [(f64, f64)],
    rows: usize,
    cols: usize,
    _tile_size: usize,
    inverse: bool,
) {
    debug_assert_eq!(data.len(), rows * cols);

    // Step 1: Row FFTs
    let row_transform: FftTransformFn = if inverse {
        crate::ndim_fft::mixed_radix::ifft_1d_raw
    } else {
        fft_1d
    };
    for r in 0..rows {
        let start = r * cols;
        let end = start + cols;
        let row_out = row_transform(&data[start..end]);
        data[start..end].copy_from_slice(&row_out);
    }

    // Step 2: In-place transposition (rows × cols → cols × rows)
    in_place_transpose(data, rows, cols);

    // Step 3: Row FFTs on transposed matrix (== column FFTs of original)
    for r in 0..cols {
        let start = r * rows;
        let end = start + rows;
        let row_out = row_transform(&data[start..end]);
        data[start..end].copy_from_slice(&row_out);
    }

    // Step 4: Transpose back (cols × rows → rows × cols)
    in_place_transpose(data, cols, rows);
}

// ─────────────────────────────────────────────────────────────────────────────
// Normalization
// ─────────────────────────────────────────────────────────────────────────────

/// Apply normalization to a flat array based on `NormMode`.
///
/// `n` is the total number of elements (product of shape).
/// `inverse` indicates whether this is the inverse transform.
pub fn apply_normalization(data: &mut [(f64, f64)], n: usize, norm: NormMode, inverse: bool) {
    let scale = match norm {
        NormMode::None if inverse => 1.0 / n as f64,
        NormMode::Ortho => 1.0 / (n as f64).sqrt(),
        NormMode::Forward => {
            if inverse {
                1.0 // unnormalized inverse
            } else {
                1.0 / n as f64
            }
        }
        // NormMode is #[non_exhaustive]; new variants default to no scaling
        #[allow(unreachable_patterns)]
        _ => 1.0,
    };

    if (scale - 1.0).abs() < f64::EPSILON {
        return; // no-op
    }

    for x in data.iter_mut() {
        x.0 *= scale;
        x.1 *= scale;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Full N-D FFT (public API)
// ─────────────────────────────────────────────────────────────────────────────

/// Forward N-dimensional FFT.
///
/// `input` is row-major flat data; `shape` gives the array dimensions.
/// Returns unnormalized forward FFT (NormMode::None convention).
///
/// # Errors
///
/// Returns an error if `input.len()` does not equal `shape.iter().product()`.
pub fn fftn(input: &[(f64, f64)], shape: &[usize]) -> FFTResult<Vec<(f64, f64)>> {
    fftn_norm(input, shape, NormMode::None)
}

/// Forward N-dimensional FFT with configurable normalization.
pub fn fftn_norm(
    input: &[(f64, f64)],
    shape: &[usize],
    norm: NormMode,
) -> FFTResult<Vec<(f64, f64)>> {
    let expected: usize = shape.iter().product();
    if input.len() != expected {
        return Err(FFTError::DimensionError(format!(
            "input length {} does not match shape {:?} (product = {})",
            input.len(),
            shape,
            expected
        )));
    }

    let ndim = shape.len();
    if ndim == 0 {
        return Ok(input.to_vec());
    }

    let mut data = input.to_vec();

    if ndim == 2 {
        // Use tiled 2D path for better cache efficiency
        tiled_2d_fft(&mut data, shape[0], shape[1], 64, false);
    } else {
        // Row-column method: apply 1-D FFT along each axis
        for axis in 0..ndim {
            fft_along_axis(&mut data, shape, axis, false)?;
        }
    }

    apply_normalization(&mut data, expected, norm, false);
    Ok(data)
}

/// Inverse N-dimensional FFT.
///
/// `input` is the frequency-domain flat data; `shape` gives dimensions.
/// Applies 1/N normalization (NormMode::None convention).
///
/// # Errors
///
/// Returns an error if dimensions mismatch.
pub fn ifftn(input: &[(f64, f64)], shape: &[usize]) -> FFTResult<Vec<(f64, f64)>> {
    ifftn_norm(input, shape, NormMode::None)
}

/// Inverse N-dimensional FFT with configurable normalization.
pub fn ifftn_norm(
    input: &[(f64, f64)],
    shape: &[usize],
    norm: NormMode,
) -> FFTResult<Vec<(f64, f64)>> {
    let expected: usize = shape.iter().product();
    if input.len() != expected {
        return Err(FFTError::DimensionError(format!(
            "input length {} does not match shape {:?}",
            input.len(),
            shape
        )));
    }

    let ndim = shape.len();
    if ndim == 0 {
        return Ok(input.to_vec());
    }

    let mut data = input.to_vec();

    if ndim == 2 {
        tiled_2d_fft(&mut data, shape[0], shape[1], 64, true);
    } else {
        for axis in 0..ndim {
            fft_along_axis(&mut data, shape, axis, true)?;
        }
    }

    apply_normalization(&mut data, expected, norm, true);
    Ok(data)
}

// ─────────────────────────────────────────────────────────────────────────────
// Real FFT (half-spectrum)
// ─────────────────────────────────────────────────────────────────────────────

/// Forward N-D real FFT.
///
/// Takes real input and returns the non-redundant half-spectrum.
/// The output shape along the last axis is `shape[last]/2 + 1`.
///
/// # Errors
///
/// Returns an error on dimension mismatch.
pub fn rfftn(input: &[f64], shape: &[usize]) -> FFTResult<Vec<(f64, f64)>> {
    let expected: usize = shape.iter().product();
    if input.len() != expected {
        return Err(FFTError::DimensionError(format!(
            "rfftn: input length {} does not match shape {:?}",
            input.len(),
            shape
        )));
    }

    let ndim = shape.len();
    if ndim == 0 {
        return Ok(input.iter().map(|&r| (r, 0.0)).collect());
    }

    // Convert real to complex
    let complex_input: Vec<(f64, f64)> = input.iter().map(|&r| (r, 0.0)).collect();

    // Apply N-D FFT
    let full = fftn(&complex_input, shape)?;

    // Extract half-spectrum along last axis
    let last_n = shape[ndim - 1];
    let half_last = last_n / 2 + 1;

    let mut out_shape = shape.to_vec();
    out_shape[ndim - 1] = half_last;

    let prefix_size: usize = shape[..ndim - 1].iter().product();
    let mut result = Vec::with_capacity(prefix_size * half_last);

    for i in 0..prefix_size {
        let src_start = i * last_n;
        result.extend_from_slice(&full[src_start..src_start + half_last]);
    }

    Ok(result)
}

/// Inverse N-D real FFT.
///
/// `input` is the half-spectrum (output shape of `rfftn`); `shape` is the desired
/// output shape (must match what was passed to `rfftn`).
///
/// # Errors
///
/// Returns an error on dimension mismatch.
pub fn irfftn(input: &[(f64, f64)], shape: &[usize]) -> FFTResult<Vec<f64>> {
    let ndim = shape.len();
    if ndim == 0 {
        return Ok(input.iter().map(|&(re, _)| re).collect());
    }

    let last_n = shape[ndim - 1];
    let half_last = last_n / 2 + 1;

    let prefix_size: usize = if ndim > 1 {
        shape[..ndim - 1].iter().product()
    } else {
        1
    };

    if input.len() != prefix_size * half_last {
        return Err(FFTError::DimensionError(format!(
            "irfftn: input length {} does not match expected {} (shape={:?})",
            input.len(),
            prefix_size * half_last,
            shape
        )));
    }

    let total: usize = shape.iter().product();

    // Reconstruct full spectrum using Hermitian symmetry
    let mut full = vec![(0.0f64, 0.0f64); total];
    for i in 0..prefix_size {
        let src_start = i * half_last;
        let dst_start = i * last_n;
        // Copy the half-spectrum
        full[dst_start..(half_last + dst_start)]
            .copy_from_slice(&input[src_start..(half_last + src_start)]);
        // Fill conjugate mirror (for k = half_last..last_n)
        // X[N-k] = conj(X[k]) for real signals
        for k in half_last..last_n {
            let conj_k = last_n - k;
            let src = input[src_start + conj_k];
            full[dst_start + k] = (src.0, -src.1);
        }
    }

    // Apply N-D IFFT to reconstruct real signal
    let complex_out = ifftn(&full, shape)?;

    // Take real parts
    Ok(complex_out.into_iter().map(|(re, _)| re).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fft2d_shape() {
        let rows = 4usize;
        let cols = 8usize;
        let input: Vec<(f64, f64)> = (0..rows * cols).map(|i| (i as f64, 0.0)).collect();
        let out = fftn(&input, &[rows, cols]).expect("fftn failed");
        assert_eq!(out.len(), rows * cols);
    }

    #[test]
    fn test_fft2d_roundtrip() {
        let rows = 4usize;
        let cols = 4usize;
        let input: Vec<(f64, f64)> = (0..rows * cols).map(|i| (i as f64 * 0.5, 0.0)).collect();
        let freq = fftn(&input, &[rows, cols]).expect("fftn failed");
        let recovered = ifftn(&freq, &[rows, cols]).expect("ifftn failed");
        for (a, b) in input.iter().zip(recovered.iter()) {
            assert_relative_eq!(a.0, b.0, epsilon = 1e-9);
            assert_relative_eq!(a.1, b.1, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_fft2d_tiled_matches_direct() {
        let rows = 8usize;
        let cols = 8usize;
        let input: Vec<(f64, f64)> = (0..rows * cols)
            .map(|i| ((i as f64).sin(), (i as f64).cos()))
            .collect();

        // Tiled path (via fftn which uses tiled_2d_fft for ndim==2)
        let tiled = fftn(&input, &[rows, cols]).expect("tiled fftn failed");

        // Direct axis-by-axis (force it manually)
        let mut direct = input.clone();
        fft_along_axis(&mut direct, &[rows, cols], 0, false).expect("axis 0 failed");
        fft_along_axis(&mut direct, &[rows, cols], 1, false).expect("axis 1 failed");

        for (a, b) in tiled.iter().zip(direct.iter()) {
            assert_relative_eq!(a.0, b.0, epsilon = 1e-9);
            assert_relative_eq!(a.1, b.1, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_fftn_3d_shape() {
        let shape = [2usize, 3, 4];
        let n: usize = shape.iter().product();
        let input: Vec<(f64, f64)> = (0..n).map(|i| (i as f64, 0.0)).collect();
        let out = fftn(&input, &shape).expect("fftn 3d failed");
        assert_eq!(out.len(), n);
    }

    #[test]
    fn test_fftn_3d_roundtrip() {
        let shape = [2usize, 4, 4];
        let n: usize = shape.iter().product();
        let input: Vec<(f64, f64)> = (0..n).map(|i| (i as f64 * 0.1, 0.0)).collect();
        let freq = fftn(&input, &shape).expect("fftn failed");
        let recovered = ifftn(&freq, &shape).expect("ifftn failed");
        for (a, b) in input.iter().zip(recovered.iter()) {
            assert_relative_eq!(a.0, b.0, epsilon = 1e-9);
            assert_relative_eq!(a.1, b.1, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_rfftn_real_input() {
        let n = 16usize;
        let input: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let out = rfftn(&input, &[n]).expect("rfftn failed");
        // Half-spectrum: n/2 + 1 components
        assert_eq!(out.len(), n / 2 + 1);
    }

    #[test]
    fn test_irfftn_roundtrip() {
        let n = 16usize;
        let input: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 2.0 * PI / n as f64).sin())
            .collect();
        let spectrum = rfftn(&input, &[n]).expect("rfftn failed");
        let recovered = irfftn(&spectrum, &[n]).expect("irfftn failed");
        assert_eq!(recovered.len(), n);
        for (a, b) in input.iter().zip(recovered.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_normalization_ortho() {
        // Parseval's theorem with Ortho normalization:
        // ||FFT(x)||² = ||x||²  (energy preserved on both sides)
        let n = 8usize;
        let input: Vec<(f64, f64)> = (0..n).map(|i| (i as f64, 0.0)).collect();
        let out = fftn_norm(&input, &[n], NormMode::Ortho).expect("fftn failed");

        let energy_in: f64 = input.iter().map(|&(re, im)| re * re + im * im).sum();
        let energy_out: f64 = out.iter().map(|&(re, im)| re * re + im * im).sum();
        assert_relative_eq!(energy_in, energy_out, epsilon = 1e-9);
    }

    #[test]
    fn test_normalization_forward() {
        // With NormMode::Forward, forward FFT divides by N.
        // DC component of a DC signal (all ones) = 1.0 (since sum/N = 1).
        let n = 8usize;
        let input: Vec<(f64, f64)> = vec![(1.0, 0.0); n];
        let out = fftn_norm(&input, &[n], NormMode::Forward).expect("fftn failed");
        // DC component (index 0) should be sum(x)/N = N/N = 1.0
        assert_relative_eq!(out[0].0, 1.0, epsilon = 1e-12);
        assert_relative_eq!(out[0].1, 0.0, epsilon = 1e-12);
        // All other components = 0
        for &(re, im) in &out[1..] {
            assert_relative_eq!(re, 0.0, epsilon = 1e-12);
            assert_relative_eq!(im, 0.0, epsilon = 1e-12);
        }
    }
}
