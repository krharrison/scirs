//! Parallel N-dimensional FFT using `std::thread::scope`.
//!
//! Splits the outermost dimension (rows for 2-D) across threads, applying
//! 1-D transforms independently in each partition.

use crate::error::{FFTError, FFTResult};
use crate::ndim_fft::mixed_radix::{fft_1d, ifft_1d_raw};
use crate::ndim_fft::ndim::{apply_normalization, compute_strides, tiled_2d_fft};
use crate::ndim_fft::types::NormMode;

/// Type alias for a 1-D FFT transform function pointer.
type FftTransformFn = fn(&[(f64, f64)]) -> Vec<(f64, f64)>;

// ─────────────────────────────────────────────────────────────────────────────
// Thread count helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Determine effective thread count.
///
/// `n_threads == 0` means auto-detect from `std::thread::available_parallelism`.
fn effective_threads(n_threads: usize) -> usize {
    if n_threads == 0 {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
    } else {
        n_threads.max(1)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel row FFTs
// ─────────────────────────────────────────────────────────────────────────────

/// Apply 1-D FFT (or raw IFFT) to each row of a `rows × cols` matrix in parallel.
///
/// The matrix is stored in row-major order in `data`.
/// `n_threads` controls parallelism (0 = auto).
pub fn parallel_fft_rows(
    data: &mut Vec<(f64, f64)>,
    rows: usize,
    cols: usize,
    n_threads: usize,
    inverse: bool,
) {
    debug_assert_eq!(data.len(), rows * cols);
    let threads = effective_threads(n_threads).min(rows);

    if threads <= 1 {
        // Serial fallback
        let transform: FftTransformFn = if inverse { ifft_1d_raw } else { fft_1d };
        for r in 0..rows {
            let start = r * cols;
            let end = start + cols;
            let out = transform(&data[start..end]);
            data[start..end].copy_from_slice(&out);
        }
        return;
    }

    // Partition rows among threads and run in parallel with thread::scope
    // We need mutable disjoint slices — split at row boundaries.
    let chunk_size = rows.div_ceil(threads);

    // Safety: each thread receives a disjoint mutable slice of the data.
    // We achieve this by splitting the flat slice at row boundaries.
    let row_slices: Vec<&mut [(f64, f64)]> = {
        let mut slices = Vec::new();
        let mut rest = data.as_mut_slice();
        let mut remaining_rows = rows;
        while remaining_rows > 0 {
            let this_chunk = chunk_size.min(remaining_rows);
            let (head, tail) = rest.split_at_mut(this_chunk * cols);
            slices.push(head);
            rest = tail;
            remaining_rows -= this_chunk;
        }
        slices
    };

    std::thread::scope(|s| {
        for chunk in row_slices {
            s.spawn(move || {
                let transform: FftTransformFn = if inverse { ifft_1d_raw } else { fft_1d };
                let row_count = chunk.len() / cols;
                for r in 0..row_count {
                    let start = r * cols;
                    let end = start + cols;
                    let out = transform(&chunk[start..end]);
                    chunk[start..end].copy_from_slice(&out);
                }
            });
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel 2-D FFT
// ─────────────────────────────────────────────────────────────────────────────

/// Parallel 2-D FFT using tiling + thread-scoped row parallelism.
///
/// The algorithm:
/// 1. Parallel row FFTs (`n_threads`).
/// 2. Sequential in-place transpose.
/// 3. Parallel column FFTs (treating transposed matrix as rows).
/// 4. Sequential transpose back.
///
/// # Errors
///
/// Returns an error if `data.len() != rows * cols`.
pub fn parallel_fft_2d(
    data: &mut Vec<(f64, f64)>,
    rows: usize,
    cols: usize,
    n_threads: usize,
    inverse: bool,
) -> FFTResult<()> {
    if data.len() != rows * cols {
        return Err(FFTError::DimensionError(format!(
            "parallel_fft_2d: data length {} != {} × {} = {}",
            data.len(),
            rows,
            cols,
            rows * cols
        )));
    }

    // Step 1: Parallel row FFTs
    parallel_fft_rows(data, rows, cols, n_threads, inverse);

    // Step 2: Transpose (rows × cols → cols × rows)
    crate::ndim_fft::ndim::in_place_transpose(data, rows, cols);

    // Step 3: Parallel column FFTs (now rows of the transposed matrix)
    parallel_fft_rows(data, cols, rows, n_threads, inverse);

    // Step 4: Transpose back
    crate::ndim_fft::ndim::in_place_transpose(data, cols, rows);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel N-D FFT
// ─────────────────────────────────────────────────────────────────────────────

/// Parallel N-dimensional forward FFT.
///
/// For 1-D and 2-D, uses optimized parallel paths.  For N > 2, parallelizes
/// the outermost axis: each thread handles a contiguous slab of the array
/// and applies (N-1)-D serial FFTs within its slab, followed by a parallel
/// axis-0 transform.
///
/// # Arguments
///
/// * `input`     — Flat row-major complex data.
/// * `shape`     — Array dimensions (product must equal `input.len()`).
/// * `n_threads` — Thread count (0 = auto-detect).
///
/// # Errors
///
/// Returns an error on dimension mismatch.
pub fn parallel_fftn(
    input: &[(f64, f64)],
    shape: &[usize],
    n_threads: usize,
) -> FFTResult<Vec<(f64, f64)>> {
    parallel_fftn_norm(input, shape, n_threads, NormMode::None, false)
}

/// Parallel N-dimensional inverse FFT (normalized by 1/N).
///
/// # Errors
///
/// Returns an error on dimension mismatch.
pub fn parallel_ifftn(
    input: &[(f64, f64)],
    shape: &[usize],
    n_threads: usize,
) -> FFTResult<Vec<(f64, f64)>> {
    parallel_fftn_norm(input, shape, n_threads, NormMode::None, true)
}

/// Parallel N-D FFT with configurable normalization and direction.
pub fn parallel_fftn_norm(
    input: &[(f64, f64)],
    shape: &[usize],
    n_threads: usize,
    norm: NormMode,
    inverse: bool,
) -> FFTResult<Vec<(f64, f64)>> {
    let expected: usize = shape.iter().product();
    if input.len() != expected {
        return Err(FFTError::DimensionError(format!(
            "parallel_fftn: input length {} != shape product {}",
            input.len(),
            expected
        )));
    }

    let ndim = shape.len();
    let threads = effective_threads(n_threads);

    match ndim {
        0 => return Ok(input.to_vec()),
        1 => {
            let mut data = input.to_vec();
            parallel_fft_rows(&mut data, 1, shape[0], threads, inverse);
            apply_normalization(&mut data, expected, norm, inverse);
            return Ok(data);
        }
        2 => {
            let mut data = input.to_vec();
            parallel_fft_2d(&mut data, shape[0], shape[1], threads, inverse)?;
            apply_normalization(&mut data, expected, norm, inverse);
            return Ok(data);
        }
        _ => {}
    }

    // General N-D case (N > 2):
    // Phase 1: Transform axes 1..N-1 within each outer slice in parallel.
    // Phase 2: Transform axis 0 serially (or parallelize the N-1-D slabs).

    let outer = shape[0];
    let inner_shape = &shape[1..];
    let inner_size: usize = inner_shape.iter().product();

    let mut data = input.to_vec();

    // Parallel: each thread handles a range of outer slices and applies
    // (N-1)-D serial FFT to each.
    let chunk_size = outer.div_ceil(threads.min(outer));
    {
        // Split data into outer-slab chunks
        let slab_slices: Vec<&mut [(f64, f64)]> = {
            let mut slices = Vec::new();
            let mut rest = data.as_mut_slice();
            let mut remaining = outer;
            while remaining > 0 {
                let this_chunk = chunk_size.min(remaining);
                let (head, tail) = rest.split_at_mut(this_chunk * inner_size);
                slices.push(head);
                rest = tail;
                remaining -= this_chunk;
            }
            slices
        };

        std::thread::scope(|s| {
            for slab in slab_slices {
                let inner_shape_ref = inner_shape;
                s.spawn(move || {
                    let slab_outer = slab.len() / inner_size;
                    for i in 0..slab_outer {
                        let slice_start = i * inner_size;
                        let slice_end = slice_start + inner_size;
                        let mut slice = slab[slice_start..slice_end].to_vec();
                        // Apply (N-1)-D serial FFT/IFFT
                        if let Ok(out) = if inverse {
                            crate::ndim_fft::ndim::ifftn(&slice, inner_shape_ref)
                        } else {
                            crate::ndim_fft::ndim::fftn(&slice, inner_shape_ref)
                        } {
                            // Undo the normalization applied by fftn/ifftn,
                            // since we'll normalize the whole array at the end.
                            let inner_n: usize = inner_shape_ref.iter().product();
                            let undo_scale = if inverse {
                                inner_n as f64 // undo the 1/inner_n from ifftn
                            } else {
                                1.0
                            };
                            slice = out;
                            if (undo_scale - 1.0).abs() > f64::EPSILON {
                                for x in slice.iter_mut() {
                                    x.0 *= undo_scale;
                                    x.1 *= undo_scale;
                                }
                            }
                        }
                        slab[slice_start..slice_end].copy_from_slice(&slice);
                    }
                });
            }
        });
    }

    // Phase 2: Transform along axis 0 (outermost)
    // For each inner index, gather elements along axis 0, FFT them, scatter back.
    let strides = compute_strides(shape);
    let axis_stride = strides[0]; // = inner_size

    for inner_idx in 0..inner_size {
        // Gather slice along axis 0
        let mut slice: Vec<(f64, f64)> = (0..outer)
            .map(|k| data[k * axis_stride + inner_idx])
            .collect();

        // Apply 1-D transform
        let out = if inverse {
            ifft_1d_raw(&slice)
        } else {
            fft_1d(&slice)
        };
        slice = out;

        // Scatter back
        for k in 0..outer {
            data[k * axis_stride + inner_idx] = slice[k];
        }
    }

    apply_normalization(&mut data, expected, norm, inverse);
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ndim_fft::ndim::fftn;
    use approx::assert_relative_eq;

    #[test]
    fn test_parallel_fft_matches_serial() {
        let rows = 8usize;
        let cols = 8usize;
        let input: Vec<(f64, f64)> = (0..rows * cols)
            .map(|i| ((i as f64 * 0.3).sin(), (i as f64 * 0.2).cos()))
            .collect();

        let serial = fftn(&input, &[rows, cols]).expect("serial fftn failed");
        let parallel = parallel_fftn(&input, &[rows, cols], 4).expect("parallel fftn failed");

        assert_eq!(serial.len(), parallel.len());
        for (a, b) in serial.iter().zip(parallel.iter()) {
            assert_relative_eq!(a.0, b.0, epsilon = 1e-9);
            assert_relative_eq!(a.1, b.1, epsilon = 1e-9);
        }
    }
}
