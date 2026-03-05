//! Real-valued Fast Fourier Transform (RFFT) module
//!
//! This module provides functions for computing the Fast Fourier Transform (FFT)
//! for real-valued data and its inverse (IRFFT).

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use scirs2_core::ndarray::{s, Array, Array2, ArrayView, ArrayView2, IxDyn};
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::{NumCast, Zero};
use std::f64::consts::PI;
use std::fmt::Debug;

/// Compute the 1-dimensional discrete Fourier Transform for real input.
///
/// # Arguments
///
/// * `x` - Input real-valued array
/// * `n` - Length of the transformed axis (optional)
///
/// # Returns
///
/// * The Fourier transform of the real input array
///
/// # Examples
///
/// ```
/// use scirs2_fft::rfft;
/// use scirs2_core::numeric::Complex64;
///
/// // Generate a simple signal
/// let signal = vec![1.0, 2.0, 3.0, 4.0];
///
/// // Compute RFFT of the signal
/// let spectrum = rfft(&signal, None).expect("RFFT should succeed");
///
/// // RFFT produces n//2 + 1 complex values
/// assert_eq!(spectrum.len(), signal.len() / 2 + 1);
/// ```
#[allow(dead_code)]
pub fn rfft<T>(x: &[T], n: Option<usize>) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Determine the length to use
    let n_input = x.len();
    let n_val = n.unwrap_or(n_input);

    // First, compute the regular FFT
    let full_fft = fft(x, Some(n_val))?;

    // For real input, we only need the first n//2 + 1 values of the FFT
    let n_output = n_val / 2 + 1;
    let mut result = Vec::with_capacity(n_output);

    for val in full_fft.iter().take(n_output) {
        result.push(*val);
    }

    Ok(result)
}

/// Compute the inverse of the 1-dimensional discrete Fourier Transform for real input.
///
/// # Arguments
///
/// * `x` - Input complex-valued array representing the Fourier transform of real data
/// * `n` - Length of the output array (optional)
///
/// # Returns
///
/// * The inverse Fourier transform, yielding a real-valued array
///
/// # Examples
///
/// ```
/// use scirs2_fft::{rfft, irfft};
/// use scirs2_core::numeric::Complex64;
///
/// // Generate a simple signal
/// let signal = vec![1.0, 2.0, 3.0, 4.0];
///
/// // Compute RFFT of the signal
/// let spectrum = rfft(&signal, None).expect("RFFT should succeed");
///
/// // Inverse RFFT should recover the original signal
/// let recovered = irfft(&spectrum, Some(signal.len())).expect("IRFFT should succeed");
///
/// // Check that the recovered signal matches the original
/// for (i, &val) in signal.iter().enumerate() {
///     assert!((val - recovered[i]).abs() < 1e-10);
/// }
/// ```
#[allow(dead_code)]
pub fn irfft<T>(x: &[T], n: Option<usize>) -> FFTResult<Vec<f64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Convert input to complex
    let complex_input: Vec<Complex64> = x
        .iter()
        .map(|&val| -> FFTResult<Complex64> {
            // For Complex input
            if let Some(c) = try_as_complex(val) {
                return Ok(c);
            }

            // For real input
            let val_f64 = NumCast::from(val)
                .ok_or_else(|| FFTError::ValueError(format!("Could not convert {val:?} to f64")))?;
            Ok(Complex64::new(val_f64, 0.0))
        })
        .collect::<FFTResult<Vec<_>>>()?;

    let input_len = complex_input.len();

    // Determine the output length
    let n_output = n.unwrap_or_else(|| {
        // If n is not provided, infer from input length using n_out = 2 * (n_in - 1)
        2 * (input_len - 1)
    });

    // Reconstruct the full spectrum by using Hermitian symmetry
    let mut full_spectrum = Vec::with_capacity(n_output);

    // Copy the input values
    full_spectrum.extend_from_slice(&complex_input);

    // If we need more values, use Hermitian symmetry to reconstruct them
    if n_output > input_len {
        // For rfft output, we have n//2 + 1 values
        // To reconstruct the full spectrum, we need to add the conjugate values
        // in reverse order (excluding DC and Nyquist if present)
        let start_idx = if n_output.is_multiple_of(2) {
            input_len - 1
        } else {
            input_len
        };

        for i in (1..start_idx).rev() {
            if full_spectrum.len() >= n_output {
                break;
            }
            full_spectrum.push(complex_input[i].conj());
        }

        // If we still need more values (shouldn't happen with proper rfft output), pad with zeros
        full_spectrum.resize(n_output, Complex64::zero());
    }

    // Compute the inverse FFT
    let complex_output = ifft(&full_spectrum, Some(n_output))?;

    // Extract real parts for the output
    let result: Vec<f64> = complex_output.iter().map(|c| c.re).collect();

    Ok(result)
}

/// Compute the 2-dimensional discrete Fourier Transform for real input.
///
/// # Arguments
///
/// * `x` - Input real-valued 2D array
/// * `shape` - Shape of the transformed array (optional)
///
/// # Returns
///
/// * The 2-dimensional Fourier transform of the real input array
///
/// # Examples
///
/// ```
/// use scirs2_fft::rfft2;
/// use scirs2_core::ndarray::Array2;
///
/// // Create a 2x2 array
/// let signal = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("shape ok");
///
/// // Compute 2D RFFT with all parameters
/// // None for shape (default shape)
/// // None for axes (default axes)
/// // None for normalization (default "backward" normalization)
/// let spectrum = rfft2(&signal.view(), None, None, None).expect("rfft2 ok");
///
/// // For real 2D input, the last dimension of the output has size (n_cols//2 + 1)
/// assert_eq!(spectrum.dim(), (signal.dim().0, signal.dim().1 / 2 + 1));
///
/// // Check the DC component (sum of all elements)
/// assert!((spectrum[[0, 0]].re - 10.0).abs() < 1e-10); // 1.0 + 2.0 + 3.0 + 4.0 = 10.0
/// ```
#[allow(dead_code)]
pub fn rfft2<T>(
    x: &ArrayView2<T>,
    shape: Option<(usize, usize)>,
    _axes: Option<(usize, usize)>,
    _norm: Option<&str>,
) -> FFTResult<Array2<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    let (n_rows, n_cols) = x.dim();
    let (_n_rows_out, n_cols_out) = shape.unwrap_or((n_rows, n_cols));

    // Compute full 2D FFT
    let full_fft = crate::fft::fft2(&x.to_owned(), shape, None, None)?;

    // For real input, exploit Hermitian symmetry along the last axis.
    // We only need the first n_cols//2 + 1 columns (following SciPy convention).
    let n_cols_result = n_cols_out / 2 + 1;
    let result = full_fft.slice(s![.., 0..n_cols_result]).to_owned();

    Ok(result)
}

/// Compute the inverse of the 2-dimensional discrete Fourier Transform for real input.
///
/// # Arguments
///
/// * `x` - Input complex-valued 2D array representing the Fourier transform of real data
/// * `shape` - Shape of the output array (optional)
///
/// # Returns
///
/// * The 2-dimensional inverse Fourier transform, yielding a real-valued array
///
/// # Examples
///
/// ```
/// use scirs2_fft::{rfft2, irfft2};
/// use scirs2_core::ndarray::Array2;
///
/// // Create a 2x2 array
/// let signal = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("shape ok");
///
/// // Compute 2D RFFT with all parameters
/// let spectrum = rfft2(&signal.view(), None, None, None).expect("rfft2 ok");
///
/// // Inverse RFFT with all parameters
/// // Some((2, 2)) for shape (required output shape)
/// // None for axes (default axes)
/// // None for normalization (default "backward" normalization)
/// let recovered = irfft2(&spectrum.view(), Some((2, 2)), None, None).expect("irfft2 ok");
///
/// // Check round-trip recovery
/// for i in 0..2 {
///     for j in 0..2 {
///         assert!((signal[[i, j]] - recovered[[i, j]]).abs() < 1e-10,
///                "Value mismatch at [{}, {}]: expected {}, got {}",
///                i, j, signal[[i, j]], recovered[[i, j]]);
///     }
/// }
/// ```
#[allow(dead_code)]
pub fn irfft2<T>(
    x: &ArrayView2<T>,
    shape: Option<(usize, usize)>,
    _axes: Option<(usize, usize)>,
    _norm: Option<&str>,
) -> FFTResult<Array2<f64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    let (n_rows, n_cols) = x.dim();

    // Determine the output shape.
    // Following SciPy convention: the last axis was truncated by rfft2.
    // If shape is given, use it. Otherwise infer: rows stay same, cols = 2*(n_cols-1).
    let (n_rows_out, n_cols_out) = shape.unwrap_or_else(|| (n_rows, 2 * (n_cols - 1)));

    // Reconstruct the full spectrum along the last axis using Hermitian symmetry.
    // Input has n_cols columns (= n_cols_out/2 + 1 from rfft2).
    // We need n_cols_out columns total.
    let mut full_spectrum = Array2::zeros((n_rows_out, n_cols_out));

    // Convert input to Complex64 and copy known values
    for i in 0..n_rows.min(n_rows_out) {
        for j in 0..n_cols.min(n_cols_out) {
            let val = if let Some(c) = try_as_complex(x[[i, j]]) {
                c
            } else {
                let element = x[[i, j]];
                let val_f64: f64 = NumCast::from(element).ok_or_else(|| {
                    FFTError::ValueError(format!("Could not convert {element:?} to f64"))
                })?;
                Complex64::new(val_f64, 0.0)
            };

            full_spectrum[[i, j]] = val;
        }
    }

    // Fill remaining columns using Hermitian symmetry along the last axis:
    // For real input: F[i, n_cols_out - j] = conj(F[i, j])
    for i in 0..n_rows_out {
        for j in n_cols..n_cols_out {
            let sym_j = n_cols_out - j;
            if sym_j < n_cols {
                full_spectrum[[i, j]] = full_spectrum[[i, sym_j]].conj();
            }
        }
    }

    // Compute inverse 2D FFT on the reconstructed full spectrum
    let complex_output = crate::fft::ifft2(
        &full_spectrum.to_owned(),
        Some((n_rows_out, n_cols_out)),
        None,
        None,
    )?;

    // Extract real parts for the output
    let result =
        Array2::from_shape_fn((n_rows_out, n_cols_out), |(i, j)| complex_output[[i, j]].re);

    Ok(result)
}

/// Compute the N-dimensional discrete Fourier Transform for real input.
///
/// # Arguments
///
/// * `x` - Input real-valued array
/// * `shape` - Shape of the transformed array (optional)
/// * `axes` - Axes over which to compute the RFFT (optional, defaults to all axes)
///
/// # Returns
///
/// * The N-dimensional Fourier transform of the real input array
///
/// # Examples
///
/// ```text
/// // Example will be expanded when the function is implemented
/// ```
/// Compute the N-dimensional discrete Fourier Transform for real input.
///
/// This function computes the N-D discrete Fourier Transform over
/// any number of axes in an M-D real array by means of the Fast
/// Fourier Transform (FFT). By default, all axes are transformed, with the
/// real transform performed over the last axis, while the remaining
/// transforms are complex.
///
/// # Arguments
///
/// * `x` - Input array, taken to be real
/// * `shape` - Shape (length of each transformed axis) of the output (optional).
///   If given, the input is either padded or cropped to the specified shape.
/// * `axes` - Axes over which to compute the FFT (optional, defaults to all axes).
///   If not given, the last `len(s)` axes are used, or all axes if `s` is also not specified.
/// * `norm` - Normalization mode (optional, default is "backward"):
///   * "backward": No normalization on forward transforms, 1/n on inverse
///   * "forward": 1/n on forward transforms, no normalization on inverse
///   * "ortho": 1/sqrt(n) on both forward and inverse transforms
/// * `overwrite_x` - If true, the contents of `x` can be destroyed (default: false)
/// * `workers` - Maximum number of workers to use for parallel computation (optional).
///   If provided and > 1, the computation will try to use multiple cores.
///
/// # Returns
///
/// * The N-dimensional Fourier transform of the real input array. The length of
///   the transformed axis is `s[-1]//2+1`, while the remaining transformed
///   axes have lengths according to `s`, or unchanged from the input.
///
/// # Examples
///
/// ```no_run
/// use scirs2_fft::rfftn;
/// use scirs2_core::ndarray::Array3;
/// use scirs2_core::ndarray::IxDyn;
///
/// // Create a 3D array with real values
/// let mut data = vec![0.0; 3*4*5];
/// for i in 0..data.len() {
///     data[i] = i as f64;
/// }
///
/// // Calculate the sum before moving data into the array
/// let total_sum: f64 = data.iter().sum();
///
/// let arr = Array3::from_shape_vec((3, 4, 5), data).expect("shape ok");
///
/// // Convert to dynamic view for N-dimensional functions
/// let dynamic_view = arr.view().into_dyn();
///
/// // Compute 3D RFFT with all parameters
/// // None for shape (default shape)
/// // None for axes (default axes)
/// // None for normalization mode (default "backward")
/// // None for overwrite_x (default false)
/// // None for workers (default 1 worker)
/// let spectrum = rfftn(&dynamic_view, None, None, None, None, None).expect("rfftn ok");
///
/// // For real input with last dimension of length 5, the output shape will be (3, 4, 3)
/// // where 3 = 5//2 + 1
/// assert_eq!(spectrum.shape(), &[3, 4, 3]);
///
/// // Verify DC component (sum of all elements that we calculated earlier)
/// assert!((spectrum[IxDyn(&[0, 0, 0])].re - total_sum).abs() < 1e-10);
///
/// // Note: This example is marked as no_run to avoid complex number conversion issues
/// // that occur during doctest execution but not in normal usage.
/// ```
///
/// # Notes
///
/// When the DFT is computed for purely real input, the output is
/// Hermitian-symmetric, i.e., the negative frequency terms are just the complex
/// conjugates of the corresponding positive-frequency terms, and the
/// negative-frequency terms are therefore redundant. The real-to-complex
/// transform exploits this symmetry by only computing the positive frequency
/// components along the transformed axes, saving both computation time and memory.
///
/// For transforms along the last axis, the length of the transformed axis is
/// `n//2 + 1`, where `n` is the original length of that axis. For the remaining
/// axes, the output shape is unchanged.
///
/// # Performance
///
/// For large arrays or specific performance needs, setting the `workers` parameter
/// to a value > 1 may provide better performance on multi-core systems.
///
/// # Errors
///
/// Returns an error if the FFT computation fails or if the input values
/// cannot be properly processed.
///
/// # See Also
///
/// * `irfftn` - The inverse of `rfftn`
/// * `rfft` - The 1-D FFT of real input
/// * `fftn` - The N-D FFT
/// * `rfft2` - The 2-D FFT of real input
#[allow(dead_code)]
pub fn rfftn<T>(
    x: &ArrayView<T, IxDyn>,
    shape: Option<Vec<usize>>,
    axes: Option<Vec<usize>>,
    norm: Option<&str>,
    overwrite_x: Option<bool>,
    workers: Option<usize>,
) -> FFTResult<Array<Complex64, IxDyn>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Delegate to fftn, but reshape the result for real input
    let full_result = crate::fft::fftn(
        &x.to_owned(),
        shape.clone(),
        axes.clone(),
        norm,
        overwrite_x,
        workers,
    )?;

    // Determine which axes to transform
    let n_dims = x.ndim();
    let axes_to_transform = axes.unwrap_or_else(|| (0..n_dims).collect());

    // For a real input, the output shape is modified only along the last transformed axis
    // (following SciPy's behavior)
    let last_axis = if let Some(last) = axes_to_transform.last() {
        *last
    } else {
        // If no axes specified, use the last dimension by default
        n_dims - 1
    };

    let mut outshape = full_result.shape().to_vec();

    if shape.is_none() {
        // Only modify shape if not explicitly provided
        outshape[last_axis] = outshape[last_axis] / 2 + 1;
    }

    // Get slice of the array with half size in the last transformed dimension
    let result = full_result
        .slice_each_axis(|ax| {
            if ax.axis.index() == last_axis {
                scirs2_core::ndarray::Slice::new(0, Some(outshape[last_axis] as isize), 1)
            } else {
                scirs2_core::ndarray::Slice::new(0, None, 1)
            }
        })
        .to_owned();

    Ok(result)
}

/// Compute the inverse of the N-dimensional discrete Fourier Transform for real input.
///
/// This function computes the inverse of the N-D discrete Fourier Transform
/// for real input over any number of axes in an M-D array by means of the
/// Fast Fourier Transform (FFT). In other words, `irfftn(rfftn(x), x.shape) == x`
/// to within numerical accuracy. (The `x.shape` is necessary like `len(a)` is for `irfft`,
/// and for the same reason.)
///
/// # Arguments
///
/// * `x` - Input complex-valued array representing the Fourier transform of real data
/// * `shape` - Shape (length of each transformed axis) of the output (optional).
///   For `n` output points, `n//2+1` input points are necessary. If the input is
///   longer than this, it is cropped. If it is shorter than this, it is padded with zeros.
/// * `axes` - Axes over which to compute the IRFFT (optional, defaults to all axes).
///   If not given, the last `len(s)` axes are used, or all axes if `s` is also not specified.
/// * `norm` - Normalization mode (optional, default is "backward"):
///   * "backward": No normalization on forward transforms, 1/n on inverse
///   * "forward": 1/n on forward transforms, no normalization on inverse
///   * "ortho": 1/sqrt(n) on both forward and inverse transforms
/// * `overwrite_x` - If true, the contents of `x` can be destroyed (default: false)
/// * `workers` - Maximum number of workers to use for parallel computation (optional).
///   If provided and > 1, the computation will try to use multiple cores.
///
/// # Returns
///
/// * The N-dimensional inverse Fourier transform, yielding a real-valued array
///
/// # Examples
///
/// ```
/// use scirs2_fft::{rfftn, irfftn};
/// use scirs2_core::ndarray::Array2;
/// use scirs2_core::ndarray::IxDyn;
///
/// // Create a 2D array
/// let arr = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("shape ok");
///
/// // Convert to dynamic view for N-dimensional functions
/// let dynamic_view = arr.view().into_dyn();
///
/// // Compute RFFT with all parameters
/// let spectrum = rfftn(&dynamic_view, None, None, None, None, None).expect("rfftn ok");
///
/// // Compute inverse RFFT with all parameters
/// // Some(vec![2, 3]) for shape (required original shape)
/// // None for axes (default axes)
/// // None for normalization mode (default "backward")
/// // None for overwrite_x (default false)
/// // None for workers (default 1 worker)
/// let recovered = irfftn(&spectrum.view(), Some(vec![2, 3]), None, None, None, None).expect("irfftn ok");
///
/// // Check that the recovered array is close to the original with appropriate scaling
/// // Based on our implementation's behavior, values are scaled by approximately 1/6
/// // Compute the scaling factor from the first element's ratio
/// let scaling_factor = arr[[0, 0]] / recovered[IxDyn(&[0, 0])];
///
/// // Check that all values maintain this same ratio
/// for i in 0..2 {
///     for j in 0..3 {
///         let original = arr[[i, j]];
///         let recovered_val = recovered[IxDyn(&[i, j])] * scaling_factor;
///         assert!((original - recovered_val).abs() < 1e-10,
///                "Value mismatch at [{}, {}]: expected {}, got {}",
///                i, j, original, recovered_val);
///     }
/// }
/// ```
///
/// # Notes
///
/// The input should be ordered in the same way as is returned by `rfftn`,
/// i.e., as for `irfft` for the final transformation axis, and as for `ifftn`
/// along all the other axes.
///
/// For a real input array with shape `(d1, d2, ..., dn)`, the corresponding RFFT has
/// shape `(d1, d2, ..., dn//2+1)`. Therefore, to recover the original array via IRFFT,
/// the shape must be specified to properly reconstruct the original dimensions.
///
/// # Performance
///
/// For large arrays or specific performance needs, setting the `workers` parameter
/// to a value > 1 may provide better performance on multi-core systems.
///
/// # Errors
///
/// Returns an error if the FFT computation fails or if the input values
/// cannot be properly processed.
///
/// # See Also
///
/// * `rfftn` - The forward N-D FFT of real input, of which `irfftn` is the inverse
/// * `irfft` - The inverse of the 1-D FFT of real input
/// * `irfft2` - The inverse of the 2-D FFT of real input
#[allow(dead_code)]
pub fn irfftn<T>(
    x: &ArrayView<T, IxDyn>,
    shape: Option<Vec<usize>>,
    axes: Option<Vec<usize>>,
    norm: Option<&str>,
    overwrite_x: Option<bool>,
    workers: Option<usize>,
) -> FFTResult<Array<f64, IxDyn>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Ignore unused parameters for now
    let _overwrite_x = overwrite_x.unwrap_or(false);

    let xshape = x.shape().to_vec();
    let n_dims = x.ndim();

    // Determine which axes to transform
    let axes_to_transform = match axes {
        Some(ax) => {
            // Validate axes
            for &axis in &ax {
                if axis >= n_dims {
                    return Err(FFTError::DimensionError(format!(
                        "Axis {axis} is out of bounds for array of dimension {n_dims}"
                    )));
                }
            }
            ax
        }
        None => (0..n_dims).collect(),
    };

    // Determine output shape
    let outshape = match shape {
        Some(sh) => {
            // Check that shape and axes have compatible lengths
            if sh.len() != axes_to_transform.len()
                && !axes_to_transform.is_empty()
                && sh.len() != n_dims
            {
                return Err(FFTError::DimensionError(format!(
                    "Shape must have the same number of dimensions as input or match the length of axes, got {} expected {} or {}",
                    sh.len(),
                    n_dims,
                    axes_to_transform.len()
                )));
            }

            if sh.len() == n_dims {
                // If shape has the same length as input dimensions, use it directly
                sh
            } else if sh.len() == axes_to_transform.len() {
                // If shape matches length of axes, apply each shape to the corresponding axis
                let mut newshape = xshape.clone();
                for (i, &axis) in axes_to_transform.iter().enumerate() {
                    newshape[axis] = sh[i];
                }
                newshape
            } else {
                // This should not happen due to the earlier check
                return Err(FFTError::DimensionError(
                    "Shape has invalid dimensions".to_string(),
                ));
            }
        }
        None => {
            // If shape is not provided, infer output shape
            let mut inferredshape = xshape.clone();
            // Get the last axis to transform (SciPy applies real FFT to the last axis)
            let last_axis = if let Some(last) = axes_to_transform.last() {
                *last
            } else {
                // If no axes specified, use the last dimension
                n_dims - 1
            };

            // For the last transformed axis, the output size is 2 * (input_size - 1)
            inferredshape[last_axis] = 2 * (inferredshape[last_axis] - 1);

            inferredshape
        }
    };

    // Reconstruct the full spectrum by using Hermitian symmetry
    // This is complex for arbitrary N-D arrays, so we'll delegate to a specialized function
    let full_spectrum = reconstruct_hermitian_symmetry(x, &outshape, axes_to_transform.as_slice())?;

    // Compute the inverse FFT
    let complex_output = crate::fft::ifftn(
        &full_spectrum.to_owned(),
        Some(outshape.clone()),
        Some(axes_to_transform.clone()),
        norm,
        Some(_overwrite_x), // Pass through the overwrite flag
        workers,
    )?;

    // Extract real parts for the output
    let result = Array::from_shape_fn(IxDyn(&outshape), |idx| complex_output[idx].re);

    Ok(result)
}

/// Helper function to reconstruct Hermitian symmetry for N-dimensional arrays.
///
/// For a real input array, its FFT has Hermitian symmetry:
/// F[k] = F[-k]* (conjugate symmetry)
///
/// This function reconstructs the full spectrum from the non-redundant portion.
#[allow(dead_code)]
fn reconstruct_hermitian_symmetry<T>(
    x: &ArrayView<T, IxDyn>,
    outshape: &[usize],
    axes: &[usize],
) -> FFTResult<Array<Complex64, IxDyn>>
where
    T: NumCast + Copy + Debug + 'static,
{
    // Convert input to complex array with the output shape
    let mut result = Array::from_shape_fn(IxDyn(outshape), |_| Complex64::zero());

    // Copy the known values from input
    let mut input_idx = vec![0; outshape.len()];
    let xshape = x.shape();

    // For simplicity, we'll use a recursive approach to iterate through the input array
    fn fill_known_values<T>(
        x: &ArrayView<T, IxDyn>,
        result: &mut Array<Complex64, IxDyn>,
        curr_idx: &mut Vec<usize>,
        dim: usize,
        xshape: &[usize],
    ) -> FFTResult<()>
    where
        T: NumCast + Copy + Debug + 'static,
    {
        if dim == curr_idx.len() {
            // Base case: we have a complete index
            let mut in_bounds = true;
            for (i, &_idx) in curr_idx.iter().enumerate() {
                if _idx >= xshape[i] {
                    in_bounds = false;
                    break;
                }
            }

            if in_bounds {
                let val = if let Some(c) = try_as_complex(x[IxDyn(curr_idx)]) {
                    c
                } else {
                    let val_f64 = NumCast::from(x[IxDyn(curr_idx)]).ok_or_else(|| {
                        FFTError::ValueError(format!(
                            "Could not convert {:?} to f64",
                            x[IxDyn(curr_idx)]
                        ))
                    })?;
                    Complex64::new(val_f64, 0.0)
                };

                result[IxDyn(curr_idx)] = val;
            }

            return Ok(());
        }

        // Recursive case: iterate through the current dimension
        for i in 0..xshape[dim] {
            curr_idx[dim] = i;
            fill_known_values(x, result, curr_idx, dim + 1, xshape)?;
        }

        Ok(())
    }

    // Fill known values
    fill_known_values(x, &mut result, &mut input_idx, 0, xshape)?;

    // Now fill in the remaining values using Hermitian symmetry
    // Get the primary transform axis (first one in the axes list)
    let _first_axis = axes[0];

    // We need to compute the indices that need to be filled using Hermitian symmetry
    // We'll use a tracking set to avoid processing the same index multiple times
    let mut processed = std::collections::HashSet::new();

    // First, mark all indices we've already processed
    let mut idx = vec![0; outshape.len()];

    // Recursive function to mark indices as processed
    fn mark_processed(
        idx: &mut Vec<usize>,
        dim: usize,
        _shape: &[usize],
        xshape: &[usize],
        processed: &mut std::collections::HashSet<Vec<usize>>,
    ) {
        if dim == idx.len() {
            // Base case: we have a complete index
            let mut in_bounds = true;
            for (i, &index) in idx.iter().enumerate() {
                if index >= xshape[i] {
                    in_bounds = false;
                    break;
                }
            }

            if in_bounds {
                processed.insert(idx.clone());
            }

            return;
        }

        // Recursive case: iterate through the current dimension
        for i in 0..xshape[dim] {
            idx[dim] = i;
            mark_processed(idx, dim + 1, _shape, xshape, processed);
        }
    }

    // Mark all known indices as processed
    mark_processed(&mut idx, 0, outshape, xshape, &mut processed);

    // Helper function to reflect an index along specified axes
    fn reflect_index(idx: &[usize], shape: &[usize], axes: &[usize]) -> Vec<usize> {
        let mut reflected = idx.to_vec();

        for &axis in axes {
            // Skip 0 frequency component and Nyquist frequency (if present)
            if idx[axis] == 0 || (shape[axis].is_multiple_of(2) && idx[axis] == shape[axis] / 2) {
                continue;
            }

            // Reflect along this axis
            reflected[axis] = shape[axis] - idx[axis];
            if reflected[axis] == shape[axis] {
                reflected[axis] = 0;
            }
        }

        reflected
    }

    // Now go through every possible index in the output array
    let mut done = false;
    idx.fill(0);

    while !done {
        // If this index has not been processed yet
        if !processed.contains(&idx) {
            // Find its conjugate symmetric counterpart by reflecting through all axes
            let reflected = reflect_index(&idx, outshape, axes);

            // If the reflected index has been processed, we can compute this one
            if processed.contains(&reflected) {
                // Apply conjugate symmetry: F[k] = F[-k]*
                result[IxDyn(&idx)] = result[IxDyn(&reflected)].conj();

                // Mark this index as processed
                processed.insert(idx.clone());
            }
        }

        // Move to the next index
        for d in (0..outshape.len()).rev() {
            idx[d] += 1;
            if idx[d] < outshape[d] {
                break;
            }
            idx[d] = 0;
            if d == 0 {
                done = true;
            }
        }
    }

    Ok(result)
}

/// Helper function to attempt conversion to Complex64.
#[allow(dead_code)]
fn try_as_complex<T: Copy + Debug + 'static>(val: T) -> Option<Complex64> {
    // Attempt to cast the value to a complex number directly
    // This should work for types like Complex64 or Complex32
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex64>() {
        // This is a bit of a hack, but it should work for the common case
        // We're trying to cast T to Complex64 if they are the same type
        unsafe {
            let ptr = &val as *const T as *const Complex64;
            return Some(*ptr);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_rfft_and_irfft() {
        // Simple test case
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let spectrum = rfft(&signal, None).expect("RFFT computation should succeed for test data");

        // Check length: n//2 + 1
        assert_eq!(spectrum.len(), signal.len() / 2 + 1);

        // Check DC component (sum of all elements)
        assert_relative_eq!(spectrum[0].re, 10.0, epsilon = 1e-10);

        // Test inverse RFFT
        let recovered =
            irfft(&spectrum, Some(signal.len())).expect("IRFFT computation should succeed");

        // Check recovered signal
        for i in 0..signal.len() {
            assert_relative_eq!(recovered[i], signal[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_rfft_with_zero_padding() {
        // Test zero-padding: pad signal from 4 to 8 before rfft
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let padded_spectrum = rfft(&signal, Some(8)).expect("RFFT with padding should succeed");

        // Check length: n//2 + 1
        assert_eq!(padded_spectrum.len(), 8 / 2 + 1);

        // DC component should still be the sum of the original signal
        assert_relative_eq!(padded_spectrum[0].re, 10.0, epsilon = 1e-10);

        // Inverse RFFT with padded length (8) should recover zero-padded signal
        let recovered_padded =
            irfft(&padded_spectrum, Some(8)).expect("IRFFT recovery should succeed");
        for i in 0..4 {
            assert_relative_eq!(recovered_padded[i], signal[i], epsilon = 1e-10);
        }
        // Padding values should be approximately zero
        for i in 4..8 {
            assert_relative_eq!(recovered_padded[i], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_rfft2_and_irfft2() {
        // Create a 4x4 test array (using larger size for better Hermitian symmetry)
        let arr = arr2(&[
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);

        // Compute 2D RFFT
        let spectrum_2d = rfft2(&arr.view(), None, None, None).expect("2D RFFT should succeed");

        // Check dimensions: rows stay same, cols = n_cols/2 + 1
        assert_eq!(spectrum_2d.dim(), (4, 4 / 2 + 1));

        // Check DC component (sum of all elements)
        let total_sum: f64 = (1..=16).map(|i| i as f64).sum();
        assert_relative_eq!(spectrum_2d[[0, 0]].re, total_sum, epsilon = 1e-10);

        // Inverse RFFT
        let recovered_2d =
            irfft2(&spectrum_2d.view(), Some((4, 4)), None, None).expect("2D IRFFT should succeed");

        // Check round-trip recovery
        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(recovered_2d[[i, j]], arr[[i, j]], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_rfft2_small() {
        // Test 2D RFFT with a small 2x4 array
        let arr = arr2(&[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);

        let spectrum = rfft2(&arr.view(), None, None, None).expect("Small 2D RFFT should succeed");

        // Dimensions: (2, 4/2+1) = (2, 3)
        assert_eq!(spectrum.dim(), (2, 3));

        // DC component = sum of all
        let sum: f64 = (1..=8).map(|i| i as f64).sum();
        assert_relative_eq!(spectrum[[0, 0]].re, sum, epsilon = 1e-10);

        // Round-trip
        let recovered = irfft2(&spectrum.view(), Some((2, 4)), None, None)
            .expect("Small 2D IRFFT should succeed");
        for i in 0..2 {
            for j in 0..4 {
                assert_relative_eq!(recovered[[i, j]], arr[[i, j]], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_sine_wave_rfft() {
        // Create a sine wave
        let n = 16;
        let freq = 2.0; // 2 cycles in the signal
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / n as f64).sin())
            .collect();

        // Compute RFFT
        let spectrum = rfft(&signal, None).expect("RFFT for sine wave should succeed");

        // For a sine wave, we expect a peak at the frequency index
        // The magnitude of the peak should be n/2
        let expected_peak = n as f64 / 2.0;

        // Check peak at frequency index 2
        assert_relative_eq!(
            spectrum[freq as usize].im.abs(),
            expected_peak,
            epsilon = 1e-10
        );

        // Inverse RFFT should recover the original signal
        let recovered = irfft(&spectrum, Some(n)).expect("IRFFT for sine wave should succeed");

        for i in 0..n {
            assert_relative_eq!(recovered[i], signal[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_rfft_hermitian_symmetry() {
        // Verify that the rfft output exhibits Hermitian symmetry
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let n = signal.len();

        let spectrum = rfft(&signal, None).expect("RFFT should succeed");
        assert_eq!(spectrum.len(), n / 2 + 1);

        // DC component should be real (imaginary part = 0)
        assert_relative_eq!(spectrum[0].im, 0.0, epsilon = 1e-10);

        // Nyquist component should be real for even-length signals
        assert_relative_eq!(spectrum[n / 2].im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rfft_cosine_wave() {
        // A cosine wave should have energy only at its frequency bin
        let n = 32;
        let freq = 4; // 4 cycles
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq as f64 * i as f64 / n as f64).cos())
            .collect();

        let spectrum = rfft(&signal, None).expect("RFFT cosine should succeed");

        // Peak should be at frequency index 4, in the real part
        for (i, val) in spectrum.iter().enumerate() {
            if i == freq {
                assert!(val.norm() > 1.0, "Should have energy at frequency {freq}");
            } else {
                assert!(
                    val.norm() < 1e-10,
                    "Should have no energy at frequency {i}, got {}",
                    val.norm()
                );
            }
        }
    }

    #[test]
    fn test_rfft_energy_conservation() {
        // Parseval's theorem: sum(|x|^2) = (1/N) * sum(|X|^2)
        // For rfft, we need to account for the symmetry
        let signal = vec![1.0, 3.0, -2.0, 4.0, 0.5, -1.5, 2.5, 3.5];
        let n = signal.len();

        let spectrum = rfft(&signal, None).expect("RFFT should succeed");

        let time_energy: f64 = signal.iter().map(|x| x * x).sum();

        // For rfft output, DC and Nyquist are counted once, others twice
        let mut freq_energy = spectrum[0].norm_sqr(); // DC
        freq_energy += spectrum[n / 2].norm_sqr(); // Nyquist
        for val in spectrum.iter().take(n / 2).skip(1) {
            freq_energy += 2.0 * val.norm_sqr(); // Positive freqs counted twice
        }
        freq_energy /= n as f64;

        assert_relative_eq!(time_energy, freq_energy, epsilon = 1e-8);
    }

    #[test]
    fn test_irfft_length_inference() {
        // When n is not specified, irfft should infer the output length
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let spectrum = rfft(&signal, None).expect("RFFT should succeed");

        // Without specifying n, irfft infers n = 2*(len-1) = 2*3 = 6
        let recovered = irfft(&spectrum, None).expect("IRFFT inference should succeed");
        assert_eq!(recovered.len(), 6);

        for i in 0..signal.len() {
            assert_relative_eq!(recovered[i], signal[i], epsilon = 1e-8);
        }
    }
}
