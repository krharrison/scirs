//! Python bindings for scirs2-ndimage
//!
//! This module provides Python bindings for N-dimensional image processing,
//! including filters, morphology, interpolation, and measurements.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

// NumPy types for Python array interface
use scirs2_numpy::{IntoPyArray, PyArray2, PyArrayMethods};

// Direct imports from scirs2-ndimage
use scirs2_ndimage::{
    // Analysis
    analysis::{image_entropy, peak_signal_to_noise_ratio, structural_similarity_index},
    // Features
    features::{canny, harris_corners},
    // Filters
    filters::{
        bilateral_filter, gaussian_filter, laplace, maximum_filter, median_filter, minimum_filter,
        sobel, uniform_filter, BorderMode,
    },
    // Interpolation
    interpolation::{rotate, shift, zoom},
    // Measurements
    measurements::{center_of_mass, moments},
    // Morphology
    morphology::{
        binary_closing, binary_dilation, binary_erosion, binary_opening, distance_transform_edt,
        grey_dilation, grey_erosion, label,
    },
    // Segmentation
    segmentation::{otsu_threshold, threshold_binary, watershed},
};

// ========================================
// FILTER OPERATIONS
// ========================================

/// Gaussian filter (blur)
#[pyfunction]
#[pyo3(signature = (input, sigma, mode="reflect"))]
fn gaussian_filter_py(
    py: Python,
    input: &Bound<'_, PyArray2<f64>>,
    sigma: f64,
    mode: &str,
) -> PyResult<Py<PyArray2<f64>>> {
    let binding = input.readonly();
    let data = binding.as_array().to_owned();

    let border_mode = match mode {
        "reflect" => BorderMode::Reflect,
        "constant" => BorderMode::Constant,
        "nearest" => BorderMode::Nearest,
        "mirror" => BorderMode::Mirror,
        "wrap" => BorderMode::Wrap,
        _ => return Err(PyValueError::new_err(format!("Unknown mode: {}", mode))),
    };

    let result = gaussian_filter(&data, sigma, Some(border_mode), None)
        .map_err(|e| PyRuntimeError::new_err(format!("Gaussian filter failed: {}", e)))?;

    Ok(result.into_pyarray(py).unbind())
}

/// Median filter
#[pyfunction]
#[pyo3(signature = (input, size, mode="reflect"))]
fn median_filter_py(
    py: Python,
    input: &Bound<'_, PyArray2<f64>>,
    size: usize,
    mode: &str,
) -> PyResult<Py<PyArray2<f64>>> {
    let binding = input.readonly();
    let data = binding.as_array().to_owned();

    let border_mode = match mode {
        "reflect" => BorderMode::Reflect,
        "constant" => BorderMode::Constant,
        "nearest" => BorderMode::Nearest,
        "mirror" => BorderMode::Mirror,
        "wrap" => BorderMode::Wrap,
        _ => return Err(PyValueError::new_err(format!("Unknown mode: {}", mode))),
    };

    // median_filter expects &[usize] for size
    let size_arr = [size, size];
    let result = median_filter(&data, &size_arr, Some(border_mode))
        .map_err(|e| PyRuntimeError::new_err(format!("Median filter failed: {}", e)))?;

    Ok(result.into_pyarray(py).unbind())
}

/// Uniform filter (box filter)
#[pyfunction]
#[pyo3(signature = (input, size, mode="reflect"))]
fn uniform_filter_py(
    py: Python,
    input: &Bound<'_, PyArray2<f64>>,
    size: usize,
    mode: &str,
) -> PyResult<Py<PyArray2<f64>>> {
    let binding = input.readonly();
    let data = binding.as_array().to_owned();

    let border_mode = match mode {
        "reflect" => BorderMode::Reflect,
        "constant" => BorderMode::Constant,
        "nearest" => BorderMode::Nearest,
        "mirror" => BorderMode::Mirror,
        "wrap" => BorderMode::Wrap,
        _ => return Err(PyValueError::new_err(format!("Unknown mode: {}", mode))),
    };

    // uniform_filter expects &[usize] for size and Option for mode and origin
    let size_arr = [size, size];
    let result = uniform_filter(&data, &size_arr, Some(border_mode), None)
        .map_err(|e| PyRuntimeError::new_err(format!("Uniform filter failed: {}", e)))?;

    Ok(result.into_pyarray(py).unbind())
}

/// Sobel edge detection
#[pyfunction]
#[pyo3(signature = (input, axis=0))]
fn sobel_py(
    py: Python,
    input: &Bound<'_, PyArray2<f64>>,
    axis: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let binding = input.readonly();
    let data = binding.as_array().to_owned();

    let result = sobel(&data, axis, None)
        .map_err(|e| PyRuntimeError::new_err(format!("Sobel filter failed: {}", e)))?;

    Ok(result.into_pyarray(py).unbind())
}

/// Laplacian filter
#[pyfunction]
fn laplace_py(py: Python, input: &Bound<'_, PyArray2<f64>>) -> PyResult<Py<PyArray2<f64>>> {
    let binding = input.readonly();
    let data = binding.as_array().to_owned();

    let result = laplace(&data, None, None)
        .map_err(|e| PyRuntimeError::new_err(format!("Laplace filter failed: {}", e)))?;

    Ok(result.into_pyarray(py).unbind())
}

/// Bilateral filter (edge-preserving smoothing)
#[pyfunction]
#[pyo3(signature = (input, sigma_spatial, sigma_intensity))]
fn bilateral_filter_py(
    py: Python,
    input: &Bound<'_, PyArray2<f64>>,
    sigma_spatial: f64,
    sigma_intensity: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let binding = input.readonly();
    let data = binding.as_array().to_owned();

    let result = bilateral_filter(&data, sigma_spatial, sigma_intensity, None)
        .map_err(|e| PyRuntimeError::new_err(format!("Bilateral filter failed: {}", e)))?;

    Ok(result.into_pyarray(py).unbind())
}

/// Maximum filter
#[pyfunction]
#[pyo3(signature = (input, size))]
fn maximum_filter_py(
    py: Python,
    input: &Bound<'_, PyArray2<f64>>,
    size: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let binding = input.readonly();
    let data = binding.as_array().to_owned();

    // maximum_filter expects (input, size, mode, origin) - 4 args
    let size_arr = [size, size];
    let result = maximum_filter(&data, &size_arr, None, None)
        .map_err(|e| PyRuntimeError::new_err(format!("Maximum filter failed: {}", e)))?;

    Ok(result.into_pyarray(py).unbind())
}

/// Minimum filter
#[pyfunction]
#[pyo3(signature = (input, size))]
fn minimum_filter_py(
    py: Python,
    input: &Bound<'_, PyArray2<f64>>,
    size: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let binding = input.readonly();
    let data = binding.as_array().to_owned();

    // minimum_filter expects (input, size, mode, origin) - 4 args
    let size_arr = [size, size];
    let result = minimum_filter(&data, &size_arr, None, None)
        .map_err(|e| PyRuntimeError::new_err(format!("Minimum filter failed: {}", e)))?;

    Ok(result.into_pyarray(py).unbind())
}

// ========================================
// MORPHOLOGICAL OPERATIONS
// ========================================

/// Binary erosion
#[pyfunction]
#[pyo3(signature = (input, iterations=1))]
fn binary_erosion_py(
    py: Python,
    input: &Bound<'_, PyArray2<u8>>,
    iterations: usize,
) -> PyResult<Py<PyArray2<u8>>> {
    let binding = input.readonly();
    let data = binding.as_array();

    // Convert to bool
    let bool_data = data.mapv(|x| x != 0);

    // Use the generic binary_erosion with all optional parameters as None
    let result = binary_erosion(&bool_data, None, Some(iterations), None, None, None, None)
        .map_err(|e| PyRuntimeError::new_err(format!("Binary erosion failed: {}", e)))?;

    // Convert back to u8
    let u8_result = result.mapv(|x| if x { 1u8 } else { 0u8 });
    Ok(u8_result.into_pyarray(py).unbind())
}

/// Binary dilation
#[pyfunction]
#[pyo3(signature = (input, iterations=1))]
fn binary_dilation_py(
    py: Python,
    input: &Bound<'_, PyArray2<u8>>,
    iterations: usize,
) -> PyResult<Py<PyArray2<u8>>> {
    let binding = input.readonly();
    let data = binding.as_array();

    let bool_data = data.mapv(|x| x != 0);

    // Use the generic binary_dilation with all optional parameters as None
    let result = binary_dilation(&bool_data, None, Some(iterations), None, None, None, None)
        .map_err(|e| PyRuntimeError::new_err(format!("Binary dilation failed: {}", e)))?;

    let u8_result = result.mapv(|x| if x { 1u8 } else { 0u8 });
    Ok(u8_result.into_pyarray(py).unbind())
}

/// Binary opening (erosion followed by dilation)
#[pyfunction]
#[pyo3(signature = (input, iterations=1))]
fn binary_opening_py(
    py: Python,
    input: &Bound<'_, PyArray2<u8>>,
    iterations: usize,
) -> PyResult<Py<PyArray2<u8>>> {
    let binding = input.readonly();
    let data = binding.as_array();

    let bool_data = data.mapv(|x| x != 0);

    // Use the generic binary_opening with all optional parameters as None
    let result = binary_opening(&bool_data, None, Some(iterations), None, None, None, None)
        .map_err(|e| PyRuntimeError::new_err(format!("Binary opening failed: {}", e)))?;

    let u8_result = result.mapv(|x| if x { 1u8 } else { 0u8 });
    Ok(u8_result.into_pyarray(py).unbind())
}

/// Binary closing (dilation followed by erosion)
#[pyfunction]
#[pyo3(signature = (input, iterations=1))]
fn binary_closing_py(
    py: Python,
    input: &Bound<'_, PyArray2<u8>>,
    iterations: usize,
) -> PyResult<Py<PyArray2<u8>>> {
    let binding = input.readonly();
    let data = binding.as_array();

    let bool_data = data.mapv(|x| x != 0);

    // Use the generic binary_closing with all optional parameters as None
    let result = binary_closing(&bool_data, None, Some(iterations), None, None, None, None)
        .map_err(|e| PyRuntimeError::new_err(format!("Binary closing failed: {}", e)))?;

    let u8_result = result.mapv(|x| if x { 1u8 } else { 0u8 });
    Ok(u8_result.into_pyarray(py).unbind())
}

/// Grayscale erosion
#[pyfunction]
#[pyo3(signature = (input, size=3))]
fn grey_erosion_py(
    py: Python,
    input: &Bound<'_, PyArray2<f64>>,
    size: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let binding = input.readonly();
    let data = binding.as_array().to_owned();

    // grey_erosion(input, size, structure, mode, cval, origin) - size is Option<&[usize]>
    let size_arr = [size, size];
    let result = grey_erosion(&data, Some(&size_arr), None, None, None, None)
        .map_err(|e| PyRuntimeError::new_err(format!("Grey erosion failed: {}", e)))?;

    Ok(result.into_pyarray(py).unbind())
}

/// Grayscale dilation
#[pyfunction]
#[pyo3(signature = (input, size=3))]
fn grey_dilation_py(
    py: Python,
    input: &Bound<'_, PyArray2<f64>>,
    size: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let binding = input.readonly();
    let data = binding.as_array().to_owned();

    // grey_dilation(input, size, structure, mode, cval, origin) - size is Option<&[usize]>
    let size_arr = [size, size];
    let result = grey_dilation(&data, Some(&size_arr), None, None, None, None)
        .map_err(|e| PyRuntimeError::new_err(format!("Grey dilation failed: {}", e)))?;

    Ok(result.into_pyarray(py).unbind())
}

/// Connected component labeling
#[pyfunction]
fn label_py(py: Python, input: &Bound<'_, PyArray2<u8>>) -> PyResult<Py<PyAny>> {
    let binding = input.readonly();
    let data = binding.as_array();

    let bool_data = data.mapv(|x| x != 0);
    // label(input, structure, connectivity, background) - 4 args
    let (labeled, num_features) = label(&bool_data, None, None, None)
        .map_err(|e| PyRuntimeError::new_err(format!("Label failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("labels", labeled.into_pyarray(py).unbind())?;
    dict.set_item("num_features", num_features)?;

    Ok(dict.into())
}

/// Euclidean distance transform
/// Note: Simplified implementation for 2D, converts result to 2D array
#[pyfunction]
fn distance_transform_edt_py(
    py: Python,
    input: &Bound<'_, PyArray2<u8>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let binding = input.readonly();
    let data = binding.as_array();
    let shape = data.raw_dim();

    let bool_data = data.mapv(|x| x != 0);
    // Convert to dynamic dimension for distance_transform_edt
    let bool_data_dyn = bool_data.into_dyn();

    // distance_transform_edt(input, sampling, return_distances, return_indices) - 4 args
    let (distances_opt, _indices_opt) =
        distance_transform_edt(&bool_data_dyn, None, true, false)
            .map_err(|e| PyRuntimeError::new_err(format!("Distance transform failed: {}", e)))?;

    // Extract distances and convert back to 2D
    let distances = distances_opt
        .ok_or_else(|| PyRuntimeError::new_err("Distance transform returned no distances"))?;
    let result = distances
        .into_shape_with_order(shape)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to reshape result: {}", e)))?;

    Ok(result.into_pyarray(py).unbind())
}

// ========================================
// GEOMETRIC TRANSFORMATIONS
// ========================================

/// Rotate image
#[pyfunction]
#[pyo3(signature = (input, angle, reshape=true, cval=0.0))]
fn rotate_py(
    py: Python,
    input: &Bound<'_, PyArray2<f64>>,
    angle: f64,
    reshape: bool,
    cval: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let binding = input.readonly();
    let data = binding.as_array().to_owned();

    // rotate(input, angle, axes, reshape, order, mode, cval, prefilter) - 8 args
    let result = rotate(
        &data,
        angle,
        None,
        Some(reshape),
        None,
        None,
        Some(cval),
        None,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Rotate failed: {}", e)))?;

    Ok(result.into_pyarray(py).unbind())
}

/// Zoom/rescale image
#[pyfunction]
#[pyo3(signature = (input, zoom_factor, cval=0.0))]
fn zoom_py(
    py: Python,
    input: &Bound<'_, PyArray2<f64>>,
    zoom_factor: f64,
    cval: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let binding = input.readonly();
    let data = binding.as_array().to_owned();

    // zoom(input, zoom_factor, order, mode, cval, prefilter) - 6 args
    let result = zoom(&data, zoom_factor, None, None, Some(cval), None)
        .map_err(|e| PyRuntimeError::new_err(format!("Zoom failed: {}", e)))?;

    Ok(result.into_pyarray(py).unbind())
}

/// Shift image
#[pyfunction]
#[pyo3(signature = (input, shift_values, cval=0.0))]
fn shift_py(
    py: Python,
    input: &Bound<'_, PyArray2<f64>>,
    shift_values: (f64, f64),
    cval: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let binding = input.readonly();
    let data = binding.as_array().to_owned();

    // shift(input, shift, order, mode, cval, prefilter) - 6 args
    let result = shift(
        &data,
        &[shift_values.0, shift_values.1],
        None,
        None,
        Some(cval),
        None,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Shift failed: {}", e)))?;

    Ok(result.into_pyarray(py).unbind())
}

// ========================================
// MEASUREMENTS
// ========================================

/// Calculate center of mass
#[pyfunction]
fn center_of_mass_py(py: Python, input: &Bound<'_, PyArray2<f64>>) -> PyResult<Py<PyAny>> {
    let binding = input.readonly();
    let data = binding.as_array().to_owned();

    let result = center_of_mass(&data)
        .map_err(|e| PyRuntimeError::new_err(format!("Center of mass failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("center", (result[0], result[1]))?;

    Ok(dict.into())
}

/// Calculate image moments
#[pyfunction]
#[pyo3(signature = (input, order=3))]
fn moments_py(py: Python, input: &Bound<'_, PyArray2<f64>>, order: usize) -> PyResult<Py<PyAny>> {
    let binding = input.readonly();
    let data = binding.as_array().to_owned();

    // moments(input, order) - returns 1D array of moment values
    let result = moments(&data, order)
        .map_err(|e| PyRuntimeError::new_err(format!("Moments failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("moments", result.into_pyarray(py).unbind())?;

    Ok(dict.into())
}

// ========================================
// SEGMENTATION
// ========================================

/// Watershed segmentation
#[pyfunction]
fn watershed_py(
    py: Python,
    input: &Bound<'_, PyArray2<f64>>,
    markers: &Bound<'_, PyArray2<i32>>,
) -> PyResult<Py<PyArray2<i32>>> {
    let input_binding = input.readonly();
    let input_data = input_binding.as_array().to_owned();
    let markers_binding = markers.readonly();
    let markers_data = markers_binding.as_array().to_owned();

    let result = watershed(&input_data, &markers_data)
        .map_err(|e| PyRuntimeError::new_err(format!("Watershed failed: {}", e)))?;

    Ok(result.into_pyarray(py).unbind())
}

/// Otsu's automatic thresholding
/// Returns the computed threshold value
#[pyfunction]
#[pyo3(signature = (input, bins=256))]
fn otsu_threshold_py(
    py: Python,
    input: &Bound<'_, PyArray2<f64>>,
    bins: usize,
) -> PyResult<Py<PyAny>> {
    let binding = input.readonly();
    let data = binding.as_array().to_owned();

    // otsu_threshold(image, bins) - returns (binarized_image, threshold_value)
    let (binarized, threshold_value) = otsu_threshold(&data, bins)
        .map_err(|e| PyRuntimeError::new_err(format!("Otsu threshold failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("threshold", threshold_value)?;
    dict.set_item("binarized", binarized.into_pyarray(py).unbind())?;

    Ok(dict.into())
}

/// Binary thresholding
/// Returns array with 1.0 where value > threshold, 0.0 otherwise
#[pyfunction]
fn threshold_binary_py(
    py: Python,
    input: &Bound<'_, PyArray2<f64>>,
    threshold: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let binding = input.readonly();
    let data = binding.as_array().to_owned();

    // threshold_binary returns Array<T, D> with T::one() or T::zero() (f64)
    let result = threshold_binary(&data, threshold)
        .map_err(|e| PyRuntimeError::new_err(format!("Threshold failed: {}", e)))?;

    Ok(result.into_pyarray(py).unbind())
}

// ========================================
// FEATURE DETECTION
// ========================================

/// Canny edge detection
/// Takes f32 array, returns edge map as f32 values
#[pyfunction]
#[pyo3(signature = (input, sigma=1.0, low_threshold=0.1, high_threshold=0.2))]
fn canny_py(
    py: Python,
    input: &Bound<'_, PyArray2<f32>>,
    sigma: f32,
    low_threshold: f32,
    high_threshold: f32,
) -> PyResult<Py<PyArray2<f32>>> {
    let binding = input.readonly();
    let data = binding.as_array().to_owned();

    // canny(image, sigma, low_threshold, high_threshold, method)
    let result = canny(&data, sigma, low_threshold, high_threshold, None);

    Ok(result.into_pyarray(py).unbind())
}

/// Harris corner detection
/// Takes f32 array, returns bool array of corner locations
#[pyfunction]
#[pyo3(signature = (input, block_size=2, k=0.04, threshold=0.01))]
fn harris_corners_py(
    py: Python,
    input: &Bound<'_, PyArray2<f32>>,
    block_size: usize,
    k: f32,
    threshold: f32,
) -> PyResult<Py<PyArray2<u8>>> {
    let binding = input.readonly();
    let data = binding.as_array().to_owned();

    // harris_corners(image, block_size, k, threshold) - returns Array<bool, Ix2>
    let result = harris_corners(&data, block_size, k, threshold);

    // Convert bool to u8
    let u8_result = result.mapv(|x| if x { 1u8 } else { 0u8 });
    Ok(u8_result.into_pyarray(py).unbind())
}

// ========================================
// IMAGE QUALITY METRICS
// ========================================

/// Peak signal-to-noise ratio
#[pyfunction]
fn psnr_py(image1: &Bound<'_, PyArray2<f64>>, image2: &Bound<'_, PyArray2<f64>>) -> PyResult<f64> {
    let img1_binding = image1.readonly();
    let img2_binding = image2.readonly();
    let img1_data = img1_binding.as_array();
    let img2_data = img2_binding.as_array();

    // peak_signal_to_noise_ratio takes ArrayView2 references
    peak_signal_to_noise_ratio(&img1_data, &img2_data)
        .map_err(|e| PyRuntimeError::new_err(format!("PSNR failed: {}", e)))
}

/// Structural similarity index (SSIM)
#[pyfunction]
fn ssim_py(image1: &Bound<'_, PyArray2<f64>>, image2: &Bound<'_, PyArray2<f64>>) -> PyResult<f64> {
    let img1_binding = image1.readonly();
    let img2_binding = image2.readonly();
    let img1_data = img1_binding.as_array();
    let img2_data = img2_binding.as_array();

    // structural_similarity_index takes ArrayView2 references
    structural_similarity_index(&img1_data, &img2_data)
        .map_err(|e| PyRuntimeError::new_err(format!("SSIM failed: {}", e)))
}

/// Image entropy
#[pyfunction]
fn image_entropy_py(input: &Bound<'_, PyArray2<f64>>) -> PyResult<f64> {
    let binding = input.readonly();
    let data = binding.as_array();

    // image_entropy takes ArrayView2 reference
    image_entropy(&data)
        .map_err(|e| PyRuntimeError::new_err(format!("Image entropy failed: {}", e)))
}

/// Python module registration
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Filters
    m.add_function(wrap_pyfunction!(gaussian_filter_py, m)?)?;
    m.add_function(wrap_pyfunction!(median_filter_py, m)?)?;
    m.add_function(wrap_pyfunction!(uniform_filter_py, m)?)?;
    m.add_function(wrap_pyfunction!(sobel_py, m)?)?;
    m.add_function(wrap_pyfunction!(laplace_py, m)?)?;
    m.add_function(wrap_pyfunction!(bilateral_filter_py, m)?)?;
    m.add_function(wrap_pyfunction!(maximum_filter_py, m)?)?;
    m.add_function(wrap_pyfunction!(minimum_filter_py, m)?)?;

    // Morphology
    m.add_function(wrap_pyfunction!(binary_erosion_py, m)?)?;
    m.add_function(wrap_pyfunction!(binary_dilation_py, m)?)?;
    m.add_function(wrap_pyfunction!(binary_opening_py, m)?)?;
    m.add_function(wrap_pyfunction!(binary_closing_py, m)?)?;
    m.add_function(wrap_pyfunction!(grey_erosion_py, m)?)?;
    m.add_function(wrap_pyfunction!(grey_dilation_py, m)?)?;
    m.add_function(wrap_pyfunction!(label_py, m)?)?;
    m.add_function(wrap_pyfunction!(distance_transform_edt_py, m)?)?;

    // Geometric transformations
    m.add_function(wrap_pyfunction!(rotate_py, m)?)?;
    m.add_function(wrap_pyfunction!(zoom_py, m)?)?;
    m.add_function(wrap_pyfunction!(shift_py, m)?)?;

    // Measurements
    m.add_function(wrap_pyfunction!(center_of_mass_py, m)?)?;
    m.add_function(wrap_pyfunction!(moments_py, m)?)?;

    // Segmentation
    m.add_function(wrap_pyfunction!(watershed_py, m)?)?;
    m.add_function(wrap_pyfunction!(otsu_threshold_py, m)?)?;
    m.add_function(wrap_pyfunction!(threshold_binary_py, m)?)?;

    // Feature detection
    m.add_function(wrap_pyfunction!(canny_py, m)?)?;
    m.add_function(wrap_pyfunction!(harris_corners_py, m)?)?;

    // Image quality metrics
    m.add_function(wrap_pyfunction!(psnr_py, m)?)?;
    m.add_function(wrap_pyfunction!(ssim_py, m)?)?;
    m.add_function(wrap_pyfunction!(image_entropy_py, m)?)?;

    Ok(())
}
