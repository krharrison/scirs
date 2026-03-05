use pyo3::prelude::*;
use pyo3::types::PyDict;
use scirs2_core::ndarray::Array2;
use scirs2_numpy::{IntoPyArray, PyArray2, PyArray3, PyArrayMethods};
use scirs2_vision::error::VisionError;

// Import vision functions
use image::{DynamicImage, GrayImage, ImageBuffer, Luma, Rgb};
use scirs2_vision::feature::canny::{canny, PreprocessMode};
use scirs2_vision::{
    bilateral_filter, clahe, detect_and_compute, equalize_histogram, find_homography,
    gaussian_blur, harris_corners, labels_to_color_image, laplacian_edges, median_filter,
    normalize_brightness, prewitt_edges, rgb_to_grayscale, rgb_to_hsv, sobel_edges, unsharp_mask,
    watershed,
};

// ============================================================================
// Helper Functions: NumPy ↔ DynamicImage conversion
// ============================================================================

/// Convert a NumPy array to a grayscale DynamicImage
/// Expects array with values in [0, 255] range
fn numpy_to_gray_image(arr: &Bound<'_, PyArray2<u8>>) -> Result<DynamicImage, VisionError> {
    let binding = arr.readonly();
    let array = binding.as_array();
    let (height, width) = array.dim();

    let mut img = GrayImage::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            img.put_pixel(x as u32, y as u32, Luma([array[[y, x]]]));
        }
    }

    Ok(DynamicImage::ImageLuma8(img))
}

/// Convert a NumPy array to an RGB DynamicImage
/// Expects array with shape (height, width, 3) and values in [0, 255] range
fn numpy_to_rgb_image(arr: &Bound<'_, PyArray3<u8>>) -> Result<DynamicImage, VisionError> {
    let binding = arr.readonly();
    let array = binding.as_array();
    let shape = array.shape();

    if shape.len() != 3 || shape[2] != 3 {
        return Err(VisionError::InvalidParameter(
            "Expected array with shape (height, width, 3)".to_string(),
        ));
    }

    let height = shape[0];
    let width = shape[1];

    let mut img = ImageBuffer::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            img.put_pixel(
                x as u32,
                y as u32,
                Rgb([array[[y, x, 0]], array[[y, x, 1]], array[[y, x, 2]]]),
            );
        }
    }

    Ok(DynamicImage::ImageRgb8(img))
}

/// Convert a grayscale DynamicImage to a NumPy array
fn gray_image_to_numpy(py: Python, img: &DynamicImage) -> Py<PyArray2<u8>> {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    let mut array = Array2::zeros((height as usize, width as usize));
    for y in 0..height {
        for x in 0..width {
            array[[y as usize, x as usize]] = gray.get_pixel(x, y)[0];
        }
    }

    array.into_pyarray(py).unbind()
}

/// Convert an RGB DynamicImage to a NumPy array with shape (height, width, 3)
fn rgb_image_to_numpy(py: Python, img: &DynamicImage) -> Py<PyArray3<u8>> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    let mut array = scirs2_core::ndarray::Array3::zeros((height as usize, width as usize, 3));
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);
            array[[y as usize, x as usize, 0]] = pixel[0];
            array[[y as usize, x as usize, 1]] = pixel[1];
            array[[y as usize, x as usize, 2]] = pixel[2];
        }
    }

    array.into_pyarray(py).unbind()
}

// ============================================================================
// Preprocessing Functions
// ============================================================================

/// Apply bilateral filtering for edge-preserving noise reduction
///
/// Bilateral filtering smooths images while preserving edges by considering
/// both spatial distance and intensity difference between pixels.
///
/// Args:
///     image (np.ndarray): Input grayscale image (2D uint8 array)
///     diameter (int): Diameter of pixel neighborhood (must be positive odd integer)
///     sigma_space (float): Standard deviation for spatial Gaussian kernel
///     sigma_color (float): Standard deviation for color/range Gaussian kernel
///
/// Returns:
///     np.ndarray: Filtered grayscale image (2D uint8 array)
#[pyfunction]
#[pyo3(signature = (image, diameter, sigma_space, sigma_color))]
fn bilateral_filter_py(
    py: Python,
    image: &Bound<'_, PyArray2<u8>>,
    diameter: u32,
    sigma_space: f32,
    sigma_color: f32,
) -> PyResult<Py<PyArray2<u8>>> {
    let img = numpy_to_gray_image(image).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Image conversion error: {}", e))
    })?;

    let filtered = bilateral_filter(&img, diameter, sigma_space, sigma_color).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Bilateral filter error: {}", e))
    })?;

    Ok(gray_image_to_numpy(py, &filtered))
}

/// Apply Gaussian blur to reduce noise
///
/// Args:
///     image (np.ndarray): Input grayscale image (2D uint8 array)
///     sigma (float): Standard deviation of Gaussian kernel (must be positive)
///
/// Returns:
///     np.ndarray: Blurred grayscale image (2D uint8 array)
#[pyfunction]
#[pyo3(signature = (image, sigma))]
fn gaussian_blur_py(
    py: Python,
    image: &Bound<'_, PyArray2<u8>>,
    sigma: f32,
) -> PyResult<Py<PyArray2<u8>>> {
    let img = numpy_to_gray_image(image).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Image conversion error: {}", e))
    })?;

    let blurred = gaussian_blur(&img, sigma).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Gaussian blur error: {}", e))
    })?;

    Ok(gray_image_to_numpy(py, &blurred))
}

/// Apply median filtering to remove salt-and-pepper noise
///
/// Args:
///     image (np.ndarray): Input grayscale image (2D uint8 array)
///     kernel_size (int): Size of square kernel (must be positive odd integer)
///
/// Returns:
///     np.ndarray: Filtered grayscale image (2D uint8 array)
#[pyfunction]
#[pyo3(signature = (image, kernel_size))]
fn median_filter_py(
    py: Python,
    image: &Bound<'_, PyArray2<u8>>,
    kernel_size: u32,
) -> PyResult<Py<PyArray2<u8>>> {
    let img = numpy_to_gray_image(image).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Image conversion error: {}", e))
    })?;

    let filtered = median_filter(&img, kernel_size).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Median filter error: {}", e))
    })?;

    Ok(gray_image_to_numpy(py, &filtered))
}

/// Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
///
/// Args:
///     image (np.ndarray): Input grayscale image (2D uint8 array)
///     tile_size (int): Size of grid tiles (typically 8)
///     clip_limit (float): Threshold for contrast limiting (1.0-4.0 typical)
///
/// Returns:
///     np.ndarray: Contrast-enhanced grayscale image (2D uint8 array)
#[pyfunction]
#[pyo3(signature = (image, tile_size=8, clip_limit=2.0))]
fn clahe_py(
    py: Python,
    image: &Bound<'_, PyArray2<u8>>,
    tile_size: u32,
    clip_limit: f32,
) -> PyResult<Py<PyArray2<u8>>> {
    let img = numpy_to_gray_image(image).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Image conversion error: {}", e))
    })?;

    let enhanced = clahe(&img, tile_size, clip_limit).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("CLAHE error: {}", e))
    })?;

    Ok(gray_image_to_numpy(py, &enhanced))
}

/// Apply histogram equalization to enhance contrast
///
/// Args:
///     image (np.ndarray): Input grayscale image (2D uint8 array)
///
/// Returns:
///     np.ndarray: Contrast-enhanced grayscale image (2D uint8 array)
#[pyfunction]
fn equalize_histogram_py(
    py: Python,
    image: &Bound<'_, PyArray2<u8>>,
) -> PyResult<Py<PyArray2<u8>>> {
    let img = numpy_to_gray_image(image).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Image conversion error: {}", e))
    })?;

    let equalized = equalize_histogram(&img).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Histogram equalization error: {}",
            e
        ))
    })?;

    Ok(gray_image_to_numpy(py, &equalized))
}

/// Normalize image brightness and contrast
///
/// Args:
///     image (np.ndarray): Input grayscale image (2D uint8 array)
///     min_out (float): Minimum output intensity (0.0 to 1.0)
///     max_out (float): Maximum output intensity (0.0 to 1.0)
///
/// Returns:
///     np.ndarray: Normalized grayscale image (2D uint8 array)
#[pyfunction]
#[pyo3(signature = (image, min_out=0.0, max_out=1.0))]
fn normalize_brightness_py(
    py: Python,
    image: &Bound<'_, PyArray2<u8>>,
    min_out: f32,
    max_out: f32,
) -> PyResult<Py<PyArray2<u8>>> {
    let img = numpy_to_gray_image(image).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Image conversion error: {}", e))
    })?;

    let normalized = normalize_brightness(&img, min_out, max_out).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Normalize brightness error: {}",
            e
        ))
    })?;

    Ok(gray_image_to_numpy(py, &normalized))
}

/// Apply unsharp masking to enhance edges
///
/// Args:
///     image (np.ndarray): Input grayscale image (2D uint8 array)
///     sigma (float): Standard deviation of Gaussian blur
///     amount (float): Strength of sharpening (typically 0.5 to 5.0)
///
/// Returns:
///     np.ndarray: Sharpened grayscale image (2D uint8 array)
#[pyfunction]
#[pyo3(signature = (image, sigma=1.0, amount=1.0))]
fn unsharp_mask_py(
    py: Python,
    image: &Bound<'_, PyArray2<u8>>,
    sigma: f32,
    amount: f32,
) -> PyResult<Py<PyArray2<u8>>> {
    let img = numpy_to_gray_image(image).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Image conversion error: {}", e))
    })?;

    let sharpened = unsharp_mask(&img, sigma, amount).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Unsharp mask error: {}", e))
    })?;

    Ok(gray_image_to_numpy(py, &sharpened))
}

// ============================================================================
// Color Conversion Functions
// ============================================================================

/// Convert RGB image to grayscale
///
/// Args:
///     image (np.ndarray): Input RGB image (3D uint8 array with shape (H, W, 3))
///     weights (Optional[list]): Custom RGB weights as [r_weight, g_weight, b_weight] (default: None for standard conversion)
///
/// Returns:
///     np.ndarray: Grayscale image (2D uint8 array)
#[pyfunction]
#[pyo3(signature = (image, weights=None))]
fn rgb_to_grayscale_py(
    py: Python,
    image: &Bound<'_, PyArray3<u8>>,
    weights: Option<[f32; 3]>,
) -> PyResult<Py<PyArray2<u8>>> {
    let img = numpy_to_rgb_image(image).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Image conversion error: {}", e))
    })?;

    let gray = rgb_to_grayscale(&img, weights).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("RGB to grayscale error: {}", e))
    })?;

    Ok(gray_image_to_numpy(py, &gray))
}

/// Convert RGB image to HSV color space
///
/// Args:
///     image (np.ndarray): Input RGB image (3D uint8 array with shape (H, W, 3))
///
/// Returns:
///     np.ndarray: HSV image (3D uint8 array with shape (H, W, 3))
#[pyfunction]
fn rgb_to_hsv_py(py: Python, image: &Bound<'_, PyArray3<u8>>) -> PyResult<Py<PyArray3<u8>>> {
    let img = numpy_to_rgb_image(image).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Image conversion error: {}", e))
    })?;

    let hsv = rgb_to_hsv(&img).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("RGB to HSV error: {}", e))
    })?;

    Ok(rgb_image_to_numpy(py, &hsv))
}

// ============================================================================
// Edge Detection Functions
// ============================================================================

/// Detect edges using Sobel operator
///
/// Args:
///     image (np.ndarray): Input grayscale image (2D uint8 array)
///     threshold (float): Edge detection threshold (0.0 to 1.0)
///
/// Returns:
///     np.ndarray: Edge map (2D uint8 array)
#[pyfunction]
#[pyo3(signature = (image, threshold=0.1))]
fn sobel_edges_py(
    py: Python,
    image: &Bound<'_, PyArray2<u8>>,
    threshold: f32,
) -> PyResult<Py<PyArray2<u8>>> {
    let img = numpy_to_gray_image(image).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Image conversion error: {}", e))
    })?;

    let edges = sobel_edges(&img, threshold).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Sobel edges error: {}", e))
    })?;

    Ok(gray_image_to_numpy(py, &DynamicImage::ImageLuma8(edges)))
}

/// Detect edges using Canny edge detector
///
/// Args:
///     image (np.ndarray): Input grayscale image (2D uint8 array)
///     sigma (float): Gaussian blur sigma (default: 1.4)
///     low_threshold (float): Low threshold for edge detection (0.0 to 1.0)
///     high_threshold (float): High threshold for edge detection (0.0 to 1.0)
///
/// Returns:
///     np.ndarray: Edge map (2D uint8 array)
#[pyfunction]
#[pyo3(signature = (image, sigma=1.4, low_threshold=0.05, high_threshold=0.15))]
fn canny_edges_py(
    py: Python,
    image: &Bound<'_, PyArray2<u8>>,
    sigma: f32,
    low_threshold: f32,
    high_threshold: f32,
) -> PyResult<Py<PyArray2<u8>>> {
    let img = numpy_to_gray_image(image).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Image conversion error: {}", e))
    })?;

    let edges = canny(
        &img,
        sigma,
        Some(low_threshold),
        Some(high_threshold),
        None,
        false,
        PreprocessMode::Reflect,
    )
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Canny edges error: {}", e))
    })?;

    Ok(gray_image_to_numpy(py, &DynamicImage::ImageLuma8(edges)))
}

/// Detect edges using Prewitt operator
///
/// Args:
///     image (np.ndarray): Input grayscale image (2D uint8 array)
///     threshold (float): Edge detection threshold (0.0 to 1.0)
///
/// Returns:
///     np.ndarray: Edge map (2D uint8 array)
#[pyfunction]
#[pyo3(signature = (image, threshold=0.1))]
fn prewitt_edges_py(
    py: Python,
    image: &Bound<'_, PyArray2<u8>>,
    threshold: f32,
) -> PyResult<Py<PyArray2<u8>>> {
    let img = numpy_to_gray_image(image).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Image conversion error: {}", e))
    })?;

    let edges = prewitt_edges(&img, threshold).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Prewitt edges error: {}", e))
    })?;

    Ok(gray_image_to_numpy(py, &DynamicImage::ImageLuma8(edges)))
}

/// Detect edges using Laplacian operator
///
/// Args:
///     image (np.ndarray): Input grayscale image (2D uint8 array)
///     threshold (float): Edge detection threshold (0.0 to 1.0)
///     use_diagonal (bool): Whether to include diagonal neighbors in Laplacian kernel
///
/// Returns:
///     np.ndarray: Edge map (2D uint8 array)
#[pyfunction]
#[pyo3(signature = (image, threshold=0.1, use_diagonal=true))]
fn laplacian_edges_py(
    py: Python,
    image: &Bound<'_, PyArray2<u8>>,
    threshold: f32,
    use_diagonal: bool,
) -> PyResult<Py<PyArray2<u8>>> {
    let img = numpy_to_gray_image(image).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Image conversion error: {}", e))
    })?;

    let edges = laplacian_edges(&img, threshold, use_diagonal).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Laplacian edges error: {}", e))
    })?;

    Ok(gray_image_to_numpy(py, &DynamicImage::ImageLuma8(edges)))
}

// ============================================================================
// Feature Detection Functions
// ============================================================================

/// Detect Harris corners
///
/// Args:
///     image (np.ndarray): Input grayscale image (2D uint8 array)
///     block_size (int): Size of block for corner detection (must be odd, typically 3 or 5)
///     k (float): Harris detector parameter (typically 0.04 to 0.06)
///     threshold (float): Corner detection threshold
///
/// Returns:
///     np.ndarray: Image with corners marked (2D uint8 array)
#[pyfunction]
#[pyo3(signature = (image, block_size=3, k=0.04, threshold=100.0))]
fn harris_corners_py(
    py: Python,
    image: &Bound<'_, PyArray2<u8>>,
    block_size: usize,
    k: f32,
    threshold: f32,
) -> PyResult<Py<PyArray2<u8>>> {
    let img = numpy_to_gray_image(image).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Image conversion error: {}", e))
    })?;

    let corners_img = harris_corners(&img, block_size, k, threshold).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Harris corners error: {}", e))
    })?;

    // Convert GrayImage to numpy array
    let (width, height) = corners_img.dimensions();
    let mut array = Array2::zeros((height as usize, width as usize));
    for y in 0..height {
        for x in 0..width {
            array[[y as usize, x as usize]] = corners_img.get_pixel(x, y)[0];
        }
    }

    Ok(array.into_pyarray(py).unbind())
}

/// Detect SIFT keypoints and compute descriptors
///
/// Args:
///     image (np.ndarray): Input grayscale image (2D uint8 array)
///     max_features (int): Maximum number of features to detect
///     contrast_threshold (float): Contrast threshold for feature detection
///
/// Returns:
///     list: List of dictionaries with keypoint information (x, y, scale, orientation, descriptor)
#[pyfunction]
#[pyo3(signature = (image, max_features=500, contrast_threshold=0.03))]
fn detect_and_compute_sift_py(
    py: Python,
    image: &Bound<'_, PyArray2<u8>>,
    max_features: usize,
    contrast_threshold: f32,
) -> PyResult<Vec<Py<PyDict>>> {
    let img = numpy_to_gray_image(image).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Image conversion error: {}", e))
    })?;

    let descriptors = detect_and_compute(&img, max_features, contrast_threshold).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("SIFT error: {}", e))
    })?;

    // Convert descriptors to Python dictionaries
    let mut result = Vec::new();
    for desc in descriptors {
        let dict = PyDict::new(py);
        dict.set_item("x", desc.keypoint.x)?;
        dict.set_item("y", desc.keypoint.y)?;
        dict.set_item("scale", desc.keypoint.scale)?;
        dict.set_item("orientation", desc.keypoint.orientation)?;
        dict.set_item("descriptor", desc.vector.into_pyarray(py))?;
        result.push(dict.into());
    }

    Ok(result)
}

// ============================================================================
// Image Segmentation Functions
// ============================================================================

/// Perform watershed segmentation
///
/// Args:
///     image (np.ndarray): Input grayscale image (2D uint8 array)
///     markers (Optional[np.ndarray]): Optional marker image (2D uint32 array)
///     connectivity (int): Pixel connectivity (4 or 8)
///
/// Returns:
///     np.ndarray: Segmented labels (2D uint32 array)
#[pyfunction]
#[pyo3(signature = (image, markers=None, connectivity=8))]
fn watershed_py(
    py: Python,
    image: &Bound<'_, PyArray2<u8>>,
    markers: Option<&Bound<'_, PyArray2<u32>>>,
    connectivity: u8,
) -> PyResult<Py<PyArray2<u32>>> {
    let img = numpy_to_gray_image(image).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Image conversion error: {}", e))
    })?;

    let marker_array =
        markers.map(|m: &Bound<'_, PyArray2<u32>>| m.readonly().as_array().to_owned());

    let labels = watershed(&img, marker_array.as_ref(), connectivity).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Watershed error: {}", e))
    })?;

    Ok(labels.into_pyarray(py).unbind())
}

/// Convert segmentation labels to color image for visualization
///
/// Args:
///     labels (np.ndarray): Segmentation labels (2D uint32 array)
///
/// Returns:
///     np.ndarray: Color image (3D uint8 array with shape (H, W, 3))
#[pyfunction]
fn labels_to_color_image_py(
    py: Python,
    labels: &Bound<'_, PyArray2<u32>>,
) -> PyResult<Py<PyArray3<u8>>> {
    let label_array = labels.readonly().as_array().to_owned();

    // labels_to_color_image expects Array2<u32> and returns RgbImage directly (not Result)
    let color_img = labels_to_color_image(&label_array, None);

    Ok(rgb_image_to_numpy(py, &DynamicImage::ImageRgb8(color_img)))
}

// ============================================================================
// Geometric Transform Functions
// ============================================================================

/// Find homography matrix from point correspondences using RANSAC
///
/// Args:
///     src_points (list): List of source points as (x, y) tuples
///     dst_points (list): List of destination points as (x, y) tuples
///     threshold (float): RANSAC inlier threshold
///     confidence (float): RANSAC confidence level (0.0 to 1.0)
///
/// Returns:
///     tuple: (homography_matrix, inlier_mask)
///         - homography_matrix: 3x3 numpy array
///         - inlier_mask: list of booleans indicating inliers
#[pyfunction]
#[pyo3(signature = (src_points, dst_points, threshold=3.0, confidence=0.99))]
fn find_homography_py(
    py: Python<'_>,
    src_points: Vec<(f64, f64)>,
    dst_points: Vec<(f64, f64)>,
    threshold: f64,
    confidence: f64,
) -> PyResult<(Py<PyArray2<f64>>, Vec<bool>)> {
    let (h, inliers) =
        find_homography(&src_points, &dst_points, threshold, confidence).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Find homography error: {}",
                e
            ))
        })?;

    // h is a Homography struct with a matrix field
    Ok((h.matrix.into_pyarray(py).unbind(), inliers))
}

// ============================================================================
// Module Registration
// ============================================================================

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Preprocessing functions
    m.add_function(wrap_pyfunction!(bilateral_filter_py, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_blur_py, m)?)?;
    m.add_function(wrap_pyfunction!(median_filter_py, m)?)?;
    m.add_function(wrap_pyfunction!(clahe_py, m)?)?;
    m.add_function(wrap_pyfunction!(equalize_histogram_py, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_brightness_py, m)?)?;
    m.add_function(wrap_pyfunction!(unsharp_mask_py, m)?)?;

    // Color conversion functions
    m.add_function(wrap_pyfunction!(rgb_to_grayscale_py, m)?)?;
    m.add_function(wrap_pyfunction!(rgb_to_hsv_py, m)?)?;

    // Edge detection functions
    m.add_function(wrap_pyfunction!(sobel_edges_py, m)?)?;
    m.add_function(wrap_pyfunction!(canny_edges_py, m)?)?;
    m.add_function(wrap_pyfunction!(prewitt_edges_py, m)?)?;
    m.add_function(wrap_pyfunction!(laplacian_edges_py, m)?)?;

    // Feature detection functions
    m.add_function(wrap_pyfunction!(harris_corners_py, m)?)?;
    m.add_function(wrap_pyfunction!(detect_and_compute_sift_py, m)?)?;

    // Segmentation functions
    m.add_function(wrap_pyfunction!(watershed_py, m)?)?;
    m.add_function(wrap_pyfunction!(labels_to_color_image_py, m)?)?;

    // Geometric transform functions
    m.add_function(wrap_pyfunction!(find_homography_py, m)?)?;

    Ok(())
}
