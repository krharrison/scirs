# Image Processing (scirs2-ndimage)

`scirs2-ndimage` provides N-dimensional image processing operations modeled after
`scipy.ndimage`, covering filtering, morphology, measurements, segmentation, and
texture analysis.

## Filtering

### Convolution Filters

```rust,ignore
use scirs2_ndimage::filters::{convolve, gaussian_filter, uniform_filter, median_filter};

// Gaussian smoothing (sigma = 1.5)
let smoothed = gaussian_filter(&image, 1.5)?;

// Uniform (box) filter
let smoothed = uniform_filter(&image, 3)?;  // 3x3 kernel

// Median filter (good for salt-and-pepper noise)
let filtered = median_filter(&image, 3)?;

// Custom convolution kernel
let kernel = array![[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]];
let edges = convolve(&image, &kernel)?;
```

### Edge Detection

```rust,ignore
use scirs2_ndimage::filters::{sobel, laplace, prewitt};

// Sobel edge detection (returns gradient magnitude)
let edges = sobel(&image)?;

// Laplacian edge detection
let edges = laplace(&image)?;

// Prewitt operator
let edges = prewitt(&image)?;
```

## Morphological Operations

```rust,ignore
use scirs2_ndimage::morphology::{
    binary_dilation, binary_erosion, binary_opening, binary_closing
};

// Structuring element (3x3 cross)
let se = array![[0, 1, 0], [1, 1, 1], [0, 1, 0]];

// Basic operations on binary images
let dilated = binary_dilation(&binary_image, &se, 1)?;
let eroded = binary_erosion(&binary_image, &se, 1)?;
let opened = binary_opening(&binary_image, &se, 1)?;
let closed = binary_closing(&binary_image, &se, 1)?;
```

### Grayscale Morphology

```rust,ignore
use scirs2_ndimage::morphology::{grey_dilation, grey_erosion};

let dilated = grey_dilation(&image, (3, 3))?;
let eroded = grey_erosion(&image, (3, 3))?;
```

## Measurements

```rust,ignore
use scirs2_ndimage::measurements::{label, find_objects, center_of_mass, sum as ndsum};

// Connected component labeling
let (labels, num_features) = label(&binary_image, None)?;
println!("Found {} connected components", num_features);

// Bounding boxes of each labeled region
let objects = find_objects(&labels)?;

// Center of mass for each labeled region
let centers = center_of_mass(&image, &labels, None)?;

// Sum of pixel values in each region
let sums = ndsum(&image, &labels, None)?;
```

## Segmentation

### Watershed

```rust,ignore
use scirs2_ndimage::segmentation::{watershed, distance_transform};

// Distance transform of a binary image
let distances = distance_transform(&binary_image)?;

// Watershed segmentation using distance transform as elevation map
let labels = watershed(&distances, &markers)?;
```

### Active Contours

```rust,ignore
use scirs2_ndimage::segmentation::{chan_vese, geodesic_active_contour};

// Chan-Vese segmentation (region-based, no edges needed)
let segmented = chan_vese(&image, num_iterations, mu, lambda1, lambda2)?;

// Geodesic active contour (edge-based)
let contour = geodesic_active_contour(&image, &initial_contour, num_iterations)?;
```

## Texture Analysis

### Radiomics Features

```rust,ignore
use scirs2_ndimage::texture::{glcm, glszm, ngtdm};

// Gray-Level Co-occurrence Matrix
let features = glcm(&image, distances, angles)?;

// Gray-Level Size Zone Matrix
let features = glszm(&image)?;

// Neighborhood Gray-Tone Difference Matrix
let features = ngtdm(&image, neighborhood_size)?;
```

### Laws' Texture Energy

```rust,ignore
use scirs2_ndimage::texture::laws_texture_energy;

let energy_maps = laws_texture_energy(&image)?;
```

## 3D and 4D Processing

All operations extend to higher dimensions:

```rust,ignore
use scirs2_ndimage::filters::gaussian_filter_nd;

// 3D Gaussian filter for volumetric data
let smoothed = gaussian_filter_nd(&volume, &[1.0, 1.0, 1.0])?;

// 4D spatiotemporal filtering
let filtered = gaussian_filter_nd(&spatiotemporal, &[1.0, 1.0, 1.0, 0.5])?;
```

## SciPy Equivalence Table

| SciPy | SciRS2 |
|-------|--------|
| `scipy.ndimage.gaussian_filter` | `filters::gaussian_filter` |
| `scipy.ndimage.median_filter` | `filters::median_filter` |
| `scipy.ndimage.sobel` | `filters::sobel` |
| `scipy.ndimage.laplace` | `filters::laplace` |
| `scipy.ndimage.binary_erosion` | `morphology::binary_erosion` |
| `scipy.ndimage.binary_dilation` | `morphology::binary_dilation` |
| `scipy.ndimage.label` | `measurements::label` |
| `scipy.ndimage.center_of_mass` | `measurements::center_of_mass` |
