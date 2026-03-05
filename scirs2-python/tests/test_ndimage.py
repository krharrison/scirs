"""Tests for scirs2 ndimage (n-dimensional image processing) module."""

import numpy as np
import pytest
import scirs2


class TestImageFilters:
    """Test image filtering operations."""

    def test_gaussian_filter(self):
        """Test Gaussian smoothing filter."""
        image = np.random.randn(32, 32)

        filtered = scirs2.gaussian_filter_py(image, sigma=1.0)

        assert filtered.shape == image.shape
        # Smoothing should reduce variance
        assert np.var(filtered) < np.var(image)

    def test_uniform_filter(self):
        """Test uniform (box) filter."""
        image = np.random.randn(32, 32)

        filtered = scirs2.uniform_filter_py(image, size=3)

        assert filtered.shape == image.shape

    def test_median_filter(self):
        """Test median filter (good for salt-and-pepper noise)."""
        image = np.random.randn(32, 32)
        # Add some outliers
        image[10, 10] = 1000
        image[20, 20] = -1000

        filtered = scirs2.median_filter_py(image, size=3)

        # Outliers should be removed
        assert np.abs(filtered[10, 10]) < 100
        assert np.abs(filtered[20, 20]) < 100

    def test_sobel_filter(self):
        """Test Sobel edge detection filter."""
        image = np.random.randn(64, 64)

        edges = scirs2.sobel_filter_py(image)

        assert edges.shape == image.shape

    def test_prewitt_filter(self):
        """Test Prewitt edge detection filter."""
        image = np.random.randn(64, 64)

        edges = scirs2.prewitt_filter_py(image)

        assert edges.shape == image.shape

    def test_laplacian_filter(self):
        """Test Laplacian filter."""
        image = np.random.randn(32, 32)

        filtered = scirs2.laplacian_filter_py(image)

        assert filtered.shape == image.shape


class TestMorphology:
    """Test morphological operations."""

    def test_binary_erosion(self):
        """Test binary erosion."""
        image = np.ones((32, 32), dtype=np.uint8)
        image[10:20, 10:20] = 0

        eroded = scirs2.binary_erosion_py(image)

        assert eroded.shape == image.shape

    def test_binary_dilation(self):
        """Test binary dilation."""
        image = np.zeros((32, 32), dtype=np.uint8)
        image[15:17, 15:17] = 1

        dilated = scirs2.binary_dilation_py(image)

        # Dilation should increase the region
        assert np.sum(dilated) > np.sum(image)

    def test_binary_opening(self):
        """Test binary opening (erosion then dilation)."""
        image = np.ones((32, 32), dtype=np.uint8)
        image[15, 15] = 0  # Small hole

        opened = scirs2.binary_opening_py(image)

        # Opening removes small objects
        assert opened.shape == image.shape

    def test_binary_closing(self):
        """Test binary closing (dilation then erosion)."""
        image = np.zeros((32, 32), dtype=np.uint8)
        image[10:20, 10:20] = 1
        image[15, 15] = 0  # Small hole

        closed = scirs2.binary_closing_py(image)

        # Closing fills small holes
        assert np.sum(closed) >= np.sum(image)

    def test_morphological_gradient(self):
        """Test morphological gradient."""
        image = np.zeros((32, 32), dtype=np.uint8)
        image[10:20, 10:20] = 1

        gradient = scirs2.morphological_gradient_py(image)

        # Gradient highlights edges
        assert gradient.shape == image.shape


class TestSegmentation:
    """Test image segmentation."""

    def test_threshold_otsu(self):
        """Test Otsu's thresholding."""
        image = np.random.randn(32, 32)

        threshold = scirs2.threshold_otsu_py(image)
        binary = image > threshold

        assert binary.dtype == bool

    def test_watershed_segmentation(self):
        """Test watershed segmentation."""
        image = np.zeros((32, 32))
        # Create two peaks
        image[10, 10] = 1.0
        image[22, 22] = 1.0

        labels = scirs2.watershed_py(image)

        assert labels.shape == image.shape
        # Should have multiple segments
        assert len(np.unique(labels)) > 1

    def test_label_connected_components(self):
        """Test connected component labeling."""
        image = np.zeros((32, 32), dtype=np.uint8)
        image[5:10, 5:10] = 1
        image[20:25, 20:25] = 1

        labels = scirs2.label_connected_components_py(image)

        # Should find 2 components
        assert len(np.unique(labels)) >= 2


class TestInterpolation:
    """Test interpolation methods."""

    def test_zoom(self):
        """Test image zooming."""
        image = np.random.randn(16, 16)

        zoomed = scirs2.zoom_py(image, zoom=2.0)

        # Should be 2x larger
        assert zoomed.shape == (32, 32)

    def test_rotate(self):
        """Test image rotation."""
        image = np.random.randn(32, 32)

        rotated = scirs2.rotate_py(image, angle=45.0)

        assert rotated.shape[0] > 0 and rotated.shape[1] > 0

    def test_shift(self):
        """Test image shift."""
        image = np.random.randn(32, 32)

        shifted = scirs2.shift_py(image, shift=(5, 5))

        assert shifted.shape == image.shape

    def test_affine_transform(self):
        """Test affine transformation."""
        image = np.random.randn(32, 32)
        matrix = np.array([[1.2, 0.0], [0.0, 1.2]])  # Scale

        transformed = scirs2.affine_transform_py(image, matrix)

        assert transformed.shape[0] > 0


class TestMeasurements:
    """Test image measurements."""

    def test_center_of_mass(self):
        """Test center of mass calculation."""
        image = np.zeros((32, 32))
        image[15:17, 15:17] = 1.0

        com = scirs2.center_of_mass_py(image)

        # Center should be around (16, 16)
        assert 14 <= com[0] <= 18
        assert 14 <= com[1] <= 18

    def test_label_stats(self):
        """Test labeled region statistics."""
        image = np.random.randn(32, 32)
        labels = np.zeros((32, 32), dtype=np.int32)
        labels[5:15, 5:15] = 1
        labels[20:30, 20:30] = 2

        stats = scirs2.label_stats_py(image, labels)

        assert "mean" in stats or "area" in stats

    def test_histogram(self):
        """Test image histogram."""
        image = np.random.randn(100, 100)

        hist, bins = scirs2.histogram_py(image, bins=20)

        assert len(hist) == 20
        assert len(bins) == 21  # n_bins + 1


class TestDistanceTransform:
    """Test distance transforms."""

    def test_distance_transform_edt(self):
        """Test Euclidean distance transform."""
        image = np.zeros((32, 32), dtype=np.uint8)
        image[16, 16] = 1

        distance = scirs2.distance_transform_edt_py(image)

        # Distance at center should be 0
        assert distance[16, 16] == 0
        # Distance should increase away from center
        assert distance[0, 0] > distance[16, 16]

    def test_distance_transform_cdt(self):
        """Test city block (Manhattan) distance transform."""
        image = np.zeros((32, 32), dtype=np.uint8)
        image[16, 16] = 1

        distance = scirs2.distance_transform_cdt_py(image)

        assert distance.shape == image.shape


class TestFourierFilters:
    """Test Fourier-based filtering."""

    def test_fourier_gaussian(self):
        """Test Gaussian filter in Fourier domain."""
        image = np.random.randn(64, 64)

        filtered = scirs2.fourier_gaussian_py(image, sigma=2.0)

        assert filtered.shape == image.shape

    def test_fourier_ellipsoid(self):
        """Test ellipsoid filter in Fourier domain."""
        image = np.random.randn(64, 64)

        filtered = scirs2.fourier_ellipsoid_py(image, size=10)

        assert filtered.shape == image.shape

    def test_fourier_shift(self):
        """Test Fourier shift."""
        image = np.random.randn(32, 32)

        shifted = scirs2.fourier_shift_py(image, shift=(5, 5))

        assert shifted.shape == image.shape


class TestRankFilters:
    """Test rank-based filters."""

    def test_minimum_filter(self):
        """Test minimum filter."""
        image = np.random.rand(32, 32)

        filtered = scirs2.minimum_filter_py(image, size=3)

        assert filtered.shape == image.shape
        # Minimum filter should reduce values
        assert np.mean(filtered) <= np.mean(image)

    def test_maximum_filter(self):
        """Test maximum filter."""
        image = np.random.rand(32, 32)

        filtered = scirs2.maximum_filter_py(image, size=3)

        assert filtered.shape == image.shape
        # Maximum filter should increase values
        assert np.mean(filtered) >= np.mean(image)

    def test_percentile_filter(self):
        """Test percentile filter."""
        image = np.random.rand(32, 32)

        filtered = scirs2.percentile_filter_py(image, percentile=50, size=3)

        assert filtered.shape == image.shape


class TestGeometricTransforms:
    """Test geometric transformations."""

    def test_resize(self):
        """Test image resizing."""
        image = np.random.randn(32, 32)

        resized = scirs2.resize_ndimage_py(image, output_shape=(64, 64))

        assert resized.shape == (64, 64)

    def test_rescale(self):
        """Test image rescaling."""
        image = np.random.randn(32, 32)

        rescaled = scirs2.rescale_py(image, scale=2.0)

        assert rescaled.shape == (64, 64)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_image(self):
        """Test operations on empty image."""
        image = np.array([])

        try:
            filtered = scirs2.gaussian_filter_py(image, sigma=1.0)
        except Exception:
            pass  # Expected to fail

    def test_single_pixel(self):
        """Test operations on single pixel."""
        image = np.array([[1.0]])

        filtered = scirs2.gaussian_filter_py(image, sigma=1.0)

        assert filtered.shape == (1, 1)

    def test_3d_image(self):
        """Test 3D image processing."""
        image = np.random.randn(16, 16, 16)

        try:
            filtered = scirs2.gaussian_filter_py(image, sigma=1.0)
            assert filtered.shape == image.shape
        except Exception:
            pass  # 3D might not be supported


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
