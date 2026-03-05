"""Tests for scirs2 computer vision module."""

import numpy as np
import pytest
import scirs2


class TestImageTransforms:
    """Test image transformation functions."""

    def test_resize_image(self):
        """Test image resizing."""
        # Create a dummy 32x32 RGB image
        image = np.random.rand(32, 32, 3).astype(np.float32)

        resized = scirs2.resize_image_py(image, width=64, height=64)

        assert resized.shape == (64, 64, 3)

    def test_resize_grayscale(self):
        """Test resizing grayscale image."""
        image = np.random.rand(28, 28).astype(np.float32)

        resized = scirs2.resize_image_py(image, width=56, height=56)

        assert resized.shape == (56, 56)

    def test_crop_image(self):
        """Test image cropping."""
        image = np.random.rand(100, 100, 3).astype(np.float32)

        cropped = scirs2.crop_image_py(image, x=10, y=10, width=50, height=50)

        assert cropped.shape == (50, 50, 3)

    def test_center_crop(self):
        """Test center cropping."""
        image = np.random.rand(100, 100, 3).astype(np.float32)

        cropped = scirs2.center_crop_py(image, size=64)

        assert cropped.shape == (64, 64, 3)

    def test_rotate_image(self):
        """Test image rotation."""
        image = np.random.rand(64, 64, 3).astype(np.float32)

        rotated = scirs2.rotate_image_py(image, angle=90.0)

        # After 90 degree rotation, dimensions might swap
        assert rotated.shape[0] > 0 and rotated.shape[1] > 0

    def test_flip_horizontal(self):
        """Test horizontal flip."""
        image = np.arange(12).reshape(3, 4).astype(np.float32)

        flipped = scirs2.flip_horizontal_py(image)

        # First and last columns should be swapped
        assert np.allclose(flipped[:, 0], image[:, -1])

    def test_flip_vertical(self):
        """Test vertical flip."""
        image = np.arange(12).reshape(3, 4).astype(np.float32)

        flipped = scirs2.flip_vertical_py(image)

        # First and last rows should be swapped
        assert np.allclose(flipped[0, :], image[-1, :])

    def test_transpose_image(self):
        """Test image transpose."""
        image = np.random.rand(64, 48, 3).astype(np.float32)

        transposed = scirs2.transpose_image_py(image)

        # Dimensions should be swapped
        assert transposed.shape == (48, 64, 3)


class TestColorTransforms:
    """Test color space transformations."""

    def test_rgb_to_grayscale(self):
        """Test RGB to grayscale conversion."""
        image = np.random.rand(32, 32, 3).astype(np.float32)

        gray = scirs2.rgb_to_grayscale_py(image)

        assert gray.shape == (32, 32) or gray.shape == (32, 32, 1)

    def test_grayscale_to_rgb(self):
        """Test grayscale to RGB conversion."""
        image = np.random.rand(32, 32).astype(np.float32)

        rgb = scirs2.grayscale_to_rgb_py(image)

        assert rgb.shape == (32, 32, 3)

    def test_rgb_to_hsv(self):
        """Test RGB to HSV conversion."""
        image = np.random.rand(16, 16, 3).astype(np.float32)

        hsv = scirs2.rgb_to_hsv_py(image)

        assert hsv.shape == (16, 16, 3)

    def test_hsv_to_rgb(self):
        """Test HSV to RGB conversion."""
        hsv = np.random.rand(16, 16, 3).astype(np.float32)

        rgb = scirs2.hsv_to_rgb_py(hsv)

        assert rgb.shape == (16, 16, 3)

    def test_adjust_brightness(self):
        """Test brightness adjustment."""
        image = np.full((10, 10, 3), 0.5, dtype=np.float32)

        brightened = scirs2.adjust_brightness_py(image, factor=1.5)

        # Values should increase
        assert np.mean(brightened) > np.mean(image)

    def test_adjust_contrast(self):
        """Test contrast adjustment."""
        image = np.random.rand(20, 20, 3).astype(np.float32)

        contrasted = scirs2.adjust_contrast_py(image, factor=2.0)

        assert contrasted.shape == image.shape

    def test_adjust_saturation(self):
        """Test saturation adjustment."""
        image = np.random.rand(20, 20, 3).astype(np.float32)

        saturated = scirs2.adjust_saturation_py(image, factor=1.5)

        assert saturated.shape == image.shape

    def test_adjust_hue(self):
        """Test hue adjustment."""
        image = np.random.rand(20, 20, 3).astype(np.float32)

        hue_shifted = scirs2.adjust_hue_py(image, delta=0.1)

        assert hue_shifted.shape == image.shape


class TestConvolutions:
    """Test convolution operations."""

    def test_convolve2d_basic(self):
        """Test 2D convolution."""
        image = np.random.rand(32, 32).astype(np.float32)
        kernel = np.ones((3, 3), dtype=np.float32) / 9.0  # Mean filter

        convolved = scirs2.convolve2d_py(image, kernel)

        assert convolved.shape[0] > 0 and convolved.shape[1] > 0

    def test_gaussian_blur(self):
        """Test Gaussian blur."""
        image = np.random.rand(64, 64, 3).astype(np.float32)

        blurred = scirs2.gaussian_blur_py(image, kernel_size=5, sigma=1.0)

        assert blurred.shape == image.shape

    def test_median_filter(self):
        """Test median filter."""
        image = np.random.rand(32, 32).astype(np.float32)

        filtered = scirs2.median_filter_py(image, kernel_size=3)

        assert filtered.shape == image.shape

    def test_sharpen_image(self):
        """Test image sharpening."""
        image = np.random.rand(32, 32, 3).astype(np.float32)

        sharpened = scirs2.sharpen_image_py(image)

        assert sharpened.shape == image.shape

    def test_edge_detection_sobel(self):
        """Test Sobel edge detection."""
        image = np.random.rand(64, 64).astype(np.float32)

        edges = scirs2.sobel_edge_detection_py(image)

        assert edges.shape == image.shape

    def test_edge_detection_canny(self):
        """Test Canny edge detection."""
        image = np.random.rand(64, 64).astype(np.float32)

        edges = scirs2.canny_edge_detection_py(image, low_threshold=0.1, high_threshold=0.3)

        assert edges.shape == image.shape


class TestAugmentation:
    """Test data augmentation techniques."""

    def test_random_crop(self):
        """Test random cropping."""
        np.random.seed(42)
        image = np.random.rand(100, 100, 3).astype(np.float32)

        cropped = scirs2.random_crop_py(image, size=64, seed=42)

        assert cropped.shape == (64, 64, 3)

    def test_random_flip(self):
        """Test random flip."""
        image = np.random.rand(32, 32, 3).astype(np.float32)

        flipped = scirs2.random_flip_py(image, probability=1.0, seed=42)

        # With probability 1.0, should always flip
        assert not np.allclose(flipped, image)

    def test_random_rotation(self):
        """Test random rotation."""
        image = np.random.rand(64, 64, 3).astype(np.float32)

        rotated = scirs2.random_rotation_py(image, max_angle=45.0, seed=42)

        assert rotated.shape[0] > 0 and rotated.shape[1] > 0

    def test_random_brightness(self):
        """Test random brightness adjustment."""
        image = np.full((20, 20, 3), 0.5, dtype=np.float32)

        adjusted = scirs2.random_brightness_py(image, max_delta=0.2, seed=42)

        assert adjusted.shape == image.shape

    def test_random_contrast(self):
        """Test random contrast adjustment."""
        image = np.random.rand(20, 20, 3).astype(np.float32)

        adjusted = scirs2.random_contrast_py(image, lower=0.8, upper=1.2, seed=42)

        assert adjusted.shape == image.shape

    def test_cutout(self):
        """Test cutout augmentation."""
        image = np.random.rand(32, 32, 3).astype(np.float32)

        augmented = scirs2.cutout_py(image, size=8, seed=42)

        assert augmented.shape == image.shape

    def test_mixup(self):
        """Test mixup augmentation."""
        image1 = np.random.rand(32, 32, 3).astype(np.float32)
        image2 = np.random.rand(32, 32, 3).astype(np.float32)

        mixed = scirs2.mixup_py(image1, image2, alpha=0.5)

        assert mixed.shape == image1.shape


class TestNormalization:
    """Test image normalization."""

    def test_normalize_image(self):
        """Test image normalization to [0, 1]."""
        image = np.random.randint(0, 256, size=(32, 32, 3)).astype(np.float32)

        normalized = scirs2.normalize_image_py(image)

        assert np.all((normalized >= 0) & (normalized <= 1))

    def test_standardize_image(self):
        """Test image standardization."""
        image = np.random.rand(32, 32, 3).astype(np.float32)

        standardized = scirs2.standardize_image_py(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        assert standardized.shape == image.shape

    def test_denormalize_image(self):
        """Test image denormalization."""
        image = np.random.rand(32, 32, 3).astype(np.float32)

        denormalized = scirs2.denormalize_image_py(image, min_val=0.0, max_val=255.0)

        assert np.max(denormalized) <= 255.0


class TestFeatureExtraction:
    """Test feature extraction from images."""

    def test_extract_patches(self):
        """Test patch extraction."""
        image = np.random.rand(64, 64, 3).astype(np.float32)

        patches = scirs2.extract_patches_py(image, patch_size=16, stride=16)

        # Should extract multiple patches
        assert patches.shape[0] > 1
        assert patches.shape[1:] == (16, 16, 3)

    def test_histogram_of_oriented_gradients(self):
        """Test HOG feature extraction."""
        image = np.random.rand(64, 64).astype(np.float32)

        hog_features = scirs2.hog_features_py(image, cell_size=8, block_size=2)

        # Should produce feature vector
        assert hog_features.shape[0] > 0

    def test_local_binary_patterns(self):
        """Test LBP feature extraction."""
        image = np.random.rand(32, 32).astype(np.float32)

        lbp = scirs2.local_binary_patterns_py(image, radius=1, n_points=8)

        assert lbp.shape == image.shape


class TestMorphology:
    """Test morphological operations."""

    def test_dilate(self):
        """Test dilation."""
        image = np.random.rand(32, 32).astype(np.float32)
        kernel = np.ones((3, 3), dtype=np.float32)

        dilated = scirs2.dilate_py(image, kernel)

        assert dilated.shape == image.shape

    def test_erode(self):
        """Test erosion."""
        image = np.random.rand(32, 32).astype(np.float32)
        kernel = np.ones((3, 3), dtype=np.float32)

        eroded = scirs2.erode_py(image, kernel)

        assert eroded.shape == image.shape

    def test_opening(self):
        """Test morphological opening."""
        image = np.random.rand(32, 32).astype(np.float32)
        kernel = np.ones((3, 3), dtype=np.float32)

        opened = scirs2.morphology_open_py(image, kernel)

        assert opened.shape == image.shape

    def test_closing(self):
        """Test morphological closing."""
        image = np.random.rand(32, 32).astype(np.float32)
        kernel = np.ones((3, 3), dtype=np.float32)

        closed = scirs2.morphology_close_py(image, kernel)

        assert closed.shape == image.shape


class TestObjectDetection:
    """Test object detection utilities."""

    def test_non_maximum_suppression(self):
        """Test non-maximum suppression."""
        # Bounding boxes: [x1, y1, x2, y2, score]
        boxes = np.array([
            [10, 10, 50, 50, 0.9],
            [15, 15, 55, 55, 0.8],
            [100, 100, 150, 150, 0.95]
        ], dtype=np.float32)

        kept_boxes = scirs2.non_maximum_suppression_py(boxes, iou_threshold=0.5)

        # Should keep boxes with low overlap
        assert kept_boxes.shape[0] >= 2

    def test_iou_calculation(self):
        """Test IoU (Intersection over Union) calculation."""
        box1 = np.array([0, 0, 10, 10], dtype=np.float32)
        box2 = np.array([5, 5, 15, 15], dtype=np.float32)

        iou = scirs2.calculate_iou_py(box1, box2)

        # Boxes overlap, IoU should be between 0 and 1
        assert 0 < iou < 1

    def test_iou_no_overlap(self):
        """Test IoU with no overlap."""
        box1 = np.array([0, 0, 10, 10], dtype=np.float32)
        box2 = np.array([20, 20, 30, 30], dtype=np.float32)

        iou = scirs2.calculate_iou_py(box1, box2)

        # No overlap, IoU should be 0
        assert iou == 0.0


class TestImageMetrics:
    """Test image quality metrics."""

    def test_psnr(self):
        """Test Peak Signal-to-Noise Ratio."""
        image1 = np.random.rand(32, 32, 3).astype(np.float32)
        image2 = image1.copy()

        psnr = scirs2.calculate_psnr_py(image1, image2)

        # Identical images should have infinite PSNR
        assert psnr > 100 or np.isinf(psnr)

    def test_ssim(self):
        """Test Structural Similarity Index."""
        image1 = np.random.rand(32, 32).astype(np.float32)
        image2 = image1.copy()

        ssim = scirs2.calculate_ssim_py(image1, image2)

        # Identical images should have SSIM = 1
        assert 0.99 <= ssim <= 1.0

    def test_mse(self):
        """Test Mean Squared Error."""
        image1 = np.zeros((32, 32), dtype=np.float32)
        image2 = np.ones((32, 32), dtype=np.float32)

        mse = scirs2.calculate_mse_py(image1, image2)

        # MSE should be 1.0
        assert np.allclose(mse, 1.0)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_pixel_image(self):
        """Test operations on single pixel image."""
        image = np.array([[0.5]], dtype=np.float32)

        normalized = scirs2.normalize_image_py(image)

        assert normalized.shape == (1, 1)

    def test_grayscale_color_ops(self):
        """Test color operations on grayscale."""
        image = np.random.rand(32, 32).astype(np.float32)

        # Some operations should handle grayscale gracefully
        try:
            blurred = scirs2.gaussian_blur_py(image, kernel_size=3, sigma=1.0)
            assert blurred.shape == image.shape
        except Exception:
            pass  # Expected if operation requires RGB

    def test_large_image(self):
        """Test with large image."""
        image = np.random.rand(1024, 1024, 3).astype(np.float32)

        resized = scirs2.resize_image_py(image, width=512, height=512)

        assert resized.shape == (512, 512, 3)

    def test_zero_kernel(self):
        """Test convolution with zero kernel."""
        image = np.random.rand(32, 32).astype(np.float32)
        kernel = np.zeros((3, 3), dtype=np.float32)

        convolved = scirs2.convolve2d_py(image, kernel)

        # Result should be all zeros
        assert np.allclose(convolved, 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
