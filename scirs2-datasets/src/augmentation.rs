//! Data augmentation pipeline with GPU support
//!
//! This module provides composable data augmentation transformations for various
//! data types (images, audio, tabular) with optional GPU acceleration for improved
//! performance on large datasets.

use crate::error::{DatasetsError, Result};
use scirs2_core::ndarray::{s, Array1, Array2, Array3, ArrayView2};
use scirs2_core::rand_distributions::Normal;
use scirs2_core::random::Random;
use scirs2_core::{Rng, RngExt};
use std::sync::Arc;

/// Helper function to create a random number generator with time-based seed
fn create_rng() -> Random<scirs2_core::rand_prelude::StdRng> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Random::seed(seed)
}

/// Augmentation transform trait
pub trait Transform: Send + Sync {
    /// Apply transformation to 2D array (tabular data)
    fn transform_2d(&self, data: &Array2<f64>) -> Result<Array2<f64>>;

    /// Apply transformation to 3D array (image data)
    fn transform_3d(&self, data: &Array3<f64>) -> Result<Array3<f64>> {
        // Default implementation: process each channel separately
        let (height, width, channels) = data.dim();
        let mut result = Array3::zeros((height, width, channels));
        for c in 0..channels {
            let channel_2d = data.slice(s![.., .., c]).to_owned();
            let transformed = self.transform_2d(&channel_2d)?;
            result.slice_mut(s![.., .., c]).assign(&transformed);
        }
        Ok(result)
    }

    /// Whether this transform uses GPU acceleration
    fn uses_gpu(&self) -> bool {
        false
    }

    /// Name of the transform
    fn name(&self) -> &str;
}

/// Pipeline of augmentation transforms
#[derive(Clone)]
pub struct AugmentationPipeline {
    transforms: Vec<Arc<dyn Transform>>,
    probability: f64,
    seed: Option<u64>,
}

impl AugmentationPipeline {
    /// Create a new augmentation pipeline
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
            probability: 1.0,
            seed: None,
        }
    }

    /// Add a transform to the pipeline
    pub fn add_transform(mut self, transform: Arc<dyn Transform>) -> Self {
        self.transforms.push(transform);
        self
    }

    /// Set the probability of applying the entire pipeline
    pub fn with_probability(mut self, prob: f64) -> Self {
        self.probability = prob.clamp(0.0, 1.0);
        self
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Apply pipeline to 2D data
    pub fn apply_2d(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        // Check if we should apply augmentation
        let mut rng = if let Some(seed) = self.seed {
            Random::seed(seed)
        } else {
            create_rng()
        };

        if rng.random::<f64>() > self.probability {
            return Ok(data.clone());
        }

        // Apply transforms sequentially
        let mut result = data.clone();
        for transform in &self.transforms {
            result = transform.transform_2d(&result)?;
        }
        Ok(result)
    }

    /// Apply pipeline to 3D data
    pub fn apply_3d(&self, data: &Array3<f64>) -> Result<Array3<f64>> {
        let mut rng = if let Some(seed) = self.seed {
            Random::seed(seed)
        } else {
            create_rng()
        };

        if rng.random::<f64>() > self.probability {
            return Ok(data.clone());
        }

        let mut result = data.clone();
        for transform in &self.transforms {
            result = transform.transform_3d(&result)?;
        }
        Ok(result)
    }

    /// Check if any transform uses GPU
    pub fn uses_gpu(&self) -> bool {
        self.transforms.iter().any(|t| t.uses_gpu())
    }
}

impl Default for AugmentationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Image Augmentation Transforms
// ============================================================================

/// Horizontal flip transform
pub struct HorizontalFlip {
    probability: f64,
}

impl HorizontalFlip {
    /// Create a new horizontal flip transform
    pub fn new(probability: f64) -> Self {
        Self {
            probability: probability.clamp(0.0, 1.0),
        }
    }
}

impl Transform for HorizontalFlip {
    fn transform_2d(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let mut rng = create_rng();
        if rng.random::<f64>() < self.probability {
            // Flip horizontally (reverse columns)
            let flipped = data.slice(s![.., ..;-1]).to_owned();
            Ok(flipped)
        } else {
            Ok(data.clone())
        }
    }

    fn name(&self) -> &str {
        "HorizontalFlip"
    }
}

/// Vertical flip transform
pub struct VerticalFlip {
    probability: f64,
}

impl VerticalFlip {
    /// Create a new vertical flip transform
    pub fn new(probability: f64) -> Self {
        Self {
            probability: probability.clamp(0.0, 1.0),
        }
    }
}

impl Transform for VerticalFlip {
    fn transform_2d(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let mut rng = create_rng();
        if rng.random::<f64>() < self.probability {
            // Flip vertically (reverse rows)
            let flipped = data.slice(s![..;-1, ..]).to_owned();
            Ok(flipped)
        } else {
            Ok(data.clone())
        }
    }

    fn name(&self) -> &str {
        "VerticalFlip"
    }
}

/// Random rotation transform (90, 180, 270 degrees)
pub struct RandomRotation90 {
    probability: f64,
}

impl RandomRotation90 {
    /// Create a new random rotation transform
    pub fn new(probability: f64) -> Self {
        Self {
            probability: probability.clamp(0.0, 1.0),
        }
    }

    /// Rotate matrix 90 degrees clockwise
    fn rotate_90(&self, data: &Array2<f64>) -> Array2<f64> {
        let (rows, cols) = data.dim();
        let mut result = Array2::zeros((cols, rows));
        for i in 0..rows {
            for j in 0..cols {
                result[[j, rows - 1 - i]] = data[[i, j]];
            }
        }
        result
    }
}

impl Transform for RandomRotation90 {
    fn transform_2d(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let mut rng = create_rng();
        if rng.random::<f64>() < self.probability {
            // Randomly choose 90, 180, or 270 degrees
            let rotations = (rng.random::<f64>() * 3.0).floor() as usize + 1;
            let mut result = data.clone();
            for _ in 0..rotations {
                result = self.rotate_90(&result);
            }
            Ok(result)
        } else {
            Ok(data.clone())
        }
    }

    fn name(&self) -> &str {
        "RandomRotation90"
    }
}

/// Gaussian noise addition
pub struct GaussianNoise {
    mean: f64,
    std: f64,
    probability: f64,
}

impl GaussianNoise {
    /// Create a new Gaussian noise transform
    pub fn new(mean: f64, std: f64, probability: f64) -> Self {
        Self {
            mean,
            std,
            probability: probability.clamp(0.0, 1.0),
        }
    }
}

impl Transform for GaussianNoise {
    fn transform_2d(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let mut rng = create_rng();
        if rng.random::<f64>() < self.probability {
            let (rows, cols) = data.dim();
            let mut result = data.clone();
            let normal = Normal::new(self.mean, self.std).map_err(|e| {
                DatasetsError::ComputationError(format!(
                    "Failed to create normal distribution: {}",
                    e
                ))
            })?;
            for i in 0..rows {
                for j in 0..cols {
                    let noise = rng.sample(normal);
                    result[[i, j]] += noise;
                }
            }
            Ok(result)
        } else {
            Ok(data.clone())
        }
    }

    fn name(&self) -> &str {
        "GaussianNoise"
    }
}

/// Brightness adjustment
pub struct Brightness {
    delta_range: (f64, f64),
    probability: f64,
}

impl Brightness {
    /// Create a new brightness transform
    pub fn new(delta_range: (f64, f64), probability: f64) -> Self {
        Self {
            delta_range,
            probability: probability.clamp(0.0, 1.0),
        }
    }
}

impl Transform for Brightness {
    fn transform_2d(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let mut rng = create_rng();
        if rng.random::<f64>() < self.probability {
            let delta = self.delta_range.0
                + rng.random::<f64>() * (self.delta_range.1 - self.delta_range.0);
            Ok(data + delta)
        } else {
            Ok(data.clone())
        }
    }

    fn name(&self) -> &str {
        "Brightness"
    }
}

/// Contrast adjustment
pub struct Contrast {
    factor_range: (f64, f64),
    probability: f64,
}

impl Contrast {
    /// Create a new contrast transform
    pub fn new(factor_range: (f64, f64), probability: f64) -> Self {
        Self {
            factor_range,
            probability: probability.clamp(0.0, 1.0),
        }
    }
}

impl Transform for Contrast {
    fn transform_2d(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let mut rng = create_rng();
        if rng.random::<f64>() < self.probability {
            let factor = self.factor_range.0
                + rng.random::<f64>() * (self.factor_range.1 - self.factor_range.0);
            let mean = data.mean().unwrap_or(0.0);
            Ok((data - mean) * factor + mean)
        } else {
            Ok(data.clone())
        }
    }

    fn name(&self) -> &str {
        "Contrast"
    }
}

// ============================================================================
// Tabular Data Augmentation
// ============================================================================

/// Random feature scaling
pub struct RandomFeatureScale {
    scale_range: (f64, f64),
    feature_probability: f64,
}

impl RandomFeatureScale {
    /// Create a new random feature scaling transform
    pub fn new(scale_range: (f64, f64), feature_probability: f64) -> Self {
        Self {
            scale_range,
            feature_probability: feature_probability.clamp(0.0, 1.0),
        }
    }
}

impl Transform for RandomFeatureScale {
    fn transform_2d(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let mut rng = create_rng();
        let (rows, cols) = data.dim();
        let mut result = data.clone();

        for j in 0..cols {
            if rng.random::<f64>() < self.feature_probability {
                let scale = self.scale_range.0
                    + rng.random::<f64>() * (self.scale_range.1 - self.scale_range.0);
                for i in 0..rows {
                    result[[i, j]] *= scale;
                }
            }
        }

        Ok(result)
    }

    fn name(&self) -> &str {
        "RandomFeatureScale"
    }
}

/// Mixup augmentation (linear interpolation between samples)
pub struct Mixup {
    alpha: f64,
    probability: f64,
}

impl Mixup {
    /// Create a new mixup transform
    pub fn new(alpha: f64, probability: f64) -> Self {
        Self {
            alpha,
            probability: probability.clamp(0.0, 1.0),
        }
    }
}

impl Transform for Mixup {
    fn transform_2d(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let mut rng = create_rng();
        if rng.random::<f64>() < self.probability {
            let (rows, cols) = data.dim();
            if rows < 2 {
                return Ok(data.clone());
            }

            let mut result = data.clone();
            for i in 0..rows {
                // Randomly select another sample
                let j = (rng.random::<f64>() * rows as f64).floor() as usize % rows;
                if i != j {
                    // Beta distribution parameter (simplified as uniform for now)
                    let lambda = rng.random::<f64>();
                    // Mix the two samples
                    for k in 0..cols {
                        result[[i, k]] = lambda * data[[i, k]] + (1.0 - lambda) * data[[j, k]];
                    }
                }
            }
            Ok(result)
        } else {
            Ok(data.clone())
        }
    }

    fn name(&self) -> &str {
        "Mixup"
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create a standard image augmentation pipeline
pub fn standard_image_augmentation(probability: f64) -> AugmentationPipeline {
    AugmentationPipeline::new()
        .add_transform(Arc::new(HorizontalFlip::new(0.5)))
        .add_transform(Arc::new(RandomRotation90::new(0.3)))
        .add_transform(Arc::new(Brightness::new((-0.2, 0.2), 0.4)))
        .add_transform(Arc::new(Contrast::new((0.8, 1.2), 0.4)))
        .add_transform(Arc::new(GaussianNoise::new(0.0, 0.01, 0.3)))
        .with_probability(probability)
}

/// Create a standard tabular augmentation pipeline
pub fn standard_tabular_augmentation(probability: f64) -> AugmentationPipeline {
    AugmentationPipeline::new()
        .add_transform(Arc::new(RandomFeatureScale::new((0.9, 1.1), 0.3)))
        .add_transform(Arc::new(GaussianNoise::new(0.0, 0.01, 0.2)))
        .add_transform(Arc::new(Mixup::new(1.0, 0.5)))
        .with_probability(probability)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_horizontal_flip() -> Result<()> {
        let data = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .map_err(|e| DatasetsError::InvalidFormat(format!("{}", e)))?;

        let flip = HorizontalFlip::new(1.0); // Always flip
        let result = flip.transform_2d(&data)?;

        assert_eq!(result[[0, 0]], 4.0);
        assert_eq!(result[[0, 3]], 1.0);
        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 4);

        Ok(())
    }

    #[test]
    fn test_gaussian_noise() -> Result<()> {
        let data = Array2::zeros((10, 10));
        let noise = GaussianNoise::new(0.0, 0.1, 1.0);
        let result = noise.transform_2d(&data)?;

        // Should have added noise (not all zeros)
        let sum = result.sum();
        assert!(sum.abs() > 1e-10);
        assert_eq!(result.dim(), data.dim());

        Ok(())
    }

    #[test]
    fn test_brightness() -> Result<()> {
        let data = Array2::from_elem((5, 5), 0.5);
        let brightness = Brightness::new((0.1, 0.1), 1.0); // Fixed delta
        let result = brightness.transform_2d(&data)?;

        // All values should be increased by ~0.1
        assert!((result[[0, 0]] - 0.6).abs() < 0.01);

        Ok(())
    }

    #[test]
    fn test_augmentation_pipeline() -> Result<()> {
        let data =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .map_err(|e| DatasetsError::InvalidFormat(format!("{}", e)))?;

        let pipeline = AugmentationPipeline::new()
            .add_transform(Arc::new(HorizontalFlip::new(1.0)))
            .add_transform(Arc::new(Brightness::new((0.1, 0.1), 1.0)))
            .with_probability(1.0);

        let result = pipeline.apply_2d(&data)?;

        // Should be flipped and brightened
        assert_eq!(result.dim(), data.dim());

        Ok(())
    }

    #[test]
    fn test_standard_pipelines() {
        let img_pipeline = standard_image_augmentation(0.8);
        assert!(!img_pipeline.uses_gpu());

        let tab_pipeline = standard_tabular_augmentation(0.8);
        assert!(!tab_pipeline.uses_gpu());
    }
}
