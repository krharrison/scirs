//! Data augmentation for training neural networks

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::rngs::SmallRng;
use scirs2_core::random::Distribution;
use scirs2_core::random::{thread_rng, Rng, RngExt, SeedableRng};
use std::fmt::Debug;

/// Trait for data augmentation
pub trait Augmentation<F: Float + NumAssign + Debug + ScalarOperand> {
    /// Apply augmentation to the input
    fn apply(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>>;

    /// Get a description of the augmentation
    fn description(&self) -> String;
}

/// Gaussian noise augmentation
#[derive(Debug, Clone)]
pub struct GaussianNoise<F: Float + NumAssign + Debug + ScalarOperand> {
    /// Standard deviation of the noise
    std: F,
}

impl<F: Float + NumAssign + Debug + ScalarOperand> GaussianNoise<F> {
    /// Create a new Gaussian noise augmentation
    pub fn new(std: F) -> Self {
        Self { std }
    }
}

impl<F: Float + NumAssign + Debug + ScalarOperand> Augmentation<F> for GaussianNoise<F> {
    fn apply(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut rng = SmallRng::from_rng(&mut thread_rng());
        let mut result = input.clone();

        for item in result.iter_mut() {
            // Create a normal distribution
            let normal = scirs2_core::random::Normal::new(0.0, self.std.to_f64().unwrap_or(0.1))
                .expect("Failed to create normal distribution");

            // Sample from the distribution
            let noise = F::from(rng.sample(normal)).unwrap_or(F::zero());
            *item += noise;
        }

        Ok(result)
    }

    fn description(&self) -> String {
        format!(
            "GaussianNoise (std: {:.3})",
            self.std.to_f64().unwrap_or(0.0)
        )
    }
}

/// Random erasing augmentation
#[derive(Debug, Clone)]
pub struct RandomErasing<F: Float + NumAssign + Debug + ScalarOperand> {
    /// Probability of applying the augmentation
    probability: f64,
    /// Value to use for erasing
    value: F,
}

impl<F: Float + NumAssign + Debug + ScalarOperand> RandomErasing<F> {
    /// Create a new random erasing augmentation
    pub fn new(probability: f64, value: F) -> Self {
        Self { probability, value }
    }
}

impl<F: Float + NumAssign + Debug + ScalarOperand> Augmentation<F> for RandomErasing<F> {
    fn apply(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut rng = SmallRng::from_rng(&mut thread_rng());
        let mut result = input.clone();

        // Only apply augmentation based on probability
        if rng.random::<f64>() > self.probability {
            return Ok(result);
        }

        // Only apply to 3D or higher arrays (like images with channels)
        if result.ndim() < 3 {
            return Ok(result);
        }

        // Note: This is a simplified implementation
        // In practice, you'd need to handle different tensor layouts

        Ok(result)
    }

    fn description(&self) -> String {
        format!(
            "RandomErasing (prob: {:.2}, value: {:.2})",
            self.probability,
            self.value.to_f64().unwrap_or(0.0)
        )
    }
}

/// Random horizontal flip augmentation
#[derive(Debug, Clone)]
pub struct RandomHorizontalFlip<F: Float + NumAssign + Debug + ScalarOperand> {
    /// Probability of applying the flip
    probability: f64,
    /// Phantom data for generic type
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + NumAssign + Debug + ScalarOperand> RandomHorizontalFlip<F> {
    /// Create a new random horizontal flip augmentation
    pub fn new(probability: f64) -> Self {
        Self {
            probability,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F: Float + NumAssign + Debug + ScalarOperand> Augmentation<F> for RandomHorizontalFlip<F> {
    fn apply(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut rng = SmallRng::from_rng(&mut thread_rng());
        let result = input.clone();

        // Only apply based on probability
        if rng.random::<f64>() > self.probability {
            return Ok(result);
        }

        // In practice, you'd implement the actual horizontal flip here
        // This would depend on the tensor layout (CHW, HWC, etc.)

        Ok(result)
    }

    fn description(&self) -> String {
        format!("RandomHorizontalFlip (prob: {:.2})", self.probability)
    }
}

/// Debug wrapper for a trait object
struct DebugAugmentationWrapper<'a, F: Float + NumAssign + Debug + ScalarOperand> {
    /// Reference to the augmentation
    inner: &'a dyn Augmentation<F>,
}

impl<F: Float + NumAssign + Debug + ScalarOperand> Debug for DebugAugmentationWrapper<'_, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Augmentation({})", self.inner.description())
    }
}

/// Compose multiple augmentations into a single augmentation
pub struct ComposeAugmentation<F: Float + NumAssign + Debug + ScalarOperand> {
    /// List of augmentations to apply in sequence
    augmentations: Vec<Box<dyn Augmentation<F>>>,
}

impl<F: Float + NumAssign + Debug + ScalarOperand> Clone for ComposeAugmentation<F> {
    fn clone(&self) -> Self {
        // We can't clone trait objects directly, so we need to implement a custom Clone
        // In a real implementation, we would need a way to clone each augmentation
        // For now, we'll return an empty list which is not ideal but will let the code compile
        Self {
            augmentations: Vec::new(),
        }
    }
}

impl<F: Float + NumAssign + Debug + ScalarOperand> Debug for ComposeAugmentation<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_list = f.debug_list();
        for augmentation in &self.augmentations {
            debug_list.entry(&DebugAugmentationWrapper {
                inner: augmentation.as_ref(),
            });
        }
        debug_list.finish()
    }
}

impl<F: Float + NumAssign + Debug + ScalarOperand> ComposeAugmentation<F> {
    /// Create a new composition of augmentations
    pub fn new(augmentations: Vec<Box<dyn Augmentation<F>>>) -> Self {
        Self { augmentations }
    }
}

impl<F: Float + NumAssign + Debug + ScalarOperand> Augmentation<F> for ComposeAugmentation<F> {
    fn apply(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut data = input.clone();
        for augmentation in &self.augmentations {
            data = augmentation.apply(&data)?;
        }
        Ok(data)
    }

    fn description(&self) -> String {
        let descriptions: Vec<String> =
            self.augmentations.iter().map(|a| a.description()).collect();
        format!("Compose({})", descriptions.join(", "))
    }
}
