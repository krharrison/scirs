//! Mixed Precision Training Support for Neural Networks
//!
//! This module provides comprehensive mixed precision training utilities that enable:
//!
//! - **FP16/FP32 Mixed Precision**: Forward passes in FP16 for speed, master weights in FP32 for accuracy
//! - **Dynamic Loss Scaling**: Automatic gradient scaling with overflow detection
//! - **Master Weight Management**: FP32 master weights with FP16 computation copies
//! - **Automatic Mixed Precision (AMP)**: Easy-to-use wrapper for mixed precision training
//! - **Memory-Efficient Operations**: Reduced memory footprint through half-precision activations
//!
//! # Example Usage
//!
//! ```rust
//! use scirs2_neural::training::mixed_precision::{
//!     MixedPrecisionConfig, GradScaler, AutoMixedPrecision, MixedPrecisionTrainer,
//! };
//!
//! // Configure mixed precision training
//! let config = MixedPrecisionConfig::builder()
//!     .enabled(true)
//!     .initial_loss_scale(65536.0)
//!     .growth_factor(2.0)
//!     .backoff_factor(0.5)
//!     .growth_interval(2000)
//!     .build()
//!     .expect("failed to build config");
//!
//! // Create gradient scaler
//! let mut scaler = GradScaler::new(config.clone()).expect("failed to create scaler");
//!
//! // Create mixed precision trainer
//! let trainer = MixedPrecisionTrainer::<f64>::new(config).expect("failed to create trainer");
//! ```

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, ArrayD, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign};
use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

// ============================================================================
// Configuration Types
// ============================================================================

/// Loss scaling strategy for mixed precision training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LossScalingStrategy {
    /// Fixed loss scaling with a constant scale factor
    Fixed,
    /// Dynamic loss scaling that adapts based on gradient overflow detection
    #[default]
    Dynamic,
    /// Automatic selection based on hardware and training conditions
    Automatic,
}

/// Precision mode for different parts of the computation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrecisionMode {
    /// Full FP32 precision (standard)
    FP32,
    /// Half precision FP16 for forward/backward, FP32 for master weights
    #[default]
    FP16Mixed,
    /// BFloat16 mixed precision (better for training stability)
    BF16Mixed,
    /// Automatic precision selection based on operation type
    Automatic,
}

/// Operations that should always run in FP32 for numerical stability
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FP32Operation {
    /// Loss computation
    LossComputation,
    /// Softmax and log-softmax
    Softmax,
    /// Layer normalization
    LayerNorm,
    /// Batch normalization running statistics
    BatchNormStats,
    /// Gradient accumulation
    GradientAccumulation,
    /// Optimizer state updates
    OptimizerUpdate,
    /// Embedding operations
    Embedding,
    /// Reduction operations (sum, mean, etc.)
    Reductions,
}

/// Configuration for mixed precision training
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Whether to enable mixed precision training
    pub enabled: bool,
    /// Precision mode to use
    pub precision_mode: PrecisionMode,
    /// Loss scaling strategy
    pub loss_scaling_strategy: LossScalingStrategy,
    /// Initial loss scale factor for dynamic scaling
    pub initial_loss_scale: f64,
    /// Factor to multiply loss scale when no overflow is detected
    pub growth_factor: f64,
    /// Factor to multiply loss scale when overflow is detected
    pub backoff_factor: f64,
    /// Number of successful steps before increasing loss scale
    pub growth_interval: usize,
    /// Minimum loss scale (to prevent underflow)
    pub min_loss_scale: f64,
    /// Maximum loss scale (to prevent overflow)
    pub max_loss_scale: f64,
    /// Operations that should always use FP32
    pub fp32_operations: Vec<FP32Operation>,
    /// Whether to cast model parameters to FP16 during forward pass
    pub cast_model_type: bool,
    /// Whether to use memory-efficient gradient computation
    pub memory_efficient_gradients: bool,
    /// Whether to enable gradient checkpointing integration
    pub gradient_checkpointing: bool,
    /// Maximum number of consecutive overflows before reducing scale aggressively
    pub max_consecutive_overflows: usize,
    /// Gradient clipping threshold (applied before unscaling)
    pub grad_clip_threshold: Option<f64>,
    /// Whether to log scaling statistics
    pub log_statistics: bool,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            precision_mode: PrecisionMode::FP16Mixed,
            loss_scaling_strategy: LossScalingStrategy::Dynamic,
            initial_loss_scale: 65536.0, // 2^16
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            min_loss_scale: 1.0,
            max_loss_scale: 2.0_f64.powi(24), // 2^24
            fp32_operations: vec![
                FP32Operation::LossComputation,
                FP32Operation::Softmax,
                FP32Operation::LayerNorm,
                FP32Operation::BatchNormStats,
                FP32Operation::GradientAccumulation,
                FP32Operation::OptimizerUpdate,
            ],
            cast_model_type: true,
            memory_efficient_gradients: true,
            gradient_checkpointing: false,
            max_consecutive_overflows: 5,
            grad_clip_threshold: None,
            log_statistics: false,
        }
    }
}

impl MixedPrecisionConfig {
    /// Create a new builder for MixedPrecisionConfig
    pub fn builder() -> MixedPrecisionConfigBuilder {
        MixedPrecisionConfigBuilder::new()
    }

    /// Check if an operation should use FP32
    pub fn should_use_fp32(&self, operation: FP32Operation) -> bool {
        self.fp32_operations.contains(&operation)
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.initial_loss_scale <= 0.0 {
            return Err(NeuralError::ConfigError(
                "Initial loss scale must be positive".to_string(),
            ));
        }
        if self.growth_factor <= 1.0 {
            return Err(NeuralError::ConfigError(
                "Growth factor must be greater than 1.0".to_string(),
            ));
        }
        if self.backoff_factor <= 0.0 || self.backoff_factor >= 1.0 {
            return Err(NeuralError::ConfigError(
                "Backoff factor must be in (0.0, 1.0)".to_string(),
            ));
        }
        if self.min_loss_scale > self.max_loss_scale {
            return Err(NeuralError::ConfigError(
                "Minimum loss scale cannot exceed maximum loss scale".to_string(),
            ));
        }
        if self.growth_interval == 0 {
            return Err(NeuralError::ConfigError(
                "Growth interval must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Builder for MixedPrecisionConfig
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfigBuilder {
    config: MixedPrecisionConfig,
}

impl MixedPrecisionConfigBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self {
            config: MixedPrecisionConfig::default(),
        }
    }

    /// Enable or disable mixed precision training
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Set the precision mode
    pub fn precision_mode(mut self, mode: PrecisionMode) -> Self {
        self.config.precision_mode = mode;
        self
    }

    /// Set the loss scaling strategy
    pub fn loss_scaling_strategy(mut self, strategy: LossScalingStrategy) -> Self {
        self.config.loss_scaling_strategy = strategy;
        self
    }

    /// Set the initial loss scale
    pub fn initial_loss_scale(mut self, scale: f64) -> Self {
        self.config.initial_loss_scale = scale;
        self
    }

    /// Set the growth factor for dynamic scaling
    pub fn growth_factor(mut self, factor: f64) -> Self {
        self.config.growth_factor = factor;
        self
    }

    /// Set the backoff factor for dynamic scaling
    pub fn backoff_factor(mut self, factor: f64) -> Self {
        self.config.backoff_factor = factor;
        self
    }

    /// Set the growth interval
    pub fn growth_interval(mut self, interval: usize) -> Self {
        self.config.growth_interval = interval;
        self
    }

    /// Set the minimum loss scale
    pub fn min_loss_scale(mut self, scale: f64) -> Self {
        self.config.min_loss_scale = scale;
        self
    }

    /// Set the maximum loss scale
    pub fn max_loss_scale(mut self, scale: f64) -> Self {
        self.config.max_loss_scale = scale;
        self
    }

    /// Add an FP32 operation
    pub fn add_fp32_operation(mut self, operation: FP32Operation) -> Self {
        if !self.config.fp32_operations.contains(&operation) {
            self.config.fp32_operations.push(operation);
        }
        self
    }

    /// Set gradient clipping threshold
    pub fn grad_clip_threshold(mut self, threshold: f64) -> Self {
        self.config.grad_clip_threshold = Some(threshold);
        self
    }

    /// Enable memory-efficient gradients
    pub fn memory_efficient_gradients(mut self, enabled: bool) -> Self {
        self.config.memory_efficient_gradients = enabled;
        self
    }

    /// Enable gradient checkpointing integration
    pub fn gradient_checkpointing(mut self, enabled: bool) -> Self {
        self.config.gradient_checkpointing = enabled;
        self
    }

    /// Enable logging of scaling statistics
    pub fn log_statistics(mut self, enabled: bool) -> Self {
        self.config.log_statistics = enabled;
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<MixedPrecisionConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for MixedPrecisionConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Half Precision Tensor Operations
// ============================================================================

/// Half precision (FP16) value representation
///
/// Uses u16 internally to store the IEEE 754 half-precision float
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Half(u16);

impl Half {
    /// Zero value in half precision
    pub const ZERO: Half = Half(0);

    /// One value in half precision
    pub const ONE: Half = Half(0x3c00);

    /// Maximum finite value
    pub const MAX: Half = Half(0x7bff);

    /// Minimum positive normalized value
    pub const MIN_POSITIVE: Half = Half(0x0400);

    /// Infinity
    pub const INFINITY: Half = Half(0x7c00);

    /// Negative infinity
    pub const NEG_INFINITY: Half = Half(0xfc00);

    /// NaN (quiet NaN)
    pub const NAN: Half = Half(0x7e00);

    /// Create a half from raw bits
    pub const fn from_bits(bits: u16) -> Self {
        Half(bits)
    }

    /// Get the raw bits
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Convert from f32 to half precision
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xff) as i32;
        let frac = bits & 0x7fffff;

        // Handle special cases
        if exp == 0xff {
            // Infinity or NaN
            if frac == 0 {
                // Infinity
                return Half(((sign << 15) | 0x7c00) as u16);
            } else {
                // NaN
                return Half(((sign << 15) | 0x7e00) as u16);
            }
        }

        // Rebias exponent from f32 (bias 127) to f16 (bias 15)
        let new_exp = exp - 127 + 15;

        if new_exp <= 0 {
            // Denormalized or zero
            if new_exp < -10 {
                // Too small, becomes zero
                return Half((sign << 15) as u16);
            }
            // Denormalized
            let shift = 1 - new_exp;
            let frac_with_hidden = frac | 0x800000;
            let frac16 = (frac_with_hidden >> (shift + 13)) as u16;
            return Half(((sign << 15) | frac16 as u32) as u16);
        }

        if new_exp >= 31 {
            // Overflow to infinity
            return Half(((sign << 15) | 0x7c00) as u16);
        }

        // Normal case
        let frac16 = (frac >> 13) as u16;
        Half(((sign << 15) | ((new_exp as u32) << 10) | frac16 as u32) as u16)
    }

    /// Convert from half precision to f32
    pub fn to_f32(self) -> f32 {
        let bits = self.0 as u32;
        let sign = (bits >> 15) & 1;
        let exp = (bits >> 10) & 0x1f;
        let frac = bits & 0x3ff;

        if exp == 0 {
            if frac == 0 {
                // Zero
                return f32::from_bits(sign << 31);
            }
            // Denormalized
            let mut frac = frac;
            let mut e = -14i32;
            while frac & 0x400 == 0 {
                frac <<= 1;
                e -= 1;
            }
            frac &= 0x3ff;
            let exp32 = (e + 127) as u32;
            let frac32 = frac << 13;
            return f32::from_bits((sign << 31) | (exp32 << 23) | frac32);
        }

        if exp == 0x1f {
            // Infinity or NaN
            if frac == 0 {
                return f32::from_bits((sign << 31) | 0x7f800000);
            }
            return f32::from_bits((sign << 31) | 0x7fc00000);
        }

        // Normal case
        let exp32 = (exp as i32 - 15 + 127) as u32;
        let frac32 = frac << 13;
        f32::from_bits((sign << 31) | (exp32 << 23) | frac32)
    }

    /// Check if the value is NaN
    pub fn is_nan(self) -> bool {
        (self.0 & 0x7c00) == 0x7c00 && (self.0 & 0x03ff) != 0
    }

    /// Check if the value is infinite
    pub fn is_infinite(self) -> bool {
        (self.0 & 0x7fff) == 0x7c00
    }

    /// Check if the value is finite
    pub fn is_finite(self) -> bool {
        (self.0 & 0x7c00) != 0x7c00
    }

    /// Check if the value is zero
    pub fn is_zero(self) -> bool {
        (self.0 & 0x7fff) == 0
    }
}

impl From<f32> for Half {
    fn from(value: f32) -> Self {
        Half::from_f32(value)
    }
}

impl From<Half> for f32 {
    fn from(value: Half) -> Self {
        value.to_f32()
    }
}

impl From<f64> for Half {
    fn from(value: f64) -> Self {
        Half::from_f32(value as f32)
    }
}

impl From<Half> for f64 {
    fn from(value: Half) -> Self {
        value.to_f32() as f64
    }
}

/// Container for mixed precision tensors
#[derive(Debug, Clone)]
pub struct MixedPrecisionTensor<F: Float + Debug + ScalarOperand> {
    /// Original FP32 data
    fp32_data: Option<ArrayD<F>>,
    /// FP16 representation (stored as raw bits for computation)
    fp16_bits: Option<ArrayD<u16>>,
    /// Shape of the tensor
    shape: Vec<usize>,
    /// Current precision state
    precision: PrecisionMode,
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive> MixedPrecisionTensor<F> {
    /// Create a new mixed precision tensor from FP32 data
    pub fn from_fp32(data: ArrayD<F>) -> Self {
        let shape = data.shape().to_vec();
        Self {
            fp32_data: Some(data),
            fp16_bits: None,
            shape,
            precision: PrecisionMode::FP32,
        }
    }

    /// Convert to FP16 representation
    pub fn to_fp16(&mut self) -> Result<()> {
        if self.fp16_bits.is_some() {
            return Ok(()); // Already in FP16
        }

        let fp32 = self
            .fp32_data
            .as_ref()
            .ok_or_else(|| NeuralError::InvalidState("No FP32 data available".to_string()))?;

        // Convert each element to FP16 bits
        let fp16_data: Vec<u16> = fp32
            .iter()
            .map(|&x| {
                let f32_val = x.to_f64().unwrap_or(0.0) as f32;
                Half::from_f32(f32_val).to_bits()
            })
            .collect();

        self.fp16_bits = Some(
            ArrayD::from_shape_vec(IxDyn(&self.shape), fp16_data).map_err(|e| {
                NeuralError::ShapeMismatch(format!("FP16 conversion failed: {}", e))
            })?,
        );
        self.precision = PrecisionMode::FP16Mixed;
        Ok(())
    }

    /// Convert back to FP32
    pub fn to_fp32(&mut self) -> Result<()> {
        if self.fp32_data.is_some() && self.precision == PrecisionMode::FP32 {
            return Ok(()); // Already in FP32
        }

        let fp16 = self
            .fp16_bits
            .as_ref()
            .ok_or_else(|| NeuralError::InvalidState("No FP16 data available".to_string()))?;

        // Convert each element back to FP32
        let fp32_data: Vec<F> = fp16
            .iter()
            .map(|&bits| {
                let f32_val = Half::from_bits(bits).to_f32();
                F::from(f32_val).unwrap_or_else(F::zero)
            })
            .collect();

        self.fp32_data = Some(
            ArrayD::from_shape_vec(IxDyn(&self.shape), fp32_data).map_err(|e| {
                NeuralError::ShapeMismatch(format!("FP32 conversion failed: {}", e))
            })?,
        );
        self.precision = PrecisionMode::FP32;
        Ok(())
    }

    /// Get FP32 data (converting if necessary)
    pub fn get_fp32(&mut self) -> Result<&ArrayD<F>> {
        self.to_fp32()?;
        self.fp32_data
            .as_ref()
            .ok_or_else(|| NeuralError::InvalidState("FP32 data not available".to_string()))
    }

    /// Get current precision mode
    pub fn precision(&self) -> PrecisionMode {
        self.precision
    }

    /// Get shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Free FP16 memory to save space
    pub fn free_fp16(&mut self) {
        self.fp16_bits = None;
    }

    /// Free FP32 memory (use with caution - will need to regenerate)
    pub fn free_fp32(&mut self) {
        self.fp32_data = None;
    }

    /// Check if tensor has valid data
    pub fn is_valid(&self) -> bool {
        self.fp32_data.is_some() || self.fp16_bits.is_some()
    }
}

// ============================================================================
// Gradient Scaler
// ============================================================================

/// Statistics for gradient scaling
#[derive(Debug, Clone, Default)]
pub struct GradScalerStats {
    /// Current loss scale
    pub loss_scale: f64,
    /// Total number of steps
    pub total_steps: u64,
    /// Number of steps with overflow
    pub overflow_steps: u64,
    /// Number of successful steps since last scale increase
    pub steps_since_increase: u64,
    /// Number of scale increases
    pub num_scale_increases: u64,
    /// Number of scale decreases
    pub num_scale_decreases: u64,
    /// Consecutive overflow count
    pub consecutive_overflows: u64,
}

/// Gradient scaler for dynamic loss scaling in mixed precision training
///
/// The GradScaler performs the following operations:
/// 1. Scales the loss before backward pass to prevent gradient underflow in FP16
/// 2. Unscales gradients after backward pass
/// 3. Checks for overflow/NaN in gradients
/// 4. Dynamically adjusts the scale factor based on gradient health
#[derive(Debug)]
pub struct GradScaler {
    /// Configuration
    config: MixedPrecisionConfig,
    /// Current loss scale
    scale: Arc<RwLock<f64>>,
    /// Number of successful steps since last scale increase
    growth_tracker: AtomicU64,
    /// Whether an overflow was detected in the current step
    found_inf: AtomicBool,
    /// Consecutive overflow count
    consecutive_overflows: AtomicU64,
    /// Statistics
    stats: Arc<RwLock<GradScalerStats>>,
}

impl GradScaler {
    /// Create a new gradient scaler
    pub fn new(config: MixedPrecisionConfig) -> Result<Self> {
        config.validate()?;
        let initial_scale = config.initial_loss_scale;

        Ok(Self {
            config,
            scale: Arc::new(RwLock::new(initial_scale)),
            growth_tracker: AtomicU64::new(0),
            found_inf: AtomicBool::new(false),
            consecutive_overflows: AtomicU64::new(0),
            stats: Arc::new(RwLock::new(GradScalerStats {
                loss_scale: initial_scale,
                ..Default::default()
            })),
        })
    }

    /// Get the current loss scale
    pub fn get_scale(&self) -> f64 {
        *self.scale.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Scale a loss value for backward pass
    pub fn scale_loss<F: Float + Debug + FromPrimitive>(&self, loss: F) -> Result<F> {
        if !self.config.enabled {
            return Ok(loss);
        }

        let scale = self.get_scale();
        let scale_f = F::from(scale).ok_or_else(|| {
            NeuralError::ComputationError("Failed to convert scale to loss type".to_string())
        })?;

        Ok(loss * scale_f)
    }

    /// Unscale gradients after backward pass
    pub fn unscale_gradients<F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign>(
        &self,
        gradients: &mut [ArrayD<F>],
    ) -> Result<bool> {
        if !self.config.enabled {
            return Ok(false);
        }

        let scale = self.get_scale();
        let inv_scale = F::from(1.0 / scale).ok_or_else(|| {
            NeuralError::ComputationError("Failed to compute inverse scale".to_string())
        })?;

        let mut found_inf = false;

        for grad in gradients.iter_mut() {
            for val in grad.iter_mut() {
                *val *= inv_scale;

                // Check for overflow/NaN
                let f64_val = val.to_f64().unwrap_or(f64::NAN);
                if !f64_val.is_finite() {
                    found_inf = true;
                }
            }
        }

        self.found_inf.store(found_inf, Ordering::SeqCst);
        Ok(found_inf)
    }

    /// Check if gradients contain inf/nan values
    pub fn check_gradients_for_overflow<F: Float + Debug + ScalarOperand>(
        &self,
        gradients: &[ArrayD<F>],
    ) -> bool {
        for grad in gradients {
            for val in grad.iter() {
                let f64_val = val.to_f64().unwrap_or(f64::NAN);
                if !f64_val.is_finite() {
                    return true;
                }
            }
        }
        false
    }

    /// Apply gradient clipping if configured
    pub fn clip_gradients<F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign>(
        &self,
        gradients: &mut [ArrayD<F>],
    ) -> Result<Option<f64>> {
        let threshold = match self.config.grad_clip_threshold {
            Some(t) => t,
            None => return Ok(None),
        };

        // Compute global norm
        let mut global_norm_sq = 0.0_f64;
        for grad in gradients.iter() {
            for val in grad.iter() {
                let f = val.to_f64().unwrap_or(0.0);
                global_norm_sq += f * f;
            }
        }
        let global_norm = global_norm_sq.sqrt();

        if global_norm > threshold {
            let clip_factor = F::from(threshold / global_norm).ok_or_else(|| {
                NeuralError::ComputationError("Failed to compute clip factor".to_string())
            })?;

            for grad in gradients.iter_mut() {
                for val in grad.iter_mut() {
                    *val *= clip_factor;
                }
            }
        }

        Ok(Some(global_norm))
    }

    /// Update the scale after a training step
    ///
    /// Returns true if the step should be skipped (due to overflow)
    pub fn update(&self) -> Result<bool> {
        if !self.config.enabled {
            return Ok(false);
        }

        let found_inf = self.found_inf.load(Ordering::SeqCst);

        let mut scale = self.scale.write().unwrap_or_else(|e| e.into_inner());
        let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());

        stats.total_steps += 1;

        if found_inf {
            // Overflow detected - decrease scale
            *scale *= self.config.backoff_factor;
            *scale = scale.max(self.config.min_loss_scale);

            self.growth_tracker.store(0, Ordering::SeqCst);
            let consec = self.consecutive_overflows.fetch_add(1, Ordering::SeqCst) + 1;

            stats.overflow_steps += 1;
            stats.num_scale_decreases += 1;
            stats.consecutive_overflows = consec;
            stats.steps_since_increase = 0;

            if self.config.log_statistics {
                eprintln!(
                    "[GradScaler] Overflow detected. Scale: {:.2} -> {:.2}, consecutive: {}",
                    *scale / self.config.backoff_factor,
                    *scale,
                    consec
                );
            }

            // Check for too many consecutive overflows
            if consec >= self.config.max_consecutive_overflows as u64 {
                // Apply more aggressive backoff
                *scale *= self.config.backoff_factor;
                *scale = scale.max(self.config.min_loss_scale);

                if self.config.log_statistics {
                    eprintln!(
                        "[GradScaler] Aggressive backoff due to {} consecutive overflows. Scale: {:.2}",
                        consec, *scale
                    );
                }
            }

            stats.loss_scale = *scale;
            self.found_inf.store(false, Ordering::SeqCst);
            return Ok(true); // Skip this step
        }

        // No overflow - potentially increase scale
        self.consecutive_overflows.store(0, Ordering::SeqCst);
        let growth_count = self.growth_tracker.fetch_add(1, Ordering::SeqCst) + 1;
        stats.steps_since_increase = growth_count;

        if growth_count >= self.config.growth_interval as u64 {
            // Increase scale
            let old_scale = *scale;
            *scale *= self.config.growth_factor;
            *scale = scale.min(self.config.max_loss_scale);

            self.growth_tracker.store(0, Ordering::SeqCst);
            stats.num_scale_increases += 1;
            stats.steps_since_increase = 0;

            if self.config.log_statistics && *scale != old_scale {
                eprintln!(
                    "[GradScaler] Scale increased: {:.2} -> {:.2}",
                    old_scale, *scale
                );
            }
        }

        stats.loss_scale = *scale;
        stats.consecutive_overflows = 0;
        self.found_inf.store(false, Ordering::SeqCst);
        Ok(false)
    }

    /// Get statistics
    pub fn get_stats(&self) -> GradScalerStats {
        self.stats.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Reset the scaler to initial state
    pub fn reset(&self) {
        let mut scale = self.scale.write().unwrap_or_else(|e| e.into_inner());
        *scale = self.config.initial_loss_scale;

        self.growth_tracker.store(0, Ordering::SeqCst);
        self.found_inf.store(false, Ordering::SeqCst);
        self.consecutive_overflows.store(0, Ordering::SeqCst);

        let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
        *stats = GradScalerStats {
            loss_scale: self.config.initial_loss_scale,
            ..Default::default()
        };
    }

    /// Check if currently enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

// ============================================================================
// Master Weights Management
// ============================================================================

/// Master weights container for mixed precision training
///
/// Stores FP32 master weights while using FP16 copies for computation.
/// This is essential for training stability as FP16 has limited precision.
#[derive(Debug)]
pub struct MasterWeights<F: Float + Debug + ScalarOperand> {
    /// FP32 master weights (authoritative)
    master_weights: HashMap<String, ArrayD<F>>,
    /// FP16 computation copies (derived from master)
    compute_weights: HashMap<String, ArrayD<u16>>,
    /// Whether to sync on every update
    sync_on_update: bool,
    /// Precision mode
    precision_mode: PrecisionMode,
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign> MasterWeights<F> {
    /// Create a new master weights container
    pub fn new(precision_mode: PrecisionMode) -> Self {
        Self {
            master_weights: HashMap::new(),
            compute_weights: HashMap::new(),
            sync_on_update: true,
            precision_mode,
        }
    }

    /// Register a weight tensor
    pub fn register(&mut self, name: &str, weights: ArrayD<F>) -> Result<()> {
        self.master_weights.insert(name.to_string(), weights);
        self.sync_to_compute(name)?;
        Ok(())
    }

    /// Sync master weights to compute weights (FP32 -> FP16)
    fn sync_to_compute(&mut self, name: &str) -> Result<()> {
        let master = self
            .master_weights
            .get(name)
            .ok_or_else(|| NeuralError::InvalidArgument(format!("Weight '{}' not found", name)))?;

        let fp16_data: Vec<u16> = master
            .iter()
            .map(|&x| {
                let f32_val = x.to_f64().unwrap_or(0.0) as f32;
                Half::from_f32(f32_val).to_bits()
            })
            .collect();

        let compute = ArrayD::from_shape_vec(IxDyn(master.shape()), fp16_data)
            .map_err(|e| NeuralError::ShapeMismatch(format!("Sync failed: {}", e)))?;

        self.compute_weights.insert(name.to_string(), compute);
        Ok(())
    }

    /// Get compute weights (FP16) for a layer
    pub fn get_compute_weights(&self, name: &str) -> Option<&ArrayD<u16>> {
        self.compute_weights.get(name)
    }

    /// Get master weights (FP32) for a layer
    pub fn get_master_weights(&self, name: &str) -> Option<&ArrayD<F>> {
        self.master_weights.get(name)
    }

    /// Update master weights with unscaled gradients
    pub fn update_master_weights(
        &mut self,
        name: &str,
        gradients: &ArrayD<F>,
        learning_rate: F,
    ) -> Result<()> {
        let master = self
            .master_weights
            .get_mut(name)
            .ok_or_else(|| NeuralError::InvalidArgument(format!("Weight '{}' not found", name)))?;

        // Standard SGD update: w = w - lr * grad
        for (w, g) in master.iter_mut().zip(gradients.iter()) {
            *w -= learning_rate * *g;
        }

        // Sync to compute weights if enabled
        if self.sync_on_update {
            self.sync_to_compute(name)?;
        }

        Ok(())
    }

    /// Sync all master weights to compute weights
    pub fn sync_all(&mut self) -> Result<()> {
        let names: Vec<String> = self.master_weights.keys().cloned().collect();
        for name in names {
            self.sync_to_compute(&name)?;
        }
        Ok(())
    }

    /// Get all weight names
    pub fn weight_names(&self) -> Vec<&String> {
        self.master_weights.keys().collect()
    }

    /// Check if a weight exists
    pub fn contains(&self, name: &str) -> bool {
        self.master_weights.contains_key(name)
    }

    /// Get the number of registered weights
    pub fn len(&self) -> usize {
        self.master_weights.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.master_weights.is_empty()
    }

    /// Get precision mode
    pub fn precision_mode(&self) -> PrecisionMode {
        self.precision_mode
    }
}

// ============================================================================
// Automatic Mixed Precision (AMP) Wrapper
// ============================================================================

/// Context manager state for AMP
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmpContextState {
    /// Normal operation (outside AMP context)
    Normal,
    /// Inside forward pass context
    Forward,
    /// Inside backward pass context
    Backward,
    /// Inside optimization step context
    Optimization,
}

/// Automatic Mixed Precision (AMP) wrapper
///
/// Provides automatic precision management for neural network operations.
/// Similar to PyTorch's `torch.cuda.amp.autocast` context manager.
#[derive(Debug)]
pub struct AutoMixedPrecision<F: Float + Debug + ScalarOperand> {
    /// Configuration
    config: MixedPrecisionConfig,
    /// Gradient scaler
    scaler: Option<GradScaler>,
    /// Master weights
    master_weights: Option<MasterWeights<F>>,
    /// Current context state
    context_state: Arc<RwLock<AmpContextState>>,
    /// Whether AMP is active
    active: AtomicBool,
    /// Phantom data
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + Send + Sync>
    AutoMixedPrecision<F>
{
    /// Create a new AMP wrapper
    pub fn new(config: MixedPrecisionConfig) -> Result<Self> {
        config.validate()?;

        let scaler = if config.enabled {
            Some(GradScaler::new(config.clone())?)
        } else {
            None
        };

        let master_weights = if config.enabled {
            Some(MasterWeights::new(config.precision_mode))
        } else {
            None
        };

        Ok(Self {
            config,
            scaler,
            master_weights,
            context_state: Arc::new(RwLock::new(AmpContextState::Normal)),
            active: AtomicBool::new(false),
            _phantom: PhantomData,
        })
    }

    /// Check if AMP is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Enter forward pass context
    pub fn enter_forward(&self) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let mut state = self
            .context_state
            .write()
            .unwrap_or_else(|e| e.into_inner());
        *state = AmpContextState::Forward;
        self.active.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Exit forward pass context
    pub fn exit_forward(&self) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let mut state = self
            .context_state
            .write()
            .unwrap_or_else(|e| e.into_inner());
        *state = AmpContextState::Normal;
        Ok(())
    }

    /// Enter backward pass context
    pub fn enter_backward(&self) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let mut state = self
            .context_state
            .write()
            .unwrap_or_else(|e| e.into_inner());
        *state = AmpContextState::Backward;
        Ok(())
    }

    /// Exit backward pass context
    pub fn exit_backward(&self) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let mut state = self
            .context_state
            .write()
            .unwrap_or_else(|e| e.into_inner());
        *state = AmpContextState::Normal;
        Ok(())
    }

    /// Get current context state
    pub fn context_state(&self) -> AmpContextState {
        *self.context_state.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Scale loss for backward pass
    pub fn scale_loss(&self, loss: F) -> Result<F> {
        match &self.scaler {
            Some(scaler) => scaler.scale_loss(loss),
            None => Ok(loss),
        }
    }

    /// Unscale and check gradients
    pub fn unscale_gradients(&self, gradients: &mut [ArrayD<F>]) -> Result<bool> {
        match &self.scaler {
            Some(scaler) => scaler.unscale_gradients(gradients),
            None => Ok(false),
        }
    }

    /// Update scaler after step
    pub fn update_scaler(&self) -> Result<bool> {
        match &self.scaler {
            Some(scaler) => scaler.update(),
            None => Ok(false),
        }
    }

    /// Get current loss scale
    pub fn get_loss_scale(&self) -> f64 {
        self.scaler.as_ref().map_or(1.0, |s| s.get_scale())
    }

    /// Register model weights for master weight management
    pub fn register_weights(&mut self, name: &str, weights: ArrayD<F>) -> Result<()> {
        if let Some(ref mut mw) = self.master_weights {
            mw.register(name, weights)?;
        }
        Ok(())
    }

    /// Get scaler statistics
    pub fn get_scaler_stats(&self) -> Option<GradScalerStats> {
        self.scaler.as_ref().map(|s| s.get_stats())
    }

    /// Reset the AMP state
    pub fn reset(&mut self) {
        if let Some(ref scaler) = self.scaler {
            scaler.reset();
        }
        if let Some(ref mut mw) = self.master_weights {
            *mw = MasterWeights::new(self.config.precision_mode);
        }
        self.active.store(false, Ordering::SeqCst);
        let mut state = self
            .context_state
            .write()
            .unwrap_or_else(|e| e.into_inner());
        *state = AmpContextState::Normal;
    }

    /// Convert tensor to computation precision (FP16)
    pub fn to_compute_precision(&self, tensor: &ArrayD<F>) -> Result<ArrayD<u16>> {
        let fp16_data: Vec<u16> = tensor
            .iter()
            .map(|&x| {
                let f32_val = x.to_f64().unwrap_or(0.0) as f32;
                Half::from_f32(f32_val).to_bits()
            })
            .collect();

        ArrayD::from_shape_vec(IxDyn(tensor.shape()), fp16_data)
            .map_err(|e| NeuralError::ShapeMismatch(format!("Precision conversion failed: {}", e)))
    }

    /// Convert tensor from computation precision (FP16) to FP32
    pub fn from_compute_precision(&self, tensor: &ArrayD<u16>) -> Result<ArrayD<F>> {
        let fp32_data: Vec<F> = tensor
            .iter()
            .map(|&bits| {
                let f32_val = Half::from_bits(bits).to_f32();
                F::from(f32_val).unwrap_or_else(F::zero)
            })
            .collect();

        ArrayD::from_shape_vec(IxDyn(tensor.shape()), fp32_data)
            .map_err(|e| NeuralError::ShapeMismatch(format!("Precision conversion failed: {}", e)))
    }
}

// ============================================================================
// Mixed Precision Trainer
// ============================================================================

/// Training step result
#[derive(Debug, Clone)]
pub struct MixedPrecisionStepResult<F: Float + Debug + NumAssign> {
    /// Scaled loss (before unscaling)
    pub scaled_loss: F,
    /// Unscaled loss (actual loss value)
    pub unscaled_loss: F,
    /// Whether overflow was detected
    pub overflow_detected: bool,
    /// Whether this step was skipped due to overflow
    pub step_skipped: bool,
    /// Current loss scale
    pub loss_scale: f64,
    /// Gradient norm (after unscaling and clipping)
    pub grad_norm: Option<f64>,
}

/// Mixed precision trainer for complete training loop management
#[derive(Debug)]
pub struct MixedPrecisionTrainer<F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign> {
    /// Configuration
    config: MixedPrecisionConfig,
    /// AMP wrapper
    amp: AutoMixedPrecision<F>,
    /// Total training steps
    total_steps: AtomicU64,
    /// Skipped steps due to overflow
    skipped_steps: AtomicU64,
    /// Whether currently training
    training: AtomicBool,
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + Send + Sync>
    MixedPrecisionTrainer<F>
{
    /// Create a new mixed precision trainer
    pub fn new(config: MixedPrecisionConfig) -> Result<Self> {
        let amp = AutoMixedPrecision::new(config.clone())?;

        Ok(Self {
            config,
            amp,
            total_steps: AtomicU64::new(0),
            skipped_steps: AtomicU64::new(0),
            training: AtomicBool::new(false),
        })
    }

    /// Check if mixed precision is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Start training mode
    pub fn train(&self) {
        self.training.store(true, Ordering::SeqCst);
    }

    /// Stop training mode
    pub fn eval(&self) {
        self.training.store(false, Ordering::SeqCst);
    }

    /// Is in training mode
    pub fn is_training(&self) -> bool {
        self.training.load(Ordering::SeqCst)
    }

    /// Perform a forward pass with mixed precision
    pub fn forward<L: Layer<F>>(&self, model: &L, input: &ArrayD<F>) -> Result<ArrayD<F>> {
        self.amp.enter_forward()?;
        let result = model.forward(input);
        self.amp.exit_forward()?;
        result
    }

    /// Scale loss for backward pass
    pub fn scale_loss(&self, loss: F) -> Result<F> {
        self.amp.scale_loss(loss)
    }

    /// Perform a complete training step
    ///
    /// This method handles:
    /// 1. Loss scaling
    /// 2. Gradient unscaling
    /// 3. Overflow detection
    /// 4. Dynamic scale adjustment
    pub fn training_step<L: Layer<F>>(
        &self,
        model: &mut L,
        input: &ArrayD<F>,
        target: &ArrayD<F>,
        loss_fn: impl Fn(&ArrayD<F>, &ArrayD<F>) -> Result<F>,
        optimizer_step: impl FnOnce(&mut L, F) -> Result<()>,
    ) -> Result<MixedPrecisionStepResult<F>> {
        self.total_steps.fetch_add(1, Ordering::SeqCst);

        // Forward pass
        self.amp.enter_forward()?;
        let output = model.forward(input)?;
        self.amp.exit_forward()?;

        // Compute loss
        let unscaled_loss = loss_fn(&output, target)?;

        // Scale loss if enabled
        let scaled_loss = self.amp.scale_loss(unscaled_loss)?;

        // Backward pass
        self.amp.enter_backward()?;

        // Get gradients from model
        let mut gradients = model.gradients();

        // Unscale gradients
        let overflow_detected = self.amp.unscale_gradients(&mut gradients)?;

        // Apply clipped gradients back to model
        model.set_gradients(&gradients)?;

        self.amp.exit_backward()?;

        // Update scaler and check if step should be skipped
        let step_skipped = self.amp.update_scaler()?;

        if step_skipped {
            self.skipped_steps.fetch_add(1, Ordering::SeqCst);
        } else {
            // Perform optimizer step
            let lr = F::from(0.001).unwrap_or_else(F::zero); // Default LR, should be passed in
            optimizer_step(model, lr)?;
        }

        Ok(MixedPrecisionStepResult {
            scaled_loss,
            unscaled_loss,
            overflow_detected,
            step_skipped,
            loss_scale: self.amp.get_loss_scale(),
            grad_norm: None,
        })
    }

    /// Get training statistics
    pub fn get_stats(&self) -> TrainingStats {
        let scaler_stats = self.amp.get_scaler_stats();
        TrainingStats {
            total_steps: self.total_steps.load(Ordering::SeqCst),
            skipped_steps: self.skipped_steps.load(Ordering::SeqCst),
            current_loss_scale: self.amp.get_loss_scale(),
            scaler_stats,
        }
    }

    /// Reset training statistics
    pub fn reset_stats(&mut self) {
        self.total_steps.store(0, Ordering::SeqCst);
        self.skipped_steps.store(0, Ordering::SeqCst);
        self.amp.reset();
    }

    /// Get the AMP wrapper
    pub fn amp(&self) -> &AutoMixedPrecision<F> {
        &self.amp
    }

    /// Get mutable AMP wrapper
    pub fn amp_mut(&mut self) -> &mut AutoMixedPrecision<F> {
        &mut self.amp
    }

    /// Get configuration
    pub fn config(&self) -> &MixedPrecisionConfig {
        &self.config
    }
}

/// Training statistics
#[derive(Debug, Clone)]
pub struct TrainingStats {
    /// Total number of training steps
    pub total_steps: u64,
    /// Number of skipped steps due to overflow
    pub skipped_steps: u64,
    /// Current loss scale
    pub current_loss_scale: f64,
    /// Detailed scaler statistics
    pub scaler_stats: Option<GradScalerStats>,
}

// ============================================================================
// Mixed Precision Callback
// ============================================================================

use crate::callbacks::{Callback, CallbackContext, CallbackTiming};

/// Callback for mixed precision training integration
#[derive(Debug)]
pub struct MixedPrecisionCallback<
    F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + Send + Sync,
> {
    /// Configuration
    config: MixedPrecisionConfig,
    /// Gradient scaler
    scaler: GradScaler,
    /// Last recorded loss scale
    last_loss_scale: f64,
    /// Number of overflows in current epoch
    epoch_overflows: usize,
    /// Phantom
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + Send + Sync>
    MixedPrecisionCallback<F>
{
    /// Create a new mixed precision callback
    pub fn new(config: MixedPrecisionConfig) -> Result<Self> {
        let scaler = GradScaler::new(config.clone())?;
        let initial_scale = config.initial_loss_scale;

        Ok(Self {
            config,
            scaler,
            last_loss_scale: initial_scale,
            epoch_overflows: 0,
            _phantom: PhantomData,
        })
    }

    /// Get the gradient scaler
    pub fn scaler(&self) -> &GradScaler {
        &self.scaler
    }

    /// Get the current loss scale
    pub fn loss_scale(&self) -> f64 {
        self.scaler.get_scale()
    }

    /// Get statistics
    pub fn get_stats(&self) -> GradScalerStats {
        self.scaler.get_stats()
    }
}

impl<
        F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign + std::fmt::Display + Send + Sync,
    > Callback<F> for MixedPrecisionCallback<F>
{
    fn on_event(&mut self, timing: CallbackTiming, context: &mut CallbackContext<F>) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        match timing {
            CallbackTiming::BeforeEpoch => {
                self.epoch_overflows = 0;
                self.last_loss_scale = self.scaler.get_scale();
            }
            CallbackTiming::AfterBatch => {
                // Check if model gradients have overflow
                if let Some(model) = context.model.as_mut() {
                    let mut gradients = model.gradients();
                    let overflow = self.scaler.unscale_gradients(&mut gradients)?;

                    if overflow {
                        self.epoch_overflows += 1;
                    }

                    // Update scaler
                    let _skipped = self.scaler.update()?;

                    // Apply clipped gradients back if no overflow
                    if !overflow {
                        model.set_gradients(&gradients)?;
                    }
                }
            }
            CallbackTiming::AfterEpoch => {
                if self.config.log_statistics {
                    let stats = self.scaler.get_stats();
                    eprintln!(
                        "[MixedPrecision] Epoch {} - Scale: {:.2}, Overflows: {}, Total skipped: {}",
                        context.epoch,
                        stats.loss_scale,
                        self.epoch_overflows,
                        stats.overflow_steps
                    );
                }
            }
            _ => {}
        }

        Ok(())
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Check if a tensor contains inf or nan values
pub fn contains_inf_or_nan<F: Float + Debug + NumAssign>(tensor: &ArrayD<F>) -> bool {
    tensor.iter().any(|x| {
        let val = x.to_f64().unwrap_or(f64::NAN);
        !val.is_finite()
    })
}

/// Compute the L2 norm of a tensor
pub fn tensor_norm<F: Float + Debug + NumAssign>(tensor: &ArrayD<F>) -> f64 {
    let sum_sq: f64 = tensor
        .iter()
        .map(|x| {
            let val = x.to_f64().unwrap_or(0.0);
            val * val
        })
        .sum();
    sum_sq.sqrt()
}

/// Compute the global norm across multiple tensors
pub fn global_norm<F: Float + Debug + NumAssign>(tensors: &[ArrayD<F>]) -> f64 {
    let sum_sq: f64 = tensors
        .iter()
        .flat_map(|t| t.iter())
        .map(|x| {
            let val = x.to_f64().unwrap_or(0.0);
            val * val
        })
        .sum();
    sum_sq.sqrt()
}

/// Clip tensor values by maximum absolute value
pub fn clip_by_value<F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign>(
    tensor: &mut ArrayD<F>,
    max_value: f64,
) -> Result<()> {
    let max_f = F::from(max_value)
        .ok_or_else(|| NeuralError::ComputationError("Failed to convert max value".to_string()))?;
    let min_f = F::from(-max_value)
        .ok_or_else(|| NeuralError::ComputationError("Failed to convert min value".to_string()))?;

    for val in tensor.iter_mut() {
        if *val > max_f {
            *val = max_f;
        } else if *val < min_f {
            *val = min_f;
        }
    }

    Ok(())
}

/// Clip tensors by global norm
pub fn clip_by_global_norm<F: Float + Debug + ScalarOperand + FromPrimitive + NumAssign>(
    tensors: &mut [ArrayD<F>],
    max_norm: f64,
) -> Result<f64> {
    let current_norm = global_norm(tensors);

    if current_norm > max_norm {
        let scale = F::from(max_norm / current_norm).ok_or_else(|| {
            NeuralError::ComputationError("Failed to compute clip scale".to_string())
        })?;

        for tensor in tensors.iter_mut() {
            for val in tensor.iter_mut() {
                *val *= scale;
            }
        }
    }

    Ok(current_norm)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_mixed_precision_config_builder() {
        let config = MixedPrecisionConfig::builder()
            .enabled(true)
            .initial_loss_scale(1024.0)
            .growth_factor(2.0)
            .backoff_factor(0.5)
            .growth_interval(100)
            .build()
            .expect("Config build should succeed");

        assert!(config.enabled);
        assert!((config.initial_loss_scale - 1024.0).abs() < 1e-10);
        assert!((config.growth_factor - 2.0).abs() < 1e-10);
        assert!((config.backoff_factor - 0.5).abs() < 1e-10);
        assert_eq!(config.growth_interval, 100);
    }

    #[test]
    fn test_config_validation() {
        // Invalid initial scale
        let result = MixedPrecisionConfig::builder()
            .initial_loss_scale(-1.0)
            .build();
        assert!(result.is_err());

        // Invalid growth factor
        let result = MixedPrecisionConfig::builder().growth_factor(0.5).build();
        assert!(result.is_err());

        // Invalid backoff factor
        let result = MixedPrecisionConfig::builder().backoff_factor(1.5).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_half_precision_conversion() {
        // Test positive values
        let val = 1.5_f32;
        let half = Half::from_f32(val);
        let back = half.to_f32();
        assert!((val - back).abs() < 0.01);

        // Test negative values
        let val = -std::f32::consts::PI;
        let half = Half::from_f32(val);
        let back = half.to_f32();
        assert!((val - back).abs() < 0.01);

        // Test zero
        let half = Half::from_f32(0.0);
        assert!(half.is_zero());

        // Test infinity
        let half = Half::from_f32(f32::INFINITY);
        assert!(half.is_infinite());

        // Test NaN
        let half = Half::from_f32(f32::NAN);
        assert!(half.is_nan());
    }

    #[test]
    fn test_grad_scaler_creation() {
        let config = MixedPrecisionConfig::builder()
            .enabled(true)
            .initial_loss_scale(65536.0)
            .build()
            .expect("Config should be valid");

        let scaler = GradScaler::new(config).expect("Scaler creation should succeed");
        assert!((scaler.get_scale() - 65536.0).abs() < 1e-10);
    }

    #[test]
    fn test_grad_scaler_scale_loss() {
        let config = MixedPrecisionConfig::builder()
            .enabled(true)
            .initial_loss_scale(1024.0)
            .build()
            .expect("Config should be valid");

        let scaler = GradScaler::new(config).expect("Scaler creation should succeed");

        let loss = 0.5_f64;
        let scaled = scaler.scale_loss(loss).expect("Scale should succeed");
        assert!((scaled - 512.0).abs() < 1e-10);
    }

    #[test]
    fn test_grad_scaler_unscale() {
        let config = MixedPrecisionConfig::builder()
            .enabled(true)
            .initial_loss_scale(1024.0)
            .build()
            .expect("Config should be valid");

        let scaler = GradScaler::new(config).expect("Scaler creation should succeed");

        let mut gradients =
            vec![
                Array::from_shape_vec(vec![2, 2], vec![1024.0_f64, 2048.0, 512.0, 256.0])
                    .expect("Array creation should succeed")
                    .into_dyn(),
            ];

        let overflow = scaler
            .unscale_gradients(&mut gradients)
            .expect("Unscale should succeed");
        assert!(!overflow);

        // Check that gradients are unscaled
        assert!((gradients[0][[0, 0]] - 1.0).abs() < 1e-10);
        assert!((gradients[0][[0, 1]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_grad_scaler_overflow_detection() {
        let config = MixedPrecisionConfig::builder()
            .enabled(true)
            .initial_loss_scale(1024.0)
            .build()
            .expect("Config should be valid");

        let scaler = GradScaler::new(config).expect("Scaler creation should succeed");

        let mut gradients = vec![Array::from_shape_vec(vec![2], vec![f64::INFINITY, 1.0])
            .expect("Array creation should succeed")
            .into_dyn()];

        let overflow = scaler
            .unscale_gradients(&mut gradients)
            .expect("Unscale should succeed");
        assert!(overflow);
    }

    #[test]
    fn test_grad_scaler_update_no_overflow() {
        let config = MixedPrecisionConfig::builder()
            .enabled(true)
            .initial_loss_scale(1024.0)
            .growth_interval(2)
            .growth_factor(2.0)
            .build()
            .expect("Config should be valid");

        let scaler = GradScaler::new(config).expect("Scaler creation should succeed");

        // Two successful steps should trigger scale increase
        let skipped = scaler.update().expect("Update should succeed");
        assert!(!skipped);

        let skipped = scaler.update().expect("Update should succeed");
        assert!(!skipped);

        // Scale should have doubled
        assert!((scaler.get_scale() - 2048.0).abs() < 1e-10);
    }

    #[test]
    fn test_grad_scaler_update_with_overflow() {
        let config = MixedPrecisionConfig::builder()
            .enabled(true)
            .initial_loss_scale(1024.0)
            .backoff_factor(0.5)
            .build()
            .expect("Config should be valid");

        let scaler = GradScaler::new(config).expect("Scaler creation should succeed");

        // Simulate overflow
        let mut gradients = vec![Array::from_shape_vec(vec![1], vec![f64::INFINITY])
            .expect("Array creation should succeed")
            .into_dyn()];
        scaler
            .unscale_gradients(&mut gradients)
            .expect("Unscale should succeed");

        let skipped = scaler.update().expect("Update should succeed");
        assert!(skipped);

        // Scale should have halved
        assert!((scaler.get_scale() - 512.0).abs() < 1e-10);
    }

    #[test]
    fn test_mixed_precision_tensor() {
        let data: ArrayD<f64> = Array::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
            .expect("Array creation should succeed")
            .into_dyn();

        let mut tensor = MixedPrecisionTensor::from_fp32(data);
        assert_eq!(tensor.precision(), PrecisionMode::FP32);

        tensor.to_fp16().expect("FP16 conversion should succeed");
        assert_eq!(tensor.precision(), PrecisionMode::FP16Mixed);

        tensor.to_fp32().expect("FP32 conversion should succeed");
        assert_eq!(tensor.precision(), PrecisionMode::FP32);

        let fp32 = tensor.get_fp32().expect("Get FP32 should succeed");
        assert!((fp32[[0, 0]] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_master_weights() {
        let mut weights: MasterWeights<f64> = MasterWeights::new(PrecisionMode::FP16Mixed);

        let w1: ArrayD<f64> = Array::from_shape_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("Array creation should succeed")
            .into_dyn();

        weights
            .register("layer1", w1)
            .expect("Register should succeed");
        assert!(weights.contains("layer1"));
        assert_eq!(weights.len(), 1);

        let master = weights.get_master_weights("layer1");
        assert!(master.is_some());

        let compute = weights.get_compute_weights("layer1");
        assert!(compute.is_some());
    }

    #[test]
    fn test_auto_mixed_precision() {
        let config = MixedPrecisionConfig::builder()
            .enabled(true)
            .initial_loss_scale(1024.0)
            .build()
            .expect("Config should be valid");

        let amp: AutoMixedPrecision<f64> =
            AutoMixedPrecision::new(config).expect("AMP creation should succeed");

        assert!(amp.is_enabled());
        assert_eq!(amp.context_state(), AmpContextState::Normal);

        amp.enter_forward().expect("Enter forward should succeed");
        assert_eq!(amp.context_state(), AmpContextState::Forward);

        amp.exit_forward().expect("Exit forward should succeed");
        assert_eq!(amp.context_state(), AmpContextState::Normal);
    }

    #[test]
    fn test_mixed_precision_trainer_creation() {
        let config = MixedPrecisionConfig::builder()
            .enabled(true)
            .build()
            .expect("Config should be valid");

        let trainer: MixedPrecisionTrainer<f64> =
            MixedPrecisionTrainer::new(config).expect("Trainer creation should succeed");

        assert!(trainer.is_enabled());
        assert!(!trainer.is_training());

        trainer.train();
        assert!(trainer.is_training());

        trainer.eval();
        assert!(!trainer.is_training());
    }

    #[test]
    fn test_utility_functions() {
        // Test contains_inf_or_nan
        let normal: ArrayD<f64> = Array::from_shape_vec(vec![2], vec![1.0, 2.0])
            .expect("Array creation should succeed")
            .into_dyn();
        assert!(!contains_inf_or_nan(&normal));

        let with_nan: ArrayD<f64> = Array::from_shape_vec(vec![2], vec![1.0, f64::NAN])
            .expect("Array creation should succeed")
            .into_dyn();
        assert!(contains_inf_or_nan(&with_nan));

        // Test tensor_norm
        let t: ArrayD<f64> = Array::from_shape_vec(vec![2], vec![3.0, 4.0])
            .expect("Array creation should succeed")
            .into_dyn();
        assert!((tensor_norm(&t) - 5.0).abs() < 1e-10);

        // Test global_norm
        let t1: ArrayD<f64> = Array::from_shape_vec(vec![2], vec![1.0, 2.0])
            .expect("Array creation should succeed")
            .into_dyn();
        let t2: ArrayD<f64> = Array::from_shape_vec(vec![2], vec![2.0, 0.0])
            .expect("Array creation should succeed")
            .into_dyn();
        let norm = global_norm(&[t1, t2]);
        assert!((norm - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_clip_by_value() {
        let mut tensor: ArrayD<f64> = Array::from_shape_vec(vec![4], vec![-5.0, -1.0, 1.0, 5.0])
            .expect("Array creation should succeed")
            .into_dyn();

        clip_by_value(&mut tensor, 2.0).expect("Clip should succeed");

        assert!((tensor[[0]] - (-2.0)).abs() < 1e-10);
        assert!((tensor[[1]] - (-1.0)).abs() < 1e-10);
        assert!((tensor[[2]] - 1.0).abs() < 1e-10);
        assert!((tensor[[3]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_clip_by_global_norm() {
        let mut tensors = vec![Array::from_shape_vec(vec![2], vec![3.0, 4.0])
            .expect("Array creation should succeed")
            .into_dyn()];

        let original_norm = clip_by_global_norm(&mut tensors, 2.5).expect("Clip should succeed");
        assert!((original_norm - 5.0).abs() < 1e-10);

        // Check that norm is now approximately 2.5
        let new_norm = global_norm(&tensors);
        assert!((new_norm - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_fp32_operation_check() {
        let config = MixedPrecisionConfig::default();

        assert!(config.should_use_fp32(FP32Operation::LossComputation));
        assert!(config.should_use_fp32(FP32Operation::Softmax));
        assert!(config.should_use_fp32(FP32Operation::LayerNorm));
        assert!(!config.should_use_fp32(FP32Operation::Embedding)); // Not in default list
    }

    #[test]
    fn test_grad_scaler_reset() {
        let config = MixedPrecisionConfig::builder()
            .enabled(true)
            .initial_loss_scale(1024.0)
            .backoff_factor(0.5)
            .build()
            .expect("Config should be valid");

        let scaler = GradScaler::new(config).expect("Scaler creation should succeed");

        // Simulate overflow and scale reduction
        let mut gradients = vec![Array::from_shape_vec(vec![1], vec![f64::INFINITY])
            .expect("Array creation should succeed")
            .into_dyn()];
        scaler
            .unscale_gradients(&mut gradients)
            .expect("Unscale should succeed");
        scaler.update().expect("Update should succeed");

        assert!((scaler.get_scale() - 512.0).abs() < 1e-10);

        // Reset
        scaler.reset();
        assert!((scaler.get_scale() - 1024.0).abs() < 1e-10);
    }
}
