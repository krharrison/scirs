//! Mixed precision training for automatic differentiation
//!
//! This module provides FP16/FP32 mixed precision training support, including:
//!
//! - **Precision policies**: Configure which operations run in which precision
//! - **Loss scaling**: Static and dynamic loss scaling to prevent gradient underflow
//! - **Master weight management**: Maintain FP32 master copies with FP16 compute copies
//! - **Gradient overflow detection**: Detect inf/nan in gradients after unscaling
//! - **Op-level precision assignment**: Fine-grained control over precision per operation
//!
//! # Example
//!
//! ```rust
//! use scirs2_autograd::mixed_precision::*;
//!
//! // Create a precision policy
//! let policy = PrecisionPolicy::default();
//! assert_eq!(policy.default_precision(), Precision::FP32);
//!
//! // Create a dynamic loss scaler
//! let scaler = DynamicLossScaler::new(
//!     DynamicScalerConfig::default()
//! );
//! assert_eq!(scaler.current_scale(), 65536.0);
//!
//! // Create a mixed precision trainer
//! let config = MixedPrecisionConfig::default();
//! let trainer = MixedPrecisionTrainer::<f64>::new(config);
//! assert_eq!(trainer.step_count(), 0);
//! ```

use crate::error::AutogradError;
use crate::{Float, NdArray, Result};
use scirs2_core::ndarray::{Array, ArrayD, IxDyn};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Precision enum
// ---------------------------------------------------------------------------

/// Floating point precision levels for mixed-precision training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    /// 16-bit floating point (half precision)
    FP16,
    /// 32-bit floating point (single precision)
    FP32,
    /// 64-bit floating point (double precision)
    FP64,
    /// Brain floating point (Google's bfloat16)
    BF16,
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Precision::FP16 => write!(f, "fp16"),
            Precision::FP32 => write!(f, "fp32"),
            Precision::FP64 => write!(f, "fp64"),
            Precision::BF16 => write!(f, "bf16"),
        }
    }
}

impl Precision {
    /// Number of bytes for this precision.
    pub fn byte_size(self) -> usize {
        match self {
            Precision::FP16 | Precision::BF16 => 2,
            Precision::FP32 => 4,
            Precision::FP64 => 8,
        }
    }

    /// Maximum representable value (approximate).
    pub fn max_value(self) -> f64 {
        match self {
            Precision::FP16 => 65504.0,
            Precision::BF16 => 3.389_531_389_251_535_2e38,
            Precision::FP32 => f32::MAX as f64,
            Precision::FP64 => f64::MAX,
        }
    }

    /// Smallest positive normal value (approximate).
    pub fn min_positive(self) -> f64 {
        match self {
            Precision::FP16 => 6.103_515_625e-5,
            Precision::BF16 => 1.175_494_350_822_287_5e-38,
            Precision::FP32 => f32::MIN_POSITIVE as f64,
            Precision::FP64 => f64::MIN_POSITIVE,
        }
    }

    /// Machine epsilon (approximate).
    pub fn epsilon(self) -> f64 {
        match self {
            Precision::FP16 => 9.765_625e-4,
            Precision::BF16 => 7.812_5e-3,
            Precision::FP32 => f32::EPSILON as f64,
            Precision::FP64 => f64::EPSILON,
        }
    }
}

// ---------------------------------------------------------------------------
// Operation categories for precision assignment
// ---------------------------------------------------------------------------

/// Categories of operations for precision assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpCategory {
    /// Matrix multiplication and linear layers
    MatMul,
    /// Convolution operations
    Convolution,
    /// Batch normalisation / layer norm
    Normalization,
    /// Softmax and cross-entropy
    Softmax,
    /// Element-wise arithmetic (+, -, *, /)
    ElementWise,
    /// Reduction operations (sum, mean, max)
    Reduction,
    /// Activation functions (relu, gelu, sigmoid)
    Activation,
    /// Embedding look-ups
    Embedding,
    /// Loss computation
    Loss,
    /// Attention mechanism (Q*K^T/sqrt(d) * V)
    Attention,
    /// Pooling operations
    Pooling,
    /// Custom / unknown ops
    Custom,
}

impl OpCategory {
    /// The recommended precision for this op category in a standard AMP policy.
    ///
    /// Operations that are numerically sensitive (norms, softmax, loss) stay in FP32,
    /// while compute-bound operations (matmul, conv) benefit from FP16.
    pub fn recommended_precision(self) -> Precision {
        match self {
            OpCategory::MatMul | OpCategory::Convolution | OpCategory::Attention => Precision::FP16,
            OpCategory::Normalization
            | OpCategory::Softmax
            | OpCategory::Loss
            | OpCategory::Reduction => Precision::FP32,
            OpCategory::ElementWise | OpCategory::Activation | OpCategory::Pooling => {
                Precision::FP16
            }
            OpCategory::Embedding | OpCategory::Custom => Precision::FP32,
        }
    }
}

// ---------------------------------------------------------------------------
// Precision policy
// ---------------------------------------------------------------------------

/// Configuration for which operations run in which precision.
///
/// The policy assigns a precision to each operation category and provides
/// a default for any unregistered categories.
#[derive(Debug, Clone)]
pub struct PrecisionPolicy {
    /// Per-category precision overrides
    overrides: HashMap<OpCategory, Precision>,
    /// Default precision for un-overridden categories
    default: Precision,
    /// Whether to keep a FP32 master copy of all weights
    keep_master_weights: bool,
    /// Whether to cast gradients back to FP32 before optimizer step
    fp32_grad_accumulation: bool,
}

impl Default for PrecisionPolicy {
    fn default() -> Self {
        Self {
            overrides: HashMap::new(),
            default: Precision::FP32,
            keep_master_weights: true,
            fp32_grad_accumulation: true,
        }
    }
}

impl PrecisionPolicy {
    /// Create a new precision policy with the given default precision.
    pub fn new(default: Precision) -> Self {
        Self {
            overrides: HashMap::new(),
            default,
            keep_master_weights: true,
            fp32_grad_accumulation: true,
        }
    }

    /// Create the standard O1-style AMP policy (matmul/conv in FP16, everything else FP32).
    pub fn amp_o1() -> Self {
        let mut overrides = HashMap::new();
        overrides.insert(OpCategory::MatMul, Precision::FP16);
        overrides.insert(OpCategory::Convolution, Precision::FP16);
        Self {
            overrides,
            default: Precision::FP32,
            keep_master_weights: true,
            fp32_grad_accumulation: true,
        }
    }

    /// Create the aggressive O2-style AMP policy (everything in FP16 except loss/softmax/norm).
    pub fn amp_o2() -> Self {
        let mut overrides = HashMap::new();
        for &cat in &[
            OpCategory::MatMul,
            OpCategory::Convolution,
            OpCategory::Attention,
            OpCategory::ElementWise,
            OpCategory::Activation,
            OpCategory::Pooling,
            OpCategory::Embedding,
        ] {
            overrides.insert(cat, Precision::FP16);
        }
        overrides.insert(OpCategory::Normalization, Precision::FP32);
        overrides.insert(OpCategory::Softmax, Precision::FP32);
        overrides.insert(OpCategory::Loss, Precision::FP32);
        overrides.insert(OpCategory::Reduction, Precision::FP32);
        Self {
            overrides,
            default: Precision::FP16,
            keep_master_weights: true,
            fp32_grad_accumulation: true,
        }
    }

    /// Create a full-FP32 policy (no mixed precision, useful as a baseline).
    pub fn full_fp32() -> Self {
        Self {
            overrides: HashMap::new(),
            default: Precision::FP32,
            keep_master_weights: false,
            fp32_grad_accumulation: false,
        }
    }

    /// Override the precision for a specific operation category.
    pub fn set_precision(&mut self, category: OpCategory, precision: Precision) {
        self.overrides.insert(category, precision);
    }

    /// Get the precision for a given operation category.
    pub fn precision_for(&self, category: OpCategory) -> Precision {
        self.overrides
            .get(&category)
            .copied()
            .unwrap_or(self.default)
    }

    /// Get the default precision.
    pub fn default_precision(&self) -> Precision {
        self.default
    }

    /// Whether to keep FP32 master weights.
    pub fn keep_master_weights(&self) -> bool {
        self.keep_master_weights
    }

    /// Whether to accumulate gradients in FP32.
    pub fn fp32_grad_accumulation(&self) -> bool {
        self.fp32_grad_accumulation
    }

    /// Set whether to keep master weights.
    pub fn set_keep_master_weights(&mut self, keep: bool) {
        self.keep_master_weights = keep;
    }

    /// Set whether to accumulate gradients in FP32.
    pub fn set_fp32_grad_accumulation(&mut self, enable: bool) {
        self.fp32_grad_accumulation = enable;
    }

    /// Return an iterator over all category overrides.
    pub fn overrides(&self) -> impl Iterator<Item = (&OpCategory, &Precision)> {
        self.overrides.iter()
    }

    /// Number of category-level overrides.
    pub fn num_overrides(&self) -> usize {
        self.overrides.len()
    }
}

// ---------------------------------------------------------------------------
// Loss scaling
// ---------------------------------------------------------------------------

/// Static loss scaler with a fixed scaling factor.
#[derive(Debug, Clone)]
pub struct StaticLossScaler {
    scale: f64,
}

impl StaticLossScaler {
    /// Create a static loss scaler with the given factor.
    pub fn new(scale: f64) -> Self {
        Self {
            scale: if scale > 0.0 { scale } else { 1.0 },
        }
    }

    /// Get the current scale factor.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Scale a loss value.
    pub fn scale_loss<F: Float>(&self, loss: &NdArray<F>) -> Result<NdArray<F>> {
        let scale_f = F::from(self.scale).ok_or_else(|| {
            AutogradError::compute_error("Cannot convert scale to target type".into())
        })?;
        Ok(loss.mapv(|v| v * scale_f))
    }

    /// Unscale gradients by dividing by the scale factor.
    pub fn unscale_gradients<F: Float>(&self, gradients: &mut [NdArray<F>]) -> Result<()> {
        let inv_scale = F::from(1.0 / self.scale).ok_or_else(|| {
            AutogradError::compute_error("Cannot convert inverse scale to target type".into())
        })?;
        for grad in gradients.iter_mut() {
            grad.mapv_inplace(|v| v * inv_scale);
        }
        Ok(())
    }
}

impl Default for StaticLossScaler {
    fn default() -> Self {
        Self { scale: 1.0 }
    }
}

/// Configuration for the dynamic loss scaler.
#[derive(Debug, Clone)]
pub struct DynamicScalerConfig {
    /// Initial scale factor (typically a large power of 2)
    pub init_scale: f64,
    /// Factor by which to increase scale after `growth_interval` clean steps
    pub growth_factor: f64,
    /// Factor by which to decrease scale after overflow
    pub backoff_factor: f64,
    /// Number of consecutive non-overflow steps before increasing scale
    pub growth_interval: usize,
    /// Minimum scale factor
    pub min_scale: f64,
    /// Maximum scale factor
    pub max_scale: f64,
}

impl Default for DynamicScalerConfig {
    fn default() -> Self {
        Self {
            init_scale: 65536.0, // 2^16
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            min_scale: 1.0,
            max_scale: 2.0f64.powi(24),
        }
    }
}

/// Dynamic loss scaler that automatically adjusts the scaling factor.
///
/// The scaler increases the scale factor after a series of clean (non-overflow)
/// steps and decreases it when overflow is detected. This maximises the use of
/// the FP16 dynamic range without causing gradient overflow.
#[derive(Debug, Clone)]
pub struct DynamicLossScaler {
    config: DynamicScalerConfig,
    current_scale: f64,
    /// Number of consecutive clean (non-overflow) steps
    growth_tracker: usize,
    /// Total number of overflow events detected
    total_overflows: usize,
    /// Total number of steps taken
    total_steps: usize,
}

impl DynamicLossScaler {
    /// Create a new dynamic loss scaler with the given configuration.
    pub fn new(config: DynamicScalerConfig) -> Self {
        let scale = config.init_scale.clamp(config.min_scale, config.max_scale);
        Self {
            current_scale: scale,
            growth_tracker: 0,
            total_overflows: 0,
            total_steps: 0,
            config,
        }
    }

    /// Get the current scale factor.
    pub fn current_scale(&self) -> f64 {
        self.current_scale
    }

    /// Get the total number of overflow events.
    pub fn total_overflows(&self) -> usize {
        self.total_overflows
    }

    /// Get the total number of steps taken.
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Scale a loss value.
    pub fn scale_loss<F: Float>(&self, loss: &NdArray<F>) -> Result<NdArray<F>> {
        let scale_f = F::from(self.current_scale).ok_or_else(|| {
            AutogradError::compute_error("Cannot convert scale to target type".into())
        })?;
        Ok(loss.mapv(|v| v * scale_f))
    }

    /// Unscale gradients and check for overflow.
    ///
    /// Returns `true` if the gradients are valid (no overflow/nan),
    /// `false` if overflow was detected and the step should be skipped.
    pub fn unscale_and_check<F: Float>(&mut self, gradients: &mut [NdArray<F>]) -> Result<bool> {
        let inv_scale = F::from(1.0 / self.current_scale).ok_or_else(|| {
            AutogradError::compute_error("Cannot convert inverse scale to target type".into())
        })?;

        // Unscale
        for grad in gradients.iter_mut() {
            grad.mapv_inplace(|v| v * inv_scale);
        }

        // Check for overflow / nan
        let has_overflow = check_gradients_overflow(gradients);

        self.total_steps += 1;

        if has_overflow {
            self.total_overflows += 1;
            self.growth_tracker = 0;
            // Backoff: reduce scale
            self.current_scale =
                (self.current_scale * self.config.backoff_factor).max(self.config.min_scale);
            Ok(false)
        } else {
            self.growth_tracker += 1;
            if self.growth_tracker >= self.config.growth_interval {
                self.growth_tracker = 0;
                // Grow: increase scale
                self.current_scale =
                    (self.current_scale * self.config.growth_factor).min(self.config.max_scale);
            }
            Ok(true)
        }
    }

    /// Update the scaler state after a step (without unscaling — use when
    /// you unscale manually via `StaticLossScaler::unscale_gradients`).
    pub fn update(&mut self, found_overflow: bool) {
        self.total_steps += 1;
        if found_overflow {
            self.total_overflows += 1;
            self.growth_tracker = 0;
            self.current_scale =
                (self.current_scale * self.config.backoff_factor).max(self.config.min_scale);
        } else {
            self.growth_tracker += 1;
            if self.growth_tracker >= self.config.growth_interval {
                self.growth_tracker = 0;
                self.current_scale =
                    (self.current_scale * self.config.growth_factor).min(self.config.max_scale);
            }
        }
    }

    /// Get the overflow ratio (overflows / total_steps).
    pub fn overflow_ratio(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            self.total_overflows as f64 / self.total_steps as f64
        }
    }

    /// Reset the scaler to its initial state.
    pub fn reset(&mut self) {
        self.current_scale = self
            .config
            .init_scale
            .clamp(self.config.min_scale, self.config.max_scale);
        self.growth_tracker = 0;
        self.total_overflows = 0;
        self.total_steps = 0;
    }
}

// ---------------------------------------------------------------------------
// Gradient overflow detection
// ---------------------------------------------------------------------------

/// Check if any gradient contains inf or nan values.
pub fn check_gradients_overflow<F: Float>(gradients: &[NdArray<F>]) -> bool {
    for grad in gradients {
        for &val in grad.iter() {
            if val.is_nan() || val.is_infinite() {
                return true;
            }
        }
    }
    false
}

/// Detailed overflow statistics for a set of gradients.
#[derive(Debug, Clone, Default)]
pub struct OverflowStats {
    /// Total number of elements checked
    pub total_elements: usize,
    /// Number of inf elements
    pub inf_count: usize,
    /// Number of nan elements
    pub nan_count: usize,
    /// Number of zero elements
    pub zero_count: usize,
    /// Number of subnormal (denormalized) elements
    pub subnormal_count: usize,
    /// Maximum absolute value
    pub max_abs: f64,
    /// Minimum absolute non-zero value
    pub min_abs_nonzero: f64,
}

impl OverflowStats {
    /// Whether any overflow was detected.
    pub fn has_overflow(&self) -> bool {
        self.inf_count > 0 || self.nan_count > 0
    }

    /// Fraction of elements that are zero.
    pub fn sparsity(&self) -> f64 {
        if self.total_elements == 0 {
            0.0
        } else {
            self.zero_count as f64 / self.total_elements as f64
        }
    }
}

/// Compute detailed overflow statistics for a set of gradients.
pub fn gradient_overflow_stats<F: Float>(gradients: &[NdArray<F>]) -> OverflowStats {
    let mut stats = OverflowStats {
        max_abs: 0.0,
        min_abs_nonzero: f64::MAX,
        ..Default::default()
    };

    for grad in gradients {
        for &val in grad.iter() {
            stats.total_elements += 1;
            let f = val.to_f64().unwrap_or(0.0);
            let abs_f = f.abs();

            if val.is_nan() {
                stats.nan_count += 1;
            } else if val.is_infinite() {
                stats.inf_count += 1;
            } else if abs_f == 0.0 {
                stats.zero_count += 1;
            } else {
                if abs_f > stats.max_abs {
                    stats.max_abs = abs_f;
                }
                if abs_f < stats.min_abs_nonzero {
                    stats.min_abs_nonzero = abs_f;
                }
                // Check for subnormal (value smaller than min positive normal)
                if abs_f < f64::MIN_POSITIVE && abs_f > 0.0 {
                    stats.subnormal_count += 1;
                }
            }
        }
    }

    if stats.min_abs_nonzero == f64::MAX {
        stats.min_abs_nonzero = 0.0;
    }

    stats
}

// ---------------------------------------------------------------------------
// Master weight management
// ---------------------------------------------------------------------------

/// Manages FP32 master copies of model weights.
///
/// In mixed-precision training, the forward and backward passes operate on
/// reduced-precision copies (FP16), but an FP32 "master" copy of each
/// parameter is maintained for the optimizer update to prevent loss of
/// small gradient updates.
#[derive(Debug, Clone)]
pub struct MasterWeightManager<F: Float> {
    /// FP32 master weights keyed by parameter name
    master_weights: HashMap<String, NdArray<F>>,
    /// Which precision the compute copies are in
    compute_precision: Precision,
}

impl<F: Float> MasterWeightManager<F> {
    /// Create a new master weight manager.
    pub fn new(compute_precision: Precision) -> Self {
        Self {
            master_weights: HashMap::new(),
            compute_precision,
        }
    }

    /// Register a parameter (stores the FP32 master copy).
    pub fn register(&mut self, name: &str, weight: NdArray<F>) {
        self.master_weights.insert(name.to_owned(), weight);
    }

    /// Get the master (FP32) copy of a parameter.
    pub fn master_weight(&self, name: &str) -> Option<&NdArray<F>> {
        self.master_weights.get(name)
    }

    /// Create a compute-precision copy of a parameter.
    ///
    /// In a real system this would cast to FP16; since we operate in generic
    /// `F`, we simulate by clamping values to the representable range of the
    /// target precision to emulate quantisation effects.
    pub fn compute_copy(&self, name: &str) -> Result<NdArray<F>> {
        let master = self.master_weights.get(name).ok_or_else(|| {
            AutogradError::OperationError(format!("Weight '{}' not registered", name))
        })?;

        match self.compute_precision {
            Precision::FP16 => {
                let max_val = F::from(Precision::FP16.max_value()).unwrap_or(F::max_value());
                let min_val = F::from(-Precision::FP16.max_value()).unwrap_or(F::min_value());
                Ok(master.mapv(|v| {
                    if v > max_val {
                        max_val
                    } else if v < min_val {
                        min_val
                    } else {
                        v
                    }
                }))
            }
            Precision::BF16 => {
                // BF16 has the same exponent range as FP32 but less mantissa precision.
                // We simulate by rounding to ~8-bit mantissa precision.
                Ok(master.mapv(|v| {
                    let f = v.to_f64().unwrap_or(0.0);
                    // Round to BF16 precision (~7.8 bits of mantissa)
                    let bits = f.to_bits();
                    let rounded_bits = bits & 0xFFFF_FFFF_FFFF_0000;
                    let rounded = f64::from_bits(rounded_bits);
                    F::from(rounded).unwrap_or(v)
                }))
            }
            _ => Ok(master.clone()),
        }
    }

    /// Update the master weight from the optimizer output.
    pub fn update_master(&mut self, name: &str, updated: NdArray<F>) -> Result<()> {
        if !self.master_weights.contains_key(name) {
            return Err(AutogradError::OperationError(format!(
                "Weight '{}' not registered",
                name
            )));
        }
        self.master_weights.insert(name.to_owned(), updated);
        Ok(())
    }

    /// Apply gradients to master weights with a learning rate.
    pub fn apply_gradients(
        &mut self,
        name: &str,
        gradient: &NdArray<F>,
        learning_rate: F,
    ) -> Result<()> {
        let master = self.master_weights.get_mut(name).ok_or_else(|| {
            AutogradError::OperationError(format!("Weight '{}' not registered", name))
        })?;

        if master.shape() != gradient.shape() {
            return Err(AutogradError::ShapeMismatch(format!(
                "Master weight shape {:?} != gradient shape {:?} for '{}'",
                master.shape(),
                gradient.shape(),
                name,
            )));
        }

        // w = w - lr * grad
        master.zip_mut_with(gradient, |w, &g| {
            *w = *w - learning_rate * g;
        });

        Ok(())
    }

    /// Number of registered parameters.
    pub fn num_params(&self) -> usize {
        self.master_weights.len()
    }

    /// Total number of elements across all master weights.
    pub fn total_elements(&self) -> usize {
        self.master_weights.values().map(|w| w.len()).sum()
    }

    /// Parameter names iterator.
    pub fn param_names(&self) -> impl Iterator<Item = &String> {
        self.master_weights.keys()
    }

    /// Remove a parameter.
    pub fn remove(&mut self, name: &str) -> Option<NdArray<F>> {
        self.master_weights.remove(name)
    }

    /// Clear all parameters.
    pub fn clear(&mut self) {
        self.master_weights.clear();
    }
}

// ---------------------------------------------------------------------------
// Mixed precision configuration
// ---------------------------------------------------------------------------

/// Top-level configuration for mixed precision training.
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Precision policy
    pub policy: PrecisionPolicy,
    /// Whether to use dynamic loss scaling
    pub dynamic_loss_scaling: bool,
    /// Dynamic scaler configuration
    pub scaler_config: DynamicScalerConfig,
    /// Static scale factor (used when dynamic_loss_scaling is false)
    pub static_scale: f64,
    /// Whether to skip optimizer step on overflow
    pub skip_on_overflow: bool,
    /// Whether to log overflow events
    pub log_overflows: bool,
    /// Maximum consecutive overflows before raising an error
    pub max_consecutive_overflows: usize,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            policy: PrecisionPolicy::amp_o1(),
            dynamic_loss_scaling: true,
            scaler_config: DynamicScalerConfig::default(),
            static_scale: 1.0,
            skip_on_overflow: true,
            log_overflows: false,
            max_consecutive_overflows: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// Mixed precision trainer
// ---------------------------------------------------------------------------

/// Orchestrates mixed precision training.
///
/// Wraps loss scaling, gradient overflow detection, master weight
/// management, and precision policy into a unified interface.
pub struct MixedPrecisionTrainer<F: Float> {
    config: MixedPrecisionConfig,
    dynamic_scaler: DynamicLossScaler,
    static_scaler: StaticLossScaler,
    master_weights: MasterWeightManager<F>,
    step_count: usize,
    consecutive_overflows: usize,
    skipped_steps: usize,
}

impl<F: Float> MixedPrecisionTrainer<F> {
    /// Create a new mixed precision trainer.
    pub fn new(config: MixedPrecisionConfig) -> Self {
        let dynamic_scaler = DynamicLossScaler::new(config.scaler_config.clone());
        let static_scaler = StaticLossScaler::new(config.static_scale);
        let compute_prec = if config.policy.default_precision() == Precision::FP32 {
            Precision::FP16
        } else {
            config.policy.default_precision()
        };
        let master_weights = MasterWeightManager::new(compute_prec);

        Self {
            config,
            dynamic_scaler,
            static_scaler,
            master_weights,
            step_count: 0,
            consecutive_overflows: 0,
            skipped_steps: 0,
        }
    }

    /// Register a parameter for master weight management.
    pub fn register_param(&mut self, name: &str, weight: NdArray<F>) {
        self.master_weights.register(name, weight);
    }

    /// Get the current loss scale factor.
    pub fn current_scale(&self) -> f64 {
        if self.config.dynamic_loss_scaling {
            self.dynamic_scaler.current_scale()
        } else {
            self.static_scaler.scale()
        }
    }

    /// Scale a loss value for mixed precision training.
    pub fn scale_loss(&self, loss: &NdArray<F>) -> Result<NdArray<F>> {
        if self.config.dynamic_loss_scaling {
            self.dynamic_scaler.scale_loss(loss)
        } else {
            self.static_scaler.scale_loss(loss)
        }
    }

    /// Unscale gradients and perform an optimizer step.
    ///
    /// Returns `true` if the step was applied, `false` if it was skipped
    /// due to gradient overflow.
    pub fn step(&mut self, gradients: &mut [NdArray<F>]) -> Result<bool> {
        // Unscale gradients
        if self.config.dynamic_loss_scaling {
            let valid = self.dynamic_scaler.unscale_and_check(gradients)?;
            if !valid {
                self.consecutive_overflows += 1;
                self.skipped_steps += 1;

                if self.consecutive_overflows >= self.config.max_consecutive_overflows {
                    return Err(AutogradError::compute_error(format!(
                        "Exceeded maximum consecutive gradient overflows ({})",
                        self.config.max_consecutive_overflows
                    )));
                }

                if self.config.skip_on_overflow {
                    return Ok(false);
                }
            } else {
                self.consecutive_overflows = 0;
            }
        } else {
            self.static_scaler.unscale_gradients(gradients)?;
            if check_gradients_overflow(gradients) {
                self.consecutive_overflows += 1;
                self.skipped_steps += 1;
                if self.config.skip_on_overflow {
                    return Ok(false);
                }
            } else {
                self.consecutive_overflows = 0;
            }
        }

        self.step_count += 1;
        Ok(true)
    }

    /// Get the precision for a given operation category.
    pub fn precision_for(&self, category: OpCategory) -> Precision {
        self.config.policy.precision_for(category)
    }

    /// Get a reference to the master weight manager.
    pub fn master_weights(&self) -> &MasterWeightManager<F> {
        &self.master_weights
    }

    /// Get a mutable reference to the master weight manager.
    pub fn master_weights_mut(&mut self) -> &mut MasterWeightManager<F> {
        &mut self.master_weights
    }

    /// Total number of successful optimizer steps.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Total number of skipped steps due to overflow.
    pub fn skipped_steps(&self) -> usize {
        self.skipped_steps
    }

    /// Get the overflow ratio from the dynamic scaler.
    pub fn overflow_ratio(&self) -> f64 {
        if self.config.dynamic_loss_scaling {
            self.dynamic_scaler.overflow_ratio()
        } else if self.step_count + self.skipped_steps == 0 {
            0.0
        } else {
            self.skipped_steps as f64 / (self.step_count + self.skipped_steps) as f64
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &MixedPrecisionConfig {
        &self.config
    }

    /// Get a reference to the dynamic scaler.
    pub fn dynamic_scaler(&self) -> &DynamicLossScaler {
        &self.dynamic_scaler
    }

    /// Reset the trainer state.
    pub fn reset(&mut self) {
        self.dynamic_scaler.reset();
        self.step_count = 0;
        self.consecutive_overflows = 0;
        self.skipped_steps = 0;
    }
}

// ---------------------------------------------------------------------------
// Precision casting utilities
// ---------------------------------------------------------------------------

/// Simulate casting an array from one precision to another.
///
/// Since we operate on generic `F`, this simulates precision loss by
/// clamping and rounding values to the target precision's representable
/// range.
pub fn simulate_precision_cast<F: Float>(
    array: &NdArray<F>,
    target: Precision,
) -> Result<NdArray<F>> {
    match target {
        Precision::FP16 => {
            let max_val = F::from(Precision::FP16.max_value()).ok_or_else(|| {
                AutogradError::compute_error("Cannot convert FP16 max to target type".into())
            })?;
            let neg_max = F::from(-Precision::FP16.max_value()).ok_or_else(|| {
                AutogradError::compute_error("Cannot convert FP16 min to target type".into())
            })?;
            let min_pos = F::from(Precision::FP16.min_positive()).ok_or_else(|| {
                AutogradError::compute_error(
                    "Cannot convert FP16 min_positive to target type".into(),
                )
            })?;

            Ok(array.mapv(|v| {
                if v.is_nan() || v.is_infinite() {
                    return v; // preserve special values
                }
                let abs_v = if v < F::zero() { F::zero() - v } else { v };
                if abs_v < min_pos && abs_v > F::zero() {
                    // Flush subnormals to zero
                    F::zero()
                } else if v > max_val {
                    max_val
                } else if v < neg_max {
                    neg_max
                } else {
                    v
                }
            }))
        }
        Precision::BF16 => Ok(array.mapv(|v| {
            let f = v.to_f64().unwrap_or(0.0);
            let bits = f.to_bits();
            let rounded_bits = bits & 0xFFFF_FFFF_FFFF_0000;
            let rounded = f64::from_bits(rounded_bits);
            F::from(rounded).unwrap_or(v)
        })),
        Precision::FP32 => Ok(array.mapv(|v| {
            let f = v.to_f64().unwrap_or(0.0);
            let f32_val = f as f32;
            F::from(f32_val).unwrap_or(v)
        })),
        Precision::FP64 => Ok(array.clone()),
    }
}

/// Compute the quantisation error from casting to a lower precision.
pub fn quantisation_error<F: Float>(array: &NdArray<F>, target: Precision) -> Result<f64> {
    let cast = simulate_precision_cast(array, target)?;
    let mut total_err = 0.0f64;
    let mut count = 0usize;

    for (&orig, &cast_v) in array.iter().zip(cast.iter()) {
        let o = orig.to_f64().unwrap_or(0.0);
        let c = cast_v.to_f64().unwrap_or(0.0);
        if o.is_finite() && c.is_finite() {
            total_err += (o - c).abs();
            count += 1;
        }
    }

    if count == 0 {
        Ok(0.0)
    } else {
        Ok(total_err / count as f64)
    }
}

// ---------------------------------------------------------------------------
// Op-level precision assignment
// ---------------------------------------------------------------------------

/// Classifies a string op name into an [`OpCategory`].
///
/// This is a best-effort heuristic based on common operation naming conventions.
pub fn classify_op(op_name: &str) -> OpCategory {
    let lower = op_name.to_lowercase();

    if lower.contains("matmul") || lower.contains("linear") || lower.contains("gemm") {
        OpCategory::MatMul
    } else if lower.contains("conv") {
        OpCategory::Convolution
    } else if lower.contains("batchnorm")
        || lower.contains("batch_norm")
        || lower.contains("layernorm")
        || lower.contains("layer_norm")
        || lower.contains("norm")
    {
        OpCategory::Normalization
    } else if lower.contains("softmax") {
        OpCategory::Softmax
    } else if lower.contains("attention") {
        OpCategory::Attention
    } else if lower.contains("relu")
        || lower.contains("gelu")
        || lower.contains("sigmoid")
        || lower.contains("tanh")
        || lower.contains("activation")
    {
        OpCategory::Activation
    } else if lower.contains("embed") {
        OpCategory::Embedding
    } else if lower.contains("loss") || lower.contains("cross_entropy") || lower.contains("mse") {
        OpCategory::Loss
    } else if lower.contains("sum")
        || lower.contains("mean")
        || lower.contains("reduce")
        || lower.contains("max")
        || lower.contains("min")
    {
        OpCategory::Reduction
    } else if lower.contains("pool") {
        OpCategory::Pooling
    } else if lower.contains("add")
        || lower.contains("mul")
        || lower.contains("sub")
        || lower.contains("div")
    {
        OpCategory::ElementWise
    } else {
        OpCategory::Custom
    }
}

/// Assigns precision to each operation in a list based on the given policy.
pub fn assign_precisions(op_names: &[&str], policy: &PrecisionPolicy) -> Vec<(String, Precision)> {
    op_names
        .iter()
        .map(|name| {
            let cat = classify_op(name);
            (name.to_string(), policy.precision_for(cat))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    // --- Precision tests ---

    #[test]
    fn test_precision_byte_size() {
        assert_eq!(Precision::FP16.byte_size(), 2);
        assert_eq!(Precision::BF16.byte_size(), 2);
        assert_eq!(Precision::FP32.byte_size(), 4);
        assert_eq!(Precision::FP64.byte_size(), 8);
    }

    #[test]
    fn test_precision_display() {
        assert_eq!(format!("{}", Precision::FP16), "fp16");
        assert_eq!(format!("{}", Precision::FP32), "fp32");
        assert_eq!(format!("{}", Precision::FP64), "fp64");
        assert_eq!(format!("{}", Precision::BF16), "bf16");
    }

    #[test]
    fn test_precision_max_value() {
        assert!(Precision::FP16.max_value() > 0.0);
        assert!(Precision::FP16.max_value() < Precision::FP32.max_value());
        assert!(Precision::FP32.max_value() < Precision::FP64.max_value());
    }

    #[test]
    fn test_precision_epsilon() {
        assert!(Precision::FP16.epsilon() > Precision::FP32.epsilon());
        assert!(Precision::FP32.epsilon() > Precision::FP64.epsilon());
        assert!(Precision::BF16.epsilon() > Precision::FP16.epsilon());
    }

    #[test]
    fn test_precision_min_positive() {
        assert!(Precision::FP16.min_positive() > Precision::FP32.min_positive());
    }

    // --- OpCategory tests ---

    #[test]
    fn test_op_category_recommended_precision() {
        assert_eq!(OpCategory::MatMul.recommended_precision(), Precision::FP16);
        assert_eq!(
            OpCategory::Convolution.recommended_precision(),
            Precision::FP16
        );
        assert_eq!(
            OpCategory::Normalization.recommended_precision(),
            Precision::FP32
        );
        assert_eq!(OpCategory::Softmax.recommended_precision(), Precision::FP32);
        assert_eq!(OpCategory::Loss.recommended_precision(), Precision::FP32);
    }

    // --- PrecisionPolicy tests ---

    #[test]
    fn test_default_precision_policy() {
        let policy = PrecisionPolicy::default();
        assert_eq!(policy.default_precision(), Precision::FP32);
        assert!(policy.keep_master_weights());
        assert!(policy.fp32_grad_accumulation());
    }

    #[test]
    fn test_amp_o1_policy() {
        let policy = PrecisionPolicy::amp_o1();
        assert_eq!(policy.precision_for(OpCategory::MatMul), Precision::FP16);
        assert_eq!(
            policy.precision_for(OpCategory::Convolution),
            Precision::FP16
        );
        assert_eq!(
            policy.precision_for(OpCategory::Normalization),
            Precision::FP32
        );
        assert_eq!(policy.precision_for(OpCategory::Loss), Precision::FP32);
    }

    #[test]
    fn test_amp_o2_policy() {
        let policy = PrecisionPolicy::amp_o2();
        assert_eq!(policy.precision_for(OpCategory::MatMul), Precision::FP16);
        assert_eq!(
            policy.precision_for(OpCategory::Activation),
            Precision::FP16
        );
        assert_eq!(
            policy.precision_for(OpCategory::Normalization),
            Precision::FP32
        );
        assert_eq!(policy.precision_for(OpCategory::Loss), Precision::FP32);
    }

    #[test]
    fn test_full_fp32_policy() {
        let policy = PrecisionPolicy::full_fp32();
        assert_eq!(policy.precision_for(OpCategory::MatMul), Precision::FP32);
        assert!(!policy.keep_master_weights());
    }

    #[test]
    fn test_policy_override() {
        let mut policy = PrecisionPolicy::default();
        policy.set_precision(OpCategory::Embedding, Precision::FP16);
        assert_eq!(policy.precision_for(OpCategory::Embedding), Precision::FP16);
        assert_eq!(policy.num_overrides(), 1);
    }

    // --- StaticLossScaler tests ---

    #[test]
    fn test_static_loss_scaler() {
        let scaler = StaticLossScaler::new(128.0);
        assert_eq!(scaler.scale(), 128.0);

        let loss = Array1::from(vec![1.0_f64, 2.0, 3.0]).into_dyn();
        let scaled = scaler.scale_loss(&loss).expect("scale_loss");
        assert!((scaled[[0]] - 128.0).abs() < 1e-10);
        assert!((scaled[[1]] - 256.0).abs() < 1e-10);
    }

    #[test]
    fn test_static_scaler_unscale() {
        let scaler = StaticLossScaler::new(4.0);
        let mut grads = vec![Array1::from(vec![4.0_f64, 8.0, 12.0]).into_dyn()];
        scaler.unscale_gradients(&mut grads).expect("unscale");
        assert!((grads[0][[0]] - 1.0).abs() < 1e-10);
        assert!((grads[0][[1]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_static_scaler_nonpositive() {
        let scaler = StaticLossScaler::new(-5.0);
        assert_eq!(scaler.scale(), 1.0); // clamps to 1.0
    }

    // --- DynamicLossScaler tests ---

    #[test]
    fn test_dynamic_scaler_initial_scale() {
        let scaler = DynamicLossScaler::new(DynamicScalerConfig::default());
        assert_eq!(scaler.current_scale(), 65536.0);
        assert_eq!(scaler.total_steps(), 0);
        assert_eq!(scaler.total_overflows(), 0);
    }

    #[test]
    fn test_dynamic_scaler_backoff_on_overflow() {
        let config = DynamicScalerConfig {
            init_scale: 1024.0,
            backoff_factor: 0.5,
            growth_interval: 10,
            ..Default::default()
        };
        let mut scaler = DynamicLossScaler::new(config);

        // Simulate overflow
        let mut grads = vec![Array1::from(vec![f64::INFINITY, 1.0]).into_dyn()];
        let valid = scaler.unscale_and_check(&mut grads).expect("check");
        assert!(!valid);
        assert_eq!(scaler.current_scale(), 512.0); // 1024 * 0.5
        assert_eq!(scaler.total_overflows(), 1);
    }

    #[test]
    fn test_dynamic_scaler_growth() {
        let config = DynamicScalerConfig {
            init_scale: 100.0,
            growth_factor: 2.0,
            growth_interval: 3,
            max_scale: 10000.0,
            ..Default::default()
        };
        let mut scaler = DynamicLossScaler::new(config);

        // 3 clean steps should trigger growth
        for _ in 0..3 {
            let mut grads = vec![Array1::from(vec![1.0_f64, 2.0]).into_dyn()];
            let valid = scaler.unscale_and_check(&mut grads).expect("check");
            assert!(valid);
        }
        assert_eq!(scaler.current_scale(), 200.0);
    }

    #[test]
    fn test_dynamic_scaler_scale_clamp() {
        let config = DynamicScalerConfig {
            init_scale: 100.0,
            backoff_factor: 0.5,
            min_scale: 80.0,
            ..Default::default()
        };
        let mut scaler = DynamicLossScaler::new(config);
        assert_eq!(scaler.current_scale(), 100.0);

        // Simulate overflow
        let mut grads = vec![Array1::from(vec![f64::NAN]).into_dyn()];
        scaler.unscale_and_check(&mut grads).expect("check");
        // 100 * 0.5 = 50, but min is 80
        assert_eq!(scaler.current_scale(), 80.0);
    }

    #[test]
    fn test_dynamic_scaler_reset() {
        let config = DynamicScalerConfig {
            init_scale: 256.0,
            ..Default::default()
        };
        let mut scaler = DynamicLossScaler::new(config);
        scaler.update(true); // overflow
        assert_eq!(scaler.total_overflows(), 1);

        scaler.reset();
        assert_eq!(scaler.current_scale(), 256.0);
        assert_eq!(scaler.total_overflows(), 0);
        assert_eq!(scaler.total_steps(), 0);
    }

    #[test]
    fn test_dynamic_scaler_overflow_ratio() {
        let mut scaler = DynamicLossScaler::new(DynamicScalerConfig::default());
        scaler.update(false);
        scaler.update(true);
        scaler.update(false);
        scaler.update(true);
        assert!((scaler.overflow_ratio() - 0.5).abs() < 1e-10);
    }

    // --- Gradient overflow detection tests ---

    #[test]
    fn test_no_overflow() {
        let grads = vec![Array1::from(vec![1.0_f64, 2.0, 3.0]).into_dyn()];
        assert!(!check_gradients_overflow(&grads));
    }

    #[test]
    fn test_inf_overflow() {
        let grads = vec![Array1::from(vec![1.0_f64, f64::INFINITY]).into_dyn()];
        assert!(check_gradients_overflow(&grads));
    }

    #[test]
    fn test_nan_overflow() {
        let grads = vec![Array1::from(vec![f64::NAN, 1.0_f64]).into_dyn()];
        assert!(check_gradients_overflow(&grads));
    }

    #[test]
    fn test_overflow_stats() {
        let grads = vec![Array1::from(vec![1.0_f64, 0.0, f64::INFINITY, f64::NAN, 0.5]).into_dyn()];
        let stats = gradient_overflow_stats(&grads);
        assert_eq!(stats.total_elements, 5);
        assert_eq!(stats.inf_count, 1);
        assert_eq!(stats.nan_count, 1);
        assert_eq!(stats.zero_count, 1);
        assert!(stats.has_overflow());
        assert!((stats.max_abs - 1.0).abs() < 1e-10);
        assert!((stats.min_abs_nonzero - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_overflow_stats_sparsity() {
        let grads = vec![Array1::from(vec![0.0_f64, 0.0, 1.0, 0.0]).into_dyn()];
        let stats = gradient_overflow_stats(&grads);
        assert!((stats.sparsity() - 0.75).abs() < 1e-10);
    }

    // --- MasterWeightManager tests ---

    #[test]
    fn test_master_weight_register() {
        let mut mgr = MasterWeightManager::<f64>::new(Precision::FP16);
        let w = Array1::from(vec![1.0, 2.0, 3.0]).into_dyn();
        mgr.register("layer1.weight", w.clone());
        assert_eq!(mgr.num_params(), 1);
        assert_eq!(mgr.total_elements(), 3);

        let retrieved = mgr.master_weight("layer1.weight").expect("exists");
        assert_eq!(retrieved.shape(), w.shape());
    }

    #[test]
    fn test_master_weight_compute_copy() {
        let mut mgr = MasterWeightManager::<f64>::new(Precision::FP16);
        let w = Array1::from(vec![0.5, 100000.0]).into_dyn();
        mgr.register("w", w);

        let copy = mgr.compute_copy("w").expect("compute_copy");
        // 100000.0 exceeds FP16 max (65504) and should be clamped
        assert!(copy[[1]] <= 65504.0);
    }

    #[test]
    fn test_master_weight_apply_gradients() {
        let mut mgr = MasterWeightManager::<f64>::new(Precision::FP16);
        let w = Array1::from(vec![1.0, 2.0, 3.0]).into_dyn();
        mgr.register("w", w);

        let grad = Array1::from(vec![0.1, 0.2, 0.3]).into_dyn();
        mgr.apply_gradients("w", &grad, 1.0).expect("apply");

        let updated = mgr.master_weight("w").expect("exists");
        assert!((updated[[0]] - 0.9).abs() < 1e-10);
        assert!((updated[[1]] - 1.8).abs() < 1e-10);
    }

    #[test]
    fn test_master_weight_shape_mismatch() {
        let mut mgr = MasterWeightManager::<f64>::new(Precision::FP16);
        let w = Array1::from(vec![1.0, 2.0]).into_dyn();
        mgr.register("w", w);

        let grad = Array1::from(vec![0.1, 0.2, 0.3]).into_dyn();
        let result = mgr.apply_gradients("w", &grad, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_master_weight_remove() {
        let mut mgr = MasterWeightManager::<f64>::new(Precision::FP16);
        mgr.register("w", Array1::from(vec![1.0]).into_dyn());
        assert_eq!(mgr.num_params(), 1);
        mgr.remove("w");
        assert_eq!(mgr.num_params(), 0);
    }

    // --- MixedPrecisionTrainer tests ---

    #[test]
    fn test_trainer_creation() {
        let config = MixedPrecisionConfig::default();
        let trainer = MixedPrecisionTrainer::<f64>::new(config);
        assert_eq!(trainer.step_count(), 0);
        assert_eq!(trainer.skipped_steps(), 0);
    }

    #[test]
    fn test_trainer_scale_loss() {
        let trainer = MixedPrecisionTrainer::<f64>::new(MixedPrecisionConfig::default());
        let loss = Array1::from(vec![1.0_f64]).into_dyn();
        let scaled = trainer.scale_loss(&loss).expect("scale_loss");
        assert!(scaled[[0]] > 1.0); // should be scaled up
    }

    #[test]
    fn test_trainer_step_clean() {
        let mut trainer = MixedPrecisionTrainer::<f64>::new(MixedPrecisionConfig::default());
        let mut grads = vec![Array1::from(vec![1.0_f64, 2.0]).into_dyn()];
        let applied = trainer.step(&mut grads).expect("step");
        assert!(applied);
        assert_eq!(trainer.step_count(), 1);
        assert_eq!(trainer.skipped_steps(), 0);
    }

    #[test]
    fn test_trainer_step_overflow_skip() {
        let mut trainer = MixedPrecisionTrainer::<f64>::new(MixedPrecisionConfig::default());
        let mut grads = vec![Array1::from(vec![f64::INFINITY]).into_dyn()];
        let applied = trainer.step(&mut grads).expect("step");
        assert!(!applied);
        assert_eq!(trainer.step_count(), 0);
        assert_eq!(trainer.skipped_steps(), 1);
    }

    #[test]
    fn test_trainer_precision_for() {
        let trainer = MixedPrecisionTrainer::<f64>::new(MixedPrecisionConfig::default());
        // O1 policy: matmul -> FP16, norm -> FP32
        assert_eq!(trainer.precision_for(OpCategory::MatMul), Precision::FP16);
        assert_eq!(
            trainer.precision_for(OpCategory::Normalization),
            Precision::FP32
        );
    }

    #[test]
    fn test_trainer_reset() {
        let mut trainer = MixedPrecisionTrainer::<f64>::new(MixedPrecisionConfig::default());
        let mut grads = vec![Array1::from(vec![1.0_f64]).into_dyn()];
        trainer.step(&mut grads).expect("step");
        assert_eq!(trainer.step_count(), 1);

        trainer.reset();
        assert_eq!(trainer.step_count(), 0);
    }

    #[test]
    fn test_trainer_register_param() {
        let mut trainer = MixedPrecisionTrainer::<f64>::new(MixedPrecisionConfig::default());
        trainer.register_param("fc1.weight", Array1::from(vec![1.0, 2.0, 3.0]).into_dyn());
        assert_eq!(trainer.master_weights().num_params(), 1);
    }

    // --- Precision casting tests ---

    #[test]
    fn test_simulate_fp16_cast_clamping() {
        let arr = Array1::from(vec![1.0_f64, 100000.0, -100000.0]).into_dyn();
        let cast = simulate_precision_cast(&arr, Precision::FP16).expect("cast");
        assert!((cast[[0]] - 1.0).abs() < 1e-3);
        assert!(cast[[1]] <= 65504.0);
        assert!(cast[[2]] >= -65504.0);
    }

    #[test]
    fn test_simulate_fp64_identity() {
        let arr = Array1::from(vec![1.0_f64, 2.0]).into_dyn();
        let cast = simulate_precision_cast(&arr, Precision::FP64).expect("cast");
        assert!((cast[[0]] - 1.0).abs() < 1e-15);
        assert!((cast[[1]] - 2.0).abs() < 1e-15);
    }

    #[test]
    fn test_quantisation_error() {
        let arr = Array1::from(vec![0.1_f64, 0.2, 0.3]).into_dyn();
        let err = quantisation_error(&arr, Precision::FP32).expect("qerr");
        // FP32 quantisation of FP64 values should have very small error
        assert!(err < 1e-6);
    }

    // --- classify_op tests ---

    #[test]
    fn test_classify_op_matmul() {
        assert_eq!(classify_op("MatMulOp"), OpCategory::MatMul);
        assert_eq!(classify_op("linear_forward"), OpCategory::MatMul);
        assert_eq!(classify_op("gemm"), OpCategory::MatMul);
    }

    #[test]
    fn test_classify_op_conv() {
        assert_eq!(classify_op("Conv2D"), OpCategory::Convolution);
        assert_eq!(classify_op("conv_transpose"), OpCategory::Convolution);
    }

    #[test]
    fn test_classify_op_norm() {
        assert_eq!(classify_op("BatchNorm"), OpCategory::Normalization);
        assert_eq!(classify_op("layer_norm"), OpCategory::Normalization);
    }

    #[test]
    fn test_classify_op_activation() {
        assert_eq!(classify_op("ReLU"), OpCategory::Activation);
        assert_eq!(classify_op("gelu"), OpCategory::Activation);
        assert_eq!(classify_op("sigmoid"), OpCategory::Activation);
    }

    #[test]
    fn test_classify_op_loss() {
        assert_eq!(classify_op("cross_entropy_loss"), OpCategory::Loss);
        assert_eq!(classify_op("mse_loss"), OpCategory::Loss);
    }

    #[test]
    fn test_classify_op_unknown() {
        assert_eq!(classify_op("my_custom_op"), OpCategory::Custom);
    }

    // --- assign_precisions tests ---

    #[test]
    fn test_assign_precisions() {
        let policy = PrecisionPolicy::amp_o1();
        let ops = &["MatMulOp", "BatchNorm", "ReLU"];
        let assignments = assign_precisions(ops, &policy);
        assert_eq!(assignments[0].1, Precision::FP16); // matmul
        assert_eq!(assignments[1].1, Precision::FP32); // norm (not overridden, default FP32)
        assert_eq!(assignments[2].1, Precision::FP32); // relu (not overridden)
    }

    #[test]
    fn test_assign_precisions_o2() {
        let policy = PrecisionPolicy::amp_o2();
        let ops = &["MatMulOp", "BatchNorm", "ReLU", "softmax"];
        let assignments = assign_precisions(ops, &policy);
        assert_eq!(assignments[0].1, Precision::FP16); // matmul
        assert_eq!(assignments[1].1, Precision::FP32); // norm
        assert_eq!(assignments[2].1, Precision::FP16); // activation -> FP16 in O2
        assert_eq!(assignments[3].1, Precision::FP32); // softmax -> FP32
    }

    // --- Mixed precision config tests ---

    #[test]
    fn test_mixed_precision_config_default() {
        let config = MixedPrecisionConfig::default();
        assert!(config.dynamic_loss_scaling);
        assert!(config.skip_on_overflow);
        assert_eq!(config.max_consecutive_overflows, 100);
    }

    #[test]
    fn test_trainer_max_consecutive_overflow_error() {
        let config = MixedPrecisionConfig {
            max_consecutive_overflows: 2,
            ..Default::default()
        };
        let mut trainer = MixedPrecisionTrainer::<f64>::new(config);

        // First overflow — skipped
        let mut grads = vec![Array1::from(vec![f64::NAN]).into_dyn()];
        let r1 = trainer.step(&mut grads);
        assert!(r1.is_ok());

        // Second overflow — hits limit
        let mut grads2 = vec![Array1::from(vec![f64::NAN]).into_dyn()];
        let r2 = trainer.step(&mut grads2);
        assert!(r2.is_err());
    }

    #[test]
    fn test_trainer_static_scaler_mode() {
        let config = MixedPrecisionConfig {
            dynamic_loss_scaling: false,
            static_scale: 256.0,
            ..Default::default()
        };
        let trainer = MixedPrecisionTrainer::<f64>::new(config);
        assert_eq!(trainer.current_scale(), 256.0);
    }
}
