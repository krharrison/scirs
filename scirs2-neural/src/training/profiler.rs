//! Training Profiler for Neural Network Performance Analysis
//!
//! Provides detailed profiling of neural network training, including:
//!
//! - **Per-layer timing**: forward and backward pass durations
//! - **Memory usage tracking**: per-layer and total memory estimates
//! - **Throughput measurement**: samples/sec, batches/sec
//! - **GPU utilization estimation**: compute vs idle time
//! - **Bottleneck identification**: finds the slowest layers
//! - **Summary report generation**: human-readable performance report
//!
//! # Example
//!
//! ```rust
//! use scirs2_neural::training::profiler::{
//!     TrainingProfiler, LayerProfile, ProfileSummary,
//! };
//!
//! let mut profiler = TrainingProfiler::new();
//!
//! // Record some layer timings
//! profiler.record_layer_forward("dense_1", std::time::Duration::from_micros(500));
//! profiler.record_layer_backward("dense_1", std::time::Duration::from_micros(800));
//! profiler.record_layer_forward("dense_2", std::time::Duration::from_micros(300));
//! profiler.record_layer_backward("dense_2", std::time::Duration::from_micros(400));
//!
//! // Record batch completion
//! profiler.record_batch(32);
//!
//! let summary = profiler.summary();
//! assert!(summary.total_batches >= 1);
//! ```

use crate::error::{NeuralError, Result};
use std::collections::HashMap;
use std::fmt::{self, Debug, Display};
use std::time::{Duration, Instant};

// ============================================================================
// Layer Profile
// ============================================================================

/// Timing and memory profile for a single layer.
#[derive(Debug, Clone)]
pub struct LayerProfile {
    /// Layer name or identifier.
    pub name: String,
    /// Cumulative forward pass time.
    pub forward_time: Duration,
    /// Cumulative backward pass time.
    pub backward_time: Duration,
    /// Number of forward passes recorded.
    pub forward_count: usize,
    /// Number of backward passes recorded.
    pub backward_count: usize,
    /// Estimated memory usage in bytes (parameters + activations).
    pub memory_bytes: usize,
    /// Number of trainable parameters in this layer.
    pub parameter_count: usize,
    /// FLOP estimate per forward pass (if available).
    pub flops_per_forward: Option<u64>,
}

impl LayerProfile {
    /// Create a new empty layer profile.
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            forward_time: Duration::ZERO,
            backward_time: Duration::ZERO,
            forward_count: 0,
            backward_count: 0,
            memory_bytes: 0,
            parameter_count: 0,
            flops_per_forward: None,
        }
    }

    /// Average forward pass duration.
    pub fn avg_forward(&self) -> Duration {
        if self.forward_count == 0 {
            Duration::ZERO
        } else {
            self.forward_time / self.forward_count as u32
        }
    }

    /// Average backward pass duration.
    pub fn avg_backward(&self) -> Duration {
        if self.backward_count == 0 {
            Duration::ZERO
        } else {
            self.backward_time / self.backward_count as u32
        }
    }

    /// Total time (forward + backward).
    pub fn total_time(&self) -> Duration {
        self.forward_time + self.backward_time
    }

    /// Average total time per step.
    pub fn avg_total(&self) -> Duration {
        let count = self.forward_count.max(self.backward_count).max(1);
        self.total_time() / count as u32
    }
}

impl Display for LayerProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: fwd={:.2}ms bwd={:.2}ms total={:.2}ms mem={}B params={}",
            self.name,
            self.avg_forward().as_secs_f64() * 1000.0,
            self.avg_backward().as_secs_f64() * 1000.0,
            self.avg_total().as_secs_f64() * 1000.0,
            self.memory_bytes,
            self.parameter_count,
        )
    }
}

// ============================================================================
// Batch Stats
// ============================================================================

/// Statistics for a single batch.
#[derive(Debug, Clone)]
pub struct BatchStats {
    /// Batch index.
    pub batch_idx: usize,
    /// Number of samples in this batch.
    pub batch_size: usize,
    /// Total batch processing time.
    pub duration: Duration,
    /// Loss value (if recorded).
    pub loss: Option<f64>,
}

/// Statistics for a single epoch.
#[derive(Debug, Clone)]
pub struct EpochStats {
    /// Epoch index.
    pub epoch: usize,
    /// Total epoch duration.
    pub duration: Duration,
    /// Number of batches.
    pub num_batches: usize,
    /// Total samples processed.
    pub total_samples: usize,
    /// Average batch duration.
    pub avg_batch_duration: Duration,
    /// Throughput: samples per second.
    pub samples_per_sec: f64,
    /// Average loss.
    pub avg_loss: Option<f64>,
}

// ============================================================================
// Profile Phase
// ============================================================================

/// Phase of training being profiled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProfilePhase {
    /// Forward pass.
    Forward,
    /// Backward pass (gradient computation).
    Backward,
    /// Optimizer step (parameter update).
    OptimizerStep,
    /// Data loading.
    DataLoading,
    /// Validation.
    Validation,
    /// Other / miscellaneous.
    Other,
}

impl Display for ProfilePhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Forward => write!(f, "Forward"),
            Self::Backward => write!(f, "Backward"),
            Self::OptimizerStep => write!(f, "OptimizerStep"),
            Self::DataLoading => write!(f, "DataLoading"),
            Self::Validation => write!(f, "Validation"),
            Self::Other => write!(f, "Other"),
        }
    }
}

// ============================================================================
// Bottleneck
// ============================================================================

/// An identified bottleneck in the training pipeline.
#[derive(Debug, Clone)]
pub struct Bottleneck {
    /// Description of the bottleneck.
    pub description: String,
    /// The component/layer causing the bottleneck.
    pub component: String,
    /// Severity: fraction of total time consumed (0.0 to 1.0).
    pub severity: f64,
    /// Suggested mitigation.
    pub suggestion: String,
}

impl Display for Bottleneck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:.0}%] {}: {} -> {}",
            self.severity * 100.0,
            self.component,
            self.description,
            self.suggestion,
        )
    }
}

// ============================================================================
// Profile Summary
// ============================================================================

/// Summary of profiling results.
#[derive(Debug, Clone)]
pub struct ProfileSummary {
    /// Total wall-clock time.
    pub total_time: Duration,
    /// Total batches processed.
    pub total_batches: usize,
    /// Total samples processed.
    pub total_samples: usize,
    /// Overall throughput (samples/sec).
    pub throughput_samples_per_sec: f64,
    /// Overall throughput (batches/sec).
    pub throughput_batches_per_sec: f64,
    /// Time breakdown by phase.
    pub phase_times: HashMap<ProfilePhase, Duration>,
    /// Per-layer profiles (sorted by total time, descending).
    pub layer_profiles: Vec<LayerProfile>,
    /// Identified bottlenecks (sorted by severity, descending).
    pub bottlenecks: Vec<Bottleneck>,
    /// Estimated GPU utilization (0.0 to 1.0), if available.
    pub gpu_utilization: Option<f64>,
    /// Estimated total memory usage in bytes.
    pub total_memory_bytes: usize,
    /// Number of epochs completed.
    pub num_epochs: usize,
    /// Per-epoch statistics.
    pub epoch_stats: Vec<EpochStats>,
}

// ============================================================================
// Training Profiler
// ============================================================================

/// A training profiler that collects timing and memory metrics.
#[derive(Debug, Clone)]
pub struct TrainingProfiler {
    /// Per-layer profiles.
    layers: HashMap<String, LayerProfile>,
    /// Insertion order for layers (to maintain ordering).
    layer_order: Vec<String>,
    /// Phase-level timings.
    phase_times: HashMap<ProfilePhase, Duration>,
    /// Active phase timers (for start/stop).
    active_timers: HashMap<String, Instant>,
    /// Batch statistics for the current epoch.
    current_epoch_batches: Vec<BatchStats>,
    /// Per-epoch statistics (completed epochs).
    epoch_stats: Vec<EpochStats>,
    /// Total samples processed.
    total_samples: usize,
    /// Total batches processed.
    total_batches: usize,
    /// Profiling start time.
    start_time: Option<Instant>,
    /// Current epoch start time.
    epoch_start: Option<Instant>,
    /// Current batch start time.
    batch_start: Option<Instant>,
    /// Current batch index within epoch.
    batch_idx: usize,
    /// Memory tracking: per-layer estimates.
    memory_estimates: HashMap<String, usize>,
    /// Whether the profiler is enabled.
    enabled: bool,
    /// GPU compute time estimate.
    gpu_compute_time: Duration,
    /// GPU idle time estimate.
    gpu_idle_time: Duration,
}

impl TrainingProfiler {
    /// Create a new training profiler.
    pub fn new() -> Self {
        Self {
            layers: HashMap::new(),
            layer_order: Vec::new(),
            phase_times: HashMap::new(),
            active_timers: HashMap::new(),
            current_epoch_batches: Vec::new(),
            epoch_stats: Vec::new(),
            total_samples: 0,
            total_batches: 0,
            start_time: None,
            epoch_start: None,
            batch_start: None,
            batch_idx: 0,
            memory_estimates: HashMap::new(),
            enabled: true,
            gpu_compute_time: Duration::ZERO,
            gpu_idle_time: Duration::ZERO,
        }
    }

    /// Create a disabled profiler (no-op; minimal overhead).
    pub fn disabled() -> Self {
        let mut p = Self::new();
        p.enabled = false;
        p
    }

    /// Whether the profiler is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable or disable the profiler.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    // ---- Phase timing ----

    /// Start timing a named phase. Call `stop_timer` to record the duration.
    pub fn start_timer(&mut self, name: &str) {
        if !self.enabled {
            return;
        }
        self.active_timers.insert(name.to_string(), Instant::now());
    }

    /// Stop a named timer and return the elapsed duration.
    pub fn stop_timer(&mut self, name: &str) -> Duration {
        if !self.enabled {
            return Duration::ZERO;
        }
        if let Some(start) = self.active_timers.remove(name) {
            let elapsed = start.elapsed();
            // Try to map to a ProfilePhase
            let phase = match name {
                "forward" | "fwd" => Some(ProfilePhase::Forward),
                "backward" | "bwd" => Some(ProfilePhase::Backward),
                "optimizer" | "optim" | "optimizer_step" => Some(ProfilePhase::OptimizerStep),
                "data_loading" | "data" => Some(ProfilePhase::DataLoading),
                "validation" | "val" => Some(ProfilePhase::Validation),
                _ => None,
            };
            if let Some(p) = phase {
                *self.phase_times.entry(p).or_insert(Duration::ZERO) += elapsed;
            }
            elapsed
        } else {
            Duration::ZERO
        }
    }

    /// Record time for a specific phase directly.
    pub fn record_phase(&mut self, phase: ProfilePhase, duration: Duration) {
        if !self.enabled {
            return;
        }
        *self.phase_times.entry(phase).or_insert(Duration::ZERO) += duration;
    }

    // ---- Layer timing ----

    fn ensure_layer(&mut self, name: &str) {
        if !self.layers.contains_key(name) {
            self.layers
                .insert(name.to_string(), LayerProfile::new(name));
            self.layer_order.push(name.to_string());
        }
    }

    /// Record a forward pass duration for a layer.
    pub fn record_layer_forward(&mut self, layer_name: &str, duration: Duration) {
        if !self.enabled {
            return;
        }
        self.ensure_layer(layer_name);
        if let Some(lp) = self.layers.get_mut(layer_name) {
            lp.forward_time += duration;
            lp.forward_count += 1;
        }
        *self
            .phase_times
            .entry(ProfilePhase::Forward)
            .or_insert(Duration::ZERO) += duration;
    }

    /// Record a backward pass duration for a layer.
    pub fn record_layer_backward(&mut self, layer_name: &str, duration: Duration) {
        if !self.enabled {
            return;
        }
        self.ensure_layer(layer_name);
        if let Some(lp) = self.layers.get_mut(layer_name) {
            lp.backward_time += duration;
            lp.backward_count += 1;
        }
        *self
            .phase_times
            .entry(ProfilePhase::Backward)
            .or_insert(Duration::ZERO) += duration;
    }

    // ---- Memory tracking ----

    /// Set the estimated memory usage for a layer.
    pub fn set_layer_memory(&mut self, layer_name: &str, bytes: usize) {
        if !self.enabled {
            return;
        }
        self.ensure_layer(layer_name);
        if let Some(lp) = self.layers.get_mut(layer_name) {
            lp.memory_bytes = bytes;
        }
        self.memory_estimates.insert(layer_name.to_string(), bytes);
    }

    /// Set the parameter count for a layer.
    pub fn set_layer_params(&mut self, layer_name: &str, count: usize) {
        if !self.enabled {
            return;
        }
        self.ensure_layer(layer_name);
        if let Some(lp) = self.layers.get_mut(layer_name) {
            lp.parameter_count = count;
        }
    }

    /// Set the FLOP estimate for a layer.
    pub fn set_layer_flops(&mut self, layer_name: &str, flops: u64) {
        if !self.enabled {
            return;
        }
        self.ensure_layer(layer_name);
        if let Some(lp) = self.layers.get_mut(layer_name) {
            lp.flops_per_forward = Some(flops);
        }
    }

    // ---- GPU utilization ----

    /// Record GPU compute time.
    pub fn record_gpu_compute(&mut self, duration: Duration) {
        if !self.enabled {
            return;
        }
        self.gpu_compute_time += duration;
    }

    /// Record GPU idle time.
    pub fn record_gpu_idle(&mut self, duration: Duration) {
        if !self.enabled {
            return;
        }
        self.gpu_idle_time += duration;
    }

    // ---- Batch / Epoch tracking ----

    /// Signal the start of profiling.
    pub fn start(&mut self) {
        if !self.enabled {
            return;
        }
        self.start_time = Some(Instant::now());
    }

    /// Signal the start of an epoch.
    pub fn start_epoch(&mut self) {
        if !self.enabled {
            return;
        }
        self.epoch_start = Some(Instant::now());
        self.current_epoch_batches.clear();
        self.batch_idx = 0;
    }

    /// Signal the start of a batch.
    pub fn start_batch(&mut self) {
        if !self.enabled {
            return;
        }
        self.batch_start = Some(Instant::now());
    }

    /// Record the completion of a batch.
    pub fn record_batch(&mut self, batch_size: usize) {
        if !self.enabled {
            return;
        }
        let duration = self
            .batch_start
            .map(|s| s.elapsed())
            .unwrap_or(Duration::ZERO);

        self.current_epoch_batches.push(BatchStats {
            batch_idx: self.batch_idx,
            batch_size,
            duration,
            loss: None,
        });

        self.total_samples += batch_size;
        self.total_batches += 1;
        self.batch_idx += 1;
        self.batch_start = None;
    }

    /// Record a batch with loss.
    pub fn record_batch_with_loss(&mut self, batch_size: usize, loss: f64) {
        if !self.enabled {
            return;
        }
        let duration = self
            .batch_start
            .map(|s| s.elapsed())
            .unwrap_or(Duration::ZERO);

        self.current_epoch_batches.push(BatchStats {
            batch_idx: self.batch_idx,
            batch_size,
            duration,
            loss: Some(loss),
        });

        self.total_samples += batch_size;
        self.total_batches += 1;
        self.batch_idx += 1;
        self.batch_start = None;
    }

    /// Signal the end of an epoch, recording statistics.
    pub fn end_epoch(&mut self) {
        if !self.enabled {
            return;
        }
        let duration = self
            .epoch_start
            .map(|s| s.elapsed())
            .unwrap_or(Duration::ZERO);

        let num_batches = self.current_epoch_batches.len();
        let total_samples: usize = self
            .current_epoch_batches
            .iter()
            .map(|b| b.batch_size)
            .sum();

        let avg_batch_duration = if num_batches > 0 {
            let total_batch_time: Duration =
                self.current_epoch_batches.iter().map(|b| b.duration).sum();
            total_batch_time / num_batches as u32
        } else {
            Duration::ZERO
        };

        let samples_per_sec = if duration.as_secs_f64() > 0.0 {
            total_samples as f64 / duration.as_secs_f64()
        } else {
            0.0
        };

        let losses: Vec<f64> = self
            .current_epoch_batches
            .iter()
            .filter_map(|b| b.loss)
            .collect();
        let avg_loss = if losses.is_empty() {
            None
        } else {
            Some(losses.iter().sum::<f64>() / losses.len() as f64)
        };

        self.epoch_stats.push(EpochStats {
            epoch: self.epoch_stats.len(),
            duration,
            num_batches,
            total_samples,
            avg_batch_duration,
            samples_per_sec,
            avg_loss,
        });

        self.epoch_start = None;
    }

    // ---- Analysis ----

    /// Identify bottlenecks in the training pipeline.
    pub fn identify_bottlenecks(&self) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();

        // Total time across all phases
        let total_phase_time: Duration = self.phase_times.values().sum();
        if total_phase_time.is_zero() {
            return bottlenecks;
        }
        let total_secs = total_phase_time.as_secs_f64();

        // Check if data loading is a bottleneck (> 30% of time)
        if let Some(&data_time) = self.phase_times.get(&ProfilePhase::DataLoading) {
            let frac = data_time.as_secs_f64() / total_secs;
            if frac > 0.3 {
                bottlenecks.push(Bottleneck {
                    description: "Data loading consumes significant time".to_string(),
                    component: "DataLoading".to_string(),
                    severity: frac,
                    suggestion: "Consider prefetching, increasing num_workers, or caching data"
                        .to_string(),
                });
            }
        }

        // Find the slowest layer
        if !self.layers.is_empty() {
            let mut sorted_layers: Vec<&LayerProfile> = self.layers.values().collect();
            sorted_layers.sort_by(|a, b| {
                b.total_time()
                    .partial_cmp(&a.total_time())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            if let Some(slowest) = sorted_layers.first() {
                let layer_frac = slowest.total_time().as_secs_f64() / total_secs;
                if layer_frac > 0.4 {
                    bottlenecks.push(Bottleneck {
                        description: format!(
                            "Layer '{}' takes {:.0}% of compute time",
                            slowest.name,
                            layer_frac * 100.0
                        ),
                        component: slowest.name.clone(),
                        severity: layer_frac,
                        suggestion:
                            "Consider reducing layer size, using mixed precision, or fusing ops"
                                .to_string(),
                    });
                }
            }

            // Check backward/forward ratio
            for layer in &sorted_layers {
                if layer.forward_count > 0 && layer.backward_count > 0 {
                    let bwd_fwd_ratio = layer.avg_backward().as_secs_f64()
                        / layer.avg_forward().as_secs_f64().max(f64::EPSILON);
                    if bwd_fwd_ratio > 5.0 {
                        bottlenecks.push(Bottleneck {
                            description: format!(
                                "Layer '{}' backward is {:.1}x slower than forward",
                                layer.name, bwd_fwd_ratio
                            ),
                            component: layer.name.clone(),
                            severity: 0.3,
                            suggestion: "Consider gradient checkpointing or simpler backward ops"
                                .to_string(),
                        });
                    }
                }
            }
        }

        // Check GPU utilization
        let total_gpu = self.gpu_compute_time + self.gpu_idle_time;
        if !total_gpu.is_zero() {
            let util = self.gpu_compute_time.as_secs_f64() / total_gpu.as_secs_f64();
            if util < 0.5 {
                bottlenecks.push(Bottleneck {
                    description: format!("Low GPU utilization ({:.0}%)", util * 100.0),
                    component: "GPU".to_string(),
                    severity: 1.0 - util,
                    suggestion: "Increase batch size, use async data loading, or overlap compute"
                        .to_string(),
                });
            }
        }

        // Sort by severity (descending)
        bottlenecks.sort_by(|a, b| {
            b.severity
                .partial_cmp(&a.severity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        bottlenecks
    }

    /// Generate a complete profile summary.
    pub fn summary(&self) -> ProfileSummary {
        let total_time = self
            .start_time
            .map(|s| s.elapsed())
            .unwrap_or(Duration::ZERO);

        let total_secs = total_time.as_secs_f64().max(f64::EPSILON);

        // Layer profiles sorted by total time
        let mut layer_profiles: Vec<LayerProfile> = self
            .layer_order
            .iter()
            .filter_map(|name| self.layers.get(name).cloned())
            .collect();
        layer_profiles.sort_by(|a, b| {
            b.total_time()
                .partial_cmp(&a.total_time())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // GPU utilization
        let gpu_utilization = {
            let total_gpu = self.gpu_compute_time + self.gpu_idle_time;
            if total_gpu.is_zero() {
                None
            } else {
                Some(self.gpu_compute_time.as_secs_f64() / total_gpu.as_secs_f64())
            }
        };

        // Total memory
        let total_memory_bytes: usize = self.memory_estimates.values().sum();

        ProfileSummary {
            total_time,
            total_batches: self.total_batches,
            total_samples: self.total_samples,
            throughput_samples_per_sec: self.total_samples as f64 / total_secs,
            throughput_batches_per_sec: self.total_batches as f64 / total_secs,
            phase_times: self.phase_times.clone(),
            layer_profiles,
            bottlenecks: self.identify_bottlenecks(),
            gpu_utilization,
            total_memory_bytes,
            num_epochs: self.epoch_stats.len(),
            epoch_stats: self.epoch_stats.clone(),
        }
    }

    /// Generate a human-readable text report.
    pub fn report(&self) -> String {
        let summary = self.summary();
        let mut out = String::new();

        out.push_str("=== Training Profiler Report ===\n\n");

        // Overview
        out.push_str("--- Overview ---\n");
        out.push_str(&format!(
            "Total time: {:.2}s\n",
            summary.total_time.as_secs_f64()
        ));
        out.push_str(&format!("Epochs: {}\n", summary.num_epochs));
        out.push_str(&format!("Batches: {}\n", summary.total_batches));
        out.push_str(&format!("Samples: {}\n", summary.total_samples));
        out.push_str(&format!(
            "Throughput: {:.1} samples/sec, {:.1} batches/sec\n",
            summary.throughput_samples_per_sec, summary.throughput_batches_per_sec,
        ));
        if let Some(gpu_util) = summary.gpu_utilization {
            out.push_str(&format!("GPU utilization: {:.1}%\n", gpu_util * 100.0));
        }
        if summary.total_memory_bytes > 0 {
            out.push_str(&format!(
                "Est. memory: {:.1} MB\n",
                summary.total_memory_bytes as f64 / (1024.0 * 1024.0)
            ));
        }
        out.push('\n');

        // Phase breakdown
        if !summary.phase_times.is_empty() {
            out.push_str("--- Phase Breakdown ---\n");
            let total_phase: Duration = summary.phase_times.values().sum();
            let total_phase_secs = total_phase.as_secs_f64().max(f64::EPSILON);

            let mut phases: Vec<_> = summary.phase_times.iter().collect();
            phases.sort_by(|a, b| b.1.cmp(a.1));
            for (phase, dur) in phases {
                let pct = dur.as_secs_f64() / total_phase_secs * 100.0;
                out.push_str(&format!(
                    "  {phase}: {:.2}ms ({pct:.1}%)\n",
                    dur.as_secs_f64() * 1000.0,
                ));
            }
            out.push('\n');
        }

        // Per-layer breakdown
        if !summary.layer_profiles.is_empty() {
            out.push_str("--- Layer Breakdown (sorted by total time) ---\n");
            for lp in &summary.layer_profiles {
                out.push_str(&format!("  {lp}\n"));
            }
            out.push('\n');
        }

        // Epoch stats
        if !summary.epoch_stats.is_empty() {
            out.push_str("--- Epoch Statistics ---\n");
            for es in &summary.epoch_stats {
                out.push_str(&format!(
                    "  Epoch {}: {:.2}s, {} batches, {:.1} samples/sec",
                    es.epoch,
                    es.duration.as_secs_f64(),
                    es.num_batches,
                    es.samples_per_sec,
                ));
                if let Some(loss) = es.avg_loss {
                    out.push_str(&format!(", loss={loss:.6}"));
                }
                out.push('\n');
            }
            out.push('\n');
        }

        // Bottlenecks
        if !summary.bottlenecks.is_empty() {
            out.push_str("--- Bottlenecks ---\n");
            for b in &summary.bottlenecks {
                out.push_str(&format!("  {b}\n"));
            }
            out.push('\n');
        }

        out
    }

    /// Reset the profiler (clear all collected data).
    pub fn reset(&mut self) {
        self.layers.clear();
        self.layer_order.clear();
        self.phase_times.clear();
        self.active_timers.clear();
        self.current_epoch_batches.clear();
        self.epoch_stats.clear();
        self.total_samples = 0;
        self.total_batches = 0;
        self.start_time = None;
        self.epoch_start = None;
        self.batch_start = None;
        self.batch_idx = 0;
        self.memory_estimates.clear();
        self.gpu_compute_time = Duration::ZERO;
        self.gpu_idle_time = Duration::ZERO;
    }

    /// Get the per-layer profiles.
    pub fn layer_profiles(&self) -> &HashMap<String, LayerProfile> {
        &self.layers
    }

    /// Get epoch statistics.
    pub fn epoch_stats(&self) -> &[EpochStats] {
        &self.epoch_stats
    }

    /// Get total samples processed.
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Get total batches processed.
    pub fn total_batches(&self) -> usize {
        self.total_batches
    }
}

impl Default for TrainingProfiler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Scoped timer helper
// ============================================================================

/// RAII guard that records the elapsed time when dropped.
///
/// Use [`TrainingProfiler::start_timer`] for named timers, or create
/// a `ScopedTimer` for scoped timing.
pub struct ScopedTimer<'a> {
    profiler: &'a mut TrainingProfiler,
    name: String,
    start: Instant,
}

impl<'a> ScopedTimer<'a> {
    /// Create a new scoped timer.
    pub fn new(profiler: &'a mut TrainingProfiler, name: &str) -> Self {
        Self {
            profiler,
            name: name.to_string(),
            start: Instant::now(),
        }
    }

    /// Manually stop the timer and return the duration.
    pub fn stop(self) -> Duration {
        let elapsed = self.start.elapsed();
        // Try to map to a phase
        let phase = match self.name.as_str() {
            "forward" | "fwd" => Some(ProfilePhase::Forward),
            "backward" | "bwd" => Some(ProfilePhase::Backward),
            "optimizer" | "optim" | "optimizer_step" => Some(ProfilePhase::OptimizerStep),
            "data_loading" | "data" => Some(ProfilePhase::DataLoading),
            "validation" | "val" => Some(ProfilePhase::Validation),
            _ => None,
        };
        if let Some(p) = phase {
            self.profiler.record_phase(p, elapsed);
        }
        elapsed
    }
}

// ============================================================================
// Estimate helpers
// ============================================================================

/// Estimate the memory usage of a dense layer.
///
/// Accounts for weights, biases, gradients, and activations.
pub fn estimate_dense_memory(input_dim: usize, output_dim: usize, batch_size: usize) -> usize {
    let elem_size = std::mem::size_of::<f64>();
    let weights = input_dim * output_dim * elem_size;
    let biases = output_dim * elem_size;
    let weight_grads = weights;
    let bias_grads = biases;
    let activations = batch_size * output_dim * elem_size;
    let input_cache = batch_size * input_dim * elem_size;
    weights + biases + weight_grads + bias_grads + activations + input_cache
}

/// Estimate the memory usage of a Conv2D layer.
pub fn estimate_conv2d_memory(
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    output_h: usize,
    output_w: usize,
    batch_size: usize,
) -> usize {
    let elem_size = std::mem::size_of::<f64>();
    let weights = in_channels * out_channels * kernel_h * kernel_w * elem_size;
    let biases = out_channels * elem_size;
    let weight_grads = weights;
    let bias_grads = biases;
    let activations = batch_size * out_channels * output_h * output_w * elem_size;
    weights + biases + weight_grads + bias_grads + activations
}

/// Estimate FLOPs for a dense layer forward pass.
pub fn estimate_dense_flops(input_dim: usize, output_dim: usize, batch_size: usize) -> u64 {
    // matmul: 2 * input * output * batch (multiply + add)
    // bias: output * batch
    (2 * input_dim * output_dim * batch_size + output_dim * batch_size) as u64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let profiler = TrainingProfiler::new();
        assert!(profiler.is_enabled());
        assert_eq!(profiler.total_samples(), 0);
        assert_eq!(profiler.total_batches(), 0);
    }

    #[test]
    fn test_disabled_profiler() {
        let mut profiler = TrainingProfiler::disabled();
        assert!(!profiler.is_enabled());

        profiler.record_layer_forward("dense_1", Duration::from_micros(100));
        profiler.record_batch(32);

        // Nothing should be recorded
        assert_eq!(profiler.total_samples(), 0);
        assert_eq!(profiler.total_batches(), 0);
    }

    #[test]
    fn test_layer_forward_backward() {
        let mut profiler = TrainingProfiler::new();

        profiler.record_layer_forward("dense_1", Duration::from_micros(500));
        profiler.record_layer_backward("dense_1", Duration::from_micros(800));
        profiler.record_layer_forward("dense_1", Duration::from_micros(600));
        profiler.record_layer_backward("dense_1", Duration::from_micros(900));

        let lp = &profiler.layer_profiles()["dense_1"];
        assert_eq!(lp.forward_count, 2);
        assert_eq!(lp.backward_count, 2);
        assert_eq!(lp.forward_time, Duration::from_micros(1100));
        assert_eq!(lp.backward_time, Duration::from_micros(1700));
    }

    #[test]
    fn test_layer_avg_time() {
        let mut lp = LayerProfile::new("test");
        lp.forward_time = Duration::from_millis(100);
        lp.forward_count = 10;
        lp.backward_time = Duration::from_millis(200);
        lp.backward_count = 10;

        assert_eq!(lp.avg_forward(), Duration::from_millis(10));
        assert_eq!(lp.avg_backward(), Duration::from_millis(20));
        assert_eq!(lp.total_time(), Duration::from_millis(300));
    }

    #[test]
    fn test_layer_avg_zero_count() {
        let lp = LayerProfile::new("test");
        assert_eq!(lp.avg_forward(), Duration::ZERO);
        assert_eq!(lp.avg_backward(), Duration::ZERO);
    }

    #[test]
    fn test_batch_recording() {
        let mut profiler = TrainingProfiler::new();
        profiler.start();

        for _ in 0..5 {
            profiler.start_batch();
            // simulate work
            std::thread::sleep(Duration::from_millis(1));
            profiler.record_batch(32);
        }

        assert_eq!(profiler.total_batches(), 5);
        assert_eq!(profiler.total_samples(), 160);
    }

    #[test]
    fn test_batch_with_loss() {
        let mut profiler = TrainingProfiler::new();
        profiler.start();
        profiler.start_epoch();
        profiler.start_batch();
        profiler.record_batch_with_loss(32, 0.5);
        profiler.end_epoch();

        let stats = &profiler.epoch_stats()[0];
        assert_eq!(stats.num_batches, 1);
        assert_eq!(stats.total_samples, 32);
        assert!((stats.avg_loss.expect("has loss") - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_epoch_stats() {
        let mut profiler = TrainingProfiler::new();
        profiler.start();

        for _epoch in 0..3 {
            profiler.start_epoch();
            for _ in 0..10 {
                profiler.start_batch();
                profiler.record_batch(32);
            }
            profiler.end_epoch();
        }

        assert_eq!(profiler.epoch_stats().len(), 3);
        for es in profiler.epoch_stats() {
            assert_eq!(es.num_batches, 10);
            assert_eq!(es.total_samples, 320);
        }
    }

    #[test]
    fn test_phase_recording() {
        let mut profiler = TrainingProfiler::new();
        profiler.record_phase(ProfilePhase::Forward, Duration::from_millis(100));
        profiler.record_phase(ProfilePhase::Backward, Duration::from_millis(200));
        profiler.record_phase(ProfilePhase::DataLoading, Duration::from_millis(50));

        let summary = profiler.summary();
        assert_eq!(
            summary.phase_times[&ProfilePhase::Forward],
            Duration::from_millis(100)
        );
        assert_eq!(
            summary.phase_times[&ProfilePhase::Backward],
            Duration::from_millis(200)
        );
    }

    #[test]
    fn test_named_timer() {
        let mut profiler = TrainingProfiler::new();
        profiler.start_timer("forward");
        std::thread::sleep(Duration::from_millis(5));
        let elapsed = profiler.stop_timer("forward");
        assert!(elapsed >= Duration::from_millis(4));

        assert!(profiler
            .summary()
            .phase_times
            .contains_key(&ProfilePhase::Forward));
    }

    #[test]
    fn test_stop_nonexistent_timer() {
        let mut profiler = TrainingProfiler::new();
        let elapsed = profiler.stop_timer("nonexistent");
        assert_eq!(elapsed, Duration::ZERO);
    }

    #[test]
    fn test_memory_tracking() {
        let mut profiler = TrainingProfiler::new();
        profiler.set_layer_memory("dense_1", 1024 * 1024);
        profiler.set_layer_memory("dense_2", 512 * 1024);
        profiler.set_layer_params("dense_1", 50000);

        let summary = profiler.summary();
        assert_eq!(summary.total_memory_bytes, 1024 * 1024 + 512 * 1024);

        let lp = &profiler.layer_profiles()["dense_1"];
        assert_eq!(lp.parameter_count, 50000);
        assert_eq!(lp.memory_bytes, 1024 * 1024);
    }

    #[test]
    fn test_gpu_utilization() {
        let mut profiler = TrainingProfiler::new();
        profiler.record_gpu_compute(Duration::from_millis(800));
        profiler.record_gpu_idle(Duration::from_millis(200));

        let summary = profiler.summary();
        let util = summary.gpu_utilization.expect("should have GPU util");
        assert!((util - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_no_gpu_utilization() {
        let profiler = TrainingProfiler::new();
        let summary = profiler.summary();
        assert!(summary.gpu_utilization.is_none());
    }

    #[test]
    fn test_bottleneck_data_loading() {
        let mut profiler = TrainingProfiler::new();
        profiler.record_phase(ProfilePhase::DataLoading, Duration::from_millis(700));
        profiler.record_phase(ProfilePhase::Forward, Duration::from_millis(200));
        profiler.record_phase(ProfilePhase::Backward, Duration::from_millis(100));

        let bottlenecks = profiler.identify_bottlenecks();
        assert!(!bottlenecks.is_empty());
        assert!(bottlenecks.iter().any(|b| b.component == "DataLoading"));
    }

    #[test]
    fn test_bottleneck_slow_layer() {
        let mut profiler = TrainingProfiler::new();

        // One very slow layer, one fast layer
        profiler.record_layer_forward("slow_layer", Duration::from_millis(900));
        profiler.record_layer_backward("slow_layer", Duration::from_millis(900));
        profiler.record_layer_forward("fast_layer", Duration::from_millis(10));
        profiler.record_layer_backward("fast_layer", Duration::from_millis(10));

        let bottlenecks = profiler.identify_bottlenecks();
        assert!(!bottlenecks.is_empty());
        assert!(bottlenecks.iter().any(|b| b.component == "slow_layer"));
    }

    #[test]
    fn test_bottleneck_gpu_idle() {
        let mut profiler = TrainingProfiler::new();
        profiler.record_gpu_compute(Duration::from_millis(200));
        profiler.record_gpu_idle(Duration::from_millis(800));
        // Need some phase time to produce non-empty bottlenecks
        profiler.record_phase(ProfilePhase::Forward, Duration::from_millis(1000));

        let bottlenecks = profiler.identify_bottlenecks();
        assert!(bottlenecks.iter().any(|b| b.component == "GPU"));
    }

    #[test]
    fn test_summary_generation() {
        let mut profiler = TrainingProfiler::new();
        profiler.start();

        profiler.record_layer_forward("dense_1", Duration::from_micros(500));
        profiler.record_layer_backward("dense_1", Duration::from_micros(800));
        profiler.start_batch();
        profiler.record_batch(32);

        let summary = profiler.summary();
        assert_eq!(summary.total_batches, 1);
        assert_eq!(summary.total_samples, 32);
        assert!(!summary.layer_profiles.is_empty());
    }

    #[test]
    fn test_report_generation() {
        let mut profiler = TrainingProfiler::new();
        profiler.start();

        profiler.record_layer_forward("dense_1", Duration::from_micros(500));
        profiler.record_layer_backward("dense_1", Duration::from_micros(800));
        profiler.set_layer_memory("dense_1", 1024);
        profiler.set_layer_params("dense_1", 100);
        profiler.start_batch();
        profiler.record_batch(32);

        let report = profiler.report();
        assert!(report.contains("Training Profiler Report"));
        assert!(report.contains("dense_1"));
    }

    #[test]
    fn test_profiler_reset() {
        let mut profiler = TrainingProfiler::new();
        profiler.start();
        profiler.record_layer_forward("dense_1", Duration::from_micros(500));
        profiler.record_batch(32);

        profiler.reset();

        assert_eq!(profiler.total_samples(), 0);
        assert_eq!(profiler.total_batches(), 0);
        assert!(profiler.layer_profiles().is_empty());
    }

    #[test]
    fn test_layer_profile_display() {
        let mut lp = LayerProfile::new("test_layer");
        lp.forward_time = Duration::from_millis(10);
        lp.forward_count = 1;
        lp.backward_time = Duration::from_millis(20);
        lp.backward_count = 1;
        lp.memory_bytes = 1024;
        lp.parameter_count = 500;

        let s = format!("{lp}");
        assert!(s.contains("test_layer"));
        assert!(s.contains("fwd="));
        assert!(s.contains("bwd="));
    }

    #[test]
    fn test_phase_display() {
        assert_eq!(format!("{}", ProfilePhase::Forward), "Forward");
        assert_eq!(format!("{}", ProfilePhase::Backward), "Backward");
        assert_eq!(format!("{}", ProfilePhase::OptimizerStep), "OptimizerStep");
        assert_eq!(format!("{}", ProfilePhase::DataLoading), "DataLoading");
    }

    #[test]
    fn test_bottleneck_display() {
        let b = Bottleneck {
            description: "Slow forward".to_string(),
            component: "dense_1".to_string(),
            severity: 0.75,
            suggestion: "Use GPU".to_string(),
        };
        let s = format!("{b}");
        assert!(s.contains("75%"));
        assert!(s.contains("dense_1"));
    }

    #[test]
    fn test_estimate_dense_memory() {
        let mem = estimate_dense_memory(784, 256, 32);
        // weights: 784*256*8 = 1,605,632
        // biases: 256*8 = 2,048
        // grads: same as weights+biases
        // activations: 32*256*8 = 65,536
        // input_cache: 32*784*8 = 200,704
        let expected = (784 * 256 + 256 + 784 * 256 + 256 + 32 * 256 + 32 * 784) * 8;
        assert_eq!(mem, expected);
    }

    #[test]
    fn test_estimate_dense_flops() {
        let flops = estimate_dense_flops(784, 256, 32);
        // 2 * 784 * 256 * 32 + 256 * 32 = 12,853,248
        assert_eq!(flops, 2 * 784 * 256 * 32 + 256 * 32);
    }

    #[test]
    fn test_estimate_conv2d_memory() {
        let mem = estimate_conv2d_memory(3, 64, 3, 3, 32, 32, 16);
        assert!(mem > 0);
    }

    #[test]
    fn test_layer_flops() {
        let mut profiler = TrainingProfiler::new();
        profiler.set_layer_flops("dense_1", 1_000_000);

        let lp = &profiler.layer_profiles()["dense_1"];
        assert_eq!(lp.flops_per_forward, Some(1_000_000));
    }

    #[test]
    fn test_enable_disable() {
        let mut profiler = TrainingProfiler::new();
        assert!(profiler.is_enabled());

        profiler.set_enabled(false);
        assert!(!profiler.is_enabled());

        profiler.record_layer_forward("test", Duration::from_millis(100));
        assert!(profiler.layer_profiles().is_empty());

        profiler.set_enabled(true);
        profiler.record_layer_forward("test", Duration::from_millis(100));
        assert_eq!(profiler.layer_profiles().len(), 1);
    }

    #[test]
    fn test_multiple_layers_ordering() {
        let mut profiler = TrainingProfiler::new();
        profiler.record_layer_forward("layer_a", Duration::from_millis(10));
        profiler.record_layer_forward("layer_b", Duration::from_millis(50));
        profiler.record_layer_forward("layer_c", Duration::from_millis(30));

        let summary = profiler.summary();
        // Should be sorted by total time descending
        assert_eq!(summary.layer_profiles[0].name, "layer_b");
        assert_eq!(summary.layer_profiles[1].name, "layer_c");
        assert_eq!(summary.layer_profiles[2].name, "layer_a");
    }

    #[test]
    fn test_backward_forward_ratio_bottleneck() {
        let mut profiler = TrainingProfiler::new();

        // Layer with extremely slow backward
        profiler.record_layer_forward("attn", Duration::from_millis(10));
        profiler.record_layer_backward("attn", Duration::from_millis(100));

        let bottlenecks = profiler.identify_bottlenecks();
        let has_ratio_warning = bottlenecks
            .iter()
            .any(|b| b.description.contains("backward") && b.description.contains("slower"));
        assert!(has_ratio_warning);
    }
}
