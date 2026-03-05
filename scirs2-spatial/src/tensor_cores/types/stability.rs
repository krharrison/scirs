//! Numerical stability monitoring and performance-accuracy analysis for tensor core operations.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use crate::error::SpatialResult;
use scirs2_core::ndarray::{Array2, ArrayStatCompat};

use super::enums::{
    DynamicPrecisionConfig, NumericalErrorType, OptimizationObjective, PrecisionMode,
    ScalingStrategy, StabilityLevel, StabilityMetrics, TradeOffParams,
};

/// Performance-accuracy trade-off analyzer
#[derive(Debug)]
pub struct PerformanceAccuracyAnalyzer {
    /// Performance measurements by precision mode
    performance_data: HashMap<PrecisionMode, VecDeque<std::time::Duration>>,
    /// Accuracy measurements by precision mode
    accuracy_data: HashMap<PrecisionMode, VecDeque<f64>>,
    /// Trade-off optimization parameters
    optimization_params: TradeOffParams,
    /// Current Pareto frontier
    pub(crate) pareto_frontier: Vec<(f64, f64, PrecisionMode)>,
}

impl PerformanceAccuracyAnalyzer {
    /// Create new performance-accuracy analyzer
    pub fn new(params: TradeOffParams) -> Self {
        Self {
            performance_data: HashMap::new(),
            accuracy_data: HashMap::new(),
            optimization_params: params,
            pareto_frontier: Vec::new(),
        }
    }

    /// Record performance measurement
    pub fn record_performance(&mut self, precision: PrecisionMode, duration: std::time::Duration) {
        self.performance_data
            .entry(precision)
            .or_default()
            .push_back(duration);
        if let Some(history) = self.performance_data.get_mut(&precision) {
            if history.len() > 100 {
                history.pop_front();
            }
        }
    }

    /// Record accuracy measurement
    pub fn record_accuracy(&mut self, precision: PrecisionMode, accuracy: f64) {
        self.accuracy_data
            .entry(precision)
            .or_default()
            .push_back(accuracy);
        if let Some(history) = self.accuracy_data.get_mut(&precision) {
            if history.len() > 100 {
                history.pop_front();
            }
        }
    }

    /// Optimize precision mode based on trade-offs
    pub fn optimize_precision(&mut self) -> PrecisionMode {
        self.update_pareto_frontier();
        match self.optimization_params.objective {
            OptimizationObjective::MaxPerformance => self
                .pareto_frontier
                .iter()
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(_a, _b, mode)| *mode)
                .unwrap_or(PrecisionMode::Mixed16),
            OptimizationObjective::MaxAccuracy => self
                .pareto_frontier
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(_a, _b, mode)| *mode)
                .unwrap_or(PrecisionMode::Full32),
            OptimizationObjective::Balanced => {
                let mut best_score = f64::NEG_INFINITY;
                let mut best_mode = PrecisionMode::Mixed16;
                let performance_weight = self.optimization_params.performance_weight;
                let accuracy_weight = self.optimization_params.accuracy_weight;
                for &(perf, acc, mode) in &self.pareto_frontier {
                    let perf_score = 1.0 / (perf + 1e-9);
                    let score = performance_weight * perf_score + accuracy_weight * acc;
                    if score > best_score {
                        best_score = score;
                        best_mode = mode;
                    }
                }
                best_mode
            }
            _ => PrecisionMode::Mixed16,
        }
    }

    /// Update Pareto frontier
    pub(crate) fn update_pareto_frontier(&mut self) {
        self.pareto_frontier.clear();
        for precision in [
            PrecisionMode::Full32,
            PrecisionMode::BrainFloat16,
            PrecisionMode::Mixed16,
            PrecisionMode::Int8Dynamic,
            PrecisionMode::Int4Advanced,
        ] {
            if let (Some(perf_data), Some(acc_data)) = (
                self.performance_data.get(&precision),
                self.accuracy_data.get(&precision),
            ) {
                if !perf_data.is_empty() && !acc_data.is_empty() {
                    let avg_perf = perf_data.iter().map(|d| d.as_secs_f64()).sum::<f64>()
                        / perf_data.len() as f64;
                    let avg_acc = acc_data.iter().sum::<f64>() / acc_data.len() as f64;
                    self.pareto_frontier.push((avg_perf, avg_acc, precision));
                }
            }
        }
    }

    /// Compute weighted score for balanced optimization
    #[allow(dead_code)]
    pub(crate) fn compute_weighted_score(&mut self, performance: f64, accuracy: f64) -> f64 {
        let perf_score = 1.0 / (performance + 1e-9);
        self.optimization_params.performance_weight * perf_score
            + self.optimization_params.accuracy_weight * accuracy
    }
}

/// Real-time numerical stability monitor
#[allow(dead_code)]
#[derive(Debug)]
pub struct NumericalStabilityMonitor {
    /// Current stability metrics
    pub(crate) current_metrics: StabilityMetrics,
    /// Historical stability data
    pub(crate) stability_history: VecDeque<StabilityMetrics>,
    /// Dynamic precision configuration
    precision_config: DynamicPrecisionConfig,
    /// Current precision mode
    pub(crate) current_precision: PrecisionMode,
    /// Precision change history
    precision_history: VecDeque<(Instant, PrecisionMode, f64)>,
    /// Error recovery attempts
    #[allow(dead_code)]
    pub(crate) recovery_attempts: usize,
    /// Maximum history length
    max_history_length: usize,
    /// Last precision change time
    last_precision_change: Option<Instant>,
}

impl NumericalStabilityMonitor {
    /// Create new stability monitor
    pub fn new(config: DynamicPrecisionConfig) -> Self {
        Self {
            current_metrics: StabilityMetrics::new(),
            stability_history: VecDeque::new(),
            precision_config: config,
            current_precision: PrecisionMode::Mixed16,
            precision_history: VecDeque::new(),
            recovery_attempts: 0,
            max_history_length: 1000,
            last_precision_change: None,
        }
    }

    /// Monitor stability during computation
    pub fn monitor_stability(
        &mut self,
        data: &Array2<f64>,
        computation_result: &Array2<f64>,
    ) -> SpatialResult<()> {
        self.current_metrics.condition_number =
            NumericalStabilityMonitor::estimate_condition_number(data);
        self.current_metrics.relative_error =
            self.estimate_relative_error(data, computation_result);
        self.current_metrics.forward_error = self.estimate_forward_error(data, computation_result);
        self.current_metrics.backward_error =
            self.estimate_backward_error(data, computation_result);
        self.current_metrics.digit_loss = self.estimate_digit_loss();
        self.current_metrics.update_stability_level();
        self.current_metrics.detect_errors(computation_result);
        self.current_metrics.timestamp = Instant::now();
        self.stability_history
            .push_back(self.current_metrics.clone());
        if self.stability_history.len() > self.max_history_length {
            self.stability_history.pop_front();
        }
        Ok(())
    }

    /// Dynamically adjust precision based on stability
    pub fn adjust_precision(&mut self) -> SpatialResult<PrecisionMode> {
        if let Some(last_change) = self.last_precision_change {
            if last_change.elapsed() < self.precision_config.change_cooldown {
                return Ok(self.current_precision);
            }
        }
        let new_precision = match self.current_metrics.stability_level {
            StabilityLevel::Critical => self.precision_config.max_precision,
            StabilityLevel::Poor => {
                NumericalStabilityMonitor::increase_precision(self.current_precision)
            }
            StabilityLevel::Moderate => {
                if self.current_metrics.relative_error
                    > self.precision_config.stability_threshold_up
                {
                    NumericalStabilityMonitor::increase_precision(self.current_precision)
                } else {
                    self.current_precision
                }
            }
            StabilityLevel::Good => {
                if self.current_metrics.relative_error
                    < self.precision_config.stability_threshold_down
                {
                    NumericalStabilityMonitor::decrease_precision(self.current_precision)
                } else {
                    self.current_precision
                }
            }
            StabilityLevel::Excellent => {
                if self.precision_config.strategy == ScalingStrategy::Aggressive {
                    self.precision_config.min_precision
                } else {
                    NumericalStabilityMonitor::decrease_precision(self.current_precision)
                }
            }
        };
        if new_precision != self.current_precision {
            self.precision_history.push_back((
                Instant::now(),
                new_precision,
                self.current_metrics.relative_error,
            ));
            self.current_precision = new_precision;
            self.last_precision_change = Some(Instant::now());
        }
        Ok(new_precision)
    }

    /// Increase precision mode
    pub(crate) fn increase_precision(current: PrecisionMode) -> PrecisionMode {
        match current {
            PrecisionMode::Int4Advanced => PrecisionMode::Int8Dynamic,
            PrecisionMode::Int8Dynamic => PrecisionMode::Mixed16,
            PrecisionMode::Mixed16 => PrecisionMode::BrainFloat16,
            PrecisionMode::BrainFloat16 => PrecisionMode::Full32,
            PrecisionMode::Full32 => PrecisionMode::Full32,
            _ => PrecisionMode::Mixed16,
        }
    }

    /// Decrease precision mode
    pub(crate) fn decrease_precision(current: PrecisionMode) -> PrecisionMode {
        match current {
            PrecisionMode::Full32 => PrecisionMode::BrainFloat16,
            PrecisionMode::BrainFloat16 => PrecisionMode::Mixed16,
            PrecisionMode::Mixed16 => PrecisionMode::Int8Dynamic,
            PrecisionMode::Int8Dynamic => PrecisionMode::Int4Advanced,
            PrecisionMode::Int4Advanced => PrecisionMode::Int4Advanced,
            _ => PrecisionMode::Mixed16,
        }
    }

    /// Estimate condition number
    pub(crate) fn estimate_condition_number(data: &Array2<f64>) -> f64 {
        let max_val = data.fold(0.0f64, |acc, &x| acc.max(x.abs()));
        let min_val = data.fold(f64::INFINITY, |acc, &x| {
            if x.abs() > 1e-15 {
                acc.min(x.abs())
            } else {
                acc
            }
        });
        if min_val.is_finite() && min_val > 0.0 {
            max_val / min_val
        } else {
            1e12
        }
    }

    /// Estimate relative error
    fn estimate_relative_error(&mut self, _input: &Array2<f64>, output: &Array2<f64>) -> f64 {
        let mean_val = output.mean_or(0.0);
        if mean_val.abs() > 1e-15 {
            let machine_eps = match self.current_precision {
                PrecisionMode::Full32 => 2.22e-16,
                PrecisionMode::Mixed16 | PrecisionMode::BrainFloat16 => 9.77e-4,
                PrecisionMode::Int8Dynamic => 1.0 / 256.0,
                PrecisionMode::Int4Advanced => 1.0 / 16.0,
                _ => 1e-6,
            };
            machine_eps * self.current_metrics.condition_number
        } else {
            0.0
        }
    }

    /// Estimate forward error
    fn estimate_forward_error(&mut self, _input: &Array2<f64>, _output: &Array2<f64>) -> f64 {
        self.current_metrics.relative_error * self.current_metrics.condition_number
    }

    /// Estimate backward error
    fn estimate_backward_error(&mut self, _input: &Array2<f64>, _output: &Array2<f64>) -> f64 {
        self.current_metrics.relative_error
    }

    /// Estimate digit loss
    fn estimate_digit_loss(&self) -> f64 {
        if self.current_metrics.condition_number > 1.0 {
            self.current_metrics.condition_number.log10().max(0.0)
        } else {
            0.0
        }
    }
}
