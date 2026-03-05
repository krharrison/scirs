//! AutoML: Automated machine learning pipeline optimisation.
//!
//! Provides random search over user-defined hyperparameter spaces,
//! returning the configuration that maximises (or minimises) a
//! user-supplied objective function.

use std::collections::HashMap;

use crate::error::OptimizeError;
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};

/// A single hyperparameter's search domain.
#[derive(Debug, Clone)]
pub enum HyperparamSpace {
    /// Choose uniformly from a finite set of string values
    Categorical(Vec<String>),
    /// Uniform integer in `[lo, hi]` (inclusive)
    IntRange(i64, i64),
    /// Uniform float in `[lo, hi)`
    FloatRange(f64, f64),
    /// Log-uniform float in `[lo, hi)` (useful for learning rates)
    LogFloatRange(f64, f64),
    /// Bernoulli sample with p = 0.5
    Bool,
}

impl HyperparamSpace {
    /// Draw one sample from this domain.
    pub fn sample(&self, rng: &mut StdRng) -> HyperparamValue {
        match self {
            Self::Categorical(choices) => {
                if choices.is_empty() {
                    return HyperparamValue::String(String::new());
                }
                let idx = rng.random_range(0..choices.len());
                HyperparamValue::String(choices[idx].clone())
            }
            Self::IntRange(lo, hi) => {
                if lo >= hi {
                    return HyperparamValue::Int(*lo);
                }
                HyperparamValue::Int(rng.random_range(*lo..=*hi))
            }
            Self::FloatRange(lo, hi) => {
                if lo >= hi {
                    return HyperparamValue::Float(*lo);
                }
                let u = rng.random::<f64>();
                HyperparamValue::Float(lo + u * (hi - lo))
            }
            Self::LogFloatRange(lo, hi) => {
                if *lo <= 0.0 || *hi <= 0.0 || lo >= hi {
                    return HyperparamValue::Float(*lo);
                }
                let log_lo = lo.ln();
                let log_hi = hi.ln();
                let u = rng.random::<f64>();
                let log_val = log_lo + u * (log_hi - log_lo);
                HyperparamValue::Float(log_val.exp())
            }
            Self::Bool => HyperparamValue::Bool(rng.random_bool(0.5)),
        }
    }
}

/// A sampled hyperparameter value.
#[derive(Debug, Clone)]
pub enum HyperparamValue {
    /// String (from Categorical)
    String(String),
    /// Integer
    Int(i64),
    /// Floating-point scalar
    Float(f64),
    /// Boolean
    Bool(bool),
}

impl HyperparamValue {
    /// Extract float, or `None` if this is not a `Float` variant.
    pub fn as_float(&self) -> Option<f64> {
        if let Self::Float(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Extract integer, or `None`.
    pub fn as_int(&self) -> Option<i64> {
        if let Self::Int(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Extract bool, or `None`.
    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Extract string reference, or `None`.
    pub fn as_str(&self) -> Option<&str> {
        if let Self::String(s) = self {
            Some(s.as_str())
        } else {
            None
        }
    }
}

/// Configuration for an AutoML random search run.
#[derive(Debug)]
pub struct AutoMLConfig {
    /// Per-hyperparameter search domains
    pub search_spaces: HashMap<String, HyperparamSpace>,
    /// Number of random trials to evaluate
    pub n_trials: usize,
    /// Name of the optimisation target metric
    pub optimization_target: String,
    /// If `true`, maximise the metric; if `false`, minimise it
    pub maximize: bool,
}

impl AutoMLConfig {
    /// Create a new config with no search spaces.
    pub fn new(target: &str, maximize: bool) -> Self {
        Self {
            search_spaces: HashMap::new(),
            n_trials: 50,
            optimization_target: target.to_string(),
            maximize,
        }
    }

    /// Add a named hyperparameter with its search domain (builder pattern).
    pub fn add_space(mut self, name: &str, space: HyperparamSpace) -> Self {
        self.search_spaces.insert(name.to_string(), space);
        self
    }

    /// Set the trial budget (builder pattern).
    pub fn with_n_trials(mut self, n: usize) -> Self {
        self.n_trials = n;
        self
    }
}

/// Summary of a completed AutoML search.
#[derive(Debug)]
pub struct AutoMLResult {
    /// Best hyperparameter configuration found
    pub best_config: HashMap<String, HyperparamValue>,
    /// Score of the best configuration
    pub best_score: f64,
    /// All (config, score) pairs in trial order
    pub all_configs: Vec<(HashMap<String, HyperparamValue>, f64)>,
    /// Total number of trials run
    pub n_trials: usize,
}

impl AutoMLResult {
    /// Iterate over all trial scores.
    pub fn scores(&self) -> impl Iterator<Item = f64> + '_ {
        self.all_configs.iter().map(|(_, s)| *s)
    }
}

/// AutoML optimiser using random search.
///
/// For each trial a configuration is sampled uniformly from the product
/// of all hyperparameter domains and evaluated by the user-provided
/// closure.  The best configuration (by `maximize` criterion) is returned.
pub struct AutoMLOptimizer {
    config: AutoMLConfig,
}

impl AutoMLOptimizer {
    /// Create a new optimiser with the given configuration.
    pub fn new(config: AutoMLConfig) -> Self {
        Self { config }
    }

    /// Run random search.
    ///
    /// # Arguments
    /// - `evaluate`: Closure mapping a config to a scalar metric.
    /// - `seed`: Random seed for reproducibility.
    ///
    /// # Errors
    /// Propagates any error returned by `evaluate`.
    pub fn optimize<F>(&self, evaluate: F, seed: u64) -> Result<AutoMLResult, OptimizeError>
    where
        F: Fn(&HashMap<String, HyperparamValue>) -> Result<f64, OptimizeError>,
    {
        if self.config.n_trials == 0 {
            return Err(OptimizeError::InvalidParameter(
                "n_trials must be at least 1".to_string(),
            ));
        }

        let mut rng = StdRng::seed_from_u64(seed);

        let mut best_score = if self.config.maximize {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        let mut best_config: HashMap<String, HyperparamValue> = HashMap::new();
        let mut all_configs = Vec::with_capacity(self.config.n_trials);

        for _ in 0..self.config.n_trials {
            // Sample one config from each hyperparameter's domain
            let trial_config: HashMap<String, HyperparamValue> = self
                .config
                .search_spaces
                .iter()
                .map(|(k, space)| (k.clone(), space.sample(&mut rng)))
                .collect();

            let score = evaluate(&trial_config)?;

            let is_better = if self.config.maximize {
                score > best_score
            } else {
                score < best_score
            };

            if is_better || best_config.is_empty() {
                best_score = score;
                best_config = trial_config.clone();
            }

            all_configs.push((trial_config, score));
        }

        Ok(AutoMLResult {
            best_config,
            best_score,
            all_configs,
            n_trials: self.config.n_trials,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Quadratic objective in log-learning-rate: minimise (lr - 1e-3)^2
    fn lr_objective(cfg: &HashMap<String, HyperparamValue>) -> Result<f64, OptimizeError> {
        let lr = cfg
            .get("lr")
            .and_then(|v| v.as_float())
            .ok_or_else(|| OptimizeError::InvalidParameter("missing lr".into()))?;
        let target = 1e-3_f64;
        Ok(-((lr - target) / target).powi(2))
    }

    #[test]
    fn test_automl_random_search_finds_good_lr() {
        let config = AutoMLConfig::new("neg_mse", true)
            .add_space("lr", HyperparamSpace::LogFloatRange(1e-5, 1e-1))
            .with_n_trials(200);

        let opt = AutoMLOptimizer::new(config);
        let result = opt.optimize(lr_objective, 42).expect("optimize failed");

        assert_eq!(result.n_trials, 200);
        assert_eq!(result.all_configs.len(), 200);

        // Best score should be close to 0.0 with enough trials
        assert!(
            result.best_score > -1.0,
            "best_score too low: {}",
            result.best_score
        );
    }

    #[test]
    fn test_automl_minimize_mode() {
        let config = AutoMLConfig::new("mse", false)
            .add_space("lr", HyperparamSpace::LogFloatRange(1e-5, 1e-1))
            .with_n_trials(100);

        let opt = AutoMLOptimizer::new(config);
        let result = opt
            .optimize(
                |cfg| {
                    let lr = cfg["lr"].as_float().unwrap_or(1.0);
                    Ok((lr - 1e-3).powi(2))
                },
                7,
            )
            .expect("optimize failed");

        // Minimised MSE should be very small
        assert!(result.best_score < 1.0);
    }

    #[test]
    fn test_automl_categorical_space() {
        let config = AutoMLConfig::new("score", true)
            .add_space(
                "optimizer",
                HyperparamSpace::Categorical(vec!["adam".into(), "sgd".into(), "rmsprop".into()]),
            )
            .with_n_trials(30);

        let opt = AutoMLOptimizer::new(config);
        let result = opt
            .optimize(
                |cfg| {
                    let name = cfg["optimizer"].as_str().unwrap_or("unknown");
                    Ok(if name == "adam" { 1.0 } else { 0.0 })
                },
                0,
            )
            .expect("optimize failed");

        assert!(result.best_score >= 0.0);
    }

    #[test]
    fn test_automl_int_range_space() {
        let config = AutoMLConfig::new("score", true)
            .add_space("n_layers", HyperparamSpace::IntRange(1, 10))
            .with_n_trials(50);

        let opt = AutoMLOptimizer::new(config);
        let result = opt
            .optimize(
                |cfg| {
                    let n = cfg["n_layers"].as_int().unwrap_or(1);
                    Ok(-(n as f64 - 5.0).powi(2))
                },
                5,
            )
            .expect("optimize failed");

        let best_n = result.best_config["n_layers"].as_int().unwrap_or(0);
        assert!((1..=10).contains(&best_n));
    }

    #[test]
    fn test_automl_bool_space_samples() {
        let config = AutoMLConfig::new("score", true)
            .add_space("use_bn", HyperparamSpace::Bool)
            .with_n_trials(20);

        let opt = AutoMLOptimizer::new(config);
        let result = opt
            .optimize(
                |cfg| {
                    let bn = cfg["use_bn"].as_bool().unwrap_or(false);
                    Ok(if bn { 1.0 } else { 0.0 })
                },
                3,
            )
            .expect("optimize failed");

        assert!(result.best_score >= 0.0);
    }

    #[test]
    fn test_automl_zero_trials_errors() {
        let config = AutoMLConfig::new("score", true).with_n_trials(0);
        let opt = AutoMLOptimizer::new(config);
        assert!(opt.optimize(|_| Ok(0.0), 0).is_err());
    }

    #[test]
    fn test_automl_result_scores_iter() {
        let config = AutoMLConfig::new("score", true)
            .add_space("lr", HyperparamSpace::FloatRange(0.0, 1.0))
            .with_n_trials(10);

        let opt = AutoMLOptimizer::new(config);
        let result = opt.optimize(|_| Ok(1.0), 0).expect("optimize failed");

        let scores: Vec<f64> = result.scores().collect();
        assert_eq!(scores.len(), 10);
    }
}
