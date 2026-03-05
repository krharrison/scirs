//! Hyperparameter Tuner for Neural Network Training
//!
//! Provides automated hyperparameter optimization with multiple strategies:
//!
//! - **Grid Search**: exhaustive search over a parameter grid
//! - **Random Search**: sample random configurations
//! - **Successive Halving** (Hyperband-style): allocate more resources to promising configs
//! - **Tree-structured Parzen Estimator (TPE)**: Bayesian optimization with density models
//! - **Early termination**: stop bad trials early based on intermediate results
//!
//! # Example
//!
//! ```rust
//! use scirs2_neural::training::hparam_tuner::{
//!     HParamSpace, HParamValue, SearchStrategy, HParamTuner, TrialResult,
//! };
//! use std::collections::HashMap;
//!
//! // Define the search space
//! let space = vec![
//!     HParamSpace::log_uniform("learning_rate", 1e-5, 1e-1),
//!     HParamSpace::choice("batch_size", vec![
//!         HParamValue::Int(16), HParamValue::Int(32), HParamValue::Int(64),
//!     ]),
//! ];
//!
//! let mut tuner = HParamTuner::new(
//!     space,
//!     SearchStrategy::Random { max_trials: 10 },
//!     42,
//! );
//!
//! // Get the first trial configuration
//! let config = tuner.suggest().expect("should suggest");
//! assert!(config.contains_key("learning_rate"));
//! assert!(config.contains_key("batch_size"));
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::random::rngs::SmallRng;
use scirs2_core::random::{Rng, SeedableRng};
use std::collections::HashMap;
use std::fmt::{self, Debug, Display};

// ============================================================================
// Hyperparameter Value
// ============================================================================

/// A concrete hyperparameter value.
#[derive(Debug, Clone)]
pub enum HParamValue {
    /// Floating point value.
    Float(f64),
    /// Integer value.
    Int(i64),
    /// String/categorical value.
    Str(String),
    /// Boolean value.
    Bool(bool),
}

impl HParamValue {
    /// Try to extract as f64.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Try to extract as i64.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Try to extract as string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(v) => Some(v.as_str()),
            _ => None,
        }
    }

    /// Try to extract as bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

impl Display for HParamValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Float(v) => write!(f, "{v:.6}"),
            Self::Int(v) => write!(f, "{v}"),
            Self::Str(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
        }
    }
}

// ============================================================================
// Hyperparameter Space
// ============================================================================

/// Definition of a single hyperparameter's search space.
#[derive(Debug, Clone)]
pub struct HParamSpace {
    /// Name of the hyperparameter.
    pub name: String,
    /// The type of search space.
    pub space_type: SpaceType,
}

/// Type of search space for a hyperparameter.
#[derive(Debug, Clone)]
pub enum SpaceType {
    /// Uniform float in [low, high].
    Uniform { low: f64, high: f64 },
    /// Log-uniform float in [low, high] (sampled uniformly in log space).
    LogUniform { low: f64, high: f64 },
    /// Integer in [low, high] (inclusive).
    IntRange { low: i64, high: i64 },
    /// Categorical choice from a list of values.
    Choice { values: Vec<HParamValue> },
    /// Boolean.
    Boolean,
}

impl HParamSpace {
    /// Create a uniform float space.
    pub fn uniform(name: &str, low: f64, high: f64) -> Self {
        Self {
            name: name.to_string(),
            space_type: SpaceType::Uniform { low, high },
        }
    }

    /// Create a log-uniform float space.
    pub fn log_uniform(name: &str, low: f64, high: f64) -> Self {
        Self {
            name: name.to_string(),
            space_type: SpaceType::LogUniform { low, high },
        }
    }

    /// Create an integer range space.
    pub fn int_range(name: &str, low: i64, high: i64) -> Self {
        Self {
            name: name.to_string(),
            space_type: SpaceType::IntRange { low, high },
        }
    }

    /// Create a categorical choice space.
    pub fn choice(name: &str, values: Vec<HParamValue>) -> Self {
        Self {
            name: name.to_string(),
            space_type: SpaceType::Choice { values },
        }
    }

    /// Create a boolean space.
    pub fn boolean(name: &str) -> Self {
        Self {
            name: name.to_string(),
            space_type: SpaceType::Boolean,
        }
    }

    /// Sample a random value from this space.
    fn sample(&self, rng: &mut SmallRng) -> HParamValue {
        match &self.space_type {
            SpaceType::Uniform { low, high } => HParamValue::Float(rng.random_range(*low..*high)),
            SpaceType::LogUniform { low, high } => {
                let log_low = low.ln();
                let log_high = high.ln();
                let log_val = rng.random_range(log_low..log_high);
                HParamValue::Float(log_val.exp())
            }
            SpaceType::IntRange { low, high } => HParamValue::Int(rng.random_range(*low..=*high)),
            SpaceType::Choice { values } => {
                if values.is_empty() {
                    HParamValue::Str("empty".to_string())
                } else {
                    let idx = rng.random_range(0..values.len());
                    values[idx].clone()
                }
            }
            SpaceType::Boolean => HParamValue::Bool(rng.random_bool(0.5)),
        }
    }

    /// Get the number of distinct values (for grid search).
    /// Returns None for continuous spaces.
    fn grid_values(&self, num_points: usize) -> Vec<HParamValue> {
        match &self.space_type {
            SpaceType::Uniform { low, high } => {
                if num_points <= 1 {
                    return vec![HParamValue::Float((*low + *high) / 2.0)];
                }
                (0..num_points)
                    .map(|i| {
                        let t = i as f64 / (num_points - 1) as f64;
                        HParamValue::Float(low + (high - low) * t)
                    })
                    .collect()
            }
            SpaceType::LogUniform { low, high } => {
                if num_points <= 1 {
                    return vec![HParamValue::Float((low * high).sqrt())];
                }
                let log_low = low.ln();
                let log_high = high.ln();
                (0..num_points)
                    .map(|i| {
                        let t = i as f64 / (num_points - 1) as f64;
                        HParamValue::Float((log_low + (log_high - log_low) * t).exp())
                    })
                    .collect()
            }
            SpaceType::IntRange { low, high } => {
                let range = (*high - *low + 1) as usize;
                let step = (range as f64 / num_points as f64).max(1.0);
                let mut values = Vec::new();
                let mut v = *low as f64;
                while v <= *high as f64 && values.len() < num_points {
                    values.push(HParamValue::Int(v as i64));
                    v += step;
                }
                values
            }
            SpaceType::Choice { values } => values.clone(),
            SpaceType::Boolean => {
                vec![HParamValue::Bool(false), HParamValue::Bool(true)]
            }
        }
    }
}

// ============================================================================
// Search Strategy
// ============================================================================

/// Strategy for hyperparameter search.
#[derive(Debug, Clone)]
pub enum SearchStrategy {
    /// Exhaustive grid search.
    Grid {
        /// Number of points per continuous dimension.
        points_per_dim: usize,
    },
    /// Random sampling.
    Random {
        /// Maximum number of trials.
        max_trials: usize,
    },
    /// Successive halving (Hyperband-style).
    SuccessiveHalving {
        /// Maximum resource (e.g., epochs) per trial.
        max_resource: usize,
        /// Reduction factor (e.g., 3 = keep top 1/3).
        reduction_factor: usize,
        /// Number of initial configurations.
        num_initial: usize,
    },
    /// Tree-structured Parzen Estimator (basic).
    TPE {
        /// Maximum number of trials.
        max_trials: usize,
        /// Number of initial random trials before using TPE model.
        n_startup: usize,
        /// Quantile for splitting good/bad trials.
        gamma: f64,
    },
}

impl Display for SearchStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Grid { points_per_dim } => write!(f, "Grid({points_per_dim})"),
            Self::Random { max_trials } => write!(f, "Random({max_trials})"),
            Self::SuccessiveHalving {
                max_resource,
                reduction_factor,
                num_initial,
            } => write!(
                f,
                "SuccessiveHalving(R={max_resource}, eta={reduction_factor}, n={num_initial})"
            ),
            Self::TPE {
                max_trials,
                n_startup,
                gamma,
            } => write!(f, "TPE(n={max_trials}, startup={n_startup}, gamma={gamma})"),
        }
    }
}

// ============================================================================
// Trial Result
// ============================================================================

/// Result of a single hyperparameter trial.
#[derive(Debug, Clone)]
pub struct TrialResult {
    /// Trial identifier.
    pub trial_id: usize,
    /// Hyperparameter configuration.
    pub config: HashMap<String, HParamValue>,
    /// Objective metric value (lower is better by default).
    pub objective: f64,
    /// Additional metrics (e.g., accuracy, loss, training time).
    pub metrics: HashMap<String, f64>,
    /// Resource used (e.g., number of epochs actually trained).
    pub resource_used: usize,
    /// Whether the trial was terminated early.
    pub early_terminated: bool,
}

impl Display for TrialResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Trial #{}: obj={:.6}", self.trial_id, self.objective)?;
        if self.early_terminated {
            write!(f, " [early-stopped]")?;
        }
        write!(f, " params={{")?;
        for (i, (k, v)) in self.config.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{k}={v}")?;
        }
        write!(f, "}}")
    }
}

// ============================================================================
// Hyperparameter Tuner
// ============================================================================

/// Hyperparameter tuner that manages trial suggestions and result tracking.
#[derive(Debug)]
pub struct HParamTuner {
    /// The search space.
    space: Vec<HParamSpace>,
    /// The search strategy.
    strategy: SearchStrategy,
    /// RNG for random sampling.
    rng: SmallRng,
    /// Completed trial results.
    results: Vec<TrialResult>,
    /// Next trial ID.
    next_trial_id: usize,
    /// Grid iterator state (for Grid strategy).
    grid_configs: Option<Vec<HashMap<String, HParamValue>>>,
    /// Grid iterator index.
    grid_index: usize,
    /// Whether to minimize (true) or maximize (false) the objective.
    minimize: bool,
    /// Early termination: threshold multiplier above the best known objective.
    /// If a trial's intermediate metric exceeds best * threshold, stop it.
    early_termination_threshold: Option<f64>,
}

impl HParamTuner {
    /// Create a new hyperparameter tuner.
    pub fn new(space: Vec<HParamSpace>, strategy: SearchStrategy, seed: u64) -> Self {
        Self {
            space,
            strategy,
            rng: SmallRng::seed_from_u64(seed),
            results: Vec::new(),
            next_trial_id: 0,
            grid_configs: None,
            grid_index: 0,
            minimize: true,
            early_termination_threshold: None,
        }
    }

    /// Set whether to minimize (default) or maximize the objective.
    pub fn set_minimize(&mut self, minimize: bool) {
        self.minimize = minimize;
    }

    /// Set the early termination threshold.
    ///
    /// Trials with intermediate metric > `best * threshold` will be terminated.
    pub fn set_early_termination_threshold(&mut self, threshold: f64) {
        self.early_termination_threshold = Some(threshold);
    }

    /// Suggest a new hyperparameter configuration.
    ///
    /// Returns `None` if the search is exhausted.
    pub fn suggest(&mut self) -> Option<HashMap<String, HParamValue>> {
        match &self.strategy {
            SearchStrategy::Grid { points_per_dim } => self.suggest_grid(*points_per_dim),
            SearchStrategy::Random { max_trials } => {
                let max = *max_trials;
                self.suggest_random(max)
            }
            SearchStrategy::SuccessiveHalving { num_initial, .. } => {
                let n = *num_initial;
                self.suggest_successive_halving(n)
            }
            SearchStrategy::TPE {
                max_trials,
                n_startup,
                gamma,
            } => {
                let max = *max_trials;
                let startup = *n_startup;
                let g = *gamma;
                self.suggest_tpe(max, startup, g)
            }
        }
    }

    /// Suggest a grid search configuration.
    fn suggest_grid(&mut self, points_per_dim: usize) -> Option<HashMap<String, HParamValue>> {
        // Build grid on first call
        if self.grid_configs.is_none() {
            let mut configs = vec![HashMap::new()];
            for dim in &self.space {
                let values = dim.grid_values(points_per_dim);
                let mut new_configs = Vec::new();
                for config in &configs {
                    for val in &values {
                        let mut c = config.clone();
                        c.insert(dim.name.clone(), val.clone());
                        new_configs.push(c);
                    }
                }
                configs = new_configs;
            }
            self.grid_configs = Some(configs);
            self.grid_index = 0;
        }

        if let Some(configs) = &self.grid_configs {
            if self.grid_index < configs.len() {
                let config = configs[self.grid_index].clone();
                self.grid_index += 1;
                Some(config)
            } else {
                None // grid exhausted
            }
        } else {
            None
        }
    }

    /// Suggest a random configuration.
    fn suggest_random(&mut self, max_trials: usize) -> Option<HashMap<String, HParamValue>> {
        if self.next_trial_id >= max_trials {
            return None;
        }

        let mut config = HashMap::new();
        for dim in &self.space {
            config.insert(dim.name.clone(), dim.sample(&mut self.rng));
        }
        Some(config)
    }

    /// Suggest for successive halving.
    fn suggest_successive_halving(
        &mut self,
        num_initial: usize,
    ) -> Option<HashMap<String, HParamValue>> {
        if self.next_trial_id >= num_initial {
            return None;
        }

        let mut config = HashMap::new();
        for dim in &self.space {
            config.insert(dim.name.clone(), dim.sample(&mut self.rng));
        }
        Some(config)
    }

    /// Suggest using TPE (basic implementation).
    fn suggest_tpe(
        &mut self,
        max_trials: usize,
        n_startup: usize,
        gamma: f64,
    ) -> Option<HashMap<String, HParamValue>> {
        if self.next_trial_id >= max_trials {
            return None;
        }

        // During startup phase, use random search
        if self.results.len() < n_startup {
            return self.suggest_random(max_trials);
        }

        // TPE: split completed trials into good and bad
        let mut sorted_results: Vec<&TrialResult> = self.results.iter().collect();
        sorted_results.sort_by(|a, b| {
            if self.minimize {
                a.objective
                    .partial_cmp(&b.objective)
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                b.objective
                    .partial_cmp(&a.objective)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        let n_good = ((sorted_results.len() as f64 * gamma).ceil() as usize).max(1);
        let good_trials = &sorted_results[..n_good.min(sorted_results.len())];

        // For each dimension, sample from the good trials' distribution
        // (simplified: sample from the good trials' values with perturbation)
        let mut config = HashMap::new();
        for dim in &self.space {
            let good_values: Vec<&HParamValue> = good_trials
                .iter()
                .filter_map(|t| t.config.get(&dim.name))
                .collect();

            if good_values.is_empty() {
                config.insert(dim.name.clone(), dim.sample(&mut self.rng));
                continue;
            }

            let value = match &dim.space_type {
                SpaceType::Uniform { low, high } | SpaceType::LogUniform { low, high } => {
                    // Sample from a kernel density estimate around good values
                    let floats: Vec<f64> =
                        good_values.iter().filter_map(|v| v.as_float()).collect();
                    if floats.is_empty() {
                        dim.sample(&mut self.rng)
                    } else {
                        // Pick a random good value and add Gaussian perturbation
                        let idx = self.rng.random_range(0..floats.len());
                        let center = floats[idx];
                        let bandwidth = (high - low) * 0.1; // 10% of range
                        let perturbation =
                            (self.rng.random_range(0.0_f64..1.0) - 0.5) * bandwidth * 2.0;
                        let val = (center + perturbation).clamp(*low, *high);
                        if matches!(dim.space_type, SpaceType::LogUniform { .. }) {
                            HParamValue::Float(val.max(f64::EPSILON))
                        } else {
                            HParamValue::Float(val)
                        }
                    }
                }
                SpaceType::IntRange { low, high } => {
                    let ints: Vec<i64> = good_values.iter().filter_map(|v| v.as_int()).collect();
                    if ints.is_empty() {
                        dim.sample(&mut self.rng)
                    } else {
                        let idx = self.rng.random_range(0..ints.len());
                        let center = ints[idx];
                        let range = (*high - *low).max(1);
                        let perturbation = self.rng.random_range(-(range / 4)..=(range / 4));
                        HParamValue::Int((center + perturbation).clamp(*low, *high))
                    }
                }
                SpaceType::Choice { .. } | SpaceType::Boolean => {
                    // For categoricals, just pick from good values
                    let idx = self.rng.random_range(0..good_values.len());
                    good_values[idx].clone()
                }
            };
            config.insert(dim.name.clone(), value);
        }

        Some(config)
    }

    /// Report the result of a trial.
    pub fn report(&mut self, config: HashMap<String, HParamValue>, objective: f64) {
        let trial = TrialResult {
            trial_id: self.next_trial_id,
            config,
            objective,
            metrics: HashMap::new(),
            resource_used: 0,
            early_terminated: false,
        };
        self.results.push(trial);
        self.next_trial_id += 1;
    }

    /// Report a trial with additional details.
    pub fn report_trial(&mut self, result: TrialResult) {
        self.results.push(TrialResult {
            trial_id: self.next_trial_id,
            ..result
        });
        self.next_trial_id += 1;
    }

    /// Check if a trial should be early-terminated.
    ///
    /// Returns `true` if the intermediate metric is much worse than the best.
    pub fn should_terminate(&self, intermediate_metric: f64) -> bool {
        if let Some(threshold) = self.early_termination_threshold {
            if let Some(best) = self.best_objective() {
                if self.minimize {
                    return intermediate_metric > best * threshold;
                } else {
                    return intermediate_metric < best / threshold;
                }
            }
        }
        false
    }

    /// Get the best objective value found so far.
    pub fn best_objective(&self) -> Option<f64> {
        if self.results.is_empty() {
            return None;
        }
        if self.minimize {
            self.results.iter().map(|r| r.objective).reduce(f64::min)
        } else {
            self.results.iter().map(|r| r.objective).reduce(f64::max)
        }
    }

    /// Get the best trial result.
    pub fn best_trial(&self) -> Option<&TrialResult> {
        if self.results.is_empty() {
            return None;
        }
        if self.minimize {
            self.results.iter().min_by(|a, b| {
                a.objective
                    .partial_cmp(&b.objective)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        } else {
            self.results.iter().max_by(|a, b| {
                a.objective
                    .partial_cmp(&b.objective)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        }
    }

    /// Get all trial results.
    pub fn results(&self) -> &[TrialResult] {
        &self.results
    }

    /// Get the number of completed trials.
    pub fn num_completed(&self) -> usize {
        self.results.len()
    }

    /// Get the search strategy.
    pub fn strategy(&self) -> &SearchStrategy {
        &self.strategy
    }

    /// Get the search space.
    pub fn space(&self) -> &[HParamSpace] {
        &self.space
    }

    /// Get trials for successive halving at a given resource level.
    ///
    /// Returns the top 1/eta fraction of trials.
    pub fn successive_halving_promote(&self, reduction_factor: usize) -> Vec<&TrialResult> {
        if self.results.is_empty() || reduction_factor == 0 {
            return Vec::new();
        }

        let mut sorted: Vec<&TrialResult> = self.results.iter().collect();
        sorted.sort_by(|a, b| {
            if self.minimize {
                a.objective
                    .partial_cmp(&b.objective)
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                b.objective
                    .partial_cmp(&a.objective)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        let keep = (sorted.len() / reduction_factor).max(1);
        sorted.truncate(keep);
        sorted
    }

    /// Generate a text summary.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Hyperparameter Tuner Summary ===\n");
        out.push_str(&format!("Strategy: {}\n", self.strategy));
        out.push_str(&format!("Completed trials: {}\n", self.results.len()));
        out.push_str(&format!(
            "Objective: {}\n",
            if self.minimize {
                "minimize"
            } else {
                "maximize"
            }
        ));

        if let Some(best) = self.best_trial() {
            out.push_str(&format!("Best trial: {best}\n"));
        }

        // Top 5 results
        if !self.results.is_empty() {
            out.push_str("\nTop 5 results:\n");
            let mut sorted = self.results.clone();
            sorted.sort_by(|a, b| {
                if self.minimize {
                    a.objective
                        .partial_cmp(&b.objective)
                        .unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    b.objective
                        .partial_cmp(&a.objective)
                        .unwrap_or(std::cmp::Ordering::Equal)
                }
            });
            for trial in sorted.iter().take(5) {
                out.push_str(&format!("  {trial}\n"));
            }
        }

        out
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_space() -> Vec<HParamSpace> {
        vec![
            HParamSpace::log_uniform("lr", 1e-5, 1e-1),
            HParamSpace::choice(
                "batch_size",
                vec![
                    HParamValue::Int(16),
                    HParamValue::Int(32),
                    HParamValue::Int(64),
                ],
            ),
            HParamSpace::uniform("dropout", 0.0, 0.5),
        ]
    }

    #[test]
    fn test_hparam_value_conversions() {
        let f = HParamValue::Float(3.14);
        assert!((f.as_float().expect("float") - 3.14).abs() < 1e-10);
        assert!(f.as_str().is_none());

        let i = HParamValue::Int(42);
        assert_eq!(i.as_int().expect("int"), 42);
        assert!((i.as_float().expect("float from int") - 42.0).abs() < 1e-10);

        let s = HParamValue::Str("hello".to_string());
        assert_eq!(s.as_str().expect("str"), "hello");
        assert!(s.as_int().is_none());

        let b = HParamValue::Bool(true);
        assert_eq!(b.as_bool().expect("bool"), true);
    }

    #[test]
    fn test_hparam_value_display() {
        assert!(format!("{}", HParamValue::Float(3.14)).contains("3.14"));
        assert_eq!(format!("{}", HParamValue::Int(42)), "42");
        assert_eq!(format!("{}", HParamValue::Str("hi".into())), "hi");
        assert_eq!(format!("{}", HParamValue::Bool(true)), "true");
    }

    #[test]
    fn test_space_sampling() {
        let mut rng = SmallRng::seed_from_u64(42);

        let uniform = HParamSpace::uniform("x", 0.0, 1.0);
        for _ in 0..100 {
            let v = uniform.sample(&mut rng);
            let f = v.as_float().expect("float");
            assert!(f >= 0.0 && f < 1.0);
        }

        let log_uniform = HParamSpace::log_uniform("lr", 1e-5, 1e-1);
        for _ in 0..100 {
            let v = log_uniform.sample(&mut rng);
            let f = v.as_float().expect("float");
            assert!(f >= 1e-5 && f <= 1e-1);
        }

        let int_range = HParamSpace::int_range("n", 1, 10);
        for _ in 0..100 {
            let v = int_range.sample(&mut rng);
            let i = v.as_int().expect("int");
            assert!(i >= 1 && i <= 10);
        }

        let choice = HParamSpace::choice(
            "opt",
            vec![
                HParamValue::Str("adam".into()),
                HParamValue::Str("sgd".into()),
            ],
        );
        for _ in 0..100 {
            let v = choice.sample(&mut rng);
            let s = v.as_str().expect("str");
            assert!(s == "adam" || s == "sgd");
        }

        let boolean = HParamSpace::boolean("flag");
        let mut seen_true = false;
        let mut seen_false = false;
        for _ in 0..100 {
            let v = boolean.sample(&mut rng);
            if v.as_bool().expect("bool") {
                seen_true = true;
            } else {
                seen_false = true;
            }
        }
        assert!(seen_true && seen_false);
    }

    #[test]
    fn test_grid_values() {
        let uniform = HParamSpace::uniform("x", 0.0, 1.0);
        let vals = uniform.grid_values(5);
        assert_eq!(vals.len(), 5);
        assert!((vals[0].as_float().expect("f") - 0.0).abs() < 1e-10);
        assert!((vals[4].as_float().expect("f") - 1.0).abs() < 1e-10);

        let log_uniform = HParamSpace::log_uniform("lr", 1e-4, 1e-1);
        let vals = log_uniform.grid_values(3);
        assert_eq!(vals.len(), 3);

        let choice = HParamSpace::choice(
            "opt",
            vec![HParamValue::Str("a".into()), HParamValue::Str("b".into())],
        );
        let vals = choice.grid_values(10); // ignores num_points, returns all choices
        assert_eq!(vals.len(), 2);

        let boolean = HParamSpace::boolean("flag");
        let vals = boolean.grid_values(10);
        assert_eq!(vals.len(), 2);
    }

    #[test]
    fn test_random_search() {
        let space = test_space();
        let mut tuner = HParamTuner::new(space, SearchStrategy::Random { max_trials: 5 }, 42);

        for i in 0..5 {
            let config = tuner.suggest().expect("should suggest");
            assert!(config.contains_key("lr"));
            assert!(config.contains_key("batch_size"));
            assert!(config.contains_key("dropout"));
            tuner.report(config, 1.0 - i as f64 * 0.1);
        }

        assert_eq!(tuner.num_completed(), 5);
        assert!(tuner.suggest().is_none()); // exhausted
    }

    #[test]
    fn test_grid_search() {
        let space = vec![
            HParamSpace::choice("a", vec![HParamValue::Int(1), HParamValue::Int(2)]),
            HParamSpace::choice(
                "b",
                vec![HParamValue::Str("x".into()), HParamValue::Str("y".into())],
            ),
        ];

        let mut tuner = HParamTuner::new(space, SearchStrategy::Grid { points_per_dim: 3 }, 42);

        let mut configs = Vec::new();
        while let Some(config) = tuner.suggest() {
            configs.push(config.clone());
            tuner.report(config, 0.5);
        }

        // 2 * 2 = 4 grid configurations
        assert_eq!(configs.len(), 4);
    }

    #[test]
    fn test_best_trial() {
        let space = test_space();
        let mut tuner = HParamTuner::new(space, SearchStrategy::Random { max_trials: 10 }, 42);

        for i in 0..5 {
            let config = tuner.suggest().expect("should suggest");
            tuner.report(config, (5 - i) as f64);
        }

        let best = tuner.best_trial().expect("should have best");
        assert!((best.objective - 1.0).abs() < 1e-10);
        assert!((tuner.best_objective().expect("obj") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_best_trial_maximize() {
        let space = test_space();
        let mut tuner = HParamTuner::new(space, SearchStrategy::Random { max_trials: 10 }, 42);
        tuner.set_minimize(false);

        for i in 0..5 {
            let config = tuner.suggest().expect("should suggest");
            tuner.report(config, i as f64);
        }

        let best = tuner.best_trial().expect("should have best");
        assert!((best.objective - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_early_termination() {
        let space = test_space();
        let mut tuner = HParamTuner::new(space, SearchStrategy::Random { max_trials: 10 }, 42);
        tuner.set_early_termination_threshold(2.0);

        // Report a good trial
        let config = tuner.suggest().expect("should suggest");
        tuner.report(config, 1.0);

        // Check if a bad intermediate should be terminated
        assert!(tuner.should_terminate(3.0)); // 3.0 > 1.0 * 2.0
        assert!(!tuner.should_terminate(1.5)); // 1.5 < 1.0 * 2.0
    }

    #[test]
    fn test_successive_halving_promote() {
        let space = test_space();
        let mut tuner = HParamTuner::new(
            space,
            SearchStrategy::SuccessiveHalving {
                max_resource: 81,
                reduction_factor: 3,
                num_initial: 9,
            },
            42,
        );

        for i in 0..9 {
            let config = tuner.suggest().expect("should suggest");
            tuner.report(config, (10 - i) as f64);
        }

        assert_eq!(tuner.num_completed(), 9);

        // Promote top 1/3
        let promoted = tuner.successive_halving_promote(3);
        assert_eq!(promoted.len(), 3); // 9/3 = 3

        // Should be the best 3
        assert!(promoted[0].objective <= promoted[1].objective);
    }

    #[test]
    fn test_tpe_search() {
        let space = vec![
            HParamSpace::uniform("x", 0.0, 10.0),
            HParamSpace::int_range("n", 1, 5),
        ];

        let mut tuner = HParamTuner::new(
            space,
            SearchStrategy::TPE {
                max_trials: 20,
                n_startup: 5,
                gamma: 0.25,
            },
            42,
        );

        for _ in 0..20 {
            let config = match tuner.suggest() {
                Some(c) => c,
                None => break,
            };
            let x = config["x"].as_float().unwrap_or(5.0);
            let obj = (x - 3.0).powi(2); // minimum at x=3
            tuner.report(config, obj);
        }

        let best = tuner.best_trial().expect("should have best");
        // TPE should find something reasonably close to x=3
        let best_x = best.config["x"].as_float().unwrap_or(0.0);
        assert!(
            (best_x - 3.0).abs() < 5.0,
            "TPE best x={best_x}, expected near 3.0"
        );
    }

    #[test]
    fn test_report_trial() {
        let space = test_space();
        let mut tuner = HParamTuner::new(space, SearchStrategy::Random { max_trials: 10 }, 42);

        let config = tuner.suggest().expect("should suggest");
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.95);

        tuner.report_trial(TrialResult {
            trial_id: 0,
            config,
            objective: 0.5,
            metrics,
            resource_used: 10,
            early_terminated: false,
        });

        assert_eq!(tuner.num_completed(), 1);
        let result = &tuner.results()[0];
        assert_eq!(result.resource_used, 10);
        assert!((result.metrics["accuracy"] - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_summary_generation() {
        let space = test_space();
        let mut tuner = HParamTuner::new(space, SearchStrategy::Random { max_trials: 5 }, 42);

        for i in 0..3 {
            let config = tuner.suggest().expect("should suggest");
            tuner.report(config, i as f64);
        }

        let summary = tuner.summary();
        assert!(summary.contains("Hyperparameter Tuner Summary"));
        assert!(summary.contains("Random"));
        assert!(summary.contains("3")); // completed trials
    }

    #[test]
    fn test_strategy_display() {
        assert_eq!(
            format!("{}", SearchStrategy::Grid { points_per_dim: 5 }),
            "Grid(5)"
        );
        assert_eq!(
            format!("{}", SearchStrategy::Random { max_trials: 10 }),
            "Random(10)"
        );
        assert!(format!(
            "{}",
            SearchStrategy::SuccessiveHalving {
                max_resource: 81,
                reduction_factor: 3,
                num_initial: 9,
            }
        )
        .contains("SuccessiveHalving"));
    }

    #[test]
    fn test_trial_result_display() {
        let mut config = HashMap::new();
        config.insert("lr".to_string(), HParamValue::Float(0.001));
        let trial = TrialResult {
            trial_id: 0,
            config,
            objective: 0.5,
            metrics: HashMap::new(),
            resource_used: 10,
            early_terminated: false,
        };
        let s = format!("{trial}");
        assert!(s.contains("Trial #0"));
        assert!(s.contains("0.5"));
        assert!(s.contains("lr="));
    }

    #[test]
    fn test_trial_result_display_early_terminated() {
        let trial = TrialResult {
            trial_id: 1,
            config: HashMap::new(),
            objective: 1.0,
            metrics: HashMap::new(),
            resource_used: 5,
            early_terminated: true,
        };
        let s = format!("{trial}");
        assert!(s.contains("early-stopped"));
    }

    #[test]
    fn test_empty_tuner_best() {
        let tuner = HParamTuner::new(vec![], SearchStrategy::Random { max_trials: 5 }, 42);
        assert!(tuner.best_trial().is_none());
        assert!(tuner.best_objective().is_none());
    }

    #[test]
    fn test_empty_space_random() {
        let mut tuner = HParamTuner::new(vec![], SearchStrategy::Random { max_trials: 3 }, 42);

        let config = tuner.suggest().expect("should suggest empty config");
        assert!(config.is_empty());
        tuner.report(config, 0.5);
    }

    #[test]
    fn test_grid_single_point() {
        let space = vec![HParamSpace::uniform("x", 0.0, 1.0)];
        let mut tuner = HParamTuner::new(space, SearchStrategy::Grid { points_per_dim: 1 }, 42);

        let config = tuner.suggest().expect("should suggest");
        let x = config["x"].as_float().expect("float");
        assert!((x - 0.5).abs() < 1e-10); // midpoint
    }

    #[test]
    fn test_successive_halving_exhaustion() {
        let space = vec![HParamSpace::uniform("x", 0.0, 1.0)];
        let mut tuner = HParamTuner::new(
            space,
            SearchStrategy::SuccessiveHalving {
                max_resource: 27,
                reduction_factor: 3,
                num_initial: 3,
            },
            42,
        );

        for _ in 0..3 {
            let config = tuner.suggest().expect("should suggest");
            tuner.report(config, 0.5);
        }

        assert!(tuner.suggest().is_none());
    }

    #[test]
    fn test_early_termination_maximize() {
        let space = test_space();
        let mut tuner = HParamTuner::new(space, SearchStrategy::Random { max_trials: 10 }, 42);
        tuner.set_minimize(false);
        tuner.set_early_termination_threshold(2.0);

        let config = tuner.suggest().expect("should suggest");
        tuner.report(config, 10.0); // best = 10

        // For maximize: terminate if intermediate < best / threshold = 10 / 2 = 5
        assert!(tuner.should_terminate(3.0)); // 3 < 5
        assert!(!tuner.should_terminate(7.0)); // 7 > 5
    }

    #[test]
    fn test_no_early_termination_without_threshold() {
        let space = test_space();
        let mut tuner = HParamTuner::new(space, SearchStrategy::Random { max_trials: 10 }, 42);

        let config = tuner.suggest().expect("should suggest");
        tuner.report(config, 1.0);

        assert!(!tuner.should_terminate(100.0));
    }

    #[test]
    fn test_int_range_grid() {
        let space = HParamSpace::int_range("n", 1, 10);
        let vals = space.grid_values(5);
        assert!(!vals.is_empty());
        assert!(vals.len() <= 5);
        // First should be 1
        assert_eq!(vals[0].as_int().expect("int"), 1);
    }
}
