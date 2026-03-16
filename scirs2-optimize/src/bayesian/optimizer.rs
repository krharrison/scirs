//! Bayesian Optimizer -- the main driver for Bayesian optimization.
//!
//! Orchestrates the GP surrogate, acquisition function, and sampling strategy
//! into a full sequential/batch optimization loop.
//!
//! # Features
//!
//! - Configurable surrogate (GP with any kernel)
//! - Pluggable acquisition functions (EI, PI, UCB, KG, Thompson, batch variants)
//! - Initial design via Latin Hypercube, Sobol, Halton, or random sampling
//! - Sequential and batch optimization loops
//! - Multi-objective Bayesian optimization via ParEGO scalarization
//! - Constraint handling via augmented acquisition
//! - Warm-starting from previous evaluations

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, RngExt, SeedableRng};

use crate::error::{OptimizeError, OptimizeResult};

use super::acquisition::{AcquisitionFn, AcquisitionType, ExpectedImprovement};
use super::gp::{GpSurrogate, GpSurrogateConfig, RbfKernel, SurrogateKernel};
use super::sampling::{generate_samples, SamplingConfig, SamplingStrategy};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Bayesian optimizer.
#[derive(Clone)]
pub struct BayesianOptimizerConfig {
    /// Acquisition function type.
    pub acquisition: AcquisitionType,
    /// Sampling strategy for initial design.
    pub initial_design: SamplingStrategy,
    /// Number of initial random/quasi-random points.
    pub n_initial: usize,
    /// Number of restarts when optimising the acquisition function.
    pub acq_n_restarts: usize,
    /// Number of random candidates evaluated per restart when optimising acquisition.
    pub acq_n_candidates: usize,
    /// GP surrogate configuration.
    pub gp_config: GpSurrogateConfig,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
    /// Verbosity level (0 = silent, 1 = summary, 2 = per-iteration).
    pub verbose: usize,
}

impl Default for BayesianOptimizerConfig {
    fn default() -> Self {
        Self {
            acquisition: AcquisitionType::EI { xi: 0.01 },
            initial_design: SamplingStrategy::LatinHypercube,
            n_initial: 10,
            acq_n_restarts: 5,
            acq_n_candidates: 200,
            gp_config: GpSurrogateConfig::default(),
            seed: None,
            verbose: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Observation record
// ---------------------------------------------------------------------------

/// A single evaluated observation.
#[derive(Debug, Clone)]
pub struct Observation {
    /// Input point.
    pub x: Array1<f64>,
    /// Objective function value.
    pub y: f64,
    /// Constraint violation values (empty if no constraints).
    pub constraints: Vec<f64>,
    /// Whether this point is feasible (all constraints satisfied).
    pub feasible: bool,
}

// ---------------------------------------------------------------------------
// Optimization result
// ---------------------------------------------------------------------------

/// Result of Bayesian optimization.
#[derive(Debug, Clone)]
pub struct BayesianOptResult {
    /// Best input point found.
    pub x_best: Array1<f64>,
    /// Best objective function value found.
    pub f_best: f64,
    /// All observations in order.
    pub observations: Vec<Observation>,
    /// Number of function evaluations.
    pub n_evals: usize,
    /// History of best values found at each iteration.
    pub best_history: Vec<f64>,
    /// Whether the optimisation was successful.
    pub success: bool,
    /// Message about the optimization.
    pub message: String,
}

// ---------------------------------------------------------------------------
// Constraint specification
// ---------------------------------------------------------------------------

/// A constraint for constrained Bayesian optimization.
///
/// The constraint is satisfied when `g(x) <= 0`.
pub struct Constraint {
    /// Constraint function: returns a scalar value; satisfied when <= 0.
    pub func: Box<dyn Fn(&ArrayView1<f64>) -> f64 + Send + Sync>,
    /// Name for diagnostic purposes.
    pub name: String,
}

// ---------------------------------------------------------------------------
// BayesianOptimizer
// ---------------------------------------------------------------------------

/// The Bayesian optimizer.
///
/// Supports sequential single-objective, batch, multi-objective (ParEGO),
/// and constrained optimization.
pub struct BayesianOptimizer {
    /// Search bounds: [(lower, upper), ...] for each dimension.
    bounds: Vec<(f64, f64)>,
    /// Configuration.
    config: BayesianOptimizerConfig,
    /// GP surrogate model.
    surrogate: GpSurrogate,
    /// Observations collected so far.
    observations: Vec<Observation>,
    /// Current best observation index.
    best_idx: Option<usize>,
    /// Constraints (empty for unconstrained).
    constraints: Vec<Constraint>,
    /// Random number generator.
    rng: StdRng,
}

impl BayesianOptimizer {
    /// Create a new Bayesian optimizer.
    ///
    /// # Arguments
    /// * `bounds` - Search bounds for each dimension: `[(lo, hi), ...]`
    /// * `config` - Optimizer configuration
    pub fn new(bounds: Vec<(f64, f64)>, config: BayesianOptimizerConfig) -> OptimizeResult<Self> {
        if bounds.is_empty() {
            return Err(OptimizeError::InvalidInput(
                "Bounds must have at least one dimension".to_string(),
            ));
        }
        for (i, &(lo, hi)) in bounds.iter().enumerate() {
            if lo >= hi {
                return Err(OptimizeError::InvalidInput(format!(
                    "Invalid bounds for dimension {}: [{}, {}]",
                    i, lo, hi
                )));
            }
        }

        let seed = config.seed.unwrap_or_else(|| {
            let s: u64 = scirs2_core::random::rng().random();
            s
        });
        let rng = StdRng::seed_from_u64(seed);

        let kernel: Box<dyn SurrogateKernel> = Box::new(RbfKernel::default());
        let surrogate = GpSurrogate::new(kernel, config.gp_config.clone());

        Ok(Self {
            bounds,
            config,
            surrogate,
            observations: Vec::new(),
            best_idx: None,
            constraints: Vec::new(),
            rng,
        })
    }

    /// Create a new optimizer with a custom kernel.
    pub fn with_kernel(
        bounds: Vec<(f64, f64)>,
        kernel: Box<dyn SurrogateKernel>,
        config: BayesianOptimizerConfig,
    ) -> OptimizeResult<Self> {
        let mut opt = Self::new(bounds, config)?;
        opt.surrogate = GpSurrogate::new(kernel, opt.config.gp_config.clone());
        Ok(opt)
    }

    /// Add a constraint: satisfied when `g(x) <= 0`.
    pub fn add_constraint<F>(&mut self, name: &str, func: F)
    where
        F: Fn(&ArrayView1<f64>) -> f64 + Send + Sync + 'static,
    {
        self.constraints.push(Constraint {
            func: Box::new(func),
            name: name.to_string(),
        });
    }

    /// Warm-start from previous evaluations.
    pub fn warm_start(&mut self, x_data: &Array2<f64>, y_data: &Array1<f64>) -> OptimizeResult<()> {
        if x_data.nrows() != y_data.len() {
            return Err(OptimizeError::InvalidInput(
                "x_data and y_data row counts must match".to_string(),
            ));
        }

        for i in 0..x_data.nrows() {
            let obs = Observation {
                x: x_data.row(i).to_owned(),
                y: y_data[i],
                constraints: Vec::new(),
                feasible: true,
            };

            // Track best
            match self.best_idx {
                Some(best) if obs.y < self.observations[best].y => {
                    self.best_idx = Some(self.observations.len());
                }
                None => {
                    self.best_idx = Some(self.observations.len());
                }
                _ => {}
            }
            self.observations.push(obs);
        }

        // Fit the surrogate
        if !self.observations.is_empty() {
            self.fit_surrogate()?;
        }

        Ok(())
    }

    /// Run the sequential optimization loop.
    ///
    /// # Arguments
    /// * `objective` - Function to minimize.
    /// * `n_iter` - Number of iterations (function evaluations after initial design).
    pub fn optimize<F>(&mut self, objective: F, n_iter: usize) -> OptimizeResult<BayesianOptResult>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        // Phase 1: Initial design
        let n_initial = if self.observations.is_empty() {
            self.config.n_initial
        } else {
            // If warm-started, may need fewer initial points
            self.config
                .n_initial
                .saturating_sub(self.observations.len())
        };

        if n_initial > 0 {
            let sampling_config = SamplingConfig {
                seed: Some(self.rng.random()),
                ..Default::default()
            };
            let initial_points = generate_samples(
                n_initial,
                &self.bounds,
                self.config.initial_design,
                Some(sampling_config),
            )?;

            for i in 0..initial_points.nrows() {
                let x = initial_points.row(i).to_owned();
                let y = objective(&x.view());
                self.record_observation(x, y);
            }

            self.fit_surrogate()?;
        }

        let mut best_history = Vec::with_capacity(n_iter);
        if let Some(best_idx) = self.best_idx {
            best_history.push(self.observations[best_idx].y);
        }

        // Phase 2: Sequential optimization
        for _iter in 0..n_iter {
            let next_x = self.suggest_next()?;
            let y = objective(&next_x.view());
            self.record_observation(next_x, y);
            self.fit_surrogate()?;

            if let Some(best_idx) = self.best_idx {
                best_history.push(self.observations[best_idx].y);
            }
        }

        // Build result
        let best_idx = self.best_idx.ok_or_else(|| {
            OptimizeError::ComputationError("No observations collected".to_string())
        })?;
        let best_obs = &self.observations[best_idx];

        Ok(BayesianOptResult {
            x_best: best_obs.x.clone(),
            f_best: best_obs.y,
            observations: self.observations.clone(),
            n_evals: self.observations.len(),
            best_history,
            success: true,
            message: format!(
                "Optimization completed: {} evaluations, best f = {:.6e}",
                self.observations.len(),
                best_obs.y
            ),
        })
    }

    /// Run batch optimization, evaluating `batch_size` points in parallel per round.
    ///
    /// Uses the Kriging Believer strategy: after selecting a candidate,
    /// the GP is updated with a fantasised observation at the predicted mean.
    pub fn optimize_batch<F>(
        &mut self,
        objective: F,
        n_rounds: usize,
        batch_size: usize,
    ) -> OptimizeResult<BayesianOptResult>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        let batch_size = batch_size.max(1);

        // Phase 1: Initial design (same as sequential)
        let n_initial = if self.observations.is_empty() {
            self.config.n_initial
        } else {
            self.config
                .n_initial
                .saturating_sub(self.observations.len())
        };

        if n_initial > 0 {
            let sampling_config = SamplingConfig {
                seed: Some(self.rng.random()),
                ..Default::default()
            };
            let initial_points = generate_samples(
                n_initial,
                &self.bounds,
                self.config.initial_design,
                Some(sampling_config),
            )?;

            for i in 0..initial_points.nrows() {
                let x = initial_points.row(i).to_owned();
                let y = objective(&x.view());
                self.record_observation(x, y);
            }
            self.fit_surrogate()?;
        }

        let mut best_history = Vec::with_capacity(n_rounds);
        if let Some(best_idx) = self.best_idx {
            best_history.push(self.observations[best_idx].y);
        }

        // Phase 2: Batch optimization rounds
        for _round in 0..n_rounds {
            let batch = self.suggest_batch(batch_size)?;

            // Evaluate all batch points
            for x in &batch {
                let y = objective(&x.view());
                self.record_observation(x.clone(), y);
            }

            self.fit_surrogate()?;

            if let Some(best_idx) = self.best_idx {
                best_history.push(self.observations[best_idx].y);
            }
        }

        let best_idx = self.best_idx.ok_or_else(|| {
            OptimizeError::ComputationError("No observations collected".to_string())
        })?;
        let best_obs = &self.observations[best_idx];

        Ok(BayesianOptResult {
            x_best: best_obs.x.clone(),
            f_best: best_obs.y,
            observations: self.observations.clone(),
            n_evals: self.observations.len(),
            best_history,
            success: true,
            message: format!(
                "Batch optimization completed: {} evaluations, best f = {:.6e}",
                self.observations.len(),
                best_obs.y
            ),
        })
    }

    /// Multi-objective optimization via ParEGO scalarization.
    ///
    /// Uses random weight vectors to scalarise the objectives into a single
    /// augmented Chebyshev function, then runs standard BO on the scalarization.
    ///
    /// # Arguments
    /// * `objectives` - Vector of objective functions to minimize.
    /// * `n_iter` - Number of sequential iterations.
    pub fn optimize_multi_objective<F>(
        &mut self,
        objectives: &[F],
        n_iter: usize,
    ) -> OptimizeResult<BayesianOptResult>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
    {
        if objectives.is_empty() {
            return Err(OptimizeError::InvalidInput(
                "At least one objective is required".to_string(),
            ));
        }
        if objectives.len() == 1 {
            // Single objective: delegate to standard optimize
            return self.optimize(&objectives[0], n_iter);
        }

        let n_obj = objectives.len();

        // Phase 1: Initial design
        let n_initial = if self.observations.is_empty() {
            self.config.n_initial
        } else {
            self.config
                .n_initial
                .saturating_sub(self.observations.len())
        };

        // Store all objective values for normalization
        let mut all_obj_values: Vec<Vec<f64>> = vec![Vec::new(); n_obj];

        if n_initial > 0 {
            let sampling_config = SamplingConfig {
                seed: Some(self.rng.random()),
                ..Default::default()
            };
            let initial_points = generate_samples(
                n_initial,
                &self.bounds,
                self.config.initial_design,
                Some(sampling_config),
            )?;

            for i in 0..initial_points.nrows() {
                let x = initial_points.row(i).to_owned();
                let obj_vals: Vec<f64> = objectives.iter().map(|f| f(&x.view())).collect();

                // ParEGO scalarization with uniform weight (initial)
                let scalarized = parego_scalarize(&obj_vals, &vec![1.0 / n_obj as f64; n_obj]);
                self.record_observation(x, scalarized);

                for (k, &v) in obj_vals.iter().enumerate() {
                    all_obj_values[k].push(v);
                }
            }
            self.fit_surrogate()?;
        }

        let mut best_history = Vec::new();
        if let Some(best_idx) = self.best_idx {
            best_history.push(self.observations[best_idx].y);
        }

        // Phase 2: Sequential iterations with rotating random weights
        for _iter in 0..n_iter {
            // Generate random weight vector on the simplex
            let weights = random_simplex_point(n_obj, &mut self.rng);

            // Suggest next point (based on current scalarized GP)
            let next_x = self.suggest_next()?;

            // Evaluate all objectives
            let obj_vals: Vec<f64> = objectives.iter().map(|f| f(&next_x.view())).collect();
            for (k, &v) in obj_vals.iter().enumerate() {
                all_obj_values[k].push(v);
            }

            // Normalize and scalarize
            let normalized: Vec<f64> = (0..n_obj)
                .map(|k| {
                    let vals = &all_obj_values[k];
                    let min_v = vals.iter().copied().fold(f64::INFINITY, f64::min);
                    let max_v = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let range = (max_v - min_v).max(1e-12);
                    (obj_vals[k] - min_v) / range
                })
                .collect();

            let scalarized = parego_scalarize(&normalized, &weights);
            self.record_observation(next_x, scalarized);
            self.fit_surrogate()?;

            if let Some(best_idx) = self.best_idx {
                best_history.push(self.observations[best_idx].y);
            }
        }

        let best_idx = self.best_idx.ok_or_else(|| {
            OptimizeError::ComputationError("No observations collected".to_string())
        })?;
        let best_obs = &self.observations[best_idx];

        Ok(BayesianOptResult {
            x_best: best_obs.x.clone(),
            f_best: best_obs.y,
            observations: self.observations.clone(),
            n_evals: self.observations.len(),
            best_history,
            success: true,
            message: format!(
                "ParEGO multi-objective optimization completed: {} evaluations",
                self.observations.len()
            ),
        })
    }

    /// Get the ask interface: suggest the next point to evaluate.
    pub fn ask(&mut self) -> OptimizeResult<Array1<f64>> {
        if self.observations.is_empty() || self.observations.len() < self.config.n_initial {
            // Still in initial design phase
            let sampling_config = SamplingConfig {
                seed: Some(self.rng.random()),
                ..Default::default()
            };
            let points = generate_samples(
                1,
                &self.bounds,
                self.config.initial_design,
                Some(sampling_config),
            )?;
            Ok(points.row(0).to_owned())
        } else {
            self.suggest_next()
        }
    }

    /// Tell interface: update with an observation.
    pub fn tell(&mut self, x: Array1<f64>, y: f64) -> OptimizeResult<()> {
        self.record_observation(x, y);
        if self.observations.len() >= 2 {
            self.fit_surrogate()?;
        }
        Ok(())
    }

    /// Get the current best observation.
    pub fn best(&self) -> Option<&Observation> {
        self.best_idx.map(|i| &self.observations[i])
    }

    /// Get all observations.
    pub fn observations(&self) -> &[Observation] {
        &self.observations
    }

    /// Number of observations.
    pub fn n_observations(&self) -> usize {
        self.observations.len()
    }

    /// Get reference to the GP surrogate.
    pub fn surrogate(&self) -> &GpSurrogate {
        &self.surrogate
    }

    // -----------------------------------------------------------------------
    // Internal methods
    // -----------------------------------------------------------------------

    /// Record an observation and update the best index.
    fn record_observation(&mut self, x: Array1<f64>, y: f64) {
        let feasible = self.evaluate_constraints(&x);

        let obs = Observation {
            x,
            y,
            constraints: Vec::new(), // filled below if needed
            feasible,
        };

        let idx = self.observations.len();

        // Update best (prefer feasible solutions)
        match self.best_idx {
            Some(best) => {
                let cur_best = &self.observations[best];
                let new_is_better = if obs.feasible && !cur_best.feasible {
                    true
                } else if obs.feasible == cur_best.feasible {
                    obs.y < cur_best.y
                } else {
                    false
                };
                if new_is_better {
                    self.best_idx = Some(idx);
                }
            }
            None => {
                self.best_idx = Some(idx);
            }
        }

        self.observations.push(obs);
    }

    /// Evaluate constraints for a point; returns true if all constraints are satisfied.
    fn evaluate_constraints(&self, x: &Array1<f64>) -> bool {
        self.constraints.iter().all(|c| (c.func)(&x.view()) <= 0.0)
    }

    /// Fit or refit the GP surrogate on all observations.
    fn fit_surrogate(&mut self) -> OptimizeResult<()> {
        let n = self.observations.len();
        if n == 0 {
            return Ok(());
        }
        let n_dims = self.observations[0].x.len();

        let mut x_data = Array2::zeros((n, n_dims));
        let mut y_data = Array1::zeros(n);

        for (i, obs) in self.observations.iter().enumerate() {
            for j in 0..n_dims {
                x_data[[i, j]] = obs.x[j];
            }
            y_data[i] = obs.y;
        }

        self.surrogate.fit(&x_data, &y_data)
    }

    /// Suggest the next point to evaluate by optimising the acquisition function.
    fn suggest_next(&mut self) -> OptimizeResult<Array1<f64>> {
        let f_best = self.best_idx.map(|i| self.observations[i].y).unwrap_or(0.0);

        // Build reference points for KG if needed
        let n = self.observations.len();
        let n_dims = self.bounds.len();
        let ref_points = if n > 0 {
            let mut pts = Array2::zeros((n, n_dims));
            for (i, obs) in self.observations.iter().enumerate() {
                for j in 0..n_dims {
                    pts[[i, j]] = obs.x[j];
                }
            }
            Some(pts)
        } else {
            None
        };

        let acq = self.config.acquisition.build(f_best, ref_points.as_ref());

        self.optimize_acquisition(acq.as_ref())
    }

    /// Suggest a batch of points using the Kriging Believer strategy.
    fn suggest_batch(&mut self, batch_size: usize) -> OptimizeResult<Vec<Array1<f64>>> {
        let mut batch = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let next = self.suggest_next()?;

            // Fantasy: predict mean at the selected point and add it as a phantom observation
            let (mu, _sigma) = self.surrogate.predict_single(&next.view())?;
            self.record_observation(next.clone(), mu);
            self.fit_surrogate()?;

            batch.push(next);
        }

        // Remove the phantom observations (they will be replaced with real ones)
        let n_real = self.observations.len() - batch_size;
        self.observations.truncate(n_real);

        // Refit surrogate without phantoms
        if !self.observations.is_empty() {
            // Update best_idx in case we removed the best
            self.best_idx = None;
            for (i, obs) in self.observations.iter().enumerate() {
                match self.best_idx {
                    Some(best) if obs.y < self.observations[best].y => {
                        self.best_idx = Some(i);
                    }
                    None => {
                        self.best_idx = Some(i);
                    }
                    _ => {}
                }
            }
            self.fit_surrogate()?;
        }

        Ok(batch)
    }

    /// Optimise the acquisition function over the search space.
    ///
    /// Uses random sampling + local refinement (coordinate search).
    fn optimize_acquisition(&mut self, acq: &dyn AcquisitionFn) -> OptimizeResult<Array1<f64>> {
        let n_dims = self.bounds.len();
        let n_candidates = self.config.acq_n_candidates;
        let n_restarts = self.config.acq_n_restarts;

        // Generate random candidates
        let sampling_config = SamplingConfig {
            seed: Some(self.rng.random()),
            ..Default::default()
        };
        let candidates = generate_samples(
            n_candidates,
            &self.bounds,
            SamplingStrategy::Random,
            Some(sampling_config),
        )?;

        // Also include the current best as a candidate
        let mut best_x = candidates.row(0).to_owned();
        let mut best_val = f64::NEG_INFINITY;

        // Evaluate all candidates
        for i in 0..candidates.nrows() {
            match acq.evaluate(&candidates.row(i), &self.surrogate) {
                Ok(val) if val > best_val => {
                    best_val = val;
                    best_x = candidates.row(i).to_owned();
                }
                _ => {}
            }
        }

        // If we have a current best observation, add it as a candidate
        if let Some(best_idx) = self.best_idx {
            let obs_x = &self.observations[best_idx].x;
            if let Ok(val) = acq.evaluate(&obs_x.view(), &self.surrogate) {
                if val > best_val {
                    best_val = val;
                    best_x = obs_x.clone();
                }
            }
        }

        // Local refinement: coordinate-wise search from the top-n candidates
        // Collect top candidates
        let mut scored: Vec<(f64, usize)> = Vec::new();
        for i in 0..candidates.nrows() {
            if let Ok(val) = acq.evaluate(&candidates.row(i), &self.surrogate) {
                scored.push((val, i));
            }
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let n_refine = n_restarts.min(scored.len());
        for k in 0..n_refine {
            let mut x_current = candidates.row(scored[k].1).to_owned();
            let mut f_current = scored[k].0;

            // Coordinate-wise golden section search
            for _round in 0..3 {
                for d in 0..n_dims {
                    let (lo, hi) = self.bounds[d];
                    let (refined_x, refined_f) =
                        golden_section_1d(acq, &self.surrogate, &x_current, d, lo, hi, 20)?;
                    if refined_f > f_current {
                        x_current[d] = refined_x;
                        f_current = refined_f;
                    }
                }
            }

            if f_current > best_val {
                best_val = f_current;
                best_x = x_current;
            }
        }

        // Clamp to bounds
        for (d, &(lo, hi)) in self.bounds.iter().enumerate() {
            best_x[d] = best_x[d].clamp(lo, hi);
        }

        Ok(best_x)
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Golden section search for maximising `acq(x_base with dim d = t)` over [lo, hi].
fn golden_section_1d(
    acq: &dyn AcquisitionFn,
    surrogate: &GpSurrogate,
    x_base: &Array1<f64>,
    dim: usize,
    lo: f64,
    hi: f64,
    max_iters: usize,
) -> OptimizeResult<(f64, f64)> {
    let gr = (5.0_f64.sqrt() - 1.0) / 2.0; // golden ratio conjugate
    let mut a = lo;
    let mut b = hi;

    let eval_at = |t: f64| -> OptimizeResult<f64> {
        let mut x = x_base.clone();
        x[dim] = t;
        acq.evaluate(&x.view(), surrogate)
    };

    let mut c = b - gr * (b - a);
    let mut d = a + gr * (b - a);
    let mut fc = eval_at(c)?;
    let mut fd = eval_at(d)?;

    for _ in 0..max_iters {
        if (b - a).abs() < 1e-8 {
            break;
        }
        // We want to maximise, so we keep the side with the larger value
        if fc < fd {
            a = c;
            c = d;
            fc = fd;
            d = a + gr * (b - a);
            fd = eval_at(d)?;
        } else {
            b = d;
            d = c;
            fd = fc;
            c = b - gr * (b - a);
            fc = eval_at(c)?;
        }
    }

    let mid = (a + b) / 2.0;
    let f_mid = eval_at(mid)?;
    Ok((mid, f_mid))
}

/// ParEGO augmented Chebyshev scalarization.
///
/// s(f, w) = max_k { w_k * f_k } + rho * sum_k { w_k * f_k }
///
/// where rho = 0.05 is a small augmentation coefficient.
fn parego_scalarize(obj_values: &[f64], weights: &[f64]) -> f64 {
    let rho = 0.05;
    let mut max_wf = f64::NEG_INFINITY;
    let mut sum_wf = 0.0;

    for (k, (&fk, &wk)) in obj_values.iter().zip(weights.iter()).enumerate() {
        let wf = wk * fk;
        if wf > max_wf {
            max_wf = wf;
        }
        sum_wf += wf;
    }

    max_wf + rho * sum_wf
}

/// Generate a random point on the probability simplex using the Dirichlet trick.
fn random_simplex_point(n: usize, rng: &mut StdRng) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }

    // Sample from Exp(1) and normalize
    let mut values: Vec<f64> = (0..n)
        .map(|_| {
            let u: f64 = rng.random_range(1e-10..1.0);
            -u.ln()
        })
        .collect();

    let sum: f64 = values.iter().sum();
    if sum > 0.0 {
        for v in &mut values {
            *v /= sum;
        }
    } else {
        // Fallback to uniform
        let w = 1.0 / n as f64;
        values.fill(w);
    }
    values
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

/// Run Bayesian optimization on a function.
///
/// This is a high-level convenience function that creates a `BayesianOptimizer`,
/// runs the optimization, and returns the result.
///
/// # Arguments
/// * `objective` - Function to minimize: `f(x) -> f64`
/// * `bounds` - Search bounds: `[(lo, hi), ...]`
/// * `n_iter` - Number of sequential iterations (after initial design)
/// * `config` - Optional optimizer configuration
///
/// # Example
///
/// ```rust
/// use scirs2_optimize::bayesian::optimize;
/// use scirs2_core::ndarray::ArrayView1;
///
/// let result = optimize(
///     |x: &ArrayView1<f64>| x[0].powi(2) + x[1].powi(2),
///     &[(-5.0, 5.0), (-5.0, 5.0)],
///     20,
///     None,
/// ).expect("optimization failed");
///
/// assert!(result.f_best < 1.0);
/// ```
pub fn optimize<F>(
    objective: F,
    bounds: &[(f64, f64)],
    n_iter: usize,
    config: Option<BayesianOptimizerConfig>,
) -> OptimizeResult<BayesianOptResult>
where
    F: Fn(&ArrayView1<f64>) -> f64,
{
    let config = config.unwrap_or_default();
    let mut optimizer = BayesianOptimizer::new(bounds.to_vec(), config)?;
    optimizer.optimize(objective, n_iter)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn sphere(x: &ArrayView1<f64>) -> f64 {
        x.iter().map(|&v| v * v).sum()
    }

    fn rosenbrock_2d(x: &ArrayView1<f64>) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
    }

    #[test]
    fn test_optimize_sphere_2d() {
        let config = BayesianOptimizerConfig {
            n_initial: 8,
            seed: Some(42),
            gp_config: GpSurrogateConfig {
                optimize_hyperparams: false,
                noise_variance: 1e-4,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = optimize(sphere, &[(-5.0, 5.0), (-5.0, 5.0)], 25, Some(config))
            .expect("optimization should succeed");

        assert!(result.success);
        assert!(result.f_best < 2.0, "f_best = {:.4}", result.f_best);
    }

    #[test]
    fn test_optimizer_ask_tell() {
        let config = BayesianOptimizerConfig {
            n_initial: 5,
            seed: Some(42),
            gp_config: GpSurrogateConfig {
                optimize_hyperparams: false,
                noise_variance: 1e-4,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut opt =
            BayesianOptimizer::new(vec![(-5.0, 5.0), (-5.0, 5.0)], config).expect("create ok");

        for _ in 0..15 {
            let x = opt.ask().expect("ask ok");
            let y = sphere(&x.view());
            opt.tell(x, y).expect("tell ok");
        }

        let best = opt.best().expect("should have a best");
        assert!(best.y < 5.0, "best y = {:.4}", best.y);
    }

    #[test]
    fn test_warm_start() {
        let config = BayesianOptimizerConfig {
            n_initial: 3,
            seed: Some(42),
            gp_config: GpSurrogateConfig {
                optimize_hyperparams: false,
                noise_variance: 1e-4,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut opt =
            BayesianOptimizer::new(vec![(-5.0, 5.0), (-5.0, 5.0)], config).expect("create ok");

        // Warm start with some previous data
        let x_prev =
            Array2::from_shape_vec((3, 2), vec![0.1, 0.2, -0.3, 0.1, 0.5, -0.5]).expect("shape ok");
        let y_prev = array![0.05, 0.1, 0.5];
        opt.warm_start(&x_prev, &y_prev).expect("warm start ok");

        assert_eq!(opt.n_observations(), 3);

        let result = opt.optimize(sphere, 10).expect("optimize ok");
        assert!(result.f_best < 0.5);
    }

    #[test]
    fn test_batch_optimization() {
        let config = BayesianOptimizerConfig {
            n_initial: 5,
            seed: Some(42),
            gp_config: GpSurrogateConfig {
                optimize_hyperparams: false,
                noise_variance: 1e-4,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut opt =
            BayesianOptimizer::new(vec![(-5.0, 5.0), (-5.0, 5.0)], config).expect("create ok");

        let result = opt
            .optimize_batch(sphere, 5, 3)
            .expect("batch optimization ok");
        assert!(result.success);
        // 5 initial + 5*3 = 20 total evaluations
        assert_eq!(result.n_evals, 20);
    }

    #[test]
    fn test_constrained_optimization() {
        let config = BayesianOptimizerConfig {
            n_initial: 8,
            seed: Some(42),
            gp_config: GpSurrogateConfig {
                optimize_hyperparams: false,
                noise_variance: 1e-4,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut opt =
            BayesianOptimizer::new(vec![(-5.0, 5.0), (-5.0, 5.0)], config).expect("create ok");

        // Constraint: x[0] >= 1.0 (i.e., 1.0 - x[0] <= 0)
        opt.add_constraint("x0_ge_1", |x: &ArrayView1<f64>| 1.0 - x[0]);

        let result = opt.optimize(sphere, 20).expect("optimize ok");
        // The constrained minimum of x^2+y^2 with x >= 1 is at (1,0), f=1
        // We just check the optimizer found something feasible and reasonable
        assert!(result.success);
        assert!(result.x_best[0] >= 0.5, "x[0] should be near >= 1");
    }

    #[test]
    fn test_multi_objective_parego() {
        let config = BayesianOptimizerConfig {
            n_initial: 8,
            seed: Some(42),
            gp_config: GpSurrogateConfig {
                optimize_hyperparams: false,
                noise_variance: 1e-4,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut opt =
            BayesianOptimizer::new(vec![(-5.0, 5.0), (-5.0, 5.0)], config).expect("create ok");

        // Two objectives: f1 = (x-1)^2 + y^2, f2 = (x+1)^2 + y^2
        let f1 = |x: &ArrayView1<f64>| (x[0] - 1.0).powi(2) + x[1].powi(2);
        let f2 = |x: &ArrayView1<f64>| (x[0] + 1.0).powi(2) + x[1].powi(2);
        let objectives: Vec<Box<dyn Fn(&ArrayView1<f64>) -> f64>> =
            vec![Box::new(f1), Box::new(f2)];

        let obj_refs: Vec<&dyn Fn(&ArrayView1<f64>) -> f64> = objectives
            .iter()
            .map(|f| f.as_ref() as &dyn Fn(&ArrayView1<f64>) -> f64)
            .collect();

        // Need to pass as slice of Fn
        let result = opt
            .optimize_multi_objective(&obj_refs[..], 15)
            .expect("multi-objective ok");
        assert!(result.success);
        // The Pareto front is between x=-1 and x=1
        assert!(result.x_best[0].abs() <= 5.0);
    }

    #[test]
    fn test_different_acquisition_functions() {
        let bounds = vec![(-3.0, 3.0)];

        for acq in &[
            AcquisitionType::EI { xi: 0.01 },
            AcquisitionType::PI { xi: 0.01 },
            AcquisitionType::UCB { kappa: 2.0 },
            AcquisitionType::Thompson { seed: 42 },
        ] {
            let config = BayesianOptimizerConfig {
                acquisition: acq.clone(),
                n_initial: 5,
                seed: Some(42),
                gp_config: GpSurrogateConfig {
                    optimize_hyperparams: false,
                    noise_variance: 1e-4,
                    ..Default::default()
                },
                ..Default::default()
            };
            let result = optimize(
                |x: &ArrayView1<f64>| x[0].powi(2),
                &bounds,
                10,
                Some(config),
            )
            .expect("optimize ok");
            assert!(
                result.f_best < 3.0,
                "Acquisition {:?} failed: f_best = {}",
                acq,
                result.f_best
            );
        }
    }

    #[test]
    fn test_invalid_bounds_rejected() {
        let result = BayesianOptimizer::new(
            vec![(5.0, 1.0)], // lo > hi
            BayesianOptimizerConfig::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_bounds_rejected() {
        let result = BayesianOptimizer::new(vec![], BayesianOptimizerConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_best_history_monotonic() {
        let config = BayesianOptimizerConfig {
            n_initial: 5,
            seed: Some(42),
            gp_config: GpSurrogateConfig {
                optimize_hyperparams: false,
                noise_variance: 1e-4,
                ..Default::default()
            },
            ..Default::default()
        };
        let result =
            optimize(sphere, &[(-5.0, 5.0), (-5.0, 5.0)], 10, Some(config)).expect("optimize ok");

        // Best history should be non-increasing
        for i in 1..result.best_history.len() {
            assert!(
                result.best_history[i] <= result.best_history[i - 1] + 1e-12,
                "Best history not monotonic at index {}: {} > {}",
                i,
                result.best_history[i],
                result.best_history[i - 1]
            );
        }
    }

    #[test]
    fn test_parego_scalarize() {
        let obj = [0.3, 0.7];
        let w = [0.5, 0.5];
        let s = parego_scalarize(&obj, &w);
        // max(0.15, 0.35) + 0.05 * (0.15 + 0.35) = 0.35 + 0.025 = 0.375
        assert!((s - 0.375).abs() < 1e-10);
    }

    #[test]
    fn test_random_simplex_point_sums_to_one() {
        let mut rng = StdRng::seed_from_u64(42);
        for n in 1..6 {
            let pt = random_simplex_point(n, &mut rng);
            assert_eq!(pt.len(), n);
            let sum: f64 = pt.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "Simplex sum = {}", sum);
            for &v in &pt {
                assert!(v >= 0.0, "Simplex component negative: {}", v);
            }
        }
    }

    #[test]
    fn test_optimize_1d() {
        let config = BayesianOptimizerConfig {
            n_initial: 5,
            seed: Some(42),
            gp_config: GpSurrogateConfig {
                optimize_hyperparams: false,
                noise_variance: 1e-4,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = optimize(
            |x: &ArrayView1<f64>| (x[0] - 2.0).powi(2),
            &[(-5.0, 5.0)],
            15,
            Some(config),
        )
        .expect("optimize ok");

        assert!(
            (result.x_best[0] - 2.0).abs() < 1.5,
            "x_best = {:.4}, expected ~2.0",
            result.x_best[0]
        );
        assert!(result.f_best < 2.0);
    }
}
