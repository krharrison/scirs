//! Types for the ensemble ODE solver.

/// Configuration for batched parallel ODE integration.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct EnsembleConfig {
    /// Number of ensemble members (parameter sets / initial conditions).
    /// Default: 100.
    pub n_ensemble: usize,
    /// Number of worker threads.  Default: number of logical CPUs.
    pub n_threads: usize,
    /// Relative error tolerance for step-size control.  Default: 1e-6.
    pub rtol: f64,
    /// Absolute error tolerance for step-size control.  Default: 1e-9.
    pub atol: f64,
    /// Integration interval `(t0, t_end)`.  Default: (0.0, 1.0).
    pub t_span: (f64, f64),
    /// Maximum number of accepted steps per member.  Default: 100_000.
    pub max_steps: usize,
    /// Initial step size (0 means auto).  Default: 0.0.
    pub h_init: f64,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        let n_threads = num_cpus::get().max(1);
        Self {
            n_ensemble: 100,
            n_threads,
            rtol: 1e-6,
            atol: 1e-9,
            t_span: (0.0, 1.0),
            max_steps: 100_000,
            h_init: 0.0,
        }
    }
}

/// Result of an ensemble ODE integration.
#[derive(Debug, Clone)]
pub struct EnsembleResult {
    /// Trajectories: `trajectories[i][k][j]` = state component `j` of member `i`
    /// at time index `k`.
    pub trajectories: Vec<Vec<Vec<f64>>>,
    /// Adaptive time grids: `times[i][k]` = time of step `k` for member `i`.
    pub times: Vec<Vec<f64>>,
    /// Whether member `i` converged within `max_steps`.
    pub converged: Vec<bool>,
    /// Number of accepted steps taken by member `i`.
    pub n_steps: Vec<usize>,
}

impl EnsembleResult {
    /// Compute the element-wise mean trajectory over all ensemble members.
    ///
    /// Members are interpolated onto a common time grid formed by taking
    /// the union of all output times of member 0 (as a reference).
    /// For simplicity, returns the mean of final states only if trajectories
    /// have different lengths; uses the shortest common prefix otherwise.
    ///
    /// Returns `None` if the ensemble is empty.
    pub fn mean_trajectory(&self) -> Option<Vec<Vec<f64>>> {
        if self.trajectories.is_empty() {
            return None;
        }

        // Use the trajectory of member 0 as the reference length
        let ref_len = self.trajectories[0].len();
        if ref_len == 0 {
            return None;
        }

        let n_state = self.trajectories[0][0].len();
        let n_members = self.trajectories.len();

        // Find the minimum trajectory length across all members
        let min_len = self
            .trajectories
            .iter()
            .map(|traj| traj.len())
            .min()
            .unwrap_or(0);

        if min_len == 0 {
            return None;
        }

        let mut mean = vec![vec![0.0_f64; n_state]; min_len];
        for traj in &self.trajectories {
            for (k, step) in traj.iter().take(min_len).enumerate() {
                for (j, &val) in step.iter().enumerate() {
                    mean[k][j] += val;
                }
            }
        }
        let n_f = n_members as f64;
        for step in mean.iter_mut() {
            for val in step.iter_mut() {
                *val /= n_f;
            }
        }
        Some(mean)
    }

    /// Compute the element-wise standard deviation trajectory.
    ///
    /// Returns `None` if fewer than 2 members or if the ensemble is empty.
    pub fn std_trajectory(&self) -> Option<Vec<Vec<f64>>> {
        if self.trajectories.len() < 2 {
            return None;
        }
        let mean = self.mean_trajectory()?;
        let min_len = mean.len();
        let n_state = mean[0].len();
        let n_members = self.trajectories.len();

        let mut variance = vec![vec![0.0_f64; n_state]; min_len];
        for traj in &self.trajectories {
            for (k, step) in traj.iter().take(min_len).enumerate() {
                for (j, &val) in step.iter().enumerate() {
                    let diff = val - mean[k][j];
                    variance[k][j] += diff * diff;
                }
            }
        }
        let n_f = (n_members - 1) as f64;
        for step in variance.iter_mut() {
            for val in step.iter_mut() {
                *val = (*val / n_f).sqrt();
            }
        }
        Some(variance)
    }

    /// Return quantile trajectories at quantile `q` (e.g. 0.5 = median).
    ///
    /// Computes the quantile across ensemble members at each common time step.
    /// Returns `None` if the ensemble is empty.
    pub fn quantile_trajectories(&self, q: f64) -> Option<Vec<Vec<f64>>> {
        if self.trajectories.is_empty() {
            return None;
        }
        let min_len = self.trajectories.iter().map(|t| t.len()).min().unwrap_or(0);
        if min_len == 0 {
            return None;
        }
        let n_state = self.trajectories[0][0].len();
        let n_members = self.trajectories.len();

        let mut result = vec![vec![0.0_f64; n_state]; min_len];
        for k in 0..min_len {
            for j in 0..n_state {
                let mut vals: Vec<f64> = self
                    .trajectories
                    .iter()
                    .filter(|traj| traj.len() > k)
                    .map(|traj| traj[k][j])
                    .collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let idx = ((q * (n_members - 1) as f64).round() as usize).min(n_members - 1);
                result[k][j] = vals[idx];
            }
        }
        Some(result)
    }
}
