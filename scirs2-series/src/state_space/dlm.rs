//! Dynamic Linear Models with builder pattern (West & Harrison 1997).
//!
//! The DLM provides a flexible builder API for constructing state-space models
//! and wraps the `LinearGaussianSSM` Kalman filter/smoother infrastructure.
//!
//! Model specification:
//!   State:  θ_t = G θ_{t-1} + w_t,  w_t ~ N(0, W)
//!   Obs:    y_t = F_t' θ_t + v_t,   v_t ~ N(0, V)
//!
//! where F_t may be time-varying (e.g. for regression/intervention effects).

use super::linear_gaussian::{KalmanOutput, LinearGaussianSSM};
use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// DLMBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing a `DynamicLinearModelVec`.
///
/// Uses method chaining (builder pattern).
#[derive(Debug, Clone)]
pub struct DLMBuilder {
    dim_state: usize,
    dim_obs: usize,
    /// Time-varying observation matrices F_t [T][p][d].
    /// If only one is given it is treated as constant.
    f_matrices: Vec<Vec<Vec<f64>>>,
    /// Transition matrix G [d][d]
    g_mat: Option<Vec<Vec<f64>>>,
    /// Observation noise covariance V [p][p]
    v_mat: Option<Vec<Vec<f64>>>,
    /// State noise covariance W [d][d]
    w_mat: Option<Vec<Vec<f64>>>,
    /// Initial state mean m0 [d]
    m0: Option<Vec<f64>>,
    /// Initial state covariance C0 [d][d]
    c0: Option<Vec<Vec<f64>>>,
}

impl DLMBuilder {
    /// Create a new builder with default 1×1 dimensions.
    pub fn new() -> Self {
        Self {
            dim_state: 1,
            dim_obs: 1,
            f_matrices: vec![],
            g_mat: None,
            v_mat: None,
            w_mat: None,
            m0: None,
            c0: None,
        }
    }

    /// Set the state dimension d.
    pub fn with_state_dim(mut self, d: usize) -> Self {
        self.dim_state = d;
        self
    }

    /// Set the observation dimension p.
    pub fn with_obs_dim(mut self, p: usize) -> Self {
        self.dim_obs = p;
        self
    }

    /// Set the (constant) transition matrix G of shape `d` x `d`.
    pub fn with_transition(mut self, g: Vec<Vec<f64>>) -> Self {
        self.g_mat = Some(g);
        self
    }

    /// Add an observation matrix F_t of shape `p` x `d`.
    ///
    /// Call once for a constant F or T times for time-varying F.
    pub fn with_observation(mut self, f: Vec<Vec<f64>>) -> Self {
        self.f_matrices.push(f);
        self
    }

    /// Set observation noise covariance V of shape `p` x `p`.
    pub fn with_obs_variance(mut self, v: Vec<Vec<f64>>) -> Self {
        self.v_mat = Some(v);
        self
    }

    /// Set state evolution noise covariance W of shape `d` x `d`.
    pub fn with_state_variance(mut self, w: Vec<Vec<f64>>) -> Self {
        self.w_mat = Some(w);
        self
    }

    /// Set initial state mean m0 of length `d`.
    pub fn with_initial_mean(mut self, m0: Vec<f64>) -> Self {
        self.m0 = Some(m0);
        self
    }

    /// Set initial state covariance C0 of shape `d` x `d`.
    pub fn with_initial_cov(mut self, c0: Vec<Vec<f64>>) -> Self {
        self.c0 = Some(c0);
        self
    }

    /// Convenience: local level model (random walk + noise).
    ///
    /// dim_state = 1, G = I, F = \[1\], W = w_var * I, V = v_var * I.
    pub fn local_level(w_var: f64, v_var: f64) -> Self {
        Self::new()
            .with_state_dim(1)
            .with_obs_dim(1)
            .with_transition(vec![vec![1.0]])
            .with_observation(vec![vec![1.0]])
            .with_state_variance(vec![vec![w_var]])
            .with_obs_variance(vec![vec![v_var]])
    }

    /// Convenience: local linear trend model.
    ///
    /// State = \[level, slope\], G = \[\[1,1\],\[0,1\]\], F = \[\[1,0\]\].
    pub fn local_linear_trend(level_w: f64, slope_w: f64, v_var: f64) -> Self {
        Self::new()
            .with_state_dim(2)
            .with_obs_dim(1)
            .with_transition(vec![vec![1.0, 1.0], vec![0.0, 1.0]])
            .with_observation(vec![vec![1.0, 0.0]])
            .with_state_variance(vec![vec![level_w, 0.0], vec![0.0, slope_w]])
            .with_obs_variance(vec![vec![v_var]])
    }

    /// Build the `DynamicLinearModelVec`.
    ///
    /// Returns an error if required matrices are missing or inconsistent.
    pub fn build(self) -> Result<DynamicLinearModelVec> {
        let d = self.dim_state;
        let p = self.dim_obs;

        // G: default identity
        let g = self.g_mat.unwrap_or_else(|| {
            let mut m = vec![vec![0.0f64; d]; d];
            for i in 0..d {
                m[i][i] = 1.0;
            }
            m
        });

        // V: default identity * 1
        let v = self.v_mat.unwrap_or_else(|| {
            let mut m = vec![vec![0.0f64; p]; p];
            for i in 0..p {
                m[i][i] = 1.0;
            }
            m
        });

        // W: default identity * 1
        let w = self.w_mat.unwrap_or_else(|| {
            let mut m = vec![vec![0.0f64; d]; d];
            for i in 0..d {
                m[i][i] = 1.0;
            }
            m
        });

        // F: default [I_{p×d}]
        let f_constant = if self.f_matrices.is_empty() {
            let p_d = p.min(d);
            let mut f = vec![vec![0.0f64; d]; p];
            for i in 0..p_d {
                f[i][i] = 1.0;
            }
            f
        } else {
            // Use first matrix as constant if only one provided
            self.f_matrices[0].clone()
        };

        // m0: default zeros
        let m0 = self.m0.unwrap_or_else(|| vec![0.0; d]);
        // C0: default large diagonal
        let c0 = self.c0.unwrap_or_else(|| {
            let mut m = vec![vec![0.0f64; d]; d];
            for i in 0..d {
                m[i][i] = 1e6;
            }
            m
        });

        // Validate dimensions
        if g.len() != d {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: d,
                actual: g.len(),
            });
        }
        if f_constant.len() != p {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: p,
                actual: f_constant.len(),
            });
        }
        if v.len() != p {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: p,
                actual: v.len(),
            });
        }
        if w.len() != d {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: d,
                actual: w.len(),
            });
        }

        // Build the time-varying F list (if multiple given)
        let tv_f = if self.f_matrices.len() > 1 {
            Some(self.f_matrices)
        } else {
            None
        };

        let ssm = LinearGaussianSSM {
            dim_state: d,
            dim_obs: p,
            f_mat: g,
            h_mat: f_constant,
            q_mat: w,
            r_mat: v,
            mu0: m0,
            p0: c0,
        };

        Ok(DynamicLinearModelVec { ssm, tv_f })
    }
}

impl Default for DLMBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DynamicLinearModelVec
// ---------------------------------------------------------------------------

/// Dynamic Linear Model wrapping a `LinearGaussianSSM`.
///
/// Supports optional time-varying observation matrices F_t.
#[derive(Debug, Clone)]
pub struct DynamicLinearModelVec {
    /// Underlying linear Gaussian state-space model
    pub ssm: LinearGaussianSSM,
    /// Optional time-varying observation matrices \[T\]\[p\]\[d\]
    pub tv_f: Option<Vec<Vec<Vec<f64>>>>,
}

impl DynamicLinearModelVec {
    /// Run the Kalman filter.
    pub fn filter(&self, obs: &[Vec<f64>]) -> Result<KalmanOutput> {
        match &self.tv_f {
            None => self.ssm.filter(obs),
            Some(f_seq) => self.filter_tv(obs, f_seq),
        }
    }

    /// Run filter + RTS smoother.
    pub fn smooth(&self, obs: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>)> {
        // Note: smoother always uses the constant H (base model).
        // For time-varying F, we use a modified smoother.
        match &self.tv_f {
            None => self.ssm.smooth(obs),
            Some(f_seq) => self.smooth_tv(obs, f_seq),
        }
    }

    /// Kalman filter with time-varying observation matrices.
    fn filter_tv(&self, obs: &[Vec<f64>], f_seq: &[Vec<Vec<f64>>]) -> Result<KalmanOutput> {
        let t_len = obs.len();
        if f_seq.len() != t_len {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Time-varying F sequence length {} does not match T = {}",
                f_seq.len(),
                t_len
            )));
        }

        let d = self.ssm.dim_state;
        let p = self.ssm.dim_obs;

        use std::f64::consts::PI;

        // Local matrix helpers (inline to avoid import issues)
        let mm = |a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>| -> Result<Vec<Vec<f64>>> {
            let m = a.len();
            if m == 0 {
                return Ok(vec![]);
            }
            let k = a[0].len();
            if k != b.len() {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: k,
                    actual: b.len(),
                });
            }
            let n = if b.is_empty() { 0 } else { b[0].len() };
            let mut c = vec![vec![0.0f64; n]; m];
            for i in 0..m {
                for l in 0..k {
                    for j in 0..n {
                        c[i][j] += a[i][l] * b[l][j];
                    }
                }
            }
            Ok(c)
        };

        let mt = |a: &Vec<Vec<f64>>| -> Vec<Vec<f64>> {
            let m = a.len();
            if m == 0 {
                return vec![];
            }
            let n = a[0].len();
            let mut out = vec![vec![0.0f64; m]; n];
            for i in 0..m {
                for j in 0..n {
                    out[j][i] = a[i][j];
                }
            }
            out
        };

        let madd = |a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>| -> Result<Vec<Vec<f64>>> {
            let m = a.len();
            if m != b.len() {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: m,
                    actual: b.len(),
                });
            }
            let mut c = a.clone();
            for i in 0..m {
                for j in 0..a[i].len() {
                    c[i][j] += b[i][j];
                }
            }
            Ok(c)
        };

        let msub = |a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>| -> Result<Vec<Vec<f64>>> {
            let m = a.len();
            if m != b.len() {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: m,
                    actual: b.len(),
                });
            }
            let mut c = a.clone();
            for i in 0..m {
                for j in 0..a[i].len() {
                    c[i][j] -= b[i][j];
                }
            }
            Ok(c)
        };

        let mv = |a: &Vec<Vec<f64>>, x: &Vec<f64>| -> Result<Vec<f64>> {
            let m = a.len();
            let n = if m == 0 { 0 } else { a[0].len() };
            if x.len() != n {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: n,
                    actual: x.len(),
                });
            }
            let mut y = vec![0.0f64; m];
            for i in 0..m {
                for j in 0..n {
                    y[i] += a[i][j] * x[j];
                }
            }
            Ok(y)
        };

        let eye = |n: usize| -> Vec<Vec<f64>> {
            let mut m = vec![vec![0.0f64; n]; n];
            for i in 0..n {
                m[i][i] = 1.0;
            }
            m
        };

        let minv = |a: &Vec<Vec<f64>>| -> Result<Vec<Vec<f64>>> {
            let n = a.len();
            if n == 0 {
                return Ok(vec![]);
            }
            let mut lu: Vec<Vec<f64>> = a.clone();
            let mut rhs = eye(n);
            for col in 0..n {
                let mut max_val = lu[col][col].abs();
                let mut max_row = col;
                for row in (col + 1)..n {
                    let v = lu[row][col].abs();
                    if v > max_val {
                        max_val = v;
                        max_row = row;
                    }
                }
                if max_row != col {
                    lu.swap(col, max_row);
                    rhs.swap(col, max_row);
                }
                if lu[col][col].abs() < 1e-14 {
                    return Err(TimeSeriesError::NumericalInstability(
                        "Near-singular".to_string(),
                    ));
                }
                let pivot = lu[col][col];
                for row in (col + 1)..n {
                    let factor = lu[row][col] / pivot;
                    lu[row][col] = 0.0;
                    for j in (col + 1)..n {
                        let v = lu[row][j] - factor * lu[col][j];
                        lu[row][j] = v;
                    }
                    for j in 0..n {
                        let v = rhs[row][j] - factor * rhs[col][j];
                        rhs[row][j] = v;
                    }
                }
            }
            for k in (0..n).rev() {
                let pivot = lu[k][k];
                for j in 0..n {
                    rhs[k][j] /= pivot;
                }
                for row in 0..k {
                    let factor = lu[row][k];
                    for j in 0..n {
                        let v = rhs[row][j] - factor * rhs[k][j];
                        rhs[row][j] = v;
                    }
                }
            }
            Ok(rhs)
        };

        let log_det_fn = |a: &Vec<Vec<f64>>| -> f64 {
            let n = a.len();
            if n == 0 {
                return 0.0;
            }
            let mut lu: Vec<Vec<f64>> = a.clone();
            for col in 0..n {
                let mut max_row = col;
                let mut max_val = lu[col][col].abs();
                for row in (col + 1)..n {
                    let v = lu[row][col].abs();
                    if v > max_val {
                        max_val = v;
                        max_row = row;
                    }
                }
                if max_row != col {
                    lu.swap(col, max_row);
                }
                if lu[col][col].abs() < 1e-14 {
                    return f64::NEG_INFINITY;
                }
                let pivot = lu[col][col];
                for row in (col + 1)..n {
                    let factor = lu[row][col] / pivot;
                    for j in (col + 1)..n {
                        let v = lu[row][j] - factor * lu[col][j];
                        lu[row][j] = v;
                    }
                }
            }
            (0..n).map(|i| lu[i][i].abs().ln()).sum()
        };

        let f_base = &self.ssm.f_mat; // G (transition)
        let q = &self.ssm.q_mat; // W
        let r = &self.ssm.r_mat; // V
        let ft_base = mt(f_base);

        let mut m_state = self.ssm.mu0.clone();
        let mut p_mat = self.ssm.p0.clone();

        let mut filtered_means = Vec::with_capacity(t_len);
        let mut filtered_covs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(t_len);
        let mut predicted_means = Vec::with_capacity(t_len);
        let mut predicted_covs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(t_len);
        let mut innovations = Vec::with_capacity(t_len);
        let mut log_likelihood = 0.0_f64;

        for t in 0..t_len {
            let h_t = &f_seq[t];
            let ht = mt(h_t);

            // Predict
            let m_pred = mv(f_base, &m_state)?;
            let fp = mm(f_base, &p_mat)?;
            let p_pred = madd(&mm(&fp, &ft_base)?, q)?;

            predicted_means.push(m_pred.clone());
            predicted_covs.push(p_pred.clone());

            let y_t = &obs[t];
            let is_missing = y_t.iter().any(|v| v.is_nan());

            if is_missing {
                filtered_means.push(m_pred.clone());
                filtered_covs.push(p_pred.clone());
                innovations.push(vec![0.0; p]);
                m_state = m_pred;
                p_mat = p_pred;
                continue;
            }

            // Update
            let hm = mv(h_t, &m_pred)?;
            let v_innov: Vec<f64> = (0..p).map(|i| y_t[i] - hm[i]).collect();
            let hp = mm(h_t, &p_pred)?;
            let s_mat = madd(&mm(&hp, &ht)?, r)?;
            let pht = mm(&p_pred, &ht)?;
            let s_inv = minv(&s_mat)?;
            let k_gain = mm(&pht, &s_inv)?;
            let kv = mv(&k_gain, &v_innov)?;
            let m_upd: Vec<f64> = (0..d).map(|i| m_pred[i] + kv[i]).collect();
            let kh = mm(&k_gain, h_t)?;
            let i_kh = msub(&eye(d), &kh)?;
            let p_upd = mm(&i_kh, &p_pred)?;

            let ld = log_det_fn(&s_mat);
            let sinv_v = mv(&s_inv, &v_innov)?;
            let quad: f64 = (0..p).map(|i| v_innov[i] * sinv_v[i]).sum();
            log_likelihood += -0.5 * ((p as f64) * (2.0 * PI).ln() + ld + quad);

            filtered_means.push(m_upd.clone());
            filtered_covs.push(p_upd.clone());
            innovations.push(v_innov);
            m_state = m_upd;
            p_mat = p_upd;
        }

        Ok(KalmanOutput {
            filtered_means,
            filtered_covs,
            predicted_means,
            predicted_covs,
            innovations,
            log_likelihood,
        })
    }

    /// RTS smoother for time-varying F case.
    fn smooth_tv(
        &self,
        obs: &[Vec<f64>],
        f_seq: &[Vec<Vec<f64>>],
    ) -> Result<(Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>)> {
        let kout = self.filter_tv(obs, f_seq)?;
        // Reuse the standard smoother from ssm (uses constant G/transition)
        let t_len = kout.filtered_means.len();
        let d = self.ssm.dim_state;

        if t_len == 0 {
            return Ok((vec![], vec![]));
        }

        let f_base = &self.ssm.f_mat;

        // Reimplement RTS smoother inline using kout
        let mut sm_means: Vec<Vec<f64>> = kout.filtered_means.clone();
        let mut sm_covs: Vec<Vec<Vec<f64>>> = kout.filtered_covs.clone();

        // Helper closures
        let mm_fn = |a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>| -> Result<Vec<Vec<f64>>> {
            let m = a.len();
            if m == 0 {
                return Ok(vec![]);
            }
            let k = a[0].len();
            if k != b.len() {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: k,
                    actual: b.len(),
                });
            }
            let n = if b.is_empty() { 0 } else { b[0].len() };
            let mut c = vec![vec![0.0f64; n]; m];
            for i in 0..m {
                for l in 0..k {
                    for j in 0..n {
                        c[i][j] += a[i][l] * b[l][j];
                    }
                }
            }
            Ok(c)
        };

        let mt_fn = |a: &Vec<Vec<f64>>| -> Vec<Vec<f64>> {
            let m = a.len();
            if m == 0 {
                return vec![];
            }
            let n = a[0].len();
            let mut out = vec![vec![0.0f64; m]; n];
            for i in 0..m {
                for j in 0..n {
                    out[j][i] = a[i][j];
                }
            }
            out
        };

        let madd_fn = |a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>| -> Result<Vec<Vec<f64>>> {
            let m = a.len();
            if m != b.len() {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: m,
                    actual: b.len(),
                });
            }
            let mut c = a.clone();
            for i in 0..m {
                for j in 0..a[i].len() {
                    c[i][j] += b[i][j];
                }
            }
            Ok(c)
        };

        let msub_fn = |a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>| -> Result<Vec<Vec<f64>>> {
            let m = a.len();
            if m != b.len() {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: m,
                    actual: b.len(),
                });
            }
            let mut c = a.clone();
            for i in 0..m {
                for j in 0..a[i].len() {
                    c[i][j] -= b[i][j];
                }
            }
            Ok(c)
        };

        let mv_fn = |a: &Vec<Vec<f64>>, x: &[f64]| -> Result<Vec<f64>> {
            let m = a.len();
            let n = if m == 0 { 0 } else { a[0].len() };
            if x.len() != n {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: n,
                    actual: x.len(),
                });
            }
            let mut y = vec![0.0f64; m];
            for i in 0..m {
                for j in 0..n {
                    y[i] += a[i][j] * x[j];
                }
            }
            Ok(y)
        };

        let eye_fn = |n: usize| -> Vec<Vec<f64>> {
            let mut m = vec![vec![0.0f64; n]; n];
            for i in 0..n {
                m[i][i] = 1.0;
            }
            m
        };

        let minv_fn = |a: &Vec<Vec<f64>>| -> Vec<Vec<f64>> {
            let n = a.len();
            if n == 0 {
                return vec![];
            }
            let mut lu: Vec<Vec<f64>> = a.clone();
            let mut rhs = eye_fn(n);
            for col in 0..n {
                let mut max_row = col;
                let mut max_val = lu[col][col].abs();
                for row in (col + 1)..n {
                    let v = lu[row][col].abs();
                    if v > max_val {
                        max_val = v;
                        max_row = row;
                    }
                }
                if max_row != col {
                    lu.swap(col, max_row);
                    rhs.swap(col, max_row);
                }
                if lu[col][col].abs() < 1e-14 {
                    return eye_fn(n);
                }
                let pivot = lu[col][col];
                for row in (col + 1)..n {
                    let factor = lu[row][col] / pivot;
                    lu[row][col] = 0.0;
                    for j in (col + 1)..n {
                        let v = lu[row][j] - factor * lu[col][j];
                        lu[row][j] = v;
                    }
                    for j in 0..n {
                        let v = rhs[row][j] - factor * rhs[col][j];
                        rhs[row][j] = v;
                    }
                }
            }
            for k in (0..n).rev() {
                let pivot = lu[k][k];
                for j in 0..n {
                    rhs[k][j] /= pivot;
                }
                for row in 0..k {
                    let factor = lu[row][k];
                    for j in 0..n {
                        let v = rhs[row][j] - factor * rhs[k][j];
                        rhs[row][j] = v;
                    }
                }
            }
            rhs
        };

        let ft_base = mt_fn(f_base);

        for t in (0..t_len - 1).rev() {
            let p_filt = &kout.filtered_covs[t];
            let p_pred_next = &kout.predicted_covs[t + 1];
            let pft = mm_fn(p_filt, &ft_base)?;
            let p_pred_inv = minv_fn(p_pred_next);
            let g_gain = mm_fn(&pft, &p_pred_inv)?;

            let diff: Vec<f64> = (0..d)
                .map(|j| sm_means[t + 1][j] - kout.predicted_means[t + 1][j])
                .collect();
            let g_diff = mv_fn(&g_gain, &diff)?;
            for j in 0..d {
                sm_means[t][j] = kout.filtered_means[t][j] + g_diff[j];
            }

            let dp = msub_fn(&sm_covs[t + 1], p_pred_next)?;
            let g_dp = mm_fn(&g_gain, &dp)?;
            let g_dp_gt = mm_fn(&g_dp, &mt_fn(&g_gain))?;
            sm_covs[t] = madd_fn(p_filt, &g_dp_gt)?;
        }

        Ok((sm_means, sm_covs))
    }

    /// Forecast h steps ahead from the last filtered state.
    pub fn forecast(&self, obs: &[Vec<f64>], h: usize) -> Result<Vec<Vec<f64>>> {
        if obs.is_empty() || h == 0 {
            return Ok(vec![]);
        }
        let kout = self.filter(obs)?;
        let t = kout.filtered_means.len() - 1;
        let last_state = &kout.filtered_means[t];
        let last_cov = &kout.filtered_covs[t];
        self.ssm.forecast(last_state, last_cov, h)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn constant_series(n: usize, val: f64) -> Vec<Vec<f64>> {
        (0..n).map(|_| vec![val]).collect()
    }

    fn trend_series(n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|i| vec![i as f64 * 0.5 + 1.0]).collect()
    }

    #[test]
    fn test_builder_local_level() {
        let dlm = DLMBuilder::local_level(0.1, 0.5).build().expect("build ok");
        assert_eq!(dlm.ssm.dim_state, 1);
        assert_eq!(dlm.ssm.dim_obs, 1);
        assert_eq!(dlm.ssm.f_mat[0][0], 1.0); // G = identity
        assert_eq!(dlm.ssm.h_mat[0][0], 1.0); // F = [1]
        assert!((dlm.ssm.q_mat[0][0] - 0.1).abs() < 1e-12);
        assert!((dlm.ssm.r_mat[0][0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_builder_local_linear_trend() {
        let dlm = DLMBuilder::local_linear_trend(0.1, 0.01, 0.5)
            .build()
            .expect("build ok");
        assert_eq!(dlm.ssm.dim_state, 2);
        // G = [[1,1],[0,1]]
        assert_eq!(dlm.ssm.f_mat[0][0], 1.0);
        assert_eq!(dlm.ssm.f_mat[0][1], 1.0);
        assert_eq!(dlm.ssm.f_mat[1][0], 0.0);
        assert_eq!(dlm.ssm.f_mat[1][1], 1.0);
    }

    #[test]
    fn test_filter_constant_series() {
        let dlm = DLMBuilder::local_level(0.1, 0.5).build().expect("ok");
        let obs = constant_series(20, 3.0);
        let kout = dlm.filter(&obs).expect("filter ok");
        assert_eq!(kout.filtered_means.len(), 20);
        // After many obs of constant 3.0, filtered mean should converge to 3.0
        let final_mean = kout.filtered_means[19][0];
        assert!(
            (final_mean - 3.0).abs() < 0.5,
            "Final mean {final_mean} not close to 3.0"
        );
    }

    #[test]
    fn test_smooth_reduces_covariance() {
        let dlm = DLMBuilder::local_level(0.2, 0.8).build().expect("ok");
        let obs = trend_series(25);
        let (sm_means, sm_covs) = dlm.smooth(&obs).expect("smooth ok");
        assert_eq!(sm_means.len(), 25);

        let kout = dlm.filter(&obs).expect("filter ok");
        // Smoothed covariance at t=5 should be <= filtered
        let p_sm = sm_covs[5][0][0];
        let p_filt = kout.filtered_covs[5][0][0];
        assert!(
            p_sm <= p_filt + 1e-6,
            "Smoothed cov {p_sm} > filtered cov {p_filt}"
        );
    }

    #[test]
    fn test_time_varying_obs_matrix() {
        let n = 10usize;
        // Build a DLM with time-varying F_t
        let f_seq: Vec<Vec<Vec<f64>>> = (0..n).map(|i| vec![vec![1.0 + 0.01 * i as f64]]).collect();

        let dlm = DLMBuilder::new()
            .with_state_dim(1)
            .with_obs_dim(1)
            .with_transition(vec![vec![1.0]])
            .with_state_variance(vec![vec![0.1]])
            .with_obs_variance(vec![vec![0.5]])
            .with_observation(f_seq[0].clone()) // one F for constant base
            .build()
            .expect("ok");

        // Manually set tv_f
        let mut dlm_tv = dlm;
        dlm_tv.tv_f = Some(f_seq);

        let obs: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64 * 0.3]).collect();
        let kout = dlm_tv.filter(&obs).expect("tv filter ok");
        assert_eq!(kout.filtered_means.len(), n);
        assert!(kout.log_likelihood.is_finite());
    }

    #[test]
    fn test_forecast() {
        let dlm = DLMBuilder::local_level(0.1, 0.5).build().expect("ok");
        let obs = constant_series(15, 2.0);
        let fc = dlm.forecast(&obs, 5).expect("forecast ok");
        assert_eq!(fc.len(), 5);
        for row in &fc {
            assert_eq!(row.len(), 1);
            assert!(row[0].is_finite());
        }
    }

    #[test]
    fn test_builder_custom() {
        let dlm = DLMBuilder::new()
            .with_state_dim(2)
            .with_obs_dim(1)
            .with_transition(vec![vec![1.0, 0.5], vec![0.0, 0.9]])
            .with_observation(vec![vec![1.0, 0.0]])
            .with_state_variance(vec![vec![0.1, 0.0], vec![0.0, 0.05]])
            .with_obs_variance(vec![vec![1.0]])
            .build()
            .expect("custom build ok");

        assert_eq!(dlm.ssm.dim_state, 2);
        assert_eq!(dlm.ssm.dim_obs, 1);
        assert_eq!(dlm.ssm.f_mat[0][1], 0.5);
        assert_eq!(dlm.ssm.f_mat[1][1], 0.9);
    }

    #[test]
    fn test_filter_empty() {
        let dlm = DLMBuilder::local_level(0.1, 0.5).build().expect("ok");
        let kout = dlm.filter(&[]).expect("empty filter ok");
        assert!(kout.filtered_means.is_empty());
        assert_eq!(kout.log_likelihood, 0.0);
    }
}
