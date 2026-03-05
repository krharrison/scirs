//! Linear Gaussian State Space Model (LGSSM) with Vec-based API.
//!
//! State equation: x_t = F x_{t-1} + w_t,  w_t ~ N(0, Q)
//! Obs equation:   y_t = H x_t + v_t,       v_t ~ N(0, R)
//!
//! Provides:
//! - Kalman filter (forward pass)
//! - RTS smoother (backward pass)
//! - EM algorithm for parameter estimation
//! - h-step ahead forecasting

use crate::error::{Result, TimeSeriesError};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Low-level matrix helpers (Vec<Vec<f64>>)
// ---------------------------------------------------------------------------

/// Matrix-matrix product: C = A * B, shapes (m,k) x (k,n) -> (m,n)
fn mm(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let m = a.len();
    if m == 0 {
        return Ok(vec![]);
    }
    let k = a[0].len();
    let k2 = b.len();
    if k != k2 {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: k,
            actual: k2,
        });
    }
    let n = if b.is_empty() { 0 } else { b[0].len() };
    let mut c = vec![vec![0.0f64; n]; m];
    for i in 0..m {
        for l in 0..k {
            let a_il = a[i][l];
            for j in 0..n {
                c[i][j] += a_il * b[l][j];
            }
        }
    }
    Ok(c)
}

/// Matrix transpose: shape (m,n) -> (n,m)
fn mt(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
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
}

/// Element-wise matrix addition: A + B
fn madd(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let m = a.len();
    if m != b.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: m,
            actual: b.len(),
        });
    }
    let mut c = a.to_vec();
    for i in 0..m {
        if a[i].len() != b[i].len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: a[i].len(),
                actual: b[i].len(),
            });
        }
        for j in 0..a[i].len() {
            c[i][j] += b[i][j];
        }
    }
    Ok(c)
}

/// Element-wise matrix subtraction: A - B
fn msub(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let m = a.len();
    if m != b.len() {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: m,
            actual: b.len(),
        });
    }
    let mut c = a.to_vec();
    for i in 0..m {
        if a[i].len() != b[i].len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: a[i].len(),
                actual: b[i].len(),
            });
        }
        for j in 0..a[i].len() {
            c[i][j] -= b[i][j];
        }
    }
    Ok(c)
}

/// Matrix-vector product: y = A x
fn mv(a: &[Vec<f64>], x: &[f64]) -> Result<Vec<f64>> {
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
}

/// Identity matrix of size n
fn eye(n: usize) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        out[i][i] = 1.0;
    }
    out
}

/// Zeros matrix of shape (m, n)
fn zeros_mat(m: usize, n: usize) -> Vec<Vec<f64>> {
    vec![vec![0.0f64; n]; m]
}

/// Matrix inversion via LU decomposition with partial pivoting.
fn minv(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n = a.len();
    if n == 0 {
        return Ok(vec![]);
    }
    // Build augmented [A | I]
    let mut lu: Vec<Vec<f64>> = a.iter().map(|r| r.clone()).collect();
    let mut rhs = eye(n);

    for col in 0..n {
        // Find pivot row
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
                "Near-singular matrix in inversion".to_string(),
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

    // Back-substitution
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
}

/// Log determinant via Cholesky (falls back to LU).
fn log_det_mat(a: &[Vec<f64>]) -> Result<f64> {
    let n = a.len();
    if n == 0 {
        return Ok(0.0);
    }

    // Try Cholesky
    let mut l = zeros_mat(n, n);
    let mut ok = true;
    'outer: for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for kk in 0..j {
                s -= l[i][kk] * l[j][kk];
            }
            if i == j {
                if s <= 0.0 {
                    ok = false;
                    break 'outer;
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }

    if ok {
        let log_sum: f64 = (0..n).map(|i| l[i][i].ln()).sum();
        return Ok(2.0 * log_sum);
    }

    // LU fallback
    let mut lu: Vec<Vec<f64>> = a.iter().map(|r| r.clone()).collect();
    let mut sign = 1.0_f64;
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
            sign = -sign;
        }
        if lu[col][col].abs() < 1e-14 {
            return Ok(f64::NEG_INFINITY);
        }
        let pivot = lu[col][col];
        for row in (col + 1)..n {
            let factor = lu[row][col] / pivot;
            lu[row][col] = factor;
            for j in (col + 1)..n {
                let v = lu[row][j] - factor * lu[col][j];
                lu[row][j] = v;
            }
        }
    }

    let log_sum: f64 = (0..n).map(|i| lu[i][i].abs().ln()).sum();
    if sign < 0.0 {
        Ok(f64::NAN)
    } else {
        Ok(log_sum)
    }
}

/// Outer product: x y^T -> (m,n) matrix
fn outer(x: &[f64], y: &[f64]) -> Vec<Vec<f64>> {
    let m = x.len();
    let n = y.len();
    let mut out = zeros_mat(m, n);
    for i in 0..m {
        for j in 0..n {
            out[i][j] = x[i] * y[j];
        }
    }
    out
}

/// E[x x^T] = x x^T + Cov
fn outer_plus_cov(x: &[f64], cov: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = x.len();
    let mut out = outer(x, x);
    for i in 0..n {
        for j in 0..n {
            out[i][j] += cov[i][j];
        }
    }
    out
}

/// Scale matrix by scalar
fn mscale(a: &[Vec<f64>], s: f64) -> Vec<Vec<f64>> {
    a.iter()
        .map(|row| row.iter().map(|&v| v * s).collect())
        .collect()
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Linear Gaussian State Space Model.
///
/// State equation: x_t = F x_{t-1} + w_t,  w_t ~ N(0, Q)
/// Obs  equation:  y_t = H x_t + v_t,       v_t ~ N(0, R)
#[derive(Debug, Clone)]
pub struct LinearGaussianSSM {
    /// Number of state dimensions d
    pub dim_state: usize,
    /// Number of observation dimensions p
    pub dim_obs: usize,
    /// State transition matrix F: (d, d)
    pub f_mat: Vec<Vec<f64>>,
    /// Observation matrix H: (p, d)
    pub h_mat: Vec<Vec<f64>>,
    /// Process noise covariance Q: (d, d)
    pub q_mat: Vec<Vec<f64>>,
    /// Measurement noise covariance R: (p, p)
    pub r_mat: Vec<Vec<f64>>,
    /// Initial state mean mu_0: (d,)
    pub mu0: Vec<f64>,
    /// Initial state covariance P_0: (d, d)
    pub p0: Vec<Vec<f64>>,
}

/// Output from the Kalman filter.
#[derive(Debug, Clone)]
pub struct KalmanOutput {
    /// Filtered state means m_{t|t}: \[T\]\[d\]
    pub filtered_means: Vec<Vec<f64>>,
    /// Filtered state covariances P_{t|t}: \[T\]\[d\]\[d\]
    pub filtered_covs: Vec<Vec<Vec<f64>>>,
    /// Predicted state means m_{t|t-1}: \[T\]\[d\]
    pub predicted_means: Vec<Vec<f64>>,
    /// Predicted state covariances P_{t|t-1}: \[T\]\[d\]\[d\]
    pub predicted_covs: Vec<Vec<Vec<f64>>>,
    /// Innovation sequence v_t = y_t - H m_{t|t-1}: \[T\]\[p\]
    pub innovations: Vec<Vec<f64>>,
    /// Log-likelihood of the observation sequence
    pub log_likelihood: f64,
}

impl LinearGaussianSSM {
    /// Create a new LGSSM with identity matrices and diffuse initial covariance.
    pub fn new(dim_state: usize, dim_obs: usize) -> Self {
        let p_obs = dim_obs.min(dim_state);
        let mut h = zeros_mat(dim_obs, dim_state);
        for i in 0..p_obs {
            h[i][i] = 1.0;
        }
        Self {
            dim_state,
            dim_obs,
            f_mat: eye(dim_state),
            h_mat: h,
            q_mat: eye(dim_state),
            r_mat: eye(dim_obs),
            mu0: vec![0.0; dim_state],
            p0: mscale(&eye(dim_state), 1e6),
        }
    }

    // --- builder methods ---

    /// Set the transition matrix F.
    pub fn with_transition(mut self, f: Vec<Vec<f64>>) -> Self {
        self.f_mat = f;
        self
    }

    /// Set the observation matrix H.
    pub fn with_observation(mut self, h: Vec<Vec<f64>>) -> Self {
        self.h_mat = h;
        self
    }

    /// Set the process noise covariance Q.
    pub fn with_process_noise(mut self, q: Vec<Vec<f64>>) -> Self {
        self.q_mat = q;
        self
    }

    /// Set the measurement noise covariance R.
    pub fn with_measurement_noise(mut self, r: Vec<Vec<f64>>) -> Self {
        self.r_mat = r;
        self
    }

    /// Set the initial state mean and covariance.
    pub fn with_initial(mut self, mu0: Vec<f64>, p0: Vec<Vec<f64>>) -> Self {
        self.mu0 = mu0;
        self.p0 = p0;
        self
    }

    // --- core algorithms ---

    /// Kalman filter: forward pass returning filtered and predicted estimates.
    ///
    /// `observations` is a slice of length-T, each element is a slice of length p.
    pub fn filter(&self, observations: &[Vec<f64>]) -> Result<KalmanOutput> {
        let t_len = observations.len();
        let d = self.dim_state;
        let p = self.dim_obs;

        if t_len == 0 {
            return Ok(KalmanOutput {
                filtered_means: vec![],
                filtered_covs: vec![],
                predicted_means: vec![],
                predicted_covs: vec![],
                innovations: vec![],
                log_likelihood: 0.0,
            });
        }

        // Validate first observation
        if observations[0].len() != p {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: p,
                actual: observations[0].len(),
            });
        }

        let f = &self.f_mat;
        let h = &self.h_mat;
        let q = &self.q_mat;
        let r = &self.r_mat;
        let ft = mt(f);
        let ht = mt(h);

        let mut m = self.mu0.clone();
        let mut pmat = self.p0.clone();

        let mut filtered_means = Vec::with_capacity(t_len);
        let mut filtered_covs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(t_len);
        let mut predicted_means = Vec::with_capacity(t_len);
        let mut predicted_covs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(t_len);
        let mut innovations = Vec::with_capacity(t_len);
        let mut log_likelihood = 0.0_f64;

        for t in 0..t_len {
            // --- Prediction ---
            // m_pred = F * m_{t-1}
            let m_pred = mv(f, &m)?;
            // P_pred = F * P * F^T + Q
            let fp = mm(f, &pmat)?;
            let fpft = mm(&fp, &ft)?;
            let p_pred = madd(&fpft, q)?;

            predicted_means.push(m_pred.clone());
            predicted_covs.push(p_pred.clone());

            let y_t = &observations[t];
            let is_missing = y_t.iter().any(|v| v.is_nan());

            if is_missing {
                filtered_means.push(m_pred.clone());
                filtered_covs.push(p_pred.clone());
                innovations.push(vec![0.0; p]);
                m = m_pred;
                pmat = p_pred;
                continue;
            }

            // --- Update ---
            // v_t = y_t - H m_pred
            let hm = mv(h, &m_pred)?;
            let v: Vec<f64> = (0..p).map(|i| y_t[i] - hm[i]).collect();

            // S_t = H P_pred H^T + R
            let hp = mm(h, &p_pred)?;
            let hpht = mm(&hp, &ht)?;
            let s_mat = madd(&hpht, r)?;

            // K_t = P_pred H^T S^{-1}
            let pht = mm(&p_pred, &ht)?;
            let s_inv = minv(&s_mat)?;
            let k_gain = mm(&pht, &s_inv)?;

            // m_{t|t} = m_pred + K v
            let kv = mv(&k_gain, &v)?;
            let m_upd: Vec<f64> = (0..d).map(|i| m_pred[i] + kv[i]).collect();

            // P_{t|t} = (I - K H) P_pred  (Joseph form)
            let kh = mm(&k_gain, h)?;
            let i_kh = msub(&eye(d), &kh)?;
            let p_upd = mm(&i_kh, &p_pred)?;

            // Log-likelihood: -0.5 * (p*log(2π) + log|S| + v^T S^{-1} v)
            let ld = log_det_mat(&s_mat)?;
            let sinv_v = mv(&s_inv, &v)?;
            let quad: f64 = (0..p).map(|i| v[i] * sinv_v[i]).sum();
            log_likelihood += -0.5 * ((p as f64) * (2.0 * PI).ln() + ld + quad);

            filtered_means.push(m_upd.clone());
            filtered_covs.push(p_upd.clone());
            innovations.push(v);

            m = m_upd;
            pmat = p_upd;
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

    /// RTS smoother: returns (smoothed_means \[T\]\[d\], smoothed_covs \[T\]\[d\]\[d\]).
    pub fn smooth(&self, observations: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>)> {
        let kout = self.filter(observations)?;
        let t_len = kout.filtered_means.len();
        let d = self.dim_state;

        if t_len == 0 {
            return Ok((vec![], vec![]));
        }

        let f = &self.f_mat;
        let ft = mt(f);

        let mut sm_means: Vec<Vec<f64>> = kout.filtered_means.clone();
        let mut sm_covs: Vec<Vec<Vec<f64>>> = kout.filtered_covs.clone();

        // Backward pass: t = T-2 down to 0
        for t in (0..t_len - 1).rev() {
            let p_filt = &kout.filtered_covs[t];
            let p_pred_next = &kout.predicted_covs[t + 1];

            // G_t = P_{t|t} F^T P_{t+1|t}^{-1}
            let pft = mm(p_filt, &ft)?;
            let p_pred_inv = minv(p_pred_next)?;
            let g = mm(&pft, &p_pred_inv)?;

            // m_{t|T} = m_{t|t} + G_t (m_{t+1|T} - m_{t+1|t})
            let diff: Vec<f64> = (0..d)
                .map(|j| sm_means[t + 1][j] - kout.predicted_means[t + 1][j])
                .collect();
            let g_diff = mv(&g, &diff)?;
            for j in 0..d {
                sm_means[t][j] = kout.filtered_means[t][j] + g_diff[j];
            }

            // P_{t|T} = P_{t|t} + G_t (P_{t+1|T} - P_{t+1|t}) G_t^T
            let dp = msub(&sm_covs[t + 1], p_pred_next)?;
            let g_dp = mm(&g, &dp)?;
            let g_dp_gt = mm(&g_dp, &mt(&g))?;
            sm_covs[t] = madd(p_filt, &g_dp_gt)?;
        }

        Ok((sm_means, sm_covs))
    }

    /// EM algorithm for parameter estimation.
    ///
    /// Estimates Q and R by default; returns log-likelihood history.
    /// When `estimate_f` is true, also re-estimates F.
    /// When `estimate_h` is true, also re-estimates H.
    pub fn fit_em(&mut self, observations: &[Vec<f64>], n_iter: usize) -> Result<Vec<f64>> {
        self.fit_em_full(observations, n_iter, false, false)
    }

    /// Full EM with control over which parameters to estimate.
    pub fn fit_em_full(
        &mut self,
        observations: &[Vec<f64>],
        n_iter: usize,
        estimate_f: bool,
        estimate_h: bool,
    ) -> Result<Vec<f64>> {
        let t_len = observations.len();
        if t_len < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "EM requires at least 2 observations".to_string(),
                required: 2,
                actual: t_len,
            });
        }
        let d = self.dim_state;
        let p_obs = self.dim_obs;

        let mut ll_history = Vec::with_capacity(n_iter);
        let tol = 1e-6;

        for iter in 0..n_iter {
            // E-step: forward filter + RTS smoother
            let kout = self.filter(observations)?;
            let ll = kout.log_likelihood;
            ll_history.push(ll);

            if iter > 0 {
                let prev = ll_history[iter - 1];
                if (ll - prev).abs() < tol {
                    break;
                }
            }

            let (sm_means, sm_covs) = self.smooth(observations)?;

            // Sufficient statistics
            let mut sum_pp = zeros_mat(d, d); // sum E[x_t x_t']
            let mut sum_pp_lag = zeros_mat(d, d); // sum E[x_t x_{t-1}'] for t=1..T
            let mut sum_pp_prev = zeros_mat(d, d); // sum E[x_{t-1} x_{t-1}'] for t=1..T
            let mut sum_yp = zeros_mat(p_obs, d); // sum y_t x_t'
            let mut sum_yy = zeros_mat(p_obs, p_obs); // sum y_t y_t'

            for t in 0..t_len {
                let ext = outer_plus_cov(&sm_means[t], &sm_covs[t]);
                sum_pp = madd(&sum_pp, &ext)?;

                let y_t = &observations[t];
                if !y_t.iter().any(|v| v.is_nan()) {
                    let yp = outer(y_t, &sm_means[t]);
                    sum_yp = madd(&sum_yp, &yp)?;
                    let yy = outer(y_t, y_t);
                    sum_yy = madd(&sum_yy, &yy)?;
                }
            }

            // Lag-one cross covariance: E[x_t x_{t-1}'] using smoother gain
            let ft_ref = mt(&self.f_mat);
            for t in 1..t_len {
                let p_filt_prev = &kout.filtered_covs[t - 1];
                let p_pred_t = &kout.predicted_covs[t];
                let pft = mm(p_filt_prev, &ft_ref)?;
                let p_pred_inv = minv(p_pred_t).unwrap_or_else(|_| eye(d));
                let g_prev = mm(&pft, &p_pred_inv)?;

                // E[x_t x_{t-1}'] ≈ x_t x_{t-1}' + G_{t-1} P_{t|T}
                let cross_outer = outer(&sm_means[t], &sm_means[t - 1]);
                let g_cov = mm(&g_prev, &sm_covs[t])?;
                let p_cross = madd(&cross_outer, &g_cov)?;

                sum_pp_lag = madd(&sum_pp_lag, &p_cross)?;
                let ext_prev = outer_plus_cov(&sm_means[t - 1], &sm_covs[t - 1]);
                sum_pp_prev = madd(&sum_pp_prev, &ext_prev)?;
            }

            let t_f64 = t_len as f64;
            let tm1_f64 = (t_len - 1) as f64;

            // M-step: update H
            if estimate_h {
                let sum_pp_inv = minv(&sum_pp).unwrap_or_else(|_| eye(d));
                self.h_mat = mm(&sum_yp, &sum_pp_inv)?;
            }

            // Update R
            {
                let h_syp_t = mm(&self.h_mat, &mt(&sum_yp))?;
                let r_unnorm = msub(&sum_yy, &h_syp_t)?;
                let mut r_new = mscale(&r_unnorm, 1.0 / t_f64);
                for i in 0..p_obs {
                    if r_new[i][i] < 1e-8 {
                        r_new[i][i] = 1e-8;
                    }
                }
                self.r_mat = r_new;
            }

            // Update F
            if estimate_f {
                let sum_pp_prev_inv = minv(&sum_pp_prev).unwrap_or_else(|_| eye(d));
                self.f_mat = mm(&sum_pp_lag, &sum_pp_prev_inv)?;
            }

            // Update Q
            {
                let f_lag_t = mm(&self.f_mat, &mt(&sum_pp_lag))?;
                let q_unnorm = msub(&sum_pp, &f_lag_t)?;
                let f_spp = mm(&self.f_mat, &sum_pp_prev)?;
                let f_spp_ft = mm(&f_spp, &mt(&self.f_mat))?;
                let q_unnorm2 = msub(&madd(&q_unnorm, &f_spp_ft)?, &q_unnorm)?;
                // Simplified: Q = (sum_pp - F sum_pp_lag^T) / (T-1) projected to diagonal
                // Use full symmetric estimate
                let mut q_new: Vec<Vec<f64>> = (0..d)
                    .map(|i| {
                        (0..d)
                            .map(|j| {
                                (sum_pp[i][j] - f_lag_t[i][j] - f_lag_t[j][i] + f_spp_ft[i][j])
                                    / tm1_f64
                            })
                            .collect()
                    })
                    .collect();
                // Enforce positive diagonal
                for i in 0..d {
                    if q_new[i][i] < 1e-10 {
                        q_new[i][i] = 1e-10;
                    }
                    // Ignore unused variable warning
                    let _ = q_unnorm2;
                }
                self.q_mat = q_new;
            }
        }

        Ok(ll_history)
    }

    /// Forecast h steps ahead starting from the last observation's filtered state.
    ///
    /// Returns forecasted observation means \[h\]\[p\].
    pub fn forecast(
        &self,
        last_state: &[f64],
        last_cov: &[Vec<f64>],
        h: usize,
    ) -> Result<Vec<Vec<f64>>> {
        let d = self.dim_state;
        let p = self.dim_obs;
        let f = &self.f_mat;
        let h_mat = &self.h_mat;
        let q = &self.q_mat;
        let ft = mt(f);

        if last_state.len() != d {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: d,
                actual: last_state.len(),
            });
        }

        let mut m = last_state.to_vec();
        let mut pmat: Vec<Vec<f64>> = last_cov.to_vec();
        let mut forecasts = Vec::with_capacity(h);

        for _ in 0..h {
            // Predict
            m = mv(f, &m)?;
            let fp = mm(f, &pmat)?;
            pmat = madd(&mm(&fp, &ft)?, q)?;

            // Observation mean
            let y_mean = mv(h_mat, &m)?;
            forecasts.push(y_mean);
        }

        Ok(forecasts)
    }

    /// Forecast with prediction intervals (mean, lower 95%, upper 95%).
    ///
    /// Returns (means, lower_bounds, upper_bounds), each of shape \[h\]\[p\].
    pub fn forecast_with_intervals(
        &self,
        last_state: &[f64],
        last_cov: &[Vec<f64>],
        h: usize,
    ) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let d = self.dim_state;
        let p = self.dim_obs;
        let f = &self.f_mat;
        let h_mat = &self.h_mat;
        let ht = mt(h_mat);
        let q = &self.q_mat;
        let r = &self.r_mat;
        let ft = mt(f);
        let z95 = 1.959_963_985_f64;

        if last_state.len() != d {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: d,
                actual: last_state.len(),
            });
        }

        let mut m = last_state.to_vec();
        let mut pmat: Vec<Vec<f64>> = last_cov.to_vec();

        let mut means = Vec::with_capacity(h);
        let mut lowers = Vec::with_capacity(h);
        let mut uppers = Vec::with_capacity(h);

        for _ in 0..h {
            m = mv(f, &m)?;
            let fp = mm(f, &pmat)?;
            pmat = madd(&mm(&fp, &ft)?, q)?;

            let y_mean = mv(h_mat, &m)?;
            // S_t = H P H^T + R
            let hp = mm(h_mat, &pmat)?;
            let s_mat = madd(&mm(&hp, &ht)?, r)?;

            let low: Vec<f64> = (0..p)
                .map(|i| y_mean[i] - z95 * s_mat[i][i].max(0.0).sqrt())
                .collect();
            let up: Vec<f64> = (0..p)
                .map(|i| y_mean[i] + z95 * s_mat[i][i].max(0.0).sqrt())
                .collect();

            means.push(y_mean);
            lowers.push(low);
            uppers.push(up);
        }

        Ok((means, lowers, uppers))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn random_walk_data(n: usize, obs_noise: f64, seed_offset: f64) -> Vec<Vec<f64>> {
        // Deterministic pseudo-random walk for reproducibility
        let mut state = seed_offset;
        let mut obs = Vec::with_capacity(n);
        for i in 0..n {
            // Simple LCG-like noise
            let noise_state = ((i as f64 * 1.23 + 7.77).sin() * 0.3).max(-1.0).min(1.0);
            let noise_obs = ((i as f64 * 3.14 + 2.71).cos() * obs_noise)
                .max(-obs_noise)
                .min(obs_noise);
            state += noise_state * 0.1;
            obs.push(vec![state + noise_obs]);
        }
        obs
    }

    #[test]
    fn test_lgssm_construction() {
        let ssm = LinearGaussianSSM::new(2, 1);
        assert_eq!(ssm.dim_state, 2);
        assert_eq!(ssm.dim_obs, 1);
        assert_eq!(ssm.f_mat.len(), 2);
        assert_eq!(ssm.h_mat.len(), 1);
    }

    #[test]
    fn test_kalman_filter_random_walk() {
        // Random walk model: F=I, H=[1], Q=0.1*I, R=1.0*I
        let mut ssm = LinearGaussianSSM::new(1, 1);
        ssm.q_mat = vec![vec![0.1]];
        ssm.r_mat = vec![vec![1.0]];
        ssm.f_mat = vec![vec![1.0]];
        ssm.h_mat = vec![vec![1.0]];
        ssm.mu0 = vec![0.0];
        ssm.p0 = vec![vec![1e6]];

        let obs = random_walk_data(50, 0.5, 0.0);
        let kout = ssm.filter(&obs).expect("filter ok");

        assert_eq!(kout.filtered_means.len(), 50);
        assert_eq!(kout.filtered_covs.len(), 50);
        assert!(kout.log_likelihood.is_finite());

        // Filtered covariance should converge to Kalman steady state (< initial)
        let p_final = kout.filtered_covs[49][0][0];
        assert!(
            p_final < 1e5,
            "P should reduce from diffuse prior, got {p_final}"
        );
    }

    #[test]
    fn test_rts_smoother() {
        let mut ssm = LinearGaussianSSM::new(1, 1);
        ssm.q_mat = vec![vec![0.1]];
        ssm.r_mat = vec![vec![1.0]];

        let obs = random_walk_data(30, 0.8, 1.0);
        let (sm_means, sm_covs) = ssm.smooth(&obs).expect("smooth ok");

        assert_eq!(sm_means.len(), 30);
        assert_eq!(sm_covs.len(), 30);

        // Smoothed covariance at t=0 should be <= filtered covariance (more information)
        let kout = ssm.filter(&obs).expect("filter ok");
        let p_filt_0 = kout.filtered_covs[0][0][0];
        let p_sm_0 = sm_covs[0][0][0];
        assert!(
            p_sm_0 <= p_filt_0 + 1e-6,
            "Smoothed cov {p_sm_0} should be <= filtered cov {p_filt_0}"
        );
    }

    #[test]
    fn test_em_fit_converges() {
        let mut ssm = LinearGaussianSSM::new(1, 1);
        ssm.q_mat = vec![vec![0.5]];
        ssm.r_mat = vec![vec![0.5]];

        let obs = random_walk_data(40, 0.7, 0.5);
        let ll_hist = ssm.fit_em(&obs, 30).expect("EM ok");

        assert!(!ll_hist.is_empty());
        // Log-likelihood should be non-decreasing (monotone EM property)
        for i in 1..ll_hist.len() {
            assert!(
                ll_hist[i] >= ll_hist[i - 1] - 1e-4,
                "LL decreased at iter {i}: {} -> {}",
                ll_hist[i - 1],
                ll_hist[i]
            );
        }
    }

    #[test]
    fn test_forecast_length() {
        let ssm = LinearGaussianSSM::new(1, 1);
        let last_state = vec![1.0];
        let last_cov = vec![vec![0.1]];
        let fc = ssm
            .forecast(&last_state, &last_cov, 5)
            .expect("forecast ok");
        assert_eq!(fc.len(), 5);
        for row in &fc {
            assert_eq!(row.len(), 1);
        }
    }

    #[test]
    fn test_forecast_with_intervals() {
        let ssm = LinearGaussianSSM::new(1, 1);
        let last_state = vec![2.0];
        let last_cov = vec![vec![0.2]];
        let (means, lows, ups) = ssm
            .forecast_with_intervals(&last_state, &last_cov, 4)
            .expect("interval forecast ok");
        assert_eq!(means.len(), 4);
        for i in 0..4 {
            assert!(lows[i][0] < means[i][0]);
            assert!(ups[i][0] > means[i][0]);
        }
    }

    #[test]
    fn test_missing_observations() {
        let mut ssm = LinearGaussianSSM::new(1, 1);
        ssm.q_mat = vec![vec![0.1]];
        ssm.r_mat = vec![vec![1.0]];

        let mut obs = random_walk_data(20, 0.5, 0.0);
        // Insert NaN at positions 5 and 10
        obs[5][0] = f64::NAN;
        obs[10][0] = f64::NAN;

        let kout = ssm.filter(&obs).expect("filter with missing ok");
        assert_eq!(kout.filtered_means.len(), 20);
        // Innovation at missing time steps should be 0
        assert_eq!(kout.innovations[5][0], 0.0);
        assert_eq!(kout.innovations[10][0], 0.0);
    }
}
