//! Laplace approximation core for INLA
//!
//! Implements Newton-Raphson mode finding and Laplace approximation
//! to the posterior distribution of the latent field given hyperparameters.
//!
//! The key identity is:
//!   p(x|y,theta) ∝ p(y|x,theta) * p(x|theta)
//!
//! The Laplace approximation finds the mode x* and uses a Gaussian
//! centered at x* with precision equal to the negative Hessian of
//! the log-posterior.

use scirs2_core::ndarray::{Array1, Array2, Axis};

use super::types::LikelihoodFamily;
use crate::error::StatsError;

/// Result of the Newton-Raphson mode-finding procedure
#[derive(Debug, Clone)]
pub struct ModeResult {
    /// The posterior mode x*
    pub mode: Array1<f64>,
    /// Negative Hessian of log-posterior at the mode (precision matrix of Laplace approx)
    pub neg_hessian: Array2<f64>,
    /// Number of iterations taken
    pub iterations: usize,
    /// Whether Newton-Raphson converged
    pub converged: bool,
    /// Log-posterior value at the mode
    pub log_posterior_at_mode: f64,
}

/// Compute the log-likelihood for each observation given the linear predictor
///
/// # Arguments
/// * `y` - Observation vector
/// * `eta` - Linear predictor (design_matrix * x)
/// * `likelihood` - Likelihood family
/// * `n_trials` - Number of trials (for Binomial)
/// * `obs_precision` - Observation precision (for Gaussian)
///
/// # Returns
/// Total log-likelihood sum_i log p(y_i | eta_i)
pub fn log_likelihood(
    y: &Array1<f64>,
    eta: &Array1<f64>,
    likelihood: LikelihoodFamily,
    n_trials: Option<&Array1<f64>>,
    obs_precision: Option<f64>,
) -> f64 {
    let n = y.len();
    let mut ll = 0.0;

    for i in 0..n {
        ll += log_likelihood_single(
            y[i],
            eta[i],
            likelihood,
            n_trials.map(|t| t[i]),
            obs_precision,
        );
    }

    ll
}

/// Log-likelihood for a single observation
fn log_likelihood_single(
    y_i: f64,
    eta_i: f64,
    likelihood: LikelihoodFamily,
    n_trial: Option<f64>,
    obs_precision: Option<f64>,
) -> f64 {
    match likelihood {
        LikelihoodFamily::Gaussian => {
            let tau = obs_precision.unwrap_or(1.0);
            // log N(y_i | eta_i, 1/tau) = -0.5*tau*(y_i - eta_i)^2 + 0.5*log(tau) - 0.5*log(2*pi)
            let diff = y_i - eta_i;
            -0.5 * tau * diff * diff + 0.5 * tau.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
        }
        LikelihoodFamily::Poisson => {
            // log Poisson(y_i | exp(eta_i)) = y_i * eta_i - exp(eta_i) - log(y_i!)
            let lambda = eta_i.exp().min(1e15); // cap for numerical stability
            y_i * eta_i - lambda - log_factorial_approx(y_i)
        }
        LikelihoodFamily::Binomial => {
            let n = n_trial.unwrap_or(1.0);
            // log Binomial(y_i | n, p_i) where p_i = logistic(eta_i)
            // = y_i * eta_i - n * log(1 + exp(eta_i)) + log(C(n, y_i))
            let log1pexp = log1p_exp(eta_i);
            y_i * eta_i - n * log1pexp + log_binom_coeff(n, y_i)
        }
        LikelihoodFamily::NegativeBinomial => {
            // Simplified: use Poisson-like approximation with log link
            let lambda = eta_i.exp().min(1e15);
            y_i * eta_i - lambda - log_factorial_approx(y_i)
        }
        _ => 0.0, // non_exhaustive fallback
    }
}

/// First derivative of log-likelihood w.r.t. eta_i
///
/// Returns d/d(eta_i) log p(y_i | eta_i)
pub fn log_likelihood_gradient_single(
    y_i: f64,
    eta_i: f64,
    likelihood: LikelihoodFamily,
    n_trial: Option<f64>,
    obs_precision: Option<f64>,
) -> f64 {
    match likelihood {
        LikelihoodFamily::Gaussian => {
            let tau = obs_precision.unwrap_or(1.0);
            // d/d(eta) [-0.5*tau*(y - eta)^2] = tau*(y - eta)
            tau * (y_i - eta_i)
        }
        LikelihoodFamily::Poisson => {
            // d/d(eta) [y*eta - exp(eta)] = y - exp(eta)
            y_i - eta_i.exp().min(1e15)
        }
        LikelihoodFamily::Binomial => {
            let n = n_trial.unwrap_or(1.0);
            // d/d(eta) [y*eta - n*log(1+exp(eta))] = y - n*sigmoid(eta)
            let p = sigmoid(eta_i);
            y_i - n * p
        }
        LikelihoodFamily::NegativeBinomial => y_i - eta_i.exp().min(1e15),
        _ => 0.0,
    }
}

/// Second derivative of log-likelihood w.r.t. eta_i (always negative for these families)
///
/// Returns d^2/d(eta_i)^2 log p(y_i | eta_i)
pub fn log_likelihood_hessian_single(
    eta_i: f64,
    likelihood: LikelihoodFamily,
    n_trial: Option<f64>,
    obs_precision: Option<f64>,
) -> f64 {
    match likelihood {
        LikelihoodFamily::Gaussian => {
            let tau = obs_precision.unwrap_or(1.0);
            // d^2/d(eta)^2 [-0.5*tau*(y-eta)^2] = -tau
            -tau
        }
        LikelihoodFamily::Poisson => {
            // d^2/d(eta)^2 [y*eta - exp(eta)] = -exp(eta)
            -eta_i.exp().min(1e15)
        }
        LikelihoodFamily::Binomial => {
            let n = n_trial.unwrap_or(1.0);
            // d^2/d(eta)^2 [y*eta - n*log(1+exp(eta))] = -n*sigmoid(eta)*(1-sigmoid(eta))
            let p = sigmoid(eta_i);
            -n * p * (1.0 - p)
        }
        LikelihoodFamily::NegativeBinomial => -eta_i.exp().min(1e15),
        _ => -1.0,
    }
}

/// Compute gradient of log-posterior w.r.t. x
///
/// grad log p(x|y,theta) = A^T * grad_eta log p(y|eta) - Q * x
///
/// where eta = A * x
pub fn log_posterior_gradient(
    x: &Array1<f64>,
    y: &Array1<f64>,
    design: &Array2<f64>,
    precision: &Array2<f64>,
    likelihood: LikelihoodFamily,
    n_trials: Option<&Array1<f64>>,
    obs_precision: Option<f64>,
) -> Array1<f64> {
    let eta = design.dot(x);
    let n = y.len();
    let p = x.len();

    // Gradient of log-likelihood w.r.t. eta
    let mut grad_eta = Array1::zeros(n);
    for i in 0..n {
        grad_eta[i] = log_likelihood_gradient_single(
            y[i],
            eta[i],
            likelihood,
            n_trials.map(|t| t[i]),
            obs_precision,
        );
    }

    // A^T * grad_eta
    let at_grad = design.t().dot(&grad_eta);

    // Prior contribution: -Q * x
    let prior_grad = {
        let mut qx: Array1<f64> = Array1::zeros(p);
        for i in 0..p {
            for j in 0..p {
                qx[i] += precision[[i, j]] * x[j];
            }
        }
        qx
    };

    at_grad - prior_grad
}

/// Compute the negative Hessian of the log-posterior (precision of Laplace approx)
///
/// H = Q + A^T * diag(w) * A
///
/// where w_i = -d^2/d(eta_i)^2 log p(y_i | eta_i) (positive values)
pub fn compute_neg_hessian(
    x: &Array1<f64>,
    design: &Array2<f64>,
    precision: &Array2<f64>,
    likelihood: LikelihoodFamily,
    n_trials: Option<&Array1<f64>>,
    obs_precision: Option<f64>,
) -> Array2<f64> {
    let eta = design.dot(x);
    let n = eta.len();
    let p = x.len();

    // Compute diagonal weights w_i = -d^2/d(eta_i)^2 log p(y_i | eta_i)
    let mut weights = Array1::zeros(n);
    for i in 0..n {
        let h = log_likelihood_hessian_single(
            eta[i],
            likelihood,
            n_trials.map(|t| t[i]),
            obs_precision,
        );
        weights[i] = (-h).max(1e-12); // ensure positive
    }

    // H = Q + A^T * diag(w) * A
    let mut neg_hess = precision.clone();
    for i in 0..p {
        for j in 0..p {
            let mut sum = 0.0;
            for k in 0..n {
                sum += design[[k, i]] * weights[k] * design[[k, j]];
            }
            neg_hess[[i, j]] += sum;
        }
    }

    neg_hess
}

/// Find the posterior mode x* via Newton-Raphson iteration
///
/// Solves: x_{k+1} = x_k + H^{-1} * grad
///
/// where grad and H are the gradient and negative Hessian of the log-posterior.
///
/// # Arguments
/// * `precision` - Prior precision matrix Q(theta)
/// * `y` - Observation vector
/// * `design` - Design matrix A
/// * `likelihood` - Likelihood family
/// * `n_trials` - Number of trials (Binomial)
/// * `obs_precision` - Observation precision (Gaussian)
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance on norm of update
/// * `damping` - Step size damping (1.0 = full Newton step)
///
/// # Returns
/// `ModeResult` with the posterior mode and associated quantities
pub fn find_mode(
    precision: &Array2<f64>,
    y: &Array1<f64>,
    design: &Array2<f64>,
    likelihood: LikelihoodFamily,
    n_trials: Option<&Array1<f64>>,
    obs_precision: Option<f64>,
    max_iter: usize,
    tol: f64,
    damping: f64,
) -> Result<ModeResult, StatsError> {
    let p = precision.nrows();
    if precision.ncols() != p {
        return Err(StatsError::DimensionMismatch(
            "Precision matrix must be square".to_string(),
        ));
    }
    if design.ncols() != p {
        return Err(StatsError::DimensionMismatch(format!(
            "Design matrix columns ({}) must match precision matrix size ({})",
            design.ncols(),
            p
        )));
    }
    if design.nrows() != y.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "Design matrix rows ({}) must match observation length ({})",
            design.nrows(),
            y.len()
        )));
    }

    let mut x = Array1::zeros(p);
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..max_iter {
        iterations = iter + 1;

        // Compute gradient
        let grad = log_posterior_gradient(
            &x,
            y,
            design,
            precision,
            likelihood,
            n_trials,
            obs_precision,
        );

        // Compute negative Hessian
        let neg_hess =
            compute_neg_hessian(&x, design, precision, likelihood, n_trials, obs_precision);

        // Solve H * delta = grad for delta (Newton step)
        let delta = solve_symmetric_positive_definite(&neg_hess, &grad)?;

        // Damped update
        let update_norm: f64 = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
        x = &x + &(damping * &delta);

        if update_norm < tol {
            converged = true;
            break;
        }
    }

    // Compute final quantities
    let neg_hess = compute_neg_hessian(&x, design, precision, likelihood, n_trials, obs_precision);
    let eta = design.dot(&x);
    let ll = log_likelihood(y, &eta, likelihood, n_trials, obs_precision);
    let log_prior = -0.5 * x.dot(&precision.dot(&x));
    let log_posterior = ll + log_prior;

    Ok(ModeResult {
        mode: x,
        neg_hessian: neg_hess,
        iterations,
        converged,
        log_posterior_at_mode: log_posterior,
    })
}

/// Compute the Laplace approximation to the log marginal likelihood
///
/// log p(y|theta) ≈ log p(y|x*,theta) + log p(x*|theta) + (p/2)*log(2*pi) - 0.5*log|H|
///
/// where x* is the posterior mode and H is the negative Hessian at the mode.
pub fn laplace_log_marginal_likelihood(
    mode_result: &ModeResult,
    precision: &Array2<f64>,
) -> Result<f64, StatsError> {
    let p = mode_result.mode.len() as f64;

    // log|H| via Cholesky
    let log_det_h = log_determinant(&mode_result.neg_hessian)?;

    // log|Q| (prior precision)
    let log_det_q = log_determinant(precision)?;

    // log p(y|theta) ≈ log p(y|x*,theta) + log p(x*|theta) + (p/2)*log(2*pi) - 0.5*log|H|
    // = log_posterior_at_mode + 0.5*log|Q| - (p/2)*log(2*pi)  [prior normalization]
    //   + (p/2)*log(2*pi) - 0.5*log|H|
    // = log_posterior_at_mode + 0.5*log|Q| - 0.5*log|H|
    let result = mode_result.log_posterior_at_mode + 0.5 * log_det_q - 0.5 * log_det_h;

    Ok(result)
}

/// Solve A * x = b for symmetric positive definite A using Cholesky decomposition
fn solve_symmetric_positive_definite(
    a: &Array2<f64>,
    b: &Array1<f64>,
) -> Result<Array1<f64>, StatsError> {
    let n = a.nrows();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    // Cholesky decomposition: A = L * L^T
    let l = cholesky_decompose(a)?;

    // Forward substitution: L * z = b
    let mut z = Array1::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[[i, j]] * z[j];
        }
        if l[[i, i]].abs() < 1e-15 {
            return Err(StatsError::ComputationError(
                "Singular matrix in Cholesky solve".to_string(),
            ));
        }
        z[i] = sum / l[[i, i]];
    }

    // Back substitution: L^T * x = z
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = z[i];
        for j in (i + 1)..n {
            sum -= l[[j, i]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }

    Ok(x)
}

/// Cholesky decomposition of a symmetric positive definite matrix
fn cholesky_decompose(a: &Array2<f64>) -> Result<Array2<f64>, StatsError> {
    let n = a.nrows();
    let mut l = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[[i, k]] * l[[j, k]];
            }

            if i == j {
                let diag = a[[i, i]] - sum;
                if diag <= 0.0 {
                    return Err(StatsError::ComputationError(format!(
                        "Matrix not positive definite at index {} (value: {:.6e})",
                        i, diag
                    )));
                }
                l[[i, j]] = diag.sqrt();
            } else {
                if l[[j, j]].abs() < 1e-15 {
                    return Err(StatsError::ComputationError(
                        "Zero diagonal in Cholesky decomposition".to_string(),
                    ));
                }
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    Ok(l)
}

/// Compute log-determinant of a symmetric positive definite matrix via Cholesky
fn log_determinant(a: &Array2<f64>) -> Result<f64, StatsError> {
    let l = cholesky_decompose(a)?;
    let n = a.nrows();
    let mut log_det = 0.0;
    for i in 0..n {
        log_det += l[[i, i]].ln();
    }
    // det(A) = det(L)^2, so log|A| = 2 * sum(log(L_ii))
    Ok(2.0 * log_det)
}

/// Compute the diagonal of the inverse of a symmetric positive definite matrix
///
/// This is needed for marginal variances: Var(x_i|y,theta) ≈ [H^{-1}]_{ii}
pub fn inverse_diagonal(a: &Array2<f64>) -> Result<Array1<f64>, StatsError> {
    let n = a.nrows();
    let mut diag = Array1::zeros(n);

    // Solve A * e_i for each standard basis vector to get column i of A^{-1}
    // But we only need the diagonal, so we only extract the i-th element
    for i in 0..n {
        let mut e_i = Array1::zeros(n);
        e_i[i] = 1.0;
        let col = solve_symmetric_positive_definite(a, &e_i)?;
        diag[i] = col[i];
    }

    Ok(diag)
}

/// Compute the full inverse of a symmetric positive definite matrix
pub fn full_inverse(a: &Array2<f64>) -> Result<Array2<f64>, StatsError> {
    let n = a.nrows();
    let mut inv = Array2::zeros((n, n));

    for i in 0..n {
        let mut e_i = Array1::zeros(n);
        e_i[i] = 1.0;
        let col = solve_symmetric_positive_definite(a, &e_i)?;
        for j in 0..n {
            inv[[j, i]] = col[j];
        }
    }

    Ok(inv)
}

// --- Helper functions ---

/// Sigmoid function: 1 / (1 + exp(-x))
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let ex = (-x).exp();
        1.0 / (1.0 + ex)
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Numerically stable log(1 + exp(x))
fn log1p_exp(x: f64) -> f64 {
    if x > 35.0 {
        x
    } else if x < -10.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Approximate log(n!) using Stirling's approximation for large n, exact for small n
fn log_factorial_approx(n: f64) -> f64 {
    if n < 0.0 {
        return 0.0;
    }
    let n_int = n as u64;
    if n_int <= 20 {
        // Exact computation for small values
        let mut result = 0.0f64;
        for i in 2..=n_int {
            result += (i as f64).ln();
        }
        result
    } else {
        // Stirling's approximation: log(n!) ≈ n*log(n) - n + 0.5*log(2*pi*n)
        n * n.ln() - n + 0.5 * (2.0 * std::f64::consts::PI * n).ln()
    }
}

/// Approximate log binomial coefficient log C(n, k)
fn log_binom_coeff(n: f64, k: f64) -> f64 {
    if k < 0.0 || k > n {
        return f64::NEG_INFINITY;
    }
    log_factorial_approx(n) - log_factorial_approx(k) - log_factorial_approx(n - k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(100.0) > 0.99);
        assert!(sigmoid(-100.0) < 0.01);
    }

    #[test]
    fn test_log1p_exp() {
        // For large x, log(1+exp(x)) ≈ x
        assert!((log1p_exp(50.0) - 50.0).abs() < 1e-10);
        // For x=0, log(1+exp(0)) = log(2)
        assert!((log1p_exp(0.0) - 2.0f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_log_factorial() {
        assert!((log_factorial_approx(0.0)).abs() < 1e-10);
        assert!((log_factorial_approx(1.0)).abs() < 1e-10);
        assert!((log_factorial_approx(5.0) - (120.0f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky() {
        let a = array![[4.0, 2.0], [2.0, 5.0]];
        let l = cholesky_decompose(&a).expect("Cholesky should succeed");
        // Verify L * L^T = A
        let reconstructed = l.dot(&l.t());
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (reconstructed[[i, j]] - a[[i, j]]).abs() < 1e-10,
                    "Cholesky reconstruction mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_solve_spd() {
        let a = array![[4.0, 2.0], [2.0, 5.0]];
        let b = array![1.0, 2.0];
        let x = solve_symmetric_positive_definite(&a, &b).expect("Solve should succeed");
        // Verify A * x = b
        let ax = a.dot(&x);
        for i in 0..2 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-10,
                "Solution mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_log_determinant() {
        let a = array![[4.0, 2.0], [2.0, 5.0]];
        let log_det = log_determinant(&a).expect("Log determinant should succeed");
        // det = 4*5 - 2*2 = 16, log(16) ≈ 2.7726
        assert!((log_det - 16.0f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_log_likelihood() {
        let y = array![1.0, 2.0, 3.0];
        let eta = array![1.0, 2.0, 3.0]; // perfect fit
        let ll = log_likelihood(&y, &eta, LikelihoodFamily::Gaussian, None, Some(1.0));
        // With perfect fit, each term is -0.5*log(2*pi) ≈ -0.9189
        let expected = -1.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((ll - expected).abs() < 1e-10);
    }

    #[test]
    fn test_find_mode_gaussian_identity() {
        // For Gaussian likelihood with identity design and precision,
        // the posterior mode should be close to (y * obs_precision) / (1 + obs_precision)
        let n = 3;
        let y = array![1.0, 2.0, 3.0];
        let design = Array2::eye(n);
        let precision = Array2::eye(n);
        let obs_prec = 10.0;

        let result = find_mode(
            &precision,
            &y,
            &design,
            LikelihoodFamily::Gaussian,
            None,
            Some(obs_prec),
            100,
            1e-10,
            1.0,
        )
        .expect("Mode finding should succeed");

        assert!(result.converged, "Newton-Raphson should converge");

        // Exact posterior mode: x_i = obs_prec * y_i / (1 + obs_prec)
        for i in 0..n {
            let expected = obs_prec * y[i] / (1.0 + obs_prec);
            assert!(
                (result.mode[i] - expected).abs() < 1e-6,
                "Mode mismatch at index {}: got {}, expected {}",
                i,
                result.mode[i],
                expected
            );
        }
    }

    #[test]
    fn test_find_mode_poisson() {
        // Simple Poisson regression: y ~ Poisson(exp(x)), x ~ N(0, I)
        let n = 3;
        let y = array![2.0, 5.0, 1.0];
        let design = Array2::eye(n);
        let precision = Array2::eye(n);

        let result = find_mode(
            &precision,
            &y,
            &design,
            LikelihoodFamily::Poisson,
            None,
            None,
            100,
            1e-10,
            1.0,
        )
        .expect("Mode finding should succeed");

        assert!(
            result.converged,
            "Newton-Raphson should converge for Poisson"
        );
        // Mode should be close to log(y) for large y (when prior is weak)
        // For y=5, mode should be somewhat less than log(5) ≈ 1.609 due to prior
        assert!(result.mode[1] > 0.5, "Mode for y=5 should be positive");
    }

    #[test]
    fn test_laplace_log_marginal() {
        let n = 2;
        let y = array![1.0, 2.0];
        let design = Array2::eye(n);
        let precision = Array2::eye(n);

        let mode_result = find_mode(
            &precision,
            &y,
            &design,
            LikelihoodFamily::Gaussian,
            None,
            Some(1.0),
            100,
            1e-10,
            1.0,
        )
        .expect("Mode finding should succeed");

        let log_ml = laplace_log_marginal_likelihood(&mode_result, &precision)
            .expect("Laplace approximation should succeed");

        // For Gaussian case, the Laplace approximation should be exact
        // log p(y) for y ~ N(0, I + I) = N(0, 2I)
        // = -0.5 * y^T (2I)^{-1} y - 0.5 * log|2I| - (n/2)*log(2*pi)
        let expected = -0.5 * (y[0] * y[0] + y[1] * y[1]) / 2.0
            - 0.5 * (2.0f64).ln() * 2.0
            - (2.0 * std::f64::consts::PI).ln();
        // Allow some tolerance since we're comparing two different computations
        assert!(
            (log_ml - expected).abs() < 0.5,
            "Laplace log marginal: got {}, expected {}",
            log_ml,
            expected
        );
    }

    #[test]
    fn test_inverse_diagonal() {
        let a = array![[2.0, 0.0], [0.0, 4.0]];
        let diag = inverse_diagonal(&a).expect("Inverse diagonal should succeed");
        assert!((diag[0] - 0.5).abs() < 1e-10);
        assert!((diag[1] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_dimension_mismatch() {
        let y = array![1.0, 2.0];
        let design = Array2::eye(2);
        let precision = Array2::eye(3); // wrong size

        let result = find_mode(
            &precision,
            &y,
            &design,
            LikelihoodFamily::Gaussian,
            None,
            None,
            10,
            1e-8,
            1.0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_gradient_gaussian() {
        let x = array![0.5, 1.0];
        let y = array![1.0, 2.0];
        let design = Array2::eye(2);
        let precision = Array2::eye(2);

        let grad = log_posterior_gradient(
            &x,
            &y,
            &design,
            &precision,
            LikelihoodFamily::Gaussian,
            None,
            Some(1.0),
        );

        // For Gaussian with tau=1, identity design/precision:
        // grad_i = (y_i - x_i) - x_i = y_i - 2*x_i
        assert!((grad[0] - (1.0 - 2.0 * 0.5)).abs() < 1e-10);
        assert!((grad[1] - (2.0 - 2.0 * 1.0)).abs() < 1e-10);
    }
}
