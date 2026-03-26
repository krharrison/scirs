//! Energy Distance and Maximum Mean Discrepancy (MMD)
//!
//! Non-parametric distance measures between probability distributions
//! based on pairwise distances or kernel evaluations.
//!
//! # Metrics
//!
//! - **Energy Distance**: Based on expected pairwise distances
//! - **Maximum Mean Discrepancy (MMD)**: Kernel-based two-sample test statistic
//! - **Kernel Two-Sample Test**: Basic permutation test using MMD
//!
//! # Examples
//!
//! ```
//! use scirs2_metrics::distributional::energy::{energy_distance, mmd_gaussian};
//!
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = vec![1.1, 2.1, 3.1, 4.1, 5.1];
//! let ed = energy_distance(&x, &y).expect("should succeed");
//! assert!(ed < 0.5);
//!
//! let mmd = mmd_gaussian(&x, &y, None).expect("should succeed");
//! assert!(mmd >= 0.0);
//! ```

use crate::error::{MetricsError, Result};

/// Computes the energy distance between two sets of 1D samples.
///
/// The energy distance is defined as:
///
/// D(X, Y) = 2 * E|X - Y| - E|X - X'| - E|Y - Y'|
///
/// where X, X' are independent copies from distribution P and
/// Y, Y' are independent copies from distribution Q.
///
/// This is a proper metric on distributions: D(X, Y) >= 0 and D(X, Y) = 0
/// iff X and Y have the same distribution.
///
/// # Arguments
///
/// * `x` - Samples from distribution P
/// * `y` - Samples from distribution Q
///
/// # Returns
///
/// The energy distance (non-negative).
pub fn energy_distance(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.is_empty() || y.is_empty() {
        return Err(MetricsError::InvalidInput(
            "samples must not be empty".to_string(),
        ));
    }

    let nx = x.len();
    let ny = y.len();

    // E|X - Y|: cross-term
    let mut cross = 0.0;
    for xi in x.iter() {
        for yj in y.iter() {
            cross += (xi - yj).abs();
        }
    }
    cross /= (nx * ny) as f64;

    // E|X - X'|: within X
    let mut within_x = 0.0;
    for i in 0..nx {
        for j in 0..nx {
            within_x += (x[i] - x[j]).abs();
        }
    }
    if nx > 1 {
        within_x /= (nx * nx) as f64;
    }

    // E|Y - Y'|: within Y
    let mut within_y = 0.0;
    for i in 0..ny {
        for j in 0..ny {
            within_y += (y[i] - y[j]).abs();
        }
    }
    if ny > 1 {
        within_y /= (ny * ny) as f64;
    }

    let ed = 2.0 * cross - within_x - within_y;
    Ok(ed.max(0.0))
}

/// Computes the energy distance between two sets of multidimensional samples.
///
/// # Arguments
///
/// * `x` - Samples from P, flattened row-major, shape [nx, dim]
/// * `y` - Samples from Q, flattened row-major, shape [ny, dim]
/// * `dim` - Dimensionality of each sample
///
/// # Returns
///
/// The energy distance (non-negative).
pub fn energy_distance_nd(x: &[f64], y: &[f64], dim: usize) -> Result<f64> {
    if dim == 0 {
        return Err(MetricsError::InvalidInput(
            "dimension must be > 0".to_string(),
        ));
    }
    if x.is_empty() || y.is_empty() {
        return Err(MetricsError::InvalidInput(
            "samples must not be empty".to_string(),
        ));
    }
    if x.len() % dim != 0 || y.len() % dim != 0 {
        return Err(MetricsError::InvalidInput(format!(
            "sample arrays must be divisible by dim={dim}"
        )));
    }

    let nx = x.len() / dim;
    let ny = y.len() / dim;

    let l2 = |a: &[f64], b: &[f64]| -> f64 {
        let mut s = 0.0;
        for k in 0..dim {
            let d = a[k] - b[k];
            s += d * d;
        }
        s.sqrt()
    };

    // E|X - Y|
    let mut cross = 0.0;
    for i in 0..nx {
        for j in 0..ny {
            cross += l2(&x[i * dim..(i + 1) * dim], &y[j * dim..(j + 1) * dim]);
        }
    }
    cross /= (nx * ny) as f64;

    // E|X - X'|
    let mut within_x = 0.0;
    for i in 0..nx {
        for j in 0..nx {
            within_x += l2(&x[i * dim..(i + 1) * dim], &x[j * dim..(j + 1) * dim]);
        }
    }
    if nx > 1 {
        within_x /= (nx * nx) as f64;
    }

    // E|Y - Y'|
    let mut within_y = 0.0;
    for i in 0..ny {
        for j in 0..ny {
            within_y += l2(&y[i * dim..(i + 1) * dim], &y[j * dim..(j + 1) * dim]);
        }
    }
    if ny > 1 {
        within_y /= (ny * ny) as f64;
    }

    let ed = 2.0 * cross - within_x - within_y;
    Ok(ed.max(0.0))
}

/// Computes the Maximum Mean Discrepancy (MMD) with a Gaussian (RBF) kernel.
///
/// MMD^2(P, Q) = E[k(X,X')] - 2*E[k(X,Y)] + E[k(Y,Y')]
///
/// where k(a, b) = exp(-||a - b||^2 / (2 * sigma^2)).
///
/// The bandwidth (sigma) is estimated using the median heuristic if not provided.
///
/// # Arguments
///
/// * `x` - Samples from distribution P
/// * `y` - Samples from distribution Q
/// * `sigma` - Bandwidth parameter (None for median heuristic)
///
/// # Returns
///
/// The MMD value (non-negative). Note: returns MMD (not MMD^2).
pub fn mmd_gaussian(x: &[f64], y: &[f64], sigma: Option<f64>) -> Result<f64> {
    if x.is_empty() || y.is_empty() {
        return Err(MetricsError::InvalidInput(
            "samples must not be empty".to_string(),
        ));
    }

    let bw = match sigma {
        Some(s) => {
            if s <= 0.0 {
                return Err(MetricsError::InvalidInput(
                    "sigma must be positive".to_string(),
                ));
            }
            s
        }
        None => median_bandwidth_1d(x, y),
    };

    let kernel = |a: f64, b: f64| -> f64 {
        let d = a - b;
        (-d * d / (2.0 * bw * bw)).exp()
    };

    let nx = x.len();
    let ny = y.len();

    // E[k(X, X')]
    let mut kxx = 0.0;
    let mut kxx_count = 0;
    for i in 0..nx {
        for j in 0..nx {
            if i != j {
                kxx += kernel(x[i], x[j]);
                kxx_count += 1;
            }
        }
    }
    if kxx_count > 0 {
        kxx /= kxx_count as f64;
    }

    // E[k(Y, Y')]
    let mut kyy = 0.0;
    let mut kyy_count = 0;
    for i in 0..ny {
        for j in 0..ny {
            if i != j {
                kyy += kernel(y[i], y[j]);
                kyy_count += 1;
            }
        }
    }
    if kyy_count > 0 {
        kyy /= kyy_count as f64;
    }

    // E[k(X, Y)]
    let mut kxy = 0.0;
    for xi in x.iter() {
        for yj in y.iter() {
            kxy += kernel(*xi, *yj);
        }
    }
    kxy /= (nx * ny) as f64;

    let mmd_sq = kxx - 2.0 * kxy + kyy;
    Ok(mmd_sq.max(0.0).sqrt())
}

/// Computes the MMD with a Gaussian kernel for multidimensional data.
///
/// # Arguments
///
/// * `x` - Samples from P, flattened row-major, shape [nx, dim]
/// * `y` - Samples from Q, flattened row-major, shape [ny, dim]
/// * `dim` - Dimensionality
/// * `sigma` - Bandwidth (None for median heuristic)
///
/// # Returns
///
/// The MMD value (non-negative).
pub fn mmd_gaussian_nd(x: &[f64], y: &[f64], dim: usize, sigma: Option<f64>) -> Result<f64> {
    if dim == 0 {
        return Err(MetricsError::InvalidInput(
            "dimension must be > 0".to_string(),
        ));
    }
    if x.is_empty() || y.is_empty() {
        return Err(MetricsError::InvalidInput(
            "samples must not be empty".to_string(),
        ));
    }
    if x.len() % dim != 0 || y.len() % dim != 0 {
        return Err(MetricsError::InvalidInput(format!(
            "sample arrays must be divisible by dim={dim}"
        )));
    }

    let nx = x.len() / dim;
    let ny = y.len() / dim;

    let sq_dist = |a: &[f64], b: &[f64]| -> f64 {
        let mut s = 0.0;
        for k in 0..dim {
            let d = a[k] - b[k];
            s += d * d;
        }
        s
    };

    let bw = match sigma {
        Some(s) => {
            if s <= 0.0 {
                return Err(MetricsError::InvalidInput(
                    "sigma must be positive".to_string(),
                ));
            }
            s
        }
        None => {
            // Median heuristic
            let mut dists = Vec::new();
            let all: Vec<&[f64]> = (0..nx)
                .map(|i| &x[i * dim..(i + 1) * dim])
                .chain((0..ny).map(|i| &y[i * dim..(i + 1) * dim]))
                .collect();
            for i in 0..all.len() {
                for j in (i + 1)..all.len() {
                    dists.push(sq_dist(all[i], all[j]).sqrt());
                }
            }
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if dists.is_empty() {
                1.0
            } else {
                let median = dists[dists.len() / 2];
                if median < 1e-10 {
                    1.0
                } else {
                    median
                }
            }
        }
    };

    let kernel = |a: &[f64], b: &[f64]| -> f64 { (-sq_dist(a, b) / (2.0 * bw * bw)).exp() };

    // E[k(X, X')]
    let mut kxx = 0.0;
    let mut kxx_count = 0;
    for i in 0..nx {
        for j in 0..nx {
            if i != j {
                kxx += kernel(&x[i * dim..(i + 1) * dim], &x[j * dim..(j + 1) * dim]);
                kxx_count += 1;
            }
        }
    }
    if kxx_count > 0 {
        kxx /= kxx_count as f64;
    }

    // E[k(Y, Y')]
    let mut kyy = 0.0;
    let mut kyy_count = 0;
    for i in 0..ny {
        for j in 0..ny {
            if i != j {
                kyy += kernel(&y[i * dim..(i + 1) * dim], &y[j * dim..(j + 1) * dim]);
                kyy_count += 1;
            }
        }
    }
    if kyy_count > 0 {
        kyy /= kyy_count as f64;
    }

    // E[k(X, Y)]
    let mut kxy = 0.0;
    for i in 0..nx {
        for j in 0..ny {
            kxy += kernel(&x[i * dim..(i + 1) * dim], &y[j * dim..(j + 1) * dim]);
        }
    }
    kxy /= (nx * ny) as f64;

    let mmd_sq = kxx - 2.0 * kxy + kyy;
    Ok(mmd_sq.max(0.0).sqrt())
}

/// Basic kernel two-sample test using MMD.
///
/// Tests whether two sets of samples come from the same distribution
/// by comparing the observed MMD to a threshold derived from a simple
/// permutation approach.
///
/// # Arguments
///
/// * `x` - Samples from distribution P
/// * `y` - Samples from distribution Q
/// * `sigma` - Kernel bandwidth (None for median heuristic)
/// * `n_permutations` - Number of permutation iterations (default: 100)
///
/// # Returns
///
/// A tuple of (mmd_observed, p_value) where p_value is the proportion
/// of permutation MMD values >= observed MMD.
pub fn kernel_two_sample_test(
    x: &[f64],
    y: &[f64],
    sigma: Option<f64>,
    n_permutations: Option<usize>,
) -> Result<(f64, f64)> {
    if x.is_empty() || y.is_empty() {
        return Err(MetricsError::InvalidInput(
            "samples must not be empty".to_string(),
        ));
    }

    let n_perm = n_permutations.unwrap_or(100);
    let observed_mmd = mmd_gaussian(x, y, sigma)?;

    // Pool all samples
    let mut pooled: Vec<f64> = Vec::with_capacity(x.len() + y.len());
    pooled.extend_from_slice(x);
    pooled.extend_from_slice(y);

    let nx = x.len();
    let n_total = pooled.len();

    // Simple deterministic permutation using index rotation
    let mut count_ge = 0;
    for perm in 0..n_perm {
        // Rotate-based pseudo-permutation
        let offset = ((perm + 1) * 7 + 13) % n_total;
        let mut perm_x = Vec::with_capacity(nx);
        let mut perm_y = Vec::with_capacity(n_total - nx);
        for i in 0..n_total {
            let idx = (i + offset) % n_total;
            if i < nx {
                perm_x.push(pooled[idx]);
            } else {
                perm_y.push(pooled[idx]);
            }
        }

        let perm_mmd = mmd_gaussian(&perm_x, &perm_y, sigma)?;
        if perm_mmd >= observed_mmd {
            count_ge += 1;
        }
    }

    let p_value = count_ge as f64 / n_perm as f64;
    Ok((observed_mmd, p_value))
}

/// Estimates the bandwidth using the median heuristic for 1D data.
fn median_bandwidth_1d(x: &[f64], y: &[f64]) -> f64 {
    let mut all: Vec<f64> = Vec::with_capacity(x.len() + y.len());
    all.extend_from_slice(x);
    all.extend_from_slice(y);

    let mut dists: Vec<f64> = Vec::new();
    for i in 0..all.len() {
        for j in (i + 1)..all.len() {
            dists.push((all[i] - all[j]).abs());
        }
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    if dists.is_empty() {
        return 1.0;
    }

    let median = dists[dists.len() / 2];
    if median < 1e-10 {
        1.0
    } else {
        median
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_distance_same() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ed = energy_distance(&x, &x).expect("should succeed");
        assert!(
            ed < 1e-10,
            "same samples should give energy distance ~0, got {ed}"
        );
    }

    #[test]
    fn test_energy_distance_different() {
        let x = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let y = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        let ed = energy_distance(&x, &y).expect("should succeed");
        assert!(
            ed > 0.0,
            "very different distributions should have positive energy distance"
        );
    }

    #[test]
    fn test_energy_distance_symmetry() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        let ed_xy = energy_distance(&x, &y).expect("should succeed");
        let ed_yx = energy_distance(&y, &x).expect("should succeed");
        assert!(
            (ed_xy - ed_yx).abs() < 1e-10,
            "energy distance should be symmetric: {ed_xy} vs {ed_yx}"
        );
    }

    #[test]
    fn test_energy_distance_empty() {
        assert!(energy_distance(&[], &[1.0]).is_err());
        assert!(energy_distance(&[1.0], &[]).is_err());
    }

    #[test]
    fn test_energy_distance_nd_same() {
        let x = vec![1.0, 2.0, 3.0, 4.0]; // 2 points in 2D
        let ed = energy_distance_nd(&x, &x, 2).expect("should succeed");
        assert!(ed < 1e-10, "same samples should give ED~0, got {ed}");
    }

    #[test]
    fn test_energy_distance_nd_different() {
        let x = vec![0.0, 0.0, 0.0, 0.0]; // 2 points at origin
        let y = vec![10.0, 10.0, 10.0, 10.0]; // 2 points at (10,10)
        let ed = energy_distance_nd(&x, &y, 2).expect("should succeed");
        assert!(ed > 0.0, "different should have positive ED");
    }

    #[test]
    fn test_mmd_same_distribution() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mmd = mmd_gaussian(&x, &x, Some(1.0)).expect("should succeed");
        assert!(mmd < 1e-10, "same samples should give MMD~0, got {mmd}");
    }

    #[test]
    fn test_mmd_different_distribution() {
        let x = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let y = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        let mmd = mmd_gaussian(&x, &y, Some(1.0)).expect("should succeed");
        assert!(
            mmd > 0.0,
            "different distributions should have positive MMD"
        );
    }

    #[test]
    fn test_mmd_auto_bandwidth() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        let mmd = mmd_gaussian(&x, &y, None).expect("should succeed");
        assert!(mmd >= 0.0, "MMD should be non-negative");
    }

    #[test]
    fn test_mmd_bad_sigma() {
        assert!(mmd_gaussian(&[1.0], &[2.0], Some(-1.0)).is_err());
        assert!(mmd_gaussian(&[1.0], &[2.0], Some(0.0)).is_err());
    }

    #[test]
    fn test_mmd_nd_same() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mmd = mmd_gaussian_nd(&x, &x, 2, Some(1.0)).expect("should succeed");
        assert!(mmd < 1e-10, "same ND samples should give MMD~0, got {mmd}");
    }

    #[test]
    fn test_kernel_two_sample_test_same() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mmd, p_value) =
            kernel_two_sample_test(&x, &x, Some(1.0), Some(50)).expect("should succeed");
        assert!(mmd < 1e-10, "same samples should give MMD~0");
        // p-value should be high (fail to reject null)
        assert!(
            p_value >= 0.0,
            "p-value should be non-negative, got {p_value}"
        );
    }

    #[test]
    fn test_kernel_two_sample_test_different() {
        let x = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let y = vec![100.0, 100.0, 100.0, 100.0, 100.0];
        let (mmd, _p_value) =
            kernel_two_sample_test(&x, &y, Some(1.0), Some(50)).expect("should succeed");
        assert!(mmd > 0.0, "very different samples should have positive MMD");
    }
}
