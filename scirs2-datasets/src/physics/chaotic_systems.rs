//! Chaotic map and attractor dataset generators.
//!
//! Provides discrete-time maps (Ikeda, Hénon, Logistic) and continuous-time
//! attractor diagnostics (Lyapunov exponent, Mandelbrot set).
//!
//! # Maps and utilities
//!
//! | Function | Description |
//! |---|---|
//! | [`ikeda_map`] | Complex-plane Ikeda map |
//! | [`henon_map`] | Hénon map |
//! | [`logistic_map`] | Logistic map |
//! | [`bifurcation_diagram`] | Logistic map bifurcation diagram |
//! | [`lyapunov_exponent_lorenz`] | Max Lyapunov exponent of Lorenz 63 |
//! | [`mandelbrot_set`] | Mandelbrot set iteration counts |

use crate::error::{DatasetsError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Ikeda map
// ─────────────────────────────────────────────────────────────────────────────

/// Iterate the Ikeda map in the complex plane.
///
/// The map is defined as:
/// ```text
/// t_{n+1}     = k - p / (1 + |z_n|²)
/// z_{n+1}     = a + b · z_n · exp(i · t_{n+1})
/// ```
/// where `z_n = (x_n, y_n)` is treated as a complex number.
///
/// # Parameters
/// - `u`      — `b` parameter (classic chaotic value ≈ 0.9)
/// - `n_steps` — number of iterations to record
/// - `x0`, `y0` — initial position
///
/// # Returns
/// Vector of `(x, y)` positions, length `n_steps + 1` (includes the initial point).
pub fn ikeda_map(u: f64, n_steps: usize, x0: f64, y0: f64) -> Result<Vec<(f64, f64)>> {
    if !(0.0..=1.0).contains(&u) {
        return Err(DatasetsError::InvalidFormat(
            "ikeda_map: u must be in [0, 1]".into(),
        ));
    }
    // Classic parameters
    let a = 1.0f64;
    let b = u;
    let k = 0.4f64;
    let p = 6.0f64;

    let mut pts = Vec::with_capacity(n_steps + 1);
    let mut x = x0;
    let mut y = y0;
    pts.push((x, y));

    for _ in 0..n_steps {
        let r2 = x * x + y * y;
        let t = k - p / (1.0 + r2);
        let (sin_t, cos_t) = t.sin_cos();
        let nx = a + b * (x * cos_t - y * sin_t);
        let ny = b * (x * sin_t + y * cos_t);
        x = nx;
        y = ny;
        pts.push((x, y));
    }
    Ok(pts)
}

// ─────────────────────────────────────────────────────────────────────────────
// Hénon map
// ─────────────────────────────────────────────────────────────────────────────

/// Iterate the Hénon map.
///
/// ```text
/// x_{n+1} = 1 - a · x_n² + y_n
/// y_{n+1} = b · x_n
/// ```
///
/// Classic chaotic parameters: `a = 1.4`, `b = 0.3`.
///
/// # Returns
/// Vector of `(x, y)` positions, length `n_steps + 1` (includes initial point).
pub fn henon_map(
    a: f64,
    b: f64,
    n_steps: usize,
    x0: f64,
    y0: f64,
) -> Result<Vec<(f64, f64)>> {
    let mut pts = Vec::with_capacity(n_steps + 1);
    let mut x = x0;
    let mut y = y0;
    pts.push((x, y));

    for _ in 0..n_steps {
        let nx = 1.0 - a * x * x + y;
        let ny = b * x;
        x = nx;
        y = ny;

        if !x.is_finite() || !y.is_finite() {
            // Trajectory escaped to infinity — stop iterating.
            break;
        }
        pts.push((x, y));
    }
    Ok(pts)
}

// ─────────────────────────────────────────────────────────────────────────────
// Logistic map
// ─────────────────────────────────────────────────────────────────────────────

/// Iterate the logistic map.
///
/// ```text
/// x_{n+1} = r · x_n · (1 - x_n)
/// ```
///
/// # Parameters
/// - `r`       — growth rate, typically in `[0, 4]`
/// - `n_steps` — number of iterations to record
/// - `x0`      — initial condition, must be in `(0, 1)`
///
/// # Returns
/// Vector of `x` values, length `n_steps + 1` (includes initial point).
pub fn logistic_map(r: f64, n_steps: usize, x0: f64) -> Result<Vec<f64>> {
    if !(0.0..=4.0).contains(&r) {
        return Err(DatasetsError::InvalidFormat(
            "logistic_map: r must be in [0, 4]".into(),
        ));
    }
    if !(0.0..=1.0).contains(&x0) {
        return Err(DatasetsError::InvalidFormat(
            "logistic_map: x0 must be in [0, 1]".into(),
        ));
    }

    let mut vals = Vec::with_capacity(n_steps + 1);
    let mut x = x0;
    vals.push(x);
    for _ in 0..n_steps {
        x = r * x * (1.0 - x);
        vals.push(x);
    }
    Ok(vals)
}

// ─────────────────────────────────────────────────────────────────────────────
// Bifurcation diagram
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a bifurcation diagram for the logistic map.
///
/// For each `r` value in `[r_min, r_max]`, the map is run for `n_burn` steps
/// (discarded as transient) then `n_keep` attractor points are collected.
///
/// # Returns
/// Vector of `(r, x)` pairs for plotting.
pub fn bifurcation_diagram(
    r_min: f64,
    r_max: f64,
    n_r: usize,
    n_burn: usize,
    n_keep: usize,
) -> Result<Vec<(f64, f64)>> {
    if r_min >= r_max {
        return Err(DatasetsError::InvalidFormat(
            "bifurcation_diagram: r_min must be < r_max".into(),
        ));
    }
    if !(0.0..=4.0).contains(&r_max) || r_min < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "bifurcation_diagram: r range must be within [0, 4]".into(),
        ));
    }
    if n_r == 0 || n_keep == 0 {
        return Err(DatasetsError::InvalidFormat(
            "bifurcation_diagram: n_r and n_keep must be > 0".into(),
        ));
    }

    let mut pts = Vec::with_capacity(n_r * n_keep);
    for i in 0..n_r {
        let r = r_min + (r_max - r_min) * (i as f64) / (n_r.saturating_sub(1).max(1) as f64);
        let mut x = 0.5;
        // Burn-in
        for _ in 0..n_burn {
            x = r * x * (1.0 - x);
        }
        // Collect attractor points
        for _ in 0..n_keep {
            x = r * x * (1.0 - x);
            pts.push((r, x));
        }
    }
    Ok(pts)
}

// ─────────────────────────────────────────────────────────────────────────────
// Lyapunov exponent of Lorenz 63
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate the maximum Lyapunov exponent of the Lorenz-63 system via
/// tangent-space evolution (linearised dynamics alongside the nonlinear orbit).
///
/// The algorithm integrates the reference trajectory and a perturbed copy;
/// periodically the separation is measured, accumulated into the exponent
/// estimate, and the perturbed trajectory is rescaled back to `epsilon`
/// distance from the reference.
///
/// # Parameters
/// - `sigma`, `rho`, `beta` — Lorenz parameters
/// - `t_end` — total integration time
/// - `dt`    — step size
///
/// # Returns
/// Estimated maximum Lyapunov exponent (bits/s in natural log units).
pub fn lyapunov_exponent_lorenz(
    sigma: f64,
    rho: f64,
    beta: f64,
    t_end: f64,
    dt: f64,
) -> Result<f64> {
    if dt <= 0.0 || t_end <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "lyapunov_exponent_lorenz: dt and t_end must be > 0".into(),
        ));
    }

    let eps = 1e-8;
    let n_steps = (t_end / dt) as usize;

    // Reference trajectory
    let mut x = [1.0f64, 1.0, 1.0];
    // Perturbed trajectory
    let mut xp = [1.0 + eps, 1.0, 1.0];

    let lorenz_rhs = |y: &[f64; 3]| -> [f64; 3] {
        [
            sigma * (y[1] - y[0]),
            y[0] * (rho - y[2]) - y[1],
            y[0] * y[1] - beta * y[2],
        ]
    };

    let rk4_step = |y: &[f64; 3], h: f64, rhs: &dyn Fn(&[f64; 3]) -> [f64; 3]| -> [f64; 3] {
        let k1 = rhs(y);
        let y2 = [y[0] + 0.5 * h * k1[0], y[1] + 0.5 * h * k1[1], y[2] + 0.5 * h * k1[2]];
        let k2 = rhs(&y2);
        let y3 = [y[0] + 0.5 * h * k2[0], y[1] + 0.5 * h * k2[1], y[2] + 0.5 * h * k2[2]];
        let k3 = rhs(&y3);
        let y4 = [y[0] + h * k3[0], y[1] + h * k3[1], y[2] + h * k3[2]];
        let k4 = rhs(&y4);
        [
            y[0] + h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
            y[1] + h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
            y[2] + h / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
        ]
    };

    let mut lyap_sum = 0.0f64;
    let renorm_every = 10usize;

    for step in 0..n_steps {
        x = rk4_step(&x, dt, &lorenz_rhs);
        xp = rk4_step(&xp, dt, &lorenz_rhs);

        if (step + 1) % renorm_every == 0 {
            let dist = ((xp[0] - x[0]).powi(2)
                + (xp[1] - x[1]).powi(2)
                + (xp[2] - x[2]).powi(2))
            .sqrt();
            if dist > 0.0 {
                lyap_sum += dist.ln();
                let scale = eps / dist;
                xp[0] = x[0] + (xp[0] - x[0]) * scale;
                xp[1] = x[1] + (xp[1] - x[1]) * scale;
                xp[2] = x[2] + (xp[2] - x[2]) * scale;
            }
        }
    }

    let n_renorms = n_steps / renorm_every;
    if n_renorms == 0 {
        return Err(DatasetsError::InvalidFormat(
            "lyapunov_exponent_lorenz: not enough steps to estimate exponent".into(),
        ));
    }
    // Normalise to per-unit-time by dividing by accumulated time
    let lyap = (lyap_sum - (eps.ln() * n_renorms as f64)) / (renorm_every as f64 * dt * n_renorms as f64);
    Ok(lyap)
}

// ─────────────────────────────────────────────────────────────────────────────
// Mandelbrot set
// ─────────────────────────────────────────────────────────────────────────────

/// Compute iteration counts for the Mandelbrot set over a rectangular grid.
///
/// For each pixel `(ix, iy)` the complex number `c` is chosen such that the
/// pixel grid spans `[center.0 ± scale/2, center.1 ± scale/2]`.  The
/// iteration `z ← z² + c` is run until `|z| > 2` (escaped) or `max_iter`
/// steps are reached.  A count of `max_iter` means the point is considered
/// *inside* the set.
///
/// # Parameters
/// - `width`, `height` — pixel dimensions
/// - `max_iter`        — maximum iterations per pixel
/// - `center`          — `(re, im)` centre of the view
/// - `scale`           — width of the view (height is scaled proportionally)
///
/// # Returns
/// `Vec<Vec<u32>>` with shape `[height][width]`.
pub fn mandelbrot_set(
    width: usize,
    height: usize,
    max_iter: usize,
    center: (f64, f64),
    scale: f64,
) -> Result<Vec<Vec<u32>>> {
    if width == 0 || height == 0 {
        return Err(DatasetsError::InvalidFormat(
            "mandelbrot_set: width and height must be > 0".into(),
        ));
    }
    if scale <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "mandelbrot_set: scale must be > 0".into(),
        ));
    }
    if max_iter == 0 {
        return Err(DatasetsError::InvalidFormat(
            "mandelbrot_set: max_iter must be > 0".into(),
        ));
    }

    let aspect = height as f64 / width as f64;
    let re_min = center.0 - scale / 2.0;
    let re_max = center.0 + scale / 2.0;
    let im_min = center.1 - scale * aspect / 2.0;
    let im_max = center.1 + scale * aspect / 2.0;

    let mut grid = vec![vec![0u32; width]; height];
    for iy in 0..height {
        let im = im_min + (im_max - im_min) * (iy as f64) / (height as f64 - 1.0).max(1.0);
        for ix in 0..width {
            let re = re_min + (re_max - re_min) * (ix as f64) / (width as f64 - 1.0).max(1.0);
            let (mut zr, mut zi) = (0.0f64, 0.0f64);
            let mut count = 0u32;
            while count < max_iter as u32 {
                let zr2 = zr * zr;
                let zi2 = zi * zi;
                if zr2 + zi2 > 4.0 {
                    break;
                }
                zi = 2.0 * zr * zi + im;
                zr = zr2 - zi2 + re;
                count += 1;
            }
            grid[iy][ix] = count;
        }
    }
    Ok(grid)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ikeda_map_length() {
        let pts = ikeda_map(0.9, 100, 0.0, 0.0).expect("valid params");
        assert_eq!(pts.len(), 101);
    }

    #[test]
    fn test_ikeda_map_invalid_u() {
        assert!(ikeda_map(1.5, 10, 0.0, 0.0).is_err());
    }

    #[test]
    fn test_henon_map_classic_attractor_bounded() {
        // Classic Hénon attractor should stay bounded.
        let pts = henon_map(1.4, 0.3, 5000, 0.0, 0.0).expect("valid params");
        for (x, y) in &pts {
            assert!(x.abs() < 2.0, "x out of bounds: {x}");
            assert!(y.abs() < 1.0, "y out of bounds: {y}");
        }
    }

    #[test]
    fn test_logistic_map_length() {
        let v = logistic_map(3.9, 200, 0.5).expect("valid params");
        assert_eq!(v.len(), 201);
    }

    #[test]
    fn test_logistic_map_range() {
        let v = logistic_map(3.9, 500, 0.3).expect("valid params");
        for &x in &v {
            assert!(x >= 0.0 && x <= 1.0, "x out of [0,1]: {x}");
        }
    }

    #[test]
    fn test_logistic_map_invalid_r() {
        assert!(logistic_map(5.0, 10, 0.5).is_err());
    }

    #[test]
    fn test_bifurcation_diagram_count() {
        let pts = bifurcation_diagram(3.0, 4.0, 10, 100, 50).expect("valid params");
        assert_eq!(pts.len(), 10 * 50);
    }

    #[test]
    fn test_lyapunov_lorenz_positive() {
        // Lorenz 63 with classic params should have a positive max Lyapunov exponent.
        let lya = lyapunov_exponent_lorenz(10.0, 28.0, 8.0 / 3.0, 50.0, 0.01).expect("valid params");
        assert!(lya > 0.0, "expected positive Lyapunov exponent, got {lya}");
    }

    #[test]
    fn test_mandelbrot_set_shape() {
        let grid = mandelbrot_set(64, 48, 100, (-0.5, 0.0), 3.0).expect("valid params");
        assert_eq!(grid.len(), 48);
        assert_eq!(grid[0].len(), 64);
    }

    #[test]
    fn test_mandelbrot_origin_inside() {
        // z=0, c=0 should never escape → iteration count equals max_iter.
        let grid = mandelbrot_set(3, 3, 100, (0.0, 0.0), 0.001).expect("valid params");
        // Centre pixel should reach max_iter.
        assert_eq!(grid[1][1], 100);
    }

    #[test]
    fn test_mandelbrot_invalid_params() {
        assert!(mandelbrot_set(0, 10, 10, (0.0, 0.0), 1.0).is_err());
        assert!(mandelbrot_set(10, 10, 10, (0.0, 0.0), -1.0).is_err());
    }
}
