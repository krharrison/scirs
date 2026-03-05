//! Spatial Point Process Analysis
//!
//! Provides tools for analysing and testing spatial point patterns:
//!
//! - **CSR envelope**: Monte-Carlo envelope test against Complete Spatial Randomness
//! - **Kernel intensity**: Gaussian kernel density estimation of point intensity
//! - **G-function**: Nearest-neighbour distance distribution function
//! - **F-function**: Empty-space (void) function (contact distribution)
//!
//! # References
//! - Diggle, P.J. (2003). *Statistical Analysis of Spatial Point Patterns*, 2nd ed.
//! - Ripley, B.D. (1981). *Spatial Statistics*.
//! - Baddeley, A., Rubak, E. & Turner, R. (2015). *Spatial Point Patterns*.

use super::{SpatialError, SpatialResult};
use super::autocorrelation::ripleys_k;

// ---------------------------------------------------------------------------
// Pseudo-random number generator (Xoshiro256** — no external dependency)
// ---------------------------------------------------------------------------

/// Simple 256-bit xoshiro256** PRNG.
struct Xoshiro256pp {
    s: [u64; 4],
}

impl Xoshiro256pp {
    fn from_seed(seed: u64) -> Self {
        // SplitMix64 for seeding
        let mut x = seed;
        let mut next = || {
            x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        let s = [next(), next(), next(), next()];
        Self { s }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = self.s[1]
            .wrapping_mul(5)
            .rotate_left(7)
            .wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform sample in `[0, 1)`.
    #[inline]
    fn next_f64(&mut self) -> f64 {
        // Use upper 53 bits
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform sample in `[lo, hi)`.
    #[inline]
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }
}

// ---------------------------------------------------------------------------
// CSR Envelope
// ---------------------------------------------------------------------------

/// Monte-Carlo envelope for the Complete Spatial Randomness (CSR) test.
///
/// Contains the lower and upper pointwise simulation envelopes and the observed K
/// function for comparison.
#[derive(Debug, Clone)]
pub struct CsrEnvelope {
    /// Lower bound of the simulation envelope at each distance.
    pub lower: Vec<f64>,
    /// Upper bound of the simulation envelope at each distance.
    pub upper: Vec<f64>,
    /// Observed K-function values at each distance.
    pub observed_k: Vec<f64>,
}

/// Compute a Monte-Carlo simulation envelope for the K function under CSR.
///
/// Generates `n_simulations` independent Poisson processes inside the axis-aligned
/// bounding box derived from `points`, computes K for each, and records the
/// pointwise minimum (lower) and maximum (upper) envelopes.
///
/// # Arguments
/// * `points`       – Observed `(x, y)` point coordinates.
/// * `area`         – Area of the study region.
/// * `distances`    – Distance thresholds at which to evaluate K.
/// * `n_simulations`– Number of simulated CSR realisations.
/// * `rng_seed`     – Seed for the internal PRNG.
///
/// # Errors
/// Returns [`SpatialError`] when the observed K computation fails.
pub fn csr_envelope(
    points: &[(f64, f64)],
    area: f64,
    distances: &[f64],
    n_simulations: usize,
    rng_seed: u64,
) -> SpatialResult<CsrEnvelope> {
    if points.len() < 2 {
        return Err(SpatialError::InsufficientData(
            "CSR envelope requires at least 2 observed points".to_string(),
        ));
    }
    if area <= 0.0 {
        return Err(SpatialError::InvalidArgument(
            "area must be positive".to_string(),
        ));
    }

    let observed_k = ripleys_k(points, area, distances)?;

    if distances.is_empty() || n_simulations == 0 {
        return Ok(CsrEnvelope {
            lower: observed_k.clone(),
            upper: observed_k.clone(),
            observed_k,
        });
    }

    // Bounding box for simulations
    let (mut x_min, mut x_max, mut y_min, mut y_max) = (
        points[0].0, points[0].0,
        points[0].1, points[0].1,
    );
    for &(x, y) in points.iter().skip(1) {
        if x < x_min { x_min = x; }
        if x > x_max { x_max = x; }
        if y < y_min { y_min = y; }
        if y > y_max { y_max = y; }
    }
    let n_pts = points.len();
    let nd = distances.len();

    let mut lower = vec![f64::INFINITY; nd];
    let mut upper = vec![f64::NEG_INFINITY; nd];

    let mut rng = Xoshiro256pp::from_seed(rng_seed);

    for _ in 0..n_simulations {
        let sim_pts: Vec<(f64, f64)> = (0..n_pts)
            .map(|_| (rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)))
            .collect();

        let sim_k = match ripleys_k(&sim_pts, area, distances) {
            Ok(k) => k,
            Err(_) => continue, // skip degenerate simulations
        };

        for d_idx in 0..nd {
            if sim_k[d_idx] < lower[d_idx] {
                lower[d_idx] = sim_k[d_idx];
            }
            if sim_k[d_idx] > upper[d_idx] {
                upper[d_idx] = sim_k[d_idx];
            }
        }
    }

    // Replace any remaining infinities (all simulations may have been skipped)
    for d_idx in 0..nd {
        if lower[d_idx].is_infinite() {
            lower[d_idx] = observed_k[d_idx];
        }
        if upper[d_idx].is_infinite() {
            upper[d_idx] = observed_k[d_idx];
        }
    }

    Ok(CsrEnvelope {
        lower,
        upper,
        observed_k,
    })
}

// ---------------------------------------------------------------------------
// Kernel intensity estimation
// ---------------------------------------------------------------------------

/// Estimate the spatial intensity of a point process at a set of evaluation
/// points using an isotropic Gaussian kernel.
///
/// The estimated intensity at location `u` is:
/// ```text
/// λ̂(u) = Σ_i K_h(u - x_i)
/// ```
/// where `K_h` is the 2-D Gaussian kernel with bandwidth `h`:
/// ```text
/// K_h(x) = (1/(2π h²)) exp(-|x|²/(2h²))
/// ```
///
/// # Arguments
/// * `points`      – Observed point locations.
/// * `eval_points` – Locations at which to evaluate the intensity.
/// * `bandwidth`   – Gaussian kernel bandwidth `h > 0`.
///
/// # Returns
/// Vector of intensity estimates, one per evaluation point.
pub fn kernel_intensity(
    points: &[(f64, f64)],
    eval_points: &[(f64, f64)],
    bandwidth: f64,
) -> Vec<f64> {
    if points.is_empty() || eval_points.is_empty() || bandwidth <= 0.0 {
        return vec![0.0; eval_points.len()];
    }
    let h2 = bandwidth * bandwidth;
    let norm_const = 1.0 / (2.0 * std::f64::consts::PI * h2);

    eval_points
        .iter()
        .map(|&(ux, uy)| {
            points.iter().fold(0.0_f64, |acc, &(px, py)| {
                let dx = ux - px;
                let dy = uy - py;
                let r2 = dx * dx + dy * dy;
                acc + norm_const * (-r2 / (2.0 * h2)).exp()
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// G-function (nearest-neighbour distance distribution)
// ---------------------------------------------------------------------------

/// Compute the empirical G-function (nearest-neighbour distance CDF).
///
/// `G(r) = P(distance to nearest neighbour ≤ r)`
///
/// For CSR, `G(r) = 1 - exp(-λ π r²)` where `λ = n/area`.
///
/// # Arguments
/// * `points`    – Observed point locations.
/// * `distances` – Distance thresholds at which to evaluate G.
///
/// # Returns
/// Vector of `Ĝ(d)` values in `[0, 1]`.
pub fn g_function(points: &[(f64, f64)], distances: &[f64]) -> Vec<f64> {
    let n = points.len();
    if n == 0 || distances.is_empty() {
        return vec![0.0; distances.len()];
    }

    // For each point compute its nearest-neighbour distance
    let nn_dists: Vec<f64> = (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let dx = points[i].0 - points[j].0;
                    let dy = points[i].1 - points[j].1;
                    (dx * dx + dy * dy).sqrt()
                })
                .fold(f64::INFINITY, f64::min)
        })
        .collect();

    // For each distance threshold count proportion of nn_dists ≤ d
    distances
        .iter()
        .map(|&d| {
            let count = nn_dists.iter().filter(|&&nn| nn <= d).count();
            count as f64 / n as f64
        })
        .collect()
}

// ---------------------------------------------------------------------------
// F-function (empty-space / contact distribution function)
// ---------------------------------------------------------------------------

/// Compute the empirical F-function (empty-space function).
///
/// `F(r) = P(distance from random test point to nearest observed point ≤ r)`
///
/// For CSR with intensity `λ`, `F(r) ≈ G(r) = 1 - exp(-λ π r²)`.
///
/// # Arguments
/// * `points`        – Observed point locations.
/// * `area`          – Area of the study region (used to determine bbox if
///                     the bbox is degenerate).
/// * `distances`     – Distance thresholds.
/// * `n_test_points` – Number of random test points to use.
/// * `rng_seed`      – Seed for the internal PRNG.
///
/// # Returns
/// Vector of `F̂(d)` values in `[0, 1]`.
pub fn f_function(
    points: &[(f64, f64)],
    area: f64,
    distances: &[f64],
    n_test_points: usize,
    rng_seed: u64,
) -> Vec<f64> {
    if points.is_empty() || distances.is_empty() || n_test_points == 0 {
        return vec![0.0; distances.len()];
    }

    // Determine bounding box
    let (mut x_min, mut x_max, mut y_min, mut y_max) =
        (points[0].0, points[0].0, points[0].1, points[0].1);
    for &(x, y) in points.iter().skip(1) {
        if x < x_min { x_min = x; }
        if x > x_max { x_max = x; }
        if y < y_min { y_min = y; }
        if y > y_max { y_max = y; }
    }

    // If bbox is degenerate use a square derived from the area
    if (x_max - x_min) < 1e-12 || (y_max - y_min) < 1e-12 {
        let half = area.sqrt() / 2.0;
        x_min -= half;
        x_max += half;
        y_min -= half;
        y_max += half;
    }

    let mut rng = Xoshiro256pp::from_seed(rng_seed);

    // Generate random test points inside the bounding box
    let test_pts: Vec<(f64, f64)> = (0..n_test_points)
        .map(|_| (rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)))
        .collect();

    // For each test point compute minimum distance to any observed point
    let min_dists: Vec<f64> = test_pts
        .iter()
        .map(|&(tx, ty)| {
            points.iter().fold(f64::INFINITY, |acc, &(px, py)| {
                let d = ((tx - px).powi(2) + (ty - py).powi(2)).sqrt();
                acc.min(d)
            })
        })
        .collect();

    distances
        .iter()
        .map(|&d| {
            let count = min_dists.iter().filter(|&&m| m <= d).count();
            count as f64 / n_test_points as f64
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn regular_grid(side: usize) -> Vec<(f64, f64)> {
        let mut pts = Vec::with_capacity(side * side);
        for i in 0..side {
            for j in 0..side {
                pts.push((i as f64 + 0.5, j as f64 + 0.5));
            }
        }
        pts
    }

    #[test]
    fn test_csr_envelope_structure() {
        let pts = regular_grid(5);
        let area = 25.0_f64;
        let distances = vec![0.5, 1.0, 1.5, 2.0];
        let env = csr_envelope(&pts, area, &distances, 50, 42)
            .expect("csr_envelope failed");

        assert_eq!(env.lower.len(), distances.len());
        assert_eq!(env.upper.len(), distances.len());
        assert_eq!(env.observed_k.len(), distances.len());

        // Lower ≤ upper for each distance
        for (l, u) in env.lower.iter().zip(env.upper.iter()) {
            assert!(l <= u, "lower {} > upper {}", l, u);
        }
    }

    #[test]
    fn test_csr_envelope_monotone_k() {
        let pts = regular_grid(5);
        let area = 25.0_f64;
        let distances = vec![0.5, 1.0, 1.5, 2.0];
        let env = csr_envelope(&pts, area, &distances, 30, 7)
            .expect("csr_envelope failed");

        // Observed K should be non-decreasing in d
        for w in env.observed_k.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-9,
                "K not monotone: {} > {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_csr_envelope_too_few_points() {
        let pts = vec![(0.0_f64, 0.0_f64)];
        assert!(csr_envelope(&pts, 1.0, &[0.5], 10, 1).is_err());
    }

    #[test]
    fn test_kernel_intensity_positive() {
        let pts = vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)];
        let eval = vec![(1.5, 1.5), (5.0, 5.0)];
        let intensities = kernel_intensity(&pts, &eval, 0.5);
        assert_eq!(intensities.len(), 2);
        // Intensity near clustered points should be larger
        assert!(
            intensities[0] > intensities[1],
            "intensity near cluster should be higher"
        );
    }

    #[test]
    fn test_kernel_intensity_zero_bandwidth() {
        let pts = vec![(1.0, 1.0), (2.0, 2.0)];
        let eval = vec![(1.5, 1.5)];
        let intensities = kernel_intensity(&pts, &eval, 0.0);
        // bandwidth ≤ 0 → return zeros
        assert_eq!(intensities, vec![0.0]);
    }

    #[test]
    fn test_kernel_intensity_symmetry() {
        let pts = vec![(0.0, 0.0)];
        let eval = vec![(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
        let intensities = kernel_intensity(&pts, &eval, 0.5);
        // All four equidistant points should have identical intensity
        for i in 1..intensities.len() {
            assert!(
                (intensities[0] - intensities[i]).abs() < 1e-12,
                "symmetry violated"
            );
        }
    }

    #[test]
    fn test_g_function_monotone() {
        let pts: Vec<(f64, f64)> = (0..10)
            .map(|i| (i as f64, i as f64))
            .collect();
        let distances = vec![0.5, 1.0, 2.0, 4.0, 8.0];
        let g = g_function(&pts, &distances);
        assert_eq!(g.len(), distances.len());
        for w in g.windows(2) {
            assert!(w[1] >= w[0] - 1e-10, "G-function not monotone");
        }
    }

    #[test]
    fn test_g_function_bounds() {
        let pts: Vec<(f64, f64)> = (0..5).map(|i| (i as f64, 0.0)).collect();
        let distances = vec![0.1, 1.0, 100.0];
        let g = g_function(&pts, &distances);
        for &v in &g {
            assert!((0.0..=1.0).contains(&v), "G value {} out of [0,1]", v);
        }
        // At very large distance all NN distances should be covered
        assert_eq!(g[2], 1.0, "G(100) should be 1.0");
    }

    #[test]
    fn test_g_function_empty() {
        let g = g_function(&[], &[1.0, 2.0]);
        assert_eq!(g, vec![0.0, 0.0]);
    }

    #[test]
    fn test_f_function_monotone() {
        let pts = regular_grid(5);
        let distances = vec![0.25, 0.5, 1.0, 2.0, 4.0];
        let f = f_function(&pts, 25.0, &distances, 500, 99);
        assert_eq!(f.len(), distances.len());
        for w in f.windows(2) {
            assert!(w[1] >= w[0] - 1e-10, "F-function not monotone");
        }
    }

    #[test]
    fn test_f_function_bounds() {
        let pts = regular_grid(4);
        let distances = vec![0.1, 0.5, 1.0, 100.0];
        let f = f_function(&pts, 16.0, &distances, 200, 77);
        for &v in &f {
            assert!((0.0..=1.0).contains(&v), "F value {} out of [0,1]", v);
        }
        assert_eq!(f[3], 1.0, "F(100) should be 1.0");
    }

    #[test]
    fn test_f_function_empty_points() {
        let f = f_function(&[], 1.0, &[1.0], 100, 1);
        assert_eq!(f, vec![0.0]);
    }
}
