//! 2D Kernel Density Estimation on spatial data
//!
//! Provides grid-based and point-at-location density estimation using
//! Gaussian, Epanechnikov, or quartic kernels with automatic bandwidth
//! selection (Silverman's rule, Scott's rule).

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

use crate::error::{SpatialError, SpatialResult};

/// Kernel function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelType {
    /// Gaussian kernel: K(u) = (1/(2pi)) * exp(-u^2/2)
    Gaussian,
    /// Epanechnikov kernel: K(u) = 3/4 * (1 - u^2)  for |u| <= 1
    Epanechnikov,
    /// Quartic (biweight) kernel: K(u) = 15/16 * (1 - u^2)^2  for |u| <= 1
    Quartic,
}

/// Bandwidth selection method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandwidthMethod {
    /// Silverman's rule of thumb: h = 0.9 * min(std, IQR/1.34) * n^{-1/5}
    Silverman,
    /// Scott's rule: h = std * n^{-1/(d+4)}
    Scott,
}

/// Configuration for spatial KDE.
#[derive(Debug, Clone)]
pub struct SpatialKdeConfig {
    /// Kernel type (default: Gaussian).
    pub kernel: KernelType,
    /// Bandwidth in x-direction. If None, auto-select.
    pub bandwidth_x: Option<f64>,
    /// Bandwidth in y-direction. If None, auto-select.
    pub bandwidth_y: Option<f64>,
    /// Bandwidth selection method when bandwidth is None.
    pub bandwidth_method: BandwidthMethod,
    /// Grid resolution in x (number of cells). Default 50.
    pub grid_nx: usize,
    /// Grid resolution in y (number of cells). Default 50.
    pub grid_ny: usize,
}

impl Default for SpatialKdeConfig {
    fn default() -> Self {
        Self {
            kernel: KernelType::Gaussian,
            bandwidth_x: None,
            bandwidth_y: None,
            bandwidth_method: BandwidthMethod::Silverman,
            grid_nx: 50,
            grid_ny: 50,
        }
    }
}

/// Result of grid-based KDE evaluation.
#[derive(Debug, Clone)]
pub struct KdeGrid {
    /// Density values on the grid (grid_ny x grid_nx).
    pub density: Array2<f64>,
    /// X coordinates of the grid cell centres (grid_nx,).
    pub x_coords: Array1<f64>,
    /// Y coordinates of the grid cell centres (grid_ny,).
    pub y_coords: Array1<f64>,
    /// Bandwidth used in x.
    pub bandwidth_x: f64,
    /// Bandwidth used in y.
    pub bandwidth_y: f64,
}

// ---------------------------------------------------------------------------
// Bandwidth selection
// ---------------------------------------------------------------------------

/// Select bandwidth for one dimension using Silverman's or Scott's rule.
///
/// Returns `(hx, hy)`.
pub fn select_bandwidth(
    coordinates: &ArrayView2<f64>,
    method: BandwidthMethod,
) -> SpatialResult<(f64, f64)> {
    let n = coordinates.nrows();
    if n < 2 {
        return Err(SpatialError::ValueError(
            "Need at least 2 points for bandwidth selection".to_string(),
        ));
    }
    if coordinates.ncols() < 2 {
        return Err(SpatialError::DimensionError(
            "Coordinates must have at least 2 columns (x, y)".to_string(),
        ));
    }

    let nf = n as f64;

    let col_x: Vec<f64> = coordinates.column(0).iter().copied().collect();
    let col_y: Vec<f64> = coordinates.column(1).iter().copied().collect();
    let hx = bandwidth_1d(&col_x, nf, method);
    let hy = bandwidth_1d(&col_y, nf, method);

    if hx <= 0.0 || hy <= 0.0 {
        return Err(SpatialError::ValueError(
            "Computed bandwidth is non-positive; data may have zero variance".to_string(),
        ));
    }

    Ok((hx, hy))
}

fn bandwidth_1d(data: &[f64], nf: f64, method: BandwidthMethod) -> f64 {
    let mean: f64 = data.iter().sum::<f64>() / nf;
    let var: f64 = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / nf;
    let std = var.sqrt();

    match method {
        BandwidthMethod::Silverman => {
            let iqr = interquartile_range(data);
            let spread = std.min(iqr / 1.34);
            let spread = if spread > 0.0 { spread } else { std };
            0.9 * spread * nf.powf(-0.2)
        }
        BandwidthMethod::Scott => {
            // d=2 for spatial data => n^{-1/(2+4)} = n^{-1/6}
            std * nf.powf(-1.0 / 6.0)
        }
    }
}

fn interquartile_range(data: &[f64]) -> f64 {
    if data.len() < 4 {
        // Fallback: use range
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        return max - min;
    }
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let q1 = sorted[n / 4];
    let q3 = sorted[3 * n / 4];
    q3 - q1
}

// ---------------------------------------------------------------------------
// Kernel evaluation
// ---------------------------------------------------------------------------

fn kernel_eval(u_sq: f64, kernel: KernelType) -> f64 {
    match kernel {
        KernelType::Gaussian => {
            // 2D Gaussian: (1/(2pi)) * exp(-u^2/2)
            (1.0 / (2.0 * std::f64::consts::PI)) * (-0.5 * u_sq).exp()
        }
        KernelType::Epanechnikov => {
            if u_sq <= 1.0 {
                // 2D Epanechnikov: (2/pi) * (1 - u^2)
                (2.0 / std::f64::consts::PI) * (1.0 - u_sq)
            } else {
                0.0
            }
        }
        KernelType::Quartic => {
            if u_sq <= 1.0 {
                // 2D quartic: (3/pi) * (1 - u^2)^2
                let t = 1.0 - u_sq;
                (3.0 / std::f64::consts::PI) * t * t
            } else {
                0.0
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Grid-based KDE
// ---------------------------------------------------------------------------

/// Compute 2D KDE on a regular grid over the data extent (plus a small margin).
///
/// The grid spans `[xmin - 3*hx, xmax + 3*hx]` x `[ymin - 3*hy, ymax + 3*hy]`.
pub fn kde_on_grid(
    coordinates: &ArrayView2<f64>,
    config: &SpatialKdeConfig,
) -> SpatialResult<KdeGrid> {
    let n = coordinates.nrows();
    if n == 0 {
        return Err(SpatialError::ValueError("No data points".to_string()));
    }
    if coordinates.ncols() < 2 {
        return Err(SpatialError::DimensionError(
            "Coordinates must be 2D".to_string(),
        ));
    }

    // Bandwidths
    let (hx, hy) = match (config.bandwidth_x, config.bandwidth_y) {
        (Some(bx), Some(by)) => {
            if bx <= 0.0 || by <= 0.0 {
                return Err(SpatialError::ValueError(
                    "Bandwidths must be positive".to_string(),
                ));
            }
            (bx, by)
        }
        _ => {
            let (auto_hx, auto_hy) = select_bandwidth(coordinates, config.bandwidth_method)?;
            (
                config.bandwidth_x.unwrap_or(auto_hx),
                config.bandwidth_y.unwrap_or(auto_hy),
            )
        }
    };

    // Data extent
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;

    for i in 0..n {
        let x = coordinates[[i, 0]];
        let y = coordinates[[i, 1]];
        if x < xmin {
            xmin = x;
        }
        if x > xmax {
            xmax = x;
        }
        if y < ymin {
            ymin = y;
        }
        if y > ymax {
            ymax = y;
        }
    }

    let margin_x = 3.0 * hx;
    let margin_y = 3.0 * hy;
    xmin -= margin_x;
    xmax += margin_x;
    ymin -= margin_y;
    ymax += margin_y;

    let nx = config.grid_nx.max(2);
    let ny = config.grid_ny.max(2);

    let dx = (xmax - xmin) / (nx as f64 - 1.0);
    let dy = (ymax - ymin) / (ny as f64 - 1.0);

    let x_coords = Array1::from_shape_fn(nx, |i| xmin + i as f64 * dx);
    let y_coords = Array1::from_shape_fn(ny, |j| ymin + j as f64 * dy);

    let nf = n as f64;
    let mut density = Array2::zeros((ny, nx));

    for j in 0..ny {
        let gy = y_coords[j];
        for i in 0..nx {
            let gx = x_coords[i];

            let mut sum = 0.0;
            for k in 0..n {
                let ux = (gx - coordinates[[k, 0]]) / hx;
                let uy = (gy - coordinates[[k, 1]]) / hy;
                let u_sq = ux * ux + uy * uy;
                sum += kernel_eval(u_sq, config.kernel);
            }

            density[[j, i]] = sum / (nf * hx * hy);
        }
    }

    Ok(KdeGrid {
        density,
        x_coords,
        y_coords,
        bandwidth_x: hx,
        bandwidth_y: hy,
    })
}

// ---------------------------------------------------------------------------
// Point-at-location KDE
// ---------------------------------------------------------------------------

/// Evaluate KDE density at a single query point.
pub fn kde_at_point(
    coordinates: &ArrayView2<f64>,
    query: &[f64; 2],
    hx: f64,
    hy: f64,
    kernel: KernelType,
) -> SpatialResult<f64> {
    let n = coordinates.nrows();
    if n == 0 {
        return Err(SpatialError::ValueError("No data points".to_string()));
    }
    if hx <= 0.0 || hy <= 0.0 {
        return Err(SpatialError::ValueError(
            "Bandwidths must be positive".to_string(),
        ));
    }

    let nf = n as f64;
    let mut sum = 0.0;
    for k in 0..n {
        let ux = (query[0] - coordinates[[k, 0]]) / hx;
        let uy = (query[1] - coordinates[[k, 1]]) / hy;
        let u_sq = ux * ux + uy * uy;
        sum += kernel_eval(u_sq, kernel);
    }

    Ok(sum / (nf * hx * hy))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_kde_peak_at_concentration() {
        // Points concentrated around (0, 0) with one outlier
        let coords = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [-0.1, 0.0],
            [0.0, -0.1],
            [5.0, 5.0], // outlier
        ];

        let config = SpatialKdeConfig {
            kernel: KernelType::Gaussian,
            bandwidth_x: Some(0.5),
            bandwidth_y: Some(0.5),
            grid_nx: 20,
            grid_ny: 20,
            ..Default::default()
        };

        let grid = kde_on_grid(&coords.view(), &config).expect("kde_on_grid");

        // Density at (0,0) should be higher than at (5,5) area
        let d_origin = kde_at_point(&coords.view(), &[0.0, 0.0], 0.5, 0.5, KernelType::Gaussian)
            .expect("point kde");
        let d_far = kde_at_point(&coords.view(), &[5.0, 5.0], 0.5, 0.5, KernelType::Gaussian)
            .expect("point kde");

        assert!(
            d_origin > d_far,
            "Density at concentration ({}) should exceed outlier density ({})",
            d_origin,
            d_far
        );
    }

    #[test]
    fn test_kde_integrates_approximately_to_one() {
        // Small cluster of points
        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],];

        let config = SpatialKdeConfig {
            kernel: KernelType::Gaussian,
            bandwidth_x: Some(0.5),
            bandwidth_y: Some(0.5),
            grid_nx: 80,
            grid_ny: 80,
            ..Default::default()
        };

        let grid = kde_on_grid(&coords.view(), &config).expect("kde_on_grid");

        // Numerical integration: sum(density) * dx * dy
        let dx = (grid.x_coords[grid.x_coords.len() - 1] - grid.x_coords[0])
            / (grid.x_coords.len() as f64 - 1.0);
        let dy = (grid.y_coords[grid.y_coords.len() - 1] - grid.y_coords[0])
            / (grid.y_coords.len() as f64 - 1.0);

        let integral: f64 = grid.density.sum() * dx * dy;

        // Should be approximately 1 (Gaussian tails are cut off at +-3h margin)
        assert!(
            (integral - 1.0).abs() < 0.15,
            "KDE integral = {}, expected ~1.0",
            integral
        );
    }

    #[test]
    fn test_bandwidth_selection_silverman() {
        let coords = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.3, 0.7],
        ];

        let (hx, hy) =
            select_bandwidth(&coords.view(), BandwidthMethod::Silverman).expect("bandwidth");
        assert!(hx > 0.0, "hx should be positive");
        assert!(hy > 0.0, "hy should be positive");
    }

    #[test]
    fn test_bandwidth_selection_scott() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5],];

        let (hx, hy) = select_bandwidth(&coords.view(), BandwidthMethod::Scott).expect("bw");
        assert!(hx > 0.0);
        assert!(hy > 0.0);
    }

    #[test]
    fn test_epanechnikov_kernel() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],];

        let config = SpatialKdeConfig {
            kernel: KernelType::Epanechnikov,
            bandwidth_x: Some(2.0),
            bandwidth_y: Some(2.0),
            grid_nx: 10,
            grid_ny: 10,
            ..Default::default()
        };

        let grid = kde_on_grid(&coords.view(), &config).expect("epanechnikov kde");

        // All density values should be non-negative
        for &d in grid.density.iter() {
            assert!(d >= 0.0, "density should be non-negative");
        }
    }

    #[test]
    fn test_quartic_kernel() {
        let coords = array![[0.0, 0.0], [0.5, 0.5],];

        let d = kde_at_point(&coords.view(), &[0.25, 0.25], 1.0, 1.0, KernelType::Quartic)
            .expect("quartic");
        assert!(d > 0.0, "quartic density should be positive near data");

        // Far away => zero for compact kernel
        let d_far = kde_at_point(
            &coords.view(),
            &[100.0, 100.0],
            1.0,
            1.0,
            KernelType::Quartic,
        )
        .expect("quartic far");
        assert!(d_far < 1e-15, "quartic density should be ~0 far from data");
    }

    #[test]
    fn test_kde_errors() {
        let empty: Array2<f64> = Array2::zeros((0, 2));
        let config = SpatialKdeConfig::default();
        assert!(kde_on_grid(&empty.view(), &config).is_err());

        let single = array![[0.0, 0.0]];
        assert!(
            kde_at_point(&single.view(), &[0.0, 0.0], -1.0, 1.0, KernelType::Gaussian).is_err()
        );
    }
}
