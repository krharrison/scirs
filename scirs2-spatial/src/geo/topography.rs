//! Terrain analysis from Digital Elevation Models (DEMs).
//!
//! This module provides standard terrain-analysis algorithms operating on
//! 2-D grids of elevation data (DEMs):
//!
//! - **`slope`** — first-derivative surface slope (degrees)
//! - **`aspect`** — surface aspect / strike direction (degrees)
//! - **`hillshade`** — analytical hillshade for visualisation
//! - **`viewshed`** — line-of-sight visibility from an observer
//! - **`flow_direction`** — D8 steepest-descent flow direction
//! - **`flow_accumulation`** — upstream drainage area
//! - **`watershed_delineation`** — watershed (catchment) mask from a pour point

use crate::error::{SpatialError, SpatialResult};
use std::f64::consts::PI;

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Safely index into a 2-D row-major array.
#[inline]
fn get(dem: &[f64], rows: usize, cols: usize, r: usize, c: usize) -> f64 {
    dem[r * cols + c]
}

/// Clamp-index (returns edge value for out-of-range).
#[inline]
fn get_clamped(dem: &[f64], rows: usize, cols: usize, r: isize, c: isize) -> f64 {
    let r = r.clamp(0, rows as isize - 1) as usize;
    let c = c.clamp(0, cols as isize - 1) as usize;
    get(dem, rows, cols, r, c)
}

/// Validate DEM dimensions.
fn validate_dem(dem: &[f64], rows: usize, cols: usize) -> SpatialResult<()> {
    if dem.len() != rows * cols {
        return Err(SpatialError::DimensionError(format!(
            "DEM length {} ≠ rows({}) × cols({})",
            dem.len(),
            rows,
            cols
        )));
    }
    if rows < 3 || cols < 3 {
        return Err(SpatialError::ValueError(
            "DEM must be at least 3×3".to_string(),
        ));
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
//  1. Slope
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute **slope** (steepest angle) for each cell of a DEM.
///
/// Uses the 3×3 Horn (1981) finite-difference algorithm:
///
/// ```text
/// dz/dx = ((e + 2f + g) − (a + 2b + c)) / (8 · cell_size)
/// dz/dy = ((c + 2d + e) − (a + 2h + g)) / (8 · cell_size)
/// slope  = atan(√((dz/dx)² + (dz/dy)²))
/// ```
///
/// Neighbour labelling (row above = a,b,c; middle = h,_,f; row below = g,e,d
/// following Horn convention).
///
/// # Arguments
/// * `dem`       – elevation values in **row-major** order (row 0 = north)
/// * `rows`      – number of rows
/// * `cols`      – number of columns
/// * `cell_size` – horizontal cell resolution in the same units as elevation
///
/// # Returns
/// Flat row-major Vec of slope angles in **degrees** [0°, 90°].
pub fn slope(dem: &[f64], rows: usize, cols: usize, cell_size: f64) -> SpatialResult<Vec<f64>> {
    validate_dem(dem, rows, cols)?;
    if cell_size <= 0.0 {
        return Err(SpatialError::ValueError(
            "cell_size must be positive".to_string(),
        ));
    }
    let mut out = vec![0.0_f64; rows * cols];
    let cs8 = 8.0 * cell_size;

    for r in 0..rows {
        for c in 0..cols {
            let ri = r as isize;
            let ci = c as isize;

            // Horn 3×3 kernel  (a b c / h _ f / g e d)
            let a = get_clamped(dem, rows, cols, ri - 1, ci - 1);
            let b = get_clamped(dem, rows, cols, ri - 1, ci);
            let cc = get_clamped(dem, rows, cols, ri - 1, ci + 1);
            let d = get_clamped(dem, rows, cols, ri + 1, ci + 1);
            let e = get_clamped(dem, rows, cols, ri + 1, ci);
            let f = get_clamped(dem, rows, cols, ri, ci + 1);
            let g = get_clamped(dem, rows, cols, ri + 1, ci - 1);
            let h = get_clamped(dem, rows, cols, ri, ci - 1);

            let dzdx = ((cc + 2.0 * f + d) - (a + 2.0 * h + g)) / cs8;
            let dzdy = ((g + 2.0 * e + d) - (a + 2.0 * b + cc)) / cs8;

            let s_rad = (dzdx * dzdx + dzdy * dzdy).sqrt().atan();
            out[r * cols + c] = s_rad * 180.0 / PI;
        }
    }
    Ok(out)
}

// ═══════════════════════════════════════════════════════════════════════════════
//  2. Aspect
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute **aspect** (downslope direction) for each DEM cell.
///
/// Uses the same Horn kernel as [`slope`].
///
/// # Returns
/// Flat Vec of aspect angles in degrees **clockwise from north** [0°, 360°).
/// Flat cells (slope ≈ 0) are assigned `NaN`.
pub fn aspect(dem: &[f64], rows: usize, cols: usize, cell_size: f64) -> SpatialResult<Vec<f64>> {
    validate_dem(dem, rows, cols)?;
    if cell_size <= 0.0 {
        return Err(SpatialError::ValueError(
            "cell_size must be positive".to_string(),
        ));
    }
    let mut out = vec![0.0_f64; rows * cols];
    let cs8 = 8.0 * cell_size;

    for r in 0..rows {
        for c in 0..cols {
            let ri = r as isize;
            let ci = c as isize;

            let a = get_clamped(dem, rows, cols, ri - 1, ci - 1);
            let b = get_clamped(dem, rows, cols, ri - 1, ci);
            let cc = get_clamped(dem, rows, cols, ri - 1, ci + 1);
            let d = get_clamped(dem, rows, cols, ri + 1, ci + 1);
            let e = get_clamped(dem, rows, cols, ri + 1, ci);
            let f = get_clamped(dem, rows, cols, ri, ci + 1);
            let g = get_clamped(dem, rows, cols, ri + 1, ci - 1);
            let h = get_clamped(dem, rows, cols, ri, ci - 1);

            let dzdx = ((cc + 2.0 * f + d) - (a + 2.0 * h + g)) / cs8;
            let dzdy = ((g + 2.0 * e + d) - (a + 2.0 * b + cc)) / cs8;

            if dzdx.abs() < 1e-12 && dzdy.abs() < 1e-12 {
                out[r * cols + c] = f64::NAN;
            } else {
                // Convert from mathematical (east=0°, CCW) to compass (north=0°, CW):
                //   compass = 90° − math_angle
                // For the uphill-facing aspect:
                //   math_angle = atan2(dzdy, dzdx)
                // giving east=90°, north=0° in compass convention.
                let math_asp = dzdy.atan2(dzdx).to_degrees();
                let mut asp = 90.0 - math_asp;
                if asp < 0.0 {
                    asp += 360.0;
                }
                out[r * cols + c] = asp % 360.0;
            }
        }
    }
    Ok(out)
}

// ═══════════════════════════════════════════════════════════════════════════════
//  3. Hillshade
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute analytical **hillshade** for terrain visualisation.
///
/// Hillshade = max(0, cos(z) cos(s) + sin(z) sin(s) cos(Φ − A))
///
/// where z = zenith angle of light source, s = slope, Φ = azimuth of light
/// source, A = aspect.
///
/// # Arguments
/// * `dem`        – elevation grid (row-major)
/// * `rows`, `cols`
/// * `cell_size`  – horizontal resolution (same units as elevation)
/// * `azimuth`    – sun azimuth in degrees, clockwise from north (default 315°)
/// * `altitude`   – sun elevation above horizon in degrees (default 45°)
///
/// # Returns
/// Flat Vec of hillshade values in [0, 255] (byte-range for display).
pub fn hillshade(
    dem: &[f64],
    rows: usize,
    cols: usize,
    cell_size: f64,
    azimuth: f64,
    altitude: f64,
) -> SpatialResult<Vec<f64>> {
    validate_dem(dem, rows, cols)?;
    if cell_size <= 0.0 {
        return Err(SpatialError::ValueError(
            "cell_size must be positive".to_string(),
        ));
    }

    let slope_grid = slope(dem, rows, cols, cell_size)?;
    let aspect_grid = aspect(dem, rows, cols, cell_size)?;

    let zenith = (90.0 - altitude).to_radians();
    let az_rad = azimuth.to_radians();

    let out: Vec<f64> = (0..rows * cols)
        .map(|idx| {
            let s_rad = slope_grid[idx].to_radians();
            let a_val = aspect_grid[idx];
            if a_val.is_nan() {
                return 127.5; // flat → mid-grey
            }
            let asp_rad = a_val.to_radians();
            let hs = 255.0
                * (zenith.cos() * s_rad.cos()
                    + zenith.sin() * s_rad.sin() * (az_rad - asp_rad).cos())
                .max(0.0);
            hs
        })
        .collect();

    Ok(out)
}

// ═══════════════════════════════════════════════════════════════════════════════
//  4. Viewshed
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute a **viewshed** from an observer position.
///
/// Uses a radial line-of-sight algorithm: for each azimuth direction a ray is
/// cast from the observer, propagating outwards, tracking the maximum angle
/// seen so far.  A cell is visible if the angle from the observer to the cell
/// exceeds all prior angles along the ray.
///
/// # Arguments
/// * `dem`          – elevation grid (row-major, north = row 0)
/// * `rows`, `cols`
/// * `cell_size`    – cell resolution in metres (or same unit as elevation)
/// * `obs_row`      – observer row index
/// * `obs_col`      – observer column index
/// * `obs_height`   – observer height above ground (metres)
/// * `max_dist`     – maximum viewing distance in cells (0 = unlimited)
///
/// # Returns
/// Boolean visibility grid (same size as DEM): `true` = visible.
pub fn viewshed(
    dem: &[f64],
    rows: usize,
    cols: usize,
    cell_size: f64,
    obs_row: usize,
    obs_col: usize,
    obs_height: f64,
    max_dist: f64,
) -> SpatialResult<Vec<bool>> {
    validate_dem(dem, rows, cols)?;
    if obs_row >= rows || obs_col >= cols {
        return Err(SpatialError::ValueError(format!(
            "observer ({obs_row},{obs_col}) outside DEM ({rows}×{cols})"
        )));
    }

    let obs_elev = get(dem, rows, cols, obs_row, obs_col) + obs_height;
    let mut visible = vec![false; rows * cols];
    visible[obs_row * cols + obs_col] = true;

    let max_dist_cells = if max_dist <= 0.0 {
        ((rows * rows + cols * cols) as f64).sqrt() + 1.0
    } else {
        max_dist / cell_size
    };

    // Cast rays to every border cell, then to all interior cells via Bresenham
    // For efficiency: iterate over all target cells and trace the line.
    for tr in 0..rows {
        for tc in 0..cols {
            if tr == obs_row && tc == obs_col {
                continue;
            }
            let dr = tr as f64 - obs_row as f64;
            let dc = tc as f64 - obs_col as f64;
            let dist = (dr * dr + dc * dc).sqrt();
            if dist > max_dist_cells {
                continue;
            }

            // Trace ray using Bresenham-like interpolation
            let steps = dist.ceil() as usize;
            let mut max_angle = f64::NEG_INFINITY;
            let mut is_visible = true;

            for s in 1..steps {
                let frac = s as f64 / dist;
                let sr = (obs_row as f64 + frac * dr).round() as isize;
                let sc = (obs_col as f64 + frac * dc).round() as isize;
                if sr < 0 || sr >= rows as isize || sc < 0 || sc >= cols as isize {
                    break;
                }
                let elev = get(dem, rows, cols, sr as usize, sc as usize);
                let horiz_dist = frac * dist * cell_size;
                let angle = (elev - obs_elev) / horiz_dist;
                if angle > max_angle {
                    max_angle = angle;
                }
            }

            // Check the target cell itself
            let target_elev = get(dem, rows, cols, tr, tc);
            let target_hdist = dist * cell_size;
            let target_angle = (target_elev - obs_elev) / target_hdist;
            if target_angle < max_angle {
                is_visible = false;
            }

            visible[tr * cols + tc] = is_visible;
        }
    }

    Ok(visible)
}

// ═══════════════════════════════════════════════════════════════════════════════
//  5. Watershed delineation (D8 flow model)
// ═══════════════════════════════════════════════════════════════════════════════

/// D8 flow direction codes.
///
/// Encoded as the index into the 8 cardinal/diagonal neighbours:
///
/// ```text
/// 7  0  1
/// 6  *  2
/// 5  4  3
/// ```
///
/// A value of `u8::MAX` indicates a sink or depression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlowDirection(pub u8);

impl FlowDirection {
    /// Row offset for each of the 8 D8 directions.
    pub const ROW_OFFSETS: [isize; 8] = [-1, -1, 0, 1, 1, 1, 0, -1];
    /// Column offset for each of the 8 D8 directions.
    pub const COL_OFFSETS: [isize; 8] = [0, 1, 1, 1, 0, -1, -1, -1];
    /// Distance weights (1 for cardinal, √2 for diagonal).
    pub const DIST_WEIGHTS: [f64; 8] = [1.0, std::f64::consts::SQRT_2, 1.0, std::f64::consts::SQRT_2, 1.0, std::f64::consts::SQRT_2, 1.0, std::f64::consts::SQRT_2];
    /// No-data / sink sentinel.
    pub const SINK: u8 = u8::MAX;
}

/// Compute **D8 flow direction** for each DEM cell.
///
/// Each cell flows to the steepest-descent neighbour (maximum drop per unit
/// distance, accounting for diagonal cell distance).  Edge cells flow
/// off-grid (marked as sinks).
///
/// # Returns
/// Vec of [`FlowDirection`] values.
pub fn flow_direction(
    dem: &[f64],
    rows: usize,
    cols: usize,
    cell_size: f64,
) -> SpatialResult<Vec<FlowDirection>> {
    validate_dem(dem, rows, cols)?;
    let mut dir = vec![FlowDirection(FlowDirection::SINK); rows * cols];

    for r in 0..rows {
        for c in 0..cols {
            let elev = get(dem, rows, cols, r, c);
            let mut max_drop = 0.0_f64;
            let mut best_dir = FlowDirection::SINK;

            for d in 0..8usize {
                let nr = r as isize + FlowDirection::ROW_OFFSETS[d];
                let nc = c as isize + FlowDirection::COL_OFFSETS[d];
                if nr < 0 || nr >= rows as isize || nc < 0 || nc >= cols as isize {
                    continue;
                }
                let nelev = get(dem, rows, cols, nr as usize, nc as usize);
                let drop = (elev - nelev) / (FlowDirection::DIST_WEIGHTS[d] * cell_size);
                if drop > max_drop {
                    max_drop = drop;
                    best_dir = d as u8;
                }
            }
            dir[r * cols + c] = FlowDirection(best_dir);
        }
    }
    Ok(dir)
}

/// Compute **flow accumulation** (upstream drainage area in cells).
///
/// Each cell accumulates the count of all upstream cells whose flow paths
/// eventually pass through it.
///
/// # Arguments
/// * `flow_dir` – output of [`flow_direction`]
/// * `rows`, `cols`
///
/// # Returns
/// Vec of flow-accumulation values (usize count of upstream cells including self).
pub fn flow_accumulation(
    flow_dir: &[FlowDirection],
    rows: usize,
    cols: usize,
) -> SpatialResult<Vec<usize>> {
    if flow_dir.len() != rows * cols {
        return Err(SpatialError::DimensionError(
            "flow_dir length must equal rows × cols".to_string(),
        ));
    }

    // Topological sort: compute in-degree (number of upstream neighbours pointing here)
    let mut in_deg = vec![0_usize; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let d = flow_dir[r * cols + c].0;
            if d == FlowDirection::SINK {
                continue;
            }
            let nr = r as isize + FlowDirection::ROW_OFFSETS[d as usize];
            let nc = c as isize + FlowDirection::COL_OFFSETS[d as usize];
            if nr >= 0 && nr < rows as isize && nc >= 0 && nc < cols as isize {
                in_deg[nr as usize * cols + nc as usize] += 1;
            }
        }
    }

    // Start with cells that have no upstream contributors
    let mut queue: std::collections::VecDeque<usize> = (0..rows * cols)
        .filter(|&idx| in_deg[idx] == 0)
        .collect();

    let mut accum = vec![1_usize; rows * cols]; // each cell contributes itself

    while let Some(idx) = queue.pop_front() {
        let r = idx / cols;
        let c = idx % cols;
        let d = flow_dir[idx].0;
        if d == FlowDirection::SINK {
            continue;
        }
        let nr = r as isize + FlowDirection::ROW_OFFSETS[d as usize];
        let nc = c as isize + FlowDirection::COL_OFFSETS[d as usize];
        if nr < 0 || nr >= rows as isize || nc < 0 || nc >= cols as isize {
            continue;
        }
        let nidx = nr as usize * cols + nc as usize;
        accum[nidx] += accum[idx];
        in_deg[nidx] -= 1;
        if in_deg[nidx] == 0 {
            queue.push_back(nidx);
        }
    }

    Ok(accum)
}

/// Result of watershed delineation.
#[derive(Debug, Clone)]
pub struct WatershedResult {
    /// Flow direction for every cell.
    pub flow_dir: Vec<FlowDirection>,
    /// Flow accumulation (upstream area in cells).
    pub flow_accum: Vec<usize>,
    /// Watershed mask: `true` = cell drains to the pour point.
    pub watershed: Vec<bool>,
    /// Pour point (row, col) as supplied.
    pub pour_point: (usize, usize),
}

/// Delineate a **watershed** (catchment area) from a pour point.
///
/// Starting from the pour point, traces all cells whose flow paths eventually
/// lead to that point via [`flow_direction`].
///
/// # Arguments
/// * `dem`             – elevation grid (row-major)
/// * `rows`, `cols`
/// * `cell_size`       – horizontal cell resolution
/// * `pour_row`        – pour point row
/// * `pour_col`        – pour point column
///
/// # Returns
/// [`WatershedResult`] containing flow direction, accumulation, and mask.
pub fn watershed_delineation(
    dem: &[f64],
    rows: usize,
    cols: usize,
    cell_size: f64,
    pour_row: usize,
    pour_col: usize,
) -> SpatialResult<WatershedResult> {
    validate_dem(dem, rows, cols)?;
    if pour_row >= rows || pour_col >= cols {
        return Err(SpatialError::ValueError(format!(
            "pour point ({pour_row},{pour_col}) outside DEM ({rows}×{cols})"
        )));
    }

    let flow_dir = flow_direction(dem, rows, cols, cell_size)?;
    let flow_accum = flow_accumulation(&flow_dir, rows, cols)?;

    // Trace upstream: BFS from pour point, following cells that drain into it.
    let mut watershed = vec![false; rows * cols];
    watershed[pour_row * cols + pour_col] = true;

    // Build a reverse-direction map (which cells flow INTO cell idx?)
    let mut upstream: Vec<Vec<usize>> = vec![Vec::new(); rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let d = flow_dir[r * cols + c].0;
            if d == FlowDirection::SINK {
                continue;
            }
            let nr = r as isize + FlowDirection::ROW_OFFSETS[d as usize];
            let nc = c as isize + FlowDirection::COL_OFFSETS[d as usize];
            if nr >= 0 && nr < rows as isize && nc >= 0 && nc < cols as isize {
                let src = r * cols + c;
                let dst = nr as usize * cols + nc as usize;
                upstream[dst].push(src);
            }
        }
    }

    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    queue.push_back(pour_row * cols + pour_col);

    while let Some(idx) = queue.pop_front() {
        for &up in &upstream[idx] {
            if !watershed[up] {
                watershed[up] = true;
                queue.push_back(up);
            }
        }
    }

    Ok(WatershedResult {
        flow_dir,
        flow_accum,
        watershed,
        pour_point: (pour_row, pour_col),
    })
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple DEM: plane tilted to the east (increasing columns).
    fn flat_slope_east(rows: usize, cols: usize) -> Vec<f64> {
        (0..rows * cols)
            .map(|i| (i % cols) as f64 * 10.0)
            .collect()
    }

    /// Create a simple cone DEM (peak at centre).
    fn cone_dem(rows: usize, cols: usize) -> Vec<f64> {
        let cr = rows as f64 / 2.0;
        let cc = cols as f64 / 2.0;
        (0..rows * cols)
            .map(|i| {
                let r = (i / cols) as f64;
                let c = (i % cols) as f64;
                100.0 - ((r - cr).powi(2) + (c - cc).powi(2)).sqrt()
            })
            .collect()
    }

    #[test]
    fn test_slope_flat_dem() {
        let dem = vec![100.0; 9]; // 3×3 flat
        let s = slope(&dem, 3, 3, 1.0).expect("slope");
        for v in &s {
            assert!(v.abs() < 0.01, "slope should be 0 on flat DEM, got {v}");
        }
    }

    #[test]
    fn test_slope_tilted_east() {
        let dem = flat_slope_east(5, 5);
        let s = slope(&dem, 5, 5, 1.0).expect("slope");
        // Interior cells: dz/dx ≈ 10 / 1 = 10 → slope ≈ atan(10) ≈ 84.3°
        for r in 1..4 {
            for c in 1..4 {
                let v = s[r * 5 + c];
                assert!(v > 50.0, "interior slope must be > 50°, got {v}");
            }
        }
    }

    #[test]
    fn test_aspect_tilted_east() {
        let dem = flat_slope_east(5, 5);
        let a = aspect(&dem, 5, 5, 1.0).expect("aspect");
        // Slope towards east → aspect ≈ 90°
        for r in 1..4 {
            for c in 1..4 {
                let v = a[r * 5 + c];
                // aspect clockwise from north; east = 90
                assert!((v - 90.0).abs() < 10.0, "east-tilted aspect should be ~90°, got {v}");
            }
        }
    }

    #[test]
    fn test_aspect_flat_is_nan() {
        let dem = vec![50.0; 9];
        let a = aspect(&dem, 3, 3, 1.0).expect("aspect");
        // Interior (only 1 interior cell in 3×3 is index 4)
        assert!(a[4].is_nan(), "flat cell aspect should be NaN");
    }

    #[test]
    fn test_hillshade_range() {
        let dem = cone_dem(7, 7);
        let hs = hillshade(&dem, 7, 7, 1.0, 315.0, 45.0).expect("hillshade");
        for &v in &hs {
            assert!(v >= 0.0 && v <= 255.0, "hillshade out of range: {v}");
        }
    }

    #[test]
    fn test_viewshed_flat() {
        // On a flat DEM everything should be visible from centre
        let dem = vec![100.0; 25]; // 5×5
        let vis = viewshed(&dem, 5, 5, 1.0, 2, 2, 2.0, 0.0).expect("viewshed");
        // All 25 cells should be visible
        assert!(vis.iter().all(|&v| v), "all cells should be visible on flat DEM");
    }

    #[test]
    fn test_viewshed_blocking_ridge() {
        // Build a DEM with a high ridge at row 2 blocking rows 3..
        let mut dem = vec![0.0_f64; 25]; // 5×5
        // Ridge at row 2, columns 0-4
        for c in 0..5 {
            dem[2 * 5 + c] = 100.0;
        }
        let vis = viewshed(&dem, 5, 5, 1.0, 0, 2, 1.0, 0.0).expect("viewshed");
        // Observer at (0,2), ridge at row 2; rows 3 & 4 should be hidden
        // (cells directly behind the ridge)
        assert!(!vis[3 * 5 + 2], "cell behind ridge should be hidden");
        assert!(!vis[4 * 5 + 2], "cell far behind ridge should be hidden");
    }

    #[test]
    fn test_flow_direction_simple() {
        // 3×3 DEM: high centre, all edges lower → centre should flow off-centre
        let dem: Vec<f64> = vec![
            1.0, 1.0, 1.0,
            1.0, 5.0, 1.0,
            1.0, 1.0, 1.0,
        ];
        let fd = flow_direction(&dem, 3, 3, 1.0).expect("flow dir");
        // Centre cell should flow somewhere
        assert_ne!(fd[4].0, FlowDirection::SINK, "centre should not be a sink");
    }

    #[test]
    fn test_flow_accumulation_valley() {
        // Linear valley: all cells flow east (right)
        let dem = flat_slope_east(3, 5);
        let fd = flow_direction(&dem, 3, 5, 1.0).expect("flow dir");
        let fa = flow_accumulation(&fd, 3, 5).expect("flow accum");
        // Rightmost column should have highest accumulation
        let max_accum = fa.iter().cloned().max().unwrap_or(0);
        assert!(max_accum >= 3, "flow accumulation should reach at least 3, got {max_accum}");
    }

    #[test]
    fn test_watershed_delineation_cone() {
        let dem = cone_dem(9, 9);
        let result = watershed_delineation(&dem, 9, 9, 1.0, 4, 4).expect("watershed");
        // Pour point at summit: everything should be in the watershed
        // (cone drains outward, so from the peak only the peak itself + immediate)
        // However, we delineate by tracing UPSTREAM from pour point.
        // Since the peak is the high point and flow goes outward, nothing flows INTO it.
        assert!(result.watershed[4 * 9 + 4], "pour point must be in watershed");
        assert_eq!(result.pour_point, (4, 4));
    }

    #[test]
    fn test_watershed_delineation_drain() {
        // flat_slope_east: elevation = col * 10, so water flows WEST (downhill).
        // Pour point at column 0 (lowest elevation) where all columns drain to.
        let dem = flat_slope_east(3, 5);
        // Each row is independent; pour at left edge of middle row.
        let pour_col = 0_usize;
        let pour_row = 1_usize;
        let result =
            watershed_delineation(&dem, 3, 5, 1.0, pour_row, pour_col).expect("watershed");
        let ws_count = result.watershed.iter().filter(|&&v| v).count();
        // Row 1 cells all drain west along row 1 → pour point captures row 1 = 5 cells
        assert!(ws_count >= 3, "watershed should capture ≥3 cells, got {ws_count}");
    }

    #[test]
    fn test_validate_dem_bad_size() {
        let dem = vec![1.0, 2.0]; // too small
        assert!(slope(&dem, 2, 1, 1.0).is_err());
        assert!(slope(&dem, 1, 2, 1.0).is_err());
    }

    #[test]
    fn test_viewshed_invalid_observer() {
        let dem = vec![0.0; 25];
        assert!(viewshed(&dem, 5, 5, 1.0, 10, 0, 0.0, 0.0).is_err());
    }
}
