//! Partition of Unity (PU) Meshless Interpolation
//!
//! The partition-of-unity method decomposes the domain into a set of
//! overlapping patches, fits a *local* RBF inside each patch, and blends the
//! local solutions via smooth weight functions (bump functions).
//!
//! ## Algorithm
//!
//! 1. Cover the point cloud with `k` patches `{Ωⱼ}` (typically axis-aligned
//!    balls or boxes that overlap by a user-specified factor).
//! 2. For each patch `Ωⱼ` collect the subset of data sites `Sⱼ` that fall
//!    inside `Ωⱼ` and fit a local RBF interpolant `sⱼ`.
//! 3. Evaluate at query `x`:
//!    ```text
//!    s(x) = Σⱼ wⱼ(x) sⱼ(x) / Σⱼ wⱼ(x)
//!    ```
//!    where `wⱼ(x) = ψ(‖x − cⱼ‖ / rⱼ)` is a smooth bump function that
//!    vanishes outside `Ωⱼ`.
//!
//! ## Patch selection
//!
//! Patches are generated on a regular grid with spacing `h`, and the radius
//! is set to `overlap * h` so that neighbouring patches overlap.  Adaptive
//! sizing is also supported: the radius of patch `j` can be set to the
//! distance to its `k`-th nearest data site scaled by `overlap`.
//!
//! ## References
//!
//! - Babuška, I. & Melenk, J. M. (1997). The partition of unity method.
//!   *Int. J. Numer. Meth. Eng.* 40, 727-758.
//! - Fasshauer, G. E. (2007). *Meshfree Approximation Methods with Matlab*.
//!   World Scientific.

use crate::error::{InterpolateError, InterpolateResult};
use crate::meshless::rbf_interpolant::{GlobalRbfInterpolant, PolyDegree, PolyharmonicKernel};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

// ---------------------------------------------------------------------------
// Bump (weight) functions
// ---------------------------------------------------------------------------

/// Smooth bump functions for blending local patches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BumpFunction {
    /// Compactly-supported C∞ bump:
    /// w(r) = exp(-1/(1-r²)) for r < 1, 0 otherwise.
    Exponential,
    /// Wendland C² bump: w(r) = (1-r)⁴(4r+1)  for r < 1, 0 otherwise.
    Wendland,
    /// Conical (hat): w(r) = (1-r)  for r < 1, 0 otherwise.
    Conical,
    /// Cubic: w(r) = (1-r)³  for r < 1, 0 otherwise.
    Cubic,
}

impl BumpFunction {
    /// Evaluate at normalised distance `r = dist / radius`.
    #[inline]
    pub fn eval(&self, r: f64) -> f64 {
        if r >= 1.0 {
            return 0.0;
        }
        match self {
            BumpFunction::Exponential => {
                let d = 1.0 - r * r;
                if d <= 0.0 {
                    0.0
                } else {
                    (-1.0 / d).exp()
                }
            }
            BumpFunction::Wendland => {
                let t = 1.0 - r;
                t.powi(4) * (4.0 * r + 1.0)
            }
            BumpFunction::Conical => 1.0 - r,
            BumpFunction::Cubic => {
                let t = 1.0 - r;
                t * t * t
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Patch descriptor
// ---------------------------------------------------------------------------

struct Patch {
    center: Vec<f64>,
    radius: f64,
    /// Local RBF interpolant (None if patch has no data points).
    local_rbf: Option<GlobalRbfInterpolant>,
}

// ---------------------------------------------------------------------------
// Partition-of-Unity Interpolant
// ---------------------------------------------------------------------------

/// Partition-of-unity RBF interpolant.
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::meshless::partition_unity::{
///     PartitionUnityInterpolant, BumpFunction,
/// };
/// use scirs2_core::ndarray::{Array2, Array1};
///
/// // Regular 5×5 grid in [0,1]²
/// let n = 25_usize;
/// let mut pts_data = Vec::with_capacity(n * 2);
/// let mut vals_data = Vec::with_capacity(n);
/// for i in 0..5_usize {
///     for j in 0..5_usize {
///         let x = i as f64 / 4.0;
///         let y = j as f64 / 4.0;
///         pts_data.push(x);
///         pts_data.push(y);
///         vals_data.push(x + y);
///     }
/// }
/// let pts = Array2::from_shape_vec((n, 2), pts_data).expect("doc example: should succeed");
/// let vals = Array1::from_vec(vals_data);
///
/// let interp = PartitionUnityInterpolant::new(
///     &pts.view(), &vals.view(), 4, 1.5, BumpFunction::Wendland,
/// ).expect("doc example: should succeed");
/// let v = interp.evaluate(&[0.5, 0.5]).expect("doc example: should succeed");
/// assert!((v - 1.0).abs() < 1e-6);
/// ```
pub struct PartitionUnityInterpolant {
    patches: Vec<Patch>,
    bump: BumpFunction,
    dim: usize,
    min_coords: Vec<f64>,
    max_coords: Vec<f64>,
}

impl PartitionUnityInterpolant {
    /// Build a PU interpolant using a regular patch grid.
    ///
    /// # Arguments
    ///
    /// * `points`           – `(n, d)` data sites.
    /// * `values`           – `n` function values.
    /// * `patches_per_dim`  – Number of patch centres along each dimension.
    ///                        Total patches ≈ `patches_per_dim^d`.
    /// * `overlap_factor`   – Patch radius = `(domain_size / (patches_per_dim - 1)) * overlap_factor`.
    ///                        Values in `(1.0, 3.0]` are typical.
    /// * `bump`             – Blending weight function.
    pub fn new(
        points: &ArrayView2<f64>,
        values: &ArrayView1<f64>,
        patches_per_dim: usize,
        overlap_factor: f64,
        bump: BumpFunction,
    ) -> InterpolateResult<Self> {
        let n = points.nrows();
        let d = points.ncols();

        if values.len() != n {
            return Err(InterpolateError::DimensionMismatch(format!(
                "points has {n} rows, values has {} entries",
                values.len()
            )));
        }
        if n == 0 {
            return Err(InterpolateError::InsufficientData(
                "PU requires at least one data point".to_string(),
            ));
        }
        if patches_per_dim < 1 {
            return Err(InterpolateError::InvalidInput {
                message: "patches_per_dim must be >= 1".to_string(),
            });
        }
        if overlap_factor <= 0.0 {
            return Err(InterpolateError::InvalidInput {
                message: format!("overlap_factor must be > 0, got {overlap_factor}"),
            });
        }

        let pts_owned = points.to_owned();

        // Compute bounding box
        let mut min_coords = vec![f64::INFINITY; d];
        let mut max_coords = vec![f64::NEG_INFINITY; d];
        for i in 0..n {
            for k in 0..d {
                let v = pts_owned[[i, k]];
                if v < min_coords[k] {
                    min_coords[k] = v;
                }
                if v > max_coords[k] {
                    max_coords[k] = v;
                }
            }
        }

        // Add small margin to avoid numerical issues at boundary
        let margin_factor = 1e-6;
        for k in 0..d {
            let span = max_coords[k] - min_coords[k];
            let margin = span.max(1e-10) * margin_factor;
            min_coords[k] -= margin;
            max_coords[k] += margin;
        }

        // Generate patch centres on a regular grid
        let p = if patches_per_dim == 1 { 1 } else { patches_per_dim };
        let patch_centers = generate_patch_grid(&min_coords, &max_coords, p, d);

        // Compute patch radius (uniform): spacing × overlap
        let spacing = if p <= 1 {
            (0..d)
                .map(|k| max_coords[k] - min_coords[k])
                .fold(0.0_f64, f64::max)
                + 1.0
        } else {
            (0..d)
                .map(|k| (max_coords[k] - min_coords[k]) / (p as f64 - 1.0))
                .fold(0.0_f64, f64::max)
        };
        let radius = spacing * overlap_factor;

        // Build patches
        let mut patches = Vec::with_capacity(patch_centers.len());
        for center in patch_centers {
            let mut local_idx = Vec::new();
            for i in 0..n {
                let dist = (0..d)
                    .map(|k| (pts_owned[[i, k]] - center[k]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if dist < radius {
                    local_idx.push(i);
                }
            }

            let local_rbf = if local_idx.is_empty() {
                None
            } else {
                let lk = local_idx.len();
                let mut lpts = Array2::<f64>::zeros((lk, d));
                let mut lvals = Array1::<f64>::zeros(lk);
                for (row, &gi) in local_idx.iter().enumerate() {
                    for k in 0..d {
                        lpts[[row, k]] = pts_owned[[gi, k]];
                    }
                    lvals[row] = values[gi];
                }

                // Choose polynomial degree based on number of local points
                let deg = if lk >= 1 + d + d * (d + 1) / 2 {
                    PolyDegree::Quadratic
                } else if lk >= 1 + d {
                    PolyDegree::Linear
                } else if lk >= 1 {
                    PolyDegree::Const
                } else {
                    PolyDegree::None
                };

                // Scale local points to improve conditioning
                GlobalRbfInterpolant::new_polyharmonic(
                    &lpts.view(),
                    &lvals.view(),
                    PolyharmonicKernel::ThinPlate,
                    deg,
                )
                .ok()
            };

            patches.push(Patch {
                center,
                radius,
                local_rbf,
            });
        }

        Ok(Self {
            patches,
            bump,
            dim: d,
            min_coords,
            max_coords,
        })
    }

    /// Build a PU interpolant using adaptive patches (one patch per data site).
    ///
    /// Each data site becomes the centre of its own patch; the radius is set to
    /// `overlap_factor` times the distance to the `k_neighbours`-th nearest neighbour.
    ///
    /// # Arguments
    ///
    /// * `points`        – `(n, d)` data sites.
    /// * `values`        – `n` function values.
    /// * `k_neighbours`  – Number of neighbours determining patch radius.
    /// * `overlap_factor` – Radius multiplier (typical: 2-4).
    /// * `bump`          – Blending weight function.
    pub fn new_adaptive(
        points: &ArrayView2<f64>,
        values: &ArrayView1<f64>,
        k_neighbours: usize,
        overlap_factor: f64,
        bump: BumpFunction,
    ) -> InterpolateResult<Self> {
        let n = points.nrows();
        let d = points.ncols();

        if values.len() != n {
            return Err(InterpolateError::DimensionMismatch(format!(
                "points has {n} rows, values has {} entries",
                values.len()
            )));
        }
        if n == 0 {
            return Err(InterpolateError::InsufficientData(
                "PU requires at least one data point".to_string(),
            ));
        }
        let k_nn = k_neighbours.min(n - 1).max(1);

        let pts_owned = points.to_owned();

        // Bounding box
        let mut min_coords = vec![f64::INFINITY; d];
        let mut max_coords = vec![f64::NEG_INFINITY; d];
        for i in 0..n {
            for k in 0..d {
                let v = pts_owned[[i, k]];
                if v < min_coords[k] {
                    min_coords[k] = v;
                }
                if v > max_coords[k] {
                    max_coords[k] = v;
                }
            }
        }

        let mut patches = Vec::with_capacity(n);
        for i in 0..n {
            let center: Vec<f64> = (0..d).map(|k| pts_owned[[i, k]]).collect();

            // Find k-th nearest distance
            let mut dists: Vec<f64> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    (0..d)
                        .map(|k| (pts_owned[[i, k]] - pts_owned[[j, k]]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let kth_dist = dists.get(k_nn - 1).copied().unwrap_or(1.0);
            let radius = kth_dist * overlap_factor;

            // Gather local points within radius
            let mut local_idx: Vec<usize> = (0..n)
                .filter(|&j| {
                    let d_ij = (0..d)
                        .map(|k| (pts_owned[[i, k]] - pts_owned[[j, k]]).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    d_ij < radius
                })
                .collect();
            // Always include the centre
            if !local_idx.contains(&i) {
                local_idx.push(i);
            }

            let lk = local_idx.len();
            let mut lpts = Array2::<f64>::zeros((lk, d));
            let mut lvals = Array1::<f64>::zeros(lk);
            for (row, &gi) in local_idx.iter().enumerate() {
                for k in 0..d {
                    lpts[[row, k]] = pts_owned[[gi, k]];
                }
                lvals[row] = values[gi];
            }

            let deg = if lk >= 1 + d + d * (d + 1) / 2 {
                PolyDegree::Quadratic
            } else if lk >= 1 + d {
                PolyDegree::Linear
            } else if lk >= 1 {
                PolyDegree::Const
            } else {
                PolyDegree::None
            };

            let local_rbf = GlobalRbfInterpolant::new_polyharmonic(
                &lpts.view(),
                &lvals.view(),
                PolyharmonicKernel::ThinPlate,
                deg,
            )
            .ok();

            patches.push(Patch {
                center,
                radius,
                local_rbf,
            });
        }

        Ok(Self {
            patches,
            bump,
            dim: d,
            min_coords,
            max_coords,
        })
    }

    /// Evaluate the PU interpolant at a query point.
    pub fn evaluate(&self, query: &[f64]) -> InterpolateResult<f64> {
        if query.len() != self.dim {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query has {} dims, interpolant has {}",
                query.len(),
                self.dim
            )));
        }

        let mut weighted_sum = 0.0_f64;
        let mut weight_sum = 0.0_f64;

        for patch in &self.patches {
            let dist = (0..self.dim)
                .map(|k| (query[k] - patch.center[k]).powi(2))
                .sum::<f64>()
                .sqrt();

            let w = self.bump.eval(dist / patch.radius);
            if w <= 0.0 {
                continue;
            }

            if let Some(rbf) = &patch.local_rbf {
                let local_val = rbf.evaluate(query)?;
                weighted_sum += w * local_val;
                weight_sum += w;
            }
        }

        if weight_sum <= 0.0 {
            // No patch covers this point → find nearest patch value
            return self.nearest_patch_value(query);
        }

        Ok(weighted_sum / weight_sum)
    }

    fn nearest_patch_value(&self, query: &[f64]) -> InterpolateResult<f64> {
        let (best_patch, _) = self
            .patches
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                let dist = (0..self.dim)
                    .map(|k| (query[k] - p.center[k]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                p.local_rbf.as_ref().map(|_| (i, dist))
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| {
                InterpolateError::ComputationError(
                    "No patch covers query and no nearest patch found".to_string(),
                )
            })?;

        self.patches[best_patch]
            .local_rbf
            .as_ref()
            .ok_or_else(|| {
                InterpolateError::ComputationError(
                    "Best patch has no fitted local RBF (internal invariant violated)".to_string(),
                )
            })?
            .evaluate(query)
    }

    /// Evaluate at multiple query points.
    pub fn evaluate_batch(&self, queries: &ArrayView2<f64>) -> InterpolateResult<Array1<f64>> {
        let nq = queries.nrows();
        let mut out = Array1::<f64>::zeros(nq);
        for i in 0..nq {
            let q: Vec<f64> = (0..queries.ncols()).map(|j| queries[[i, j]]).collect();
            out[i] = self.evaluate(&q)?;
        }
        Ok(out)
    }

    /// Bounding box of the training data `([min], [max])`.
    pub fn bounding_box(&self) -> (&[f64], &[f64]) {
        (&self.min_coords, &self.max_coords)
    }
}

// ---------------------------------------------------------------------------
// Helper: generate regular patch grid
// ---------------------------------------------------------------------------

fn generate_patch_grid(
    min: &[f64],
    max: &[f64],
    p: usize,
    d: usize,
) -> Vec<Vec<f64>> {
    // Iterate over all combinations of d-dimensional grid indices
    let total = p.pow(d as u32);
    let mut centers = Vec::with_capacity(total);

    for flat_idx in 0..total {
        let mut center = Vec::with_capacity(d);
        let mut remaining = flat_idx;
        for k in 0..d {
            let idx = remaining % p;
            remaining /= p;
            let coord = if p == 1 {
                (min[k] + max[k]) * 0.5
            } else {
                min[k] + idx as f64 * (max[k] - min[k]) / (p as f64 - 1.0)
            };
            center.push(coord);
        }
        centers.push(center);
    }
    centers
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{Array1, Array2};

    fn grid_2d(n: usize) -> (Array2<f64>, Array1<f64>) {
        let side = n;
        let total = side * side;
        let mut pts = Array2::<f64>::zeros((total, 2));
        let mut vals = Array1::<f64>::zeros(total);
        let mut row = 0;
        for i in 0..side {
            for j in 0..side {
                let x = i as f64 / (side - 1) as f64;
                let y = j as f64 / (side - 1) as f64;
                pts[[row, 0]] = x;
                pts[[row, 1]] = y;
                vals[row] = x + y;
                row += 1;
            }
        }
        (pts, vals)
    }

    #[test]
    fn test_pu_regular_grid_linear_function() {
        let (pts, vals) = grid_2d(5);
        let interp = PartitionUnityInterpolant::new(
            &pts.view(),
            &vals.view(),
            4,
            1.5,
            BumpFunction::Wendland,
        )
        .expect("test: should succeed");
        let v = interp.evaluate(&[0.5, 0.5]).expect("test: should succeed");
        assert_abs_diff_eq!(v, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_pu_adaptive_linear() {
        let (pts, vals) = grid_2d(4);
        let interp = PartitionUnityInterpolant::new_adaptive(
            &pts.view(),
            &vals.view(),
            3,
            2.0,
            BumpFunction::Cubic,
        )
        .expect("test: should succeed");
        let v = interp.evaluate(&[0.5, 0.5]).expect("test: should succeed");
        assert_abs_diff_eq!(v, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_bump_functions_at_boundary() {
        for bump in [BumpFunction::Exponential, BumpFunction::Wendland, BumpFunction::Conical, BumpFunction::Cubic] {
            assert_eq!(bump.eval(1.0), 0.0, "{bump:?} should vanish at r=1");
            assert_eq!(bump.eval(1.5), 0.0, "{bump:?} should vanish beyond r=1");
            assert!(bump.eval(0.0) > 0.0, "{bump:?} should be positive at r=0");
        }
    }

    #[test]
    fn test_pu_batch_equals_individual() {
        let (pts, vals) = grid_2d(4);
        let interp = PartitionUnityInterpolant::new(
            &pts.view(),
            &vals.view(),
            3,
            1.8,
            BumpFunction::Wendland,
        )
        .expect("test: should succeed");

        let queries = Array2::from_shape_vec(
            (3, 2),
            vec![0.2, 0.3, 0.7, 0.1, 0.4, 0.6],
        )
        .expect("test: should succeed");
        let batch = interp.evaluate_batch(&queries.view()).expect("test: should succeed");
        for i in 0..3 {
            let q = vec![queries[[i, 0]], queries[[i, 1]]];
            let single = interp.evaluate(&q).expect("test: should succeed");
            assert_abs_diff_eq!(batch[i], single, epsilon = 1e-12);
        }
    }
}
