# Interpolation (scirs2-interpolate)

`scirs2-interpolate` provides interpolation and approximation methods modeled after
`scipy.interpolate`, including splines, RBFs, scattered data interpolation, and
tensor-train decompositions.

## 1D Interpolation

### Cubic Splines

```rust,ignore
use scirs2_interpolate::{CubicSpline, InterpolateResult};

// Known data points
let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
let y = vec![0.0, 1.0, 0.0, 1.0, 0.0];

// Fit a cubic spline
let spline = CubicSpline::new(&x, &y)?;

// Evaluate at new points
let y_new = spline.evaluate(1.5)?;
let y_deriv = spline.derivative(1.5, 1)?;  // first derivative
```

### B-Splines

```rust,ignore
use scirs2_interpolate::bspline::{BSpline, make_interp_spline};

// Automatic knot placement
let spline = make_interp_spline(&x, &y, 3)?;  // degree 3

// Evaluate
let y_new = spline.evaluate(1.5)?;

// Integrate the spline over an interval
let integral = spline.integrate(0.0, 4.0)?;
```

## Radial Basis Functions (RBF)

For scattered data interpolation in multiple dimensions:

```rust,ignore
use scirs2_interpolate::rbf::{RBFInterpolator, RBFKernel};

// Scattered 2D data
let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
let values = array![0.0, 1.0, 1.0, 0.5];

// Fit RBF interpolant
let rbf = RBFInterpolator::new(
    &points.view(), &values.view(), RBFKernel::ThinPlateSpline, None
)?;

// Evaluate at query points
let query = array![[0.5, 0.5]];
let result = rbf.evaluate(&query.view())?;
```

### Physics-Informed RBF

RBF interpolation with PDE constraints:

```rust,ignore
use scirs2_interpolate::rbf::PhysicsInformedRBF;

let rbf = PhysicsInformedRBF::new(
    &points.view(), &values.view(),
    pde_operator,  // e.g., Laplacian = 0
    &boundary_points.view(),
    &boundary_values.view(),
)?;
```

## Sparse Grid Interpolation

For high-dimensional problems where tensor-product grids are infeasible:

```rust,ignore
use scirs2_interpolate::sparse_grid::{SparseGridInterpolator, SmolyakGrid};

// Smolyak sparse grid in 5 dimensions
let grid = SmolyakGrid::new(5, 3)?;  // 5 dims, level 3
let interpolator = SparseGridInterpolator::new(grid, &function_values)?;

let result = interpolator.evaluate(&query_point)?;
```

## Tensor-Train Decomposition

For ultra-high-dimensional interpolation via tensor decomposition:

```rust,ignore
use scirs2_interpolate::tensor_train::{TensorTrain, TTCross};

// TT-cross approximation of a high-dimensional function
let tt = TTCross::new(dims, ranks, tolerance)?;
let approximation = tt.approximate(eval_fn)?;

// Evaluate the TT approximation
let value = approximation.evaluate(&point)?;
```

## ANOVA Decomposition

Functional ANOVA for decomposing high-dimensional functions into lower-dimensional
components:

```rust,ignore
use scirs2_interpolate::anova::{ANOVADecomposition, ANOVAOptions};

let anova = ANOVADecomposition::new(eval_fn, n_dims, ANOVAOptions::default())?;
let main_effects = anova.main_effects()?;
let interaction_effects = anova.interaction_effects()?;
```

## SciPy Equivalence Table

| SciPy | SciRS2 |
|-------|--------|
| `scipy.interpolate.CubicSpline` | `CubicSpline::new` |
| `scipy.interpolate.make_interp_spline` | `bspline::make_interp_spline` |
| `scipy.interpolate.RBFInterpolator` | `rbf::RBFInterpolator` |
| `scipy.interpolate.BSpline` | `bspline::BSpline` |
| `scipy.interpolate.interp1d` | `interp1d` |
