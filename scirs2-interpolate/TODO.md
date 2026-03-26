# scirs2-interpolate TODO

## v0.3.3 Completed

### 1D Interpolation
- [x] Linear and nearest-neighbor interpolation with boundary handling
- [x] Natural cubic spline with natural, not-a-knot, clamped, periodic BCs
- [x] Akima spline: outlier-robust local construction
- [x] PCHIP (Piecewise Cubic Hermite Interpolating Polynomial): shape- and monotonicity-preserving
- [x] B-splines: arbitrary-order, de Boor evaluation, knot insertion and removal
- [x] NURBS: Non-Uniform Rational B-Splines for exact conics and free-form curves
- [x] Bezier curves: de Casteljau; rational and polynomial
- [x] Tension splines: splines under tension
- [x] Penalized splines (P-splines): regularized B-spline fitting for noisy data
- [x] Monotone constrained splines
- [x] Hermite splines with user-specified derivatives
- [x] Floater-Hormann barycentric rational interpolation (arbitrary blending order d)

### Scattered Data Interpolation
- [x] RBF interpolation: multiquadric, thin-plate spline, Gaussian, inverse multiquadric, linear
- [x] RBF parameter optimization: cross-validation and LOOCV error
- [x] Compactly supported RBF kernels (Wendland, Wu)
- [x] Ordinary Kriging with variogram fitting and prediction variance
- [x] Universal Kriging with polynomial trend
- [x] Indicator Kriging for binary/categorical data
- [x] Bayesian Kriging with uncertainty quantification
- [x] Fast Kriging: local O(k^3), fixed-rank, sparse tapering, HODLR
- [x] Moving Least Squares (MLS): weighted polynomial fitting
- [x] Natural Neighbor (Sibson): Voronoi-based, C1 continuity
- [x] Thin-Plate Spline (TPS): global scattered data interpolant
- [x] Shepard's method: inverse distance weighting; modified Shepard variant
- [x] Scattered 2D interpolation via Delaunay triangulation (linear and cubic)

### Spherical and Parametric
- [x] Spherical harmonic interpolation on the sphere (real harmonics, arbitrary l,m)
- [x] Parametric arc-length curve interpolation of 2D/3D point sequences
- [x] Barycentric coordinates on arbitrary triangulated manifolds

### Multidimensional Grid Interpolation
- [x] Regular grid N-D interpolation (`RegularGridInterpolator`): linear and cubic
- [x] Tensor product interpolation on separable grids
- [x] Bivariate splines: smoothing and interpolating on 2D rectangular grids
- [x] B-spline surface fitting of 3D point clouds

### Adaptive Interpolation
- [x] Error-controlled adaptive refinement: subdivide until local tolerance met
- [x] Hierarchical multi-level sparse-grid construction
- [x] Meshless methods: partition-of-unity and reproducing-kernel particle method

### Performance
- [x] SIMD-accelerated de Boor B-spline evaluation
- [x] SIMD-accelerated pairwise distance computation for RBF
- [x] Parallel batch evaluation using Rayon
- [x] K-d tree for O(log n) nearest-neighbor queries
- [x] Ball tree for metric-space nearest-neighbor queries
- [x] Cache-aware memory access patterns in hot paths

### Bug Fixes (v0.3.1)
- [x] PCHIP extrapolation: switched to linear extension at endpoints to avoid polynomial blow-up (issue #96)
- [x] Bicubic Hermite: corrected 4x4 Hermite matrix transpose
- [x] CubicSpline boundary condition: fixed not-a-knot third-derivative condition

## v0.4.0 Roadmap

### GPU-Accelerated Scattered Data
- [ ] GPU-accelerated RBF solve: CUDA kernel for dense system assembly and direct solve
- [ ] GPU batch evaluation: thousands of query points in parallel for RBF and Kriging
- [ ] GPU-accelerated k-d tree queries for large scattered datasets

### Machine Learning Enhanced Interpolation
- [ ] Physics-Informed interpolation: enforce PDE residuals as constraints
- [ ] Neural-network-enhanced interpolation: learned correction terms on top of RBF
- [ ] Gaussian Process surrogate with automatic kernel structure discovery
- [ ] Deep Kriging: deep neural feature maps combined with Kriging
- [ ] Active learning for adaptive sampling: minimize interpolation error with fewest evaluations

### New Interpolation Methods
- [x] Hermite-Birkhoff interpolation: arbitrary derivative data at arbitrary points — Implemented in v0.4.0
- [x] Polyharmonic splines: higher-order thin-plate generalizations — Implemented in v0.4.0
- [x] Subdivision surfaces: Loop and Catmull-Clark subdivision for smooth surfaces — Implemented in v0.4.0
- [ ] Kernel interpolation on Lie groups and homogeneous spaces

### High-Dimensional and Tensor Methods
- [x] Sparse grid interpolation via Smolyak construction — Implemented in v0.4.0 (`sparse_grid/smolyak.rs`)
- [x] Tensor-train / TT-cross interpolation for very high-dimensional grids — Implemented in v0.4.0 (`tensor_train/` module)
- [ ] ANOVA decomposition for variance-based adaptive sparse grids
- [ ] Anchored-ANOVA: exploit low effective dimensionality

### Approximate and Streaming Interpolation
- [ ] Random feature approximation of RBF (Rahimi-Recht)
- [ ] Nystrom approximation for large Kriging systems
- [ ] Online / streaming interpolation: incremental update without full re-solve
- [ ] Out-of-core interpolation: disk-backed coefficient storage for huge datasets

### Usability and Tooling
- [ ] Automatic method selection: given data size, dimension, and smoothness estimate, recommend best method
- [ ] Extrapolation modes: nearest, linear, polynomial, reflection, periodic
- [ ] Grid resampling utilities: resample scattered data onto regular or irregular grids
- [ ] Symbolic derivative evaluation: analytically differentiate spline representations

## Known Issues

- Kriging with very large nugget values (> 0.5 * signal variance) may produce numerically unstable Cholesky factorizations; increasing nugget regularization is the current workaround.
- Natural Neighbor interpolation does not extrapolate beyond the convex hull of the data; it returns an error for out-of-hull queries.
- NURBS surface fitting is implemented for structured (grid-like) point clouds; unstructured point cloud fitting requires the scattered-2D module.
- Meshless partition-of-unity methods require a minimum patch overlap ratio of 1.5x; smaller overlaps cause oscillation in the unity partition.
- B-spline surface fitting performance degrades for large grids (> 200x200 control points) due to dense system assembly; sparse assembly is planned.
